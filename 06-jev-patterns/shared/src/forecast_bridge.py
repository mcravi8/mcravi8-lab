"""CPU-only forecasting bridge from recorded candle-recognition predictions."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

EXP = Path(__file__).resolve().parents[2]
REPO = EXP.parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(REPO))
from shared.src.baselines import numeric_features, logistic_probabilities, load_visible_window
from shared.src.data import load, validate_ohlcv
from shared.src.evaluate import evaluate_arrays, read_predictions
from research.statistics import block_bootstrap_means, percentile_ci, probability_metrics

DEFAULT_PROTOCOL = EXP / "config/forecast_bridge_v1.json"
SHAPE = [f"bar_{position}_{field}" for position in range(32)
         for field in ("open", "high", "low", "close", "volume")]
SUMMARIES = ["close_ret_1", "close_ret_5", "close_ret_10", "close_ret_19",
             "visible_return_std", "visible_range_pct", "visible_volume_ratio_5_20", "visible_drawdown"]
ARMS = ["pooled_train_prior", "uniform", "A_numeric", "B_detector", "C_jev"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_protocol(protocol, root=EXP):
    for relative, expected in protocol["input_sha256"].items():
        path = (Path(root) / relative).resolve()
        if not path.is_relative_to(Path(root).resolve()) or digest(path) != expected:
            raise ValueError(f"Frozen input checksum mismatch: {relative}")
    expected = {}
    for line in (Path(root) / "requirements.txt").read_text().splitlines():
        if "==" in line:
            package, version = line.split("==", 1)
            expected[package] = version
    versions = {package: importlib.metadata.version(package) for package in expected}
    if versions != expected:
        raise ValueError("Installed research dependencies differ from pinned requirements")
    if sys.version_info[:2] != (3, 11):
        raise ValueError("Frozen bridge requires Python 3.11")
    return versions


def align_inputs(index, predictions, options, metadata):
    required = {"file", "ticker", "window_start", "end_date", "label", "split", "bars_file", "n_bars"}
    if required - set(index) or index.empty or index.file.duplicated().any():
        raise ValueError("Canonical market index is incomplete or duplicated")
    if predictions.file.duplicated().any() or set(index.file) != set(predictions.file):
        raise ValueError("Recorded JEV and canonical index must have identical unique file coverage")
    if not index.split.isin(["train", "test"]).all():
        raise ValueError("Only original market train/test candles are eligible")
    if not index.label.isin(options).all() or not pd.to_numeric(index.n_bars).eq(20).all():
        raise ValueError("Unexpected candle label or visible bar count")
    predictions = predictions.set_index("file").loc[index.file].reset_index()
    index = index.reset_index(drop=True)
    for key in required:
        if key not in predictions or not index[key].astype(str).equals(predictions[key].astype(str)):
            raise ValueError(f"Recorded JEV identity differs from canonical index: {key}")
    for key, expected in [("data_origin", "market"), ("evidence_role", "benchmark"),
                          ("source_collection", "candles")]:
        if key not in predictions or not predictions[key].eq(expected).all():
            raise ValueError("Synthetic/showcase or unverified origin cannot enter forecasting bridge")
    if not predictions.accuracy_valid.astype(str).str.lower().eq("true").all():
        raise ValueError("Invalid-label control cannot enter forecasting bridge")
    if metadata.get("synthetic") is not False or metadata.get("prediction_origin") != "jev":
        raise ValueError("Bridge requires recorded real JEV predictions")
    p = np.array([json.loads(value) for value in predictions.probabilities], dtype=float)
    if p.shape != (len(index), len(options)) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid JEV probabilities")
    if not np.allclose(p.sum(1), 1., atol=1e-9, rtol=0):
        raise ValueError("JEV probabilities must sum to one")
    return index, p


def visible_features(window):
    if len(window) != 20:
        raise ValueError("Numeric bridge features require exactly 20 visible bars")
    values = numeric_features(window)
    close = window.Close.to_numpy(float)
    daily = close[1:] / close[:-1] - 1
    summaries = [close[-1] / close[-1-lag] - 1 for lag in (1, 5, 10, 19)]
    summaries += [daily.std(ddof=1), (window.High.max() - window.Low.min()) / close[-1],
                  window.Volume.tail(5).mean() / max(float(window.Volume.mean()), 1.),
                  float((close / np.maximum.accumulate(close) - 1).min())]
    result = np.r_[values, summaries]
    if result.shape != (168,) or not np.isfinite(result).all():
        raise ValueError("Expected finite numeric features")
    return result


def target_at(frame, position, protocol):
    """Trailing scale is causal; target is exactly five later trading sessions."""
    if position < 0 or position >= len(frame):
        raise ValueError("Invalid chart-end position")
    horizon = protocol["target"]["horizon_sessions"]
    count = protocol["target"]["volatility_observations"]
    start = position - count - horizon + 1
    if start < 0:
        return None, "insufficient_trailing_history"
    if position + horizon >= len(frame):
        return None, "missing_forward_target"
    close = frame.Close.to_numpy(float)
    history = close[start:position+1]
    sigma = float(np.std(history[horizon:] / history[:-horizon] - 1, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return None, "nonpositive_trailing_sigma"
    forward = float(close[position+horizon] / close[position] - 1)
    if not np.isfinite(forward):
        raise ValueError("Nonfinite forward return")
    label = "Flat" if abs(forward) < protocol["target"]["flat_band"] * sigma else ("Up" if forward > 0 else "Down")
    return {"target_date": str(frame.index[position+horizon].date()), "forward_return": forward,
            "sigma": sigma, "label": label, "information_start": str(frame.index[start].date())}, None


def detector_module():
    path = EXP / "06a-candles/src/detect.py"
    spec = importlib.util.spec_from_file_location("forecast_bridge_detector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_cohort(index, jev_p, options, raw, protocol, *, verify_visible_files=True):
    """Reproduce causal detector labels and visible inputs before deriving targets."""
    detector = detector_module()
    flags = {}
    for ticker, frame in raw.items():
        validate_ohlcv(frame)
        flags[ticker] = (detector.talib_detector(frame), detector.pandas_detector(frame))
    rows, excluded = [], []
    for number, record in enumerate(index.to_dict("records")):
        frame = raw[record["ticker"]]
        end, start = pd.Timestamp(record["end_date"]), pd.Timestamp(record["window_start"])
        if end not in frame.index:
            raise ValueError("Chart end is not a frozen market session")
        position = int(frame.index.get_loc(end))
        window = frame.iloc[max(0, position-19):position+1]
        if len(window) != 20 or window.index[0] != start:
            raise ValueError("Canonical window does not contain exactly the 20 declared sessions")
        if record["split"] == "train" and end > pd.Timestamp("2018-12-31"):
            raise ValueError("Original training chart crosses frozen boundary")
        if record["split"] == "test" and start < pd.Timestamp("2019-04-01"):
            raise ValueError("Test visible window crosses embargo")
        if verify_visible_files:
            saved = load_visible_window(record, EXP / "06a-candles")
            if not saved.index.equals(window.index) or not np.allclose(saved.to_numpy(), window.to_numpy(), rtol=1e-12, atol=1e-10):
                raise ValueError("Visible numeric bars differ from frozen market snapshot")
        for detected in flags[record["ticker"]]:
            labels = detected.columns[detected.loc[end].to_numpy(bool)].tolist()
            recovered = labels[0] if len(labels) == 1 else ("None" if not labels else None)
            if recovered != record["label"]:
                raise ValueError(f"Causal detector does not reproduce canonical label: {record['file']}")
        numeric = visible_features(window)
        target, reason = target_at(frame, position, protocol)
        if reason is None and record["split"] == "train" and target["target_date"] > "2018-12-31":
            reason = "train_target_after_boundary"
        if reason:
            excluded.append({"file": record["file"], "ticker": record["ticker"], "end_date": record["end_date"],
                             "split": record["split"], "reason": reason})
            continue
        row = {key: record[key] for key in ("file", "ticker", "window_start", "end_date", "split")}
        row.update(pattern_label=record["label"], **target)
        row.update(dict(zip(SHAPE+SUMMARIES, numeric)))
        row.update({f"detector_{j}": float(record["label"] == option) for j, option in enumerate(options)})
        row.update({f"jev_{j}": float(jev_p[number, j]) for j in range(len(options))})
        rows.append(row)
    if not rows:
        raise ValueError("No eligible market observations")
    return pd.DataFrame(rows).sort_values(["end_date", "ticker", "file"]).reset_index(drop=True), excluded


def partitions(frame, protocol):
    first_test = frame.loc[frame.split.eq("test"), "information_start"].min()
    train = frame[frame.split.eq("train") & frame.target_date.lt(first_test)].copy()
    test = frame[frame.split.eq("test")].copy()
    yield "fixed_holdout", "all", train, test
    cfg = protocol["secondary_evaluation"]
    for year in range(cfg["first_test_year"], cfg["last_test_year"]+1):
        test = frame[frame.split.eq("test") & frame.end_date.str.startswith(str(year))].copy()
        if test.empty:
            continue
        train = frame[frame.end_date.lt(f"{year}-01-01") & frame.target_date.lt(test.information_start.min())].copy()
        yield "annual_walkforward", str(year), train, test


def check_partition(train, test, protocol):
    if len(train) < protocol["secondary_evaluation"]["minimum_train_rows"] or test.empty:
        raise ValueError("Insufficient training/evaluation observations")
    options = protocol["target"]["options"]
    if set(train.label) != set(options):
        raise ValueError("Training fold must observe every target class")
    if not set(train.file).isdisjoint(test.file) or train.target_date.max() >= test.information_start.min():
        raise ValueError("Training targets overlap test information history")
    if train.end_date.max() >= test.end_date.min():
        raise ValueError("Chronological forecast partition is invalid")


def feature_sets(options):
    base = SHAPE + SUMMARIES
    return {"A_numeric": base, "B_detector": base + [f"detector_{i}" for i in range(len(options))],
            "C_jev": base + [f"jev_{i}" for i in range(len(options))]}


def forecast_fold(train, test, options, protocol, cache=None, prefix="test"):
    check_partition(train, test, protocol)
    prior = train.label.value_counts(normalize=True).reindex(protocol["target"]["options"], fill_value=0).to_numpy()
    predictions = {"pooled_train_prior": np.tile(prior, (len(test), 1)),
                   "uniform": np.full((len(test), len(prior)), 1/len(prior))}
    model_checks = {}
    for arm, columns in feature_sets(options).items():
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        cfg = protocol["model"]
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=cfg["C"], max_iter=cfg["max_iter"], random_state=cfg["random_seed"],
            class_weight=cfg["class_weight"]))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train[columns].to_numpy(float), train.label.to_numpy())
        p = logistic_probabilities(model, test[columns].to_numpy(float), protocol["target"]["options"])
        scaler = model.steps[0][1]
        np.testing.assert_allclose(scaler.mean_, train[columns].mean().to_numpy(), atol=1e-12, rtol=1e-12)
        check = {"feature_count": len(columns), "features": columns, "iterations": model.steps[1][1].n_iter_.tolist()}
        if cache is not None:
            path = Path(cache) / f"{prefix}_{arm}.joblib"
            joblib.dump(model, path)
            restored = logistic_probabilities(joblib.load(path), test[columns].to_numpy(float), protocol["target"]["options"])
            difference = float(np.abs(restored-p).max())
            if difference > 1e-12:
                raise ValueError("Model checkpoint reload predictions differ")
            check.update(checkpoint_sha256=digest(path), reload_max_abs_diff=difference)
        predictions[arm] = p
        model_checks[arm] = check
    return predictions, model_checks


def paired_difference(left, right, labels, dates, order, protocol):
    lm, rm = probability_metrics(left, labels, order), probability_metrics(right, labels, order)
    cfg = protocol["uncertainty"]
    brier_difference = rm["brier_rows"] - lm["brier_rows"]
    hit_difference = lm["hit_rows"].astype(float) - rm["hit_rows"].astype(float)
    result = {"brier_improvement": float(brier_difference.mean()), "accuracy_improvement": float(hit_difference.mean()),
              "direction": "Positive means left improves over right", "ci95": {}, "date_clusters": int(pd.Series(dates).nunique())}
    for metric, values in [("brier", brier_difference), ("accuracy", hit_difference)]:
        result["ci95"][metric] = {str(length): percentile_ci(block_bootstrap_means(
            values, dates, resamples=cfg["resamples"], seed=cfg["seed"], block_length=length)[:, 0])
            for length in cfg["blocks"]}
    return result


def summarize(predictions, protocol):
    order = protocol["target"]["options"]
    report = {}
    for evaluation, frame in predictions.groupby("evaluation", sort=False):
        arms, cohorts = {}, {}
        for arm, rows in frame.groupby("model", sort=False):
            rows = rows.sort_values("file").reset_index(drop=True)
            p = rows[["p_up", "p_flat", "p_down"]].to_numpy()
            metrics = evaluate_arrays(p, rows.label.to_numpy(), order)
            metrics["per_year"] = {}
            for year, annual in rows.groupby(rows.end_date.str[:4]):
                am = evaluate_arrays(annual[["p_up", "p_flat", "p_down"]].to_numpy(), annual.label.to_numpy(), order)
                metrics["per_year"][year] = {k: am[k] for k in ("n", "accuracy", "macro_f1", "brier")}
            arms[arm] = metrics
            cohorts[arm] = rows
        reference = cohorts["A_numeric"]
        if any(not reference.file.equals(rows.file) or not reference.label.equals(rows.label) for rows in cohorts.values()):
            raise ValueError("Forecast arms do not have identical paired coverage/truth")
        dates = pd.to_datetime(reference.end_date).dt.to_period("W-FRI").dt.end_time.dt.strftime("%Y-%m-%d").to_numpy()
        comparisons = {}
        for left, right in [("B_detector", "A_numeric"), ("C_jev", "A_numeric"),
                            ("A_numeric", "pooled_train_prior"), ("C_jev", "pooled_train_prior")]:
            result = paired_difference(cohorts[left][["p_up", "p_flat", "p_down"]].to_numpy(),
                                       cohorts[right][["p_up", "p_flat", "p_down"]].to_numpy(),
                                       reference.label.to_numpy(), dates, order, protocol)
            result["per_year"] = {year: arms[right]["per_year"][year]["brier"] - arms[left]["per_year"][year]["brier"]
                                  for year in arms[left]["per_year"]}
            result["years_improved"] = sum(x > 0 for x in result["per_year"].values())
            comparisons[f"{left}_vs_{right}"] = result
        report[evaluation] = {"n": len(reference), "forecast_weeks": int(pd.Series(dates).nunique()),
                              "target_counts": reference.label.value_counts().reindex(order, fill_value=0).to_dict(),
                              "arms": arms, "comparisons": comparisons}
    fixed, walk = report["fixed_holdout"], report["annual_walkforward"]
    primary = fixed["comparisons"]["C_jev_vs_A_numeric"]
    low = primary["ci95"]["brier"][str(protocol["uncertainty"]["primary_block"])][0]
    criteria = {"primary_paired_ci_above_zero": low > 0,
                "primary_C_brier_below_train_prior": fixed["arms"]["C_jev"]["brier"] < fixed["arms"]["pooled_train_prior"]["brier"],
                "walkforward_C_vs_A_improvement_positive": walk["comparisons"]["C_jev_vs_A_numeric"]["brier_improvement"] > 0}
    return {"evaluations": report, "continuation": {"criteria": criteria, "all_met": all(criteria.values()),
             "meaning": "Evidence to pursue independent future validation, not a demonstrated trading edge" if all(criteria.values())
                        else "No demonstrated case for more JEV compute under the frozen rule; not proof of no information"}}


def write_report(report, manifest, out):
    lines = ["# CPU forecasting bridge v1", "",
             "This evaluates **next-five-session Up/Flat/Down forecasts**, using recorded JEV pattern probabilities as features. It does not measure pattern-recognition accuracy.",
             "", "The protocol and implementation were committed before this return evaluation. This is a retrospective, pattern-balanced cohort, not independent confirmation or a naturally sampled trading population.",
             "", f"Eligible cohort: **{manifest['eligible_rows']} rows**; {manifest['eligible_by_split']}. Exclusions: {manifest['exclusions_by_reason']}.",
             "", "## Forecast metrics", "",
             "| Evaluation | Model | N | Accuracy | Macro-F1 | Brier ↓ | Log loss ↓ |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for evaluation, result in report["evaluations"].items():
        for arm in ARMS:
            d = result["arms"][arm]
            lines.append(f"| {evaluation} | {arm} | {d['n']} | {d['accuracy']:.2%} | {d['macro_f1']:.3f} | {d['brier']:.6f} | {d['log_loss']:.6f} |")
    lines += ["", "## Paired comparisons", "", "Positive Brier reduction means the left model improves. Intervals use 2,000 circular moving-block draws of four consecutive retained forecast-week clusters; all dates/tickers in a week stay together. Other block lengths and annual results are in metrics.json.",
              "", "| Evaluation | Comparison | Brier reduction | 95% interval | Accuracy change (pp) | Years with positive Brier reduction |",
              "| --- | --- | ---: | --- | ---: | ---: |"]
    for evaluation, result in report["evaluations"].items():
        for name, d in result["comparisons"].items():
            low, high = d["ci95"]["brier"]["4"]
            lines.append(f"| {evaluation} | {name} | {d['brier_improvement']:.6f} | [{low:.6f}, {high:.6f}] | {100*d['accuracy_improvement']:+.2f} | {d['years_improved']}/{len(d['per_year'])} |")
    lines += ["", "## Frozen continuation decision", "",
              "**All continuation criteria met. Requires new future confirmation.**" if report["continuation"]["all_met"] else "**Continuation criteria not met: no demonstrated case for additional JEV compute from this study.**", ""]
    for name, passed in report["continuation"]["criteria"].items():
        lines.append(f"- {name}: {'passed' if passed else 'failed'}")
    lines += ["", "## Interpretation and limitations", "",
              "- A uses 160 normalized visible OHLCV grid features plus eight visible-window summaries. B adds detector-label indicators; C adds the full eight-option JEV definitions probability vector. Separate fixed logistic models forecast returns; JEV was not asked to predict returns here.",
              "- Fixed holdout fits only the original pre-2019 training rows. Annual walk-forward refits on preceding observations; earlier test years become training history. Every training target precedes the earliest test information window, including the trailing target-scale history.",
              "- JEV's definitions arm was selected after reviewing recognition performance; return evaluation choices were frozen before this run. Historical periods and the hand-selected surviving universe have already informed research, so these findings are exploratory. JEV pretraining overlap is unknown.",
              "- Pattern-balanced sampling, sparse year/ticker coverage and limited training size restrict generalization. This is a fixed linear baseline comparison, not a claim about all numeric or vision models. A negative detector result does not logically rule out other JEV features.",
              "- Bootstrap intervals condition on fitted predictions, approximate stationarity and the retained-week block construction. Sparse gaps span more calendar time; intervals have no multiplicity correction. Refit uncertainty is not included.",
              "- Existing v7 probability noise and post-hoc repeatability-policy changes carry forward. No robustness control predictions were added. No costs, execution, turnover or net P&L were tested; adjusted close-to-close targets are a research convention.",
              "- No GPU, Runpod operation, HTTP inference or market-data download was used. Exp05's prospective holdout and frozen models remain unchanged.",
              "", "## Reproduction", "",
              "From the repository root with the ignored frozen market snapshots and visible bars restored:",
              "", "§§§sh", "06-jev-patterns/.venv/bin/python 06-jev-patterns/shared/src/forecast_bridge.py", "§§§", "",
              f"Protocol SHA-256: {manifest['protocol_sha256']}. Code SHA-256: {manifest['source_sha256']['forecast_bridge.py']}.",
              "See [metrics.json](metrics.json), [manifest.json](manifest.json), [folds.json](folds.json), [predictions.csv](predictions.csv), [cohort.csv](cohort.csv), and [exclusions.csv](exclusions.csv)."]
    (Path(out) / "README.md").write_text("\n".join(lines).replace("§", chr(96)) + "\n")


def plot_comparisons(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, (name, result) in zip(axes, report["evaluations"].items()):
        names = ["B_detector_vs_A_numeric", "C_jev_vs_A_numeric", "C_jev_vs_pooled_train_prior"]
        labels = ["Detector vs numeric", "JEV vs numeric", "JEV vs training prior"]
        for y, key in enumerate(names):
            d = result["comparisons"][key]
            lo, hi = d["ci95"]["brier"]["4"]
            ax.plot([lo, hi], [y, y], color="#246293", linewidth=2)
            ax.scatter(d["brier_improvement"], y, color="#246293", zorder=3)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_yticks(range(3), labels); ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Paired Brier reduction (positive is better)")
        ax.invert_yaxis()
    fig.suptitle("Five-session forecasts: exploratory 95% paired block intervals")
    fig.tight_layout()
    fig.savefig(Path(out) / "comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=EXP / "shared/results/forecast_bridge_v1")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    versions = verify_protocol(protocol)
    canonical = pd.read_csv(EXP / "06a-candles/results/index.csv", keep_default_na=False)
    recorded = EXP / "06a-candles/results/inference/frozen_v7/candles_definitions_p1.csv"
    predictions, options, metadata = read_predictions(recorded)
    if metadata["server_identity"]["model_revision"] != protocol["jev_revision"]:
        raise ValueError("JEV revision differs from frozen bridge protocol")
    index, jev_p = align_inputs(canonical, predictions, options, metadata)
    for relative, expected in metadata["image_sha256"].items():
        if digest(EXP / "06a-candles" / relative) != expected:
            raise ValueError("Recorded JEV image hash differs from current input")
    if set(metadata["image_sha256"]) != set(index.file):
        raise ValueError("Recorded JEV image hashes do not cover the entire cohort")
    raw = {ticker: load(ticker) for ticker in sorted(index.ticker.unique())}
    cohort, excluded = build_cohort(index, jev_p, options, raw, protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    cache = EXP / ".cache/forecast_bridge_v1"
    cache.mkdir(parents=True, exist_ok=True)
    outputs, fold_records = [], []
    for evaluation, fold, train, test in partitions(cohort, protocol):
        probabilities, checks = forecast_fold(train, test, options, protocol, cache, f"{evaluation}_{fold}")
        record = {"evaluation": evaluation, "fold": fold, "train_rows": len(train), "test_rows": len(test),
                  "train_chart_end_max": train.end_date.max(), "train_target_date_max": train.target_date.max(),
                  "test_information_start_min": test.information_start.min(), "test_chart_start_min": test.window_start.min(),
                  "training_target_counts": train.label.value_counts().reindex(protocol["target"]["options"], fill_value=0).to_dict(),
                  "models": checks}
        fold_records.append(record)
        for arm, p in probabilities.items():
            saved = test[["file", "ticker", "window_start", "information_start", "end_date",
                          "target_date", "split", "pattern_label", "label", "forward_return", "sigma"]].copy()
            saved["evaluation"], saved["fold"], saved["model"] = evaluation, fold, arm
            saved[["p_up", "p_flat", "p_down"]] = p
            saved["prediction"] = np.array(protocol["target"]["options"])[p.argmax(1)]
            outputs.append(saved)
        print(f"{evaluation} {fold}: {len(train)} training / {len(test)} test; checkpoints verified", flush=True)
    combined = pd.concat(outputs, ignore_index=True)
    report = summarize(combined, protocol)
    paths = {"forecast_bridge.py": Path(__file__), "baselines.py": EXP / "shared/src/baselines.py",
             "evaluate.py": EXP / "shared/src/evaluate.py", "data.py": EXP / "shared/src/data.py",
             "detect.py": EXP / "06a-candles/src/detect.py", "statistics.py": REPO / "research/statistics.py"}
    cohort.to_csv(args.out / "cohort.csv", index=False, float_format="%.17g")
    pd.DataFrame(excluded, columns=["file", "ticker", "end_date", "split", "reason"]).to_csv(args.out / "exclusions.csv", index=False)
    combined.to_csv(args.out / "predictions.csv", index=False, float_format="%.17g")
    manifest = {"status": "complete CPU-only exploratory evaluation", "protocol_sha256": digest(args.protocol),
                "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "input_sha256": protocol["input_sha256"], "source_sha256": {name: digest(path) for name, path in paths.items()},
                "versions": versions, "python": sys.version, "jev_revision": protocol["jev_revision"],
                "jev_revision_source": metadata["server_identity"]["revision_source"], "jev_runtime": metadata["execution"],
                "eligible_rows": len(cohort), "eligible_by_split": cohort.split.value_counts().to_dict(),
                "excluded_rows": len(excluded), "exclusions_by_reason": dict(Counter(row["reason"] for row in excluded)),
                "target_counts_by_split": {split: rows.label.value_counts().reindex(protocol["target"]["options"], fill_value=0).to_dict()
                                          for split, rows in cohort.groupby("split")},
                "gpu_calls": 0, "runpod_operations": 0, "downloads": 0,
                "artifacts": {name: digest(args.out / name) for name in ("cohort.csv", "exclusions.csv", "predictions.csv")}}
    write_json(args.out / "metrics.json", report)
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "folds.json", fold_records)
    write_report(report, manifest, args.out)
    plot_comparisons(report, args.out)
    print(json.dumps({"continuation": report["continuation"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
