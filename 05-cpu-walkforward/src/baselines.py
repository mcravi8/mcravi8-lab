"""Fixed CPU baselines with purged annual development folds and a frozen future model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import ROOT
from features import SHAPE, CONTEXT

sys.path.insert(0, str(ROOT.parent))
from research.statistics import paired_inference, probability_metrics

ORDERS = {"ternary_volatility": ["Up", "Flat", "Down"], "binary_direction": ["Up", "Down"]}
LABEL_COLUMN = {"ternary_volatility": "label_ternary", "binary_direction": "label_binary"}
P = ["p_up", "p_flat", "p_down"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def folds(df: pd.DataFrame, protocol: dict):
    """Fit only labels known before the first chart starts in each test year."""
    if df.end_date.max() >= protocol["prospective_first_chart_end"]:
        raise ValueError("Prospective observations cannot enter development evaluation")
    for year in range(protocol["first_development_test_year"], protocol["last_development_test_year"] + 1):
        test = df[df.end_date.str[:4] == str(year)].copy()
        if test.empty:
            continue
        first_input = test.week_start.min()
        train = df[(df.end_date >= f"{year - protocol['train_window_years']}-01-01")
                   & (df.target_date < first_input) & (df.end_date < f"{year}-01-01")].copy()
        if train.end_date.nunique() < protocol["minimum_train_weeks"]:
            continue
        assert train.target_date.max() < test.week_start.min()
        yield year, train, test


def prior(labels: pd.Series, order: list[str]) -> np.ndarray:
    if labels.empty:
        raise ValueError("Cannot fit an empty class prior")
    return labels.value_counts(normalize=True).reindex(order, fill_value=0).to_numpy(float)


def shuffled_labels(train: pd.DataFrame, column: str, seed: int) -> pd.Series:
    """Permute whole date label vectors among dates with identical ticker coverage."""
    rng = np.random.default_rng(seed)
    output = train[column].copy()
    coverage = {}
    for date, rows in train.groupby("end_date", sort=True):
        coverage.setdefault(tuple(sorted(rows.ticker)), []).append(date)
    lookup = train.set_index(["end_date", "ticker"])[column]
    for dates in coverage.values():
        partners = dict(zip(dates, rng.permutation(dates)))
        for date in dates:
            rows = train[train.end_date == date]
            output.loc[rows.index] = [lookup.loc[partners[date], ticker] for ticker in rows.ticker]
    return output


def fit_logistic(train: pd.DataFrame, column: str, features: list[str], protocol: dict, shuffled=False):
    y = shuffled_labels(train, column, protocol["random_seed"]) if shuffled else train[column]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=protocol["logistic_C"],
        max_iter=protocol["logistic_max_iter"], random_state=protocol["random_seed"]))
    model.fit(train[features].to_numpy(), y.to_numpy())
    return model


def predict_ordered(model, test: pd.DataFrame, features: list[str], order: list[str]) -> np.ndarray:
    q = model.predict_proba(test[features].to_numpy())
    return q[:, [list(model.classes_).index(label) for label in order]]


def export_model(model, features: list[str], order: list[str]) -> dict:
    scaler, logit = model.steps[0][1], model.steps[1][1]
    return {"features": features, "order": order, "classes": logit.classes_.tolist(),
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coefficients": logit.coef_.tolist(), "intercept": logit.intercept_.tolist()}


def predict_frozen(payload: dict, frame: pd.DataFrame) -> np.ndarray:
    x = (frame[payload["features"]].to_numpy(float) - payload["mean"]) / payload["scale"]
    z = x @ np.asarray(payload["coefficients"]).T + payload["intercept"]
    if len(payload["classes"]) == 2:
        z = np.column_stack([np.zeros(len(z)), z[:, 0]])
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z); q /= q.sum(axis=1, keepdims=True)
    return q[:, [payload["classes"].index(label) for label in payload["order"]]]


def model_predictions(train: pd.DataFrame, test: pd.DataFrame, task: str, protocol: dict) -> dict:
    order, column = ORDERS[task], LABEL_COLUMN[task]
    pooled = prior(train[column], order)
    predictions = {"pooled_train_prior": np.tile(pooled, (len(test), 1)),
                   "uniform": np.full((len(test), len(order)), 1 / len(order)),
                   "always_up": np.tile([1.] + [0.] * (len(order) - 1), (len(test), 1))}
    by_ticker = {ticker: prior(rows[column], order) for ticker, rows in train.groupby("ticker")}
    predictions["ticker_train_prior"] = np.array([by_ticker.get(t, pooled) for t in test.ticker])
    for name, sign in (("momentum", 1), ("reversal", -1)):
        r = sign * test.week_return.to_numpy()
        choice = np.where(r >= 0, "Up", "Down")
        if task == "ternary_volatility":
            choice = np.where(abs(r) < protocol["flat_band"] * test.sigma.to_numpy(), "Flat", choice)
        predictions[name] = np.array([[float(label == option) for option in order] for label in choice])
    for name, features, shuffled in (("logistic_shape", SHAPE, False),
                                     ("logistic_context", SHAPE + CONTEXT, False),
                                     ("logistic_shuffled_labels", SHAPE + CONTEXT, True)):
        model = fit_logistic(train, column, features, protocol, shuffled)
        probabilities = predict_ordered(model, test, features, order)
        restored = predict_frozen(export_model(model, features, order), test)
        np.testing.assert_allclose(probabilities, restored, atol=1e-12, rtol=1e-12)
        predictions[name] = probabilities
    return predictions


def freeze_prospective(df: pd.DataFrame, protocol: dict) -> dict:
    """Fit once on available past data; no prospective labels or model selection."""
    cutoff = pd.Timestamp(protocol["development_cutoff"])
    start = (cutoff - pd.DateOffset(years=protocol["train_window_years"])).date().isoformat()
    train = df[(df.end_date >= start) & (df.target_date <= str(cutoff.date()))].copy()
    task = protocol["primary_task"]; column = LABEL_COLUMN[task]
    train = train[train[column].notna()]
    features = SHAPE + CONTEXT
    model = fit_logistic(train, column, features, protocol)
    payload = {"protocol_sha256": digest(ROOT / "protocol.json"),
               "features_sha256": digest(ROOT / "data/features.csv"),
               "sklearn_version": sklearn.__version__, "train_start": train.end_date.min(),
               "train_last_target": train.target_date.max(), "train_rows": len(train),
               "train_weeks": int(train.end_date.nunique()), "task": task,
               "comparator_prior": prior(train[column], ORDERS[task]).tolist(),
               "model": export_model(model, features, ORDERS[task])}
    serialized = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    path = ROOT / "models/prospective_context.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists() and path.read_text() != serialized:
        raise ValueError("Frozen prospective model differs; preserve it and version a new protocol before replacement")
    path.write_text(serialized)
    reloaded = json.loads(path.read_text())
    np.testing.assert_allclose(predict_ordered(model, train.head(20), features, ORDERS[task]),
                               predict_frozen(reloaded["model"], train.head(20)), atol=1e-12)
    manifest = {**{k:v for k,v in payload.items() if k != "model"}, "model_file": "models/prospective_context.json",
                "model_sha256": digest(path), "reload_verified": True,
                "prospective_status": "not evaluated: future observations unavailable"}
    (ROOT / "results/frozen_model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summarize(predictions: pd.DataFrame, protocol: dict) -> dict:
    result = {"status": "exploratory development only; no independent confirmation", "tasks": {}}
    for task, order in ORDERS.items():
        d = predictions[predictions.task == task]
        reference = d[d.model == "pooled_train_prior"].set_index("observation_id")
        models = {}
        for name, rows in d.groupby("model", sort=True):
            rows = rows.sort_values(["end_date", "ticker"])
            columns = ["p_" + label.lower() for label in order]
            p = rows[columns].to_numpy()
            labels = rows.label.to_numpy()
            m = probability_metrics(p, labels, order)
            ref_p = reference.loc[rows.observation_id, columns].to_numpy()
            rm = probability_metrics(ref_p, labels, order)
            y = np.array([order.index(label) for label in labels])
            calls = p.argmax(1)
            scalar = {k:v for k,v in m.items() if not k.endswith("_rows")}
            scalar["n"] = len(rows)
            scalar["weeks"] = int(rows.end_date.nunique())
            scalar["balanced_accuracy"] = float(np.mean([(calls[y == i] == i).mean() for i in range(len(order)) if (y == i).any()]))
            scalar["bss_vs_pooled_prior"] = float(1 - np.mean(m["brier_rows"]) / np.mean(rm["brier_rows"]))
            scalar["brier_improvement_inference"] = paired_inference(np.asarray(rm["brier_rows"]) - m["brier_rows"], rows.end_date,
                resamples=protocol["bootstrap_resamples"], seed=protocol["random_seed"],
                block_lengths=protocol["bootstrap_block_lengths_forecast_weeks"])
            scalar["accuracy_improvement_inference"] = paired_inference(np.asarray(m["hit_rows"]) - rm["hit_rows"], rows.end_date,
                resamples=protocol["bootstrap_resamples"], seed=protocol["random_seed"],
                block_lengths=protocol["bootstrap_block_lengths_forecast_weeks"])
            scalar["per_year"] = {}
            for year, annual in rows.groupby("fold"):
                am = probability_metrics(annual[columns].to_numpy(), annual.label.to_numpy(), order)
                arm = probability_metrics(reference.loc[annual.observation_id, columns].to_numpy(), annual.label.to_numpy(), order)
                scalar["per_year"][str(year)] = {**{k:v for k,v in am.items() if not k.endswith("_rows")},
                    "n": len(annual), "brier_improvement": float(arm["brier"] - am["brier"]),
                    "accuracy_improvement": float(am["hit_rate"] - arm["hit_rate"])}
            scalar["years_with_positive_brier_improvement"] = sum(y["brier_improvement"] > 0 for y in scalar["per_year"].values())
            models[name] = scalar
        result["tasks"][task] = models
    return result


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    df = pd.read_csv(ROOT / "data/features.csv")
    if df.target_date.max() > protocol["development_cutoff"]:
        raise ValueError("Future labels cannot enter this study")
    frozen = freeze_prospective(df, protocol)
    records, split_records = [], []
    for task in ORDERS:
        column = LABEL_COLUMN[task]
        eligible = df[df[column].notna()].copy()
        for year, train, test in folds(eligible, protocol):
            predicted = model_predictions(train, test, task, protocol)
            split_records.append({"task": task, "fold": year, "train_rows": len(train), "test_rows": len(test),
                                  "train_weeks": int(train.end_date.nunique()), "test_weeks": int(test.end_date.nunique()),
                                  "last_train_target": train.target_date.max(), "first_test_input": test.week_start.min()})
            for name, p in predicted.items():
                out = test[["observation_id", "ticker", "end_date", "target_date"]].copy()
                out["task"], out["model"], out["fold"], out["label"] = task, name, year, test[column].to_numpy()
                for label in ["Up", "Flat", "Down"]:
                    out["p_" + label.lower()] = p[:, ORDERS[task].index(label)] if label in ORDERS[task] else 0.
                records.append(out)
            print(f"{task} {year}: fit {len(train)}, test {len(test)}; 9 CPU baselines", flush=True)
    pred = pd.concat(records, ignore_index=True)
    path = ROOT / "results/development_predictions.csv.gz"
    pred.to_csv(path, index=False, float_format="%.12g", compression={"method": "gzip", "mtime": 0})
    metrics = summarize(pred, protocol)
    metrics["provenance"] = {"protocol_sha256": digest(ROOT / "protocol.json"),
        "features_sha256": digest(ROOT / "data/features.csv"), "predictions_sha256": digest(path),
        "shared_statistics_sha256": digest(ROOT.parent / "research/statistics.py"),
        "code_sha256": {p.name:digest(p) for p in sorted((ROOT / "src").glob("*.py"))},
        "numpy_version": np.__version__, "pandas_version": pd.__version__, "sklearn_version": sklearn.__version__,
        "frozen_prospective_model": frozen, "splits": split_records}
    (ROOT / "results/metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print("Wrote CPU development metrics; prospective holdout remains unevaluated.")


if __name__ == "__main__":
    main()
