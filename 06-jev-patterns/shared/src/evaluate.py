"""Ordered multiclass metrics and paired ticker-quarter/bootstrap comparisons.

Ticker-quarter resampling keeps every row within each asset/calendar-quarter
block together. It does not preserve common shocks across assets or dependence
lasting beyond a quarter. McNemar's row independence assumption is stronger.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.statistics import exact_mcnemar, probability_metrics

REQUIRED = {"file", "ticker", "end_date", "label", "probabilities", "prediction", "latency_ms"}


def validate_options(options):
    if isinstance(options, (str, bytes)):
        raise ValueError("Options must be an ordered collection, not a single string")
    order = list(options)
    if len(order) < 2 or any(not isinstance(x, str) or not x for x in order) or len(set(order)) != len(order):
        raise ValueError("Expected at least two distinct nonempty ordered option strings")
    return order


def evaluate_arrays(probabilities, labels, options):
    """Metrics average over rows; macro-F1 includes every declared class."""
    order = validate_options(options)
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=str)
    basic = probability_metrics(p, y, order)
    calls = p.argmax(axis=1)
    truth = np.array([order.index(label) for label in y])
    confusion = np.zeros((len(order), len(order)), dtype=int)
    np.add.at(confusion, (truth, calls), 1)
    true_positive = confusion.diagonal()
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(len(order), dtype=float), where=denominator != 0)
    confidence, hits = p.max(1), np.asarray(basic["hit_rows"])
    bin_id = np.minimum((confidence * 15).astype(int), 14)
    reliability, ece = [], 0.
    for b in range(15):
        selected = bin_id == b
        count = int(selected.sum())
        accuracy = float(hits[selected].mean()) if count else None
        conf = float(confidence[selected].mean()) if count else None
        reliability.append({"lower": b / 15, "upper": (b + 1) / 15, "n": count,
                            "accuracy": accuracy, "confidence": conf})
        if count:
            ece += count / len(y) * abs(accuracy - conf)
    risk_coverage = {}
    for threshold in (.5, .7, .9):
        selected = confidence >= threshold
        count = int(selected.sum())
        accuracy = float(hits[selected].mean()) if count else None
        risk_coverage[str(threshold)] = {"n": count, "coverage": count / len(y), "accuracy": accuracy,
                                         "risk": 1 - accuracy if count else None}
    return {"n": len(y), "options": order, "accuracy": basic["hit_rate"], "macro_f1": float(f1.mean()),
            "class_f1": dict(zip(order, f1.tolist())), "confusion_matrix": confusion.tolist(),
            "confusion_axes": "rows = truth, columns = prediction; declared option order",
            "brier": basic["brier"], "log_loss": basic["log_loss"], "ece_15": float(ece),
            "reliability": reliability, "risk_coverage": risk_coverage,
            "argmax_tie_rule": "First option in the declared order",
            "absent_class_f1": "Zero; all declared classes enter macro-F1"}


def _probabilities(frame):
    values = [json.loads(value) if isinstance(value, str) else value for value in frame.probabilities]
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Probabilities must contain equal-length JSON numeric arrays") from error


def validate_frame(frame, options):
    order = validate_options(options)
    if frame.empty or REQUIRED - set(frame.columns):
        raise ValueError(f"Empty or incomplete predictions: missing {sorted(REQUIRED - set(frame.columns))}")
    if frame[["file", "label", "prediction"]].isna().any().any() or frame.file.duplicated().any():
        raise ValueError("Prediction identities must be nonmissing and unique")
    if not frame.file.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("File IDs must be nonempty strings")
    p = _probabilities(frame)
    evaluate_arrays(p, frame.label.to_numpy(), order)
    if not np.array_equal(np.asarray(order)[p.argmax(1)], frame.prediction.to_numpy()):
        raise ValueError("Stored prediction differs from the ordered probability argmax")
    latency = pd.to_numeric(frame.latency_ms, errors="raise").to_numpy(float)
    if not np.isfinite(latency).all() or (latency < 0).any():
        raise ValueError("Latency must be finite and nonnegative")
    return p


def bootstrap_groups(frame, *, synthetic=False):
    """Return whole-block row indexes in canonical file order, or an explicit no-CI reason."""
    if synthetic:
        if "template_id" not in frame or frame.template_id.isna().any() or frame.template_id.astype(str).eq("").any():
            return [], "Synthetic evaluation has no complete template_id groups; no CI is reported"
        keys = frame.template_id.astype(str).to_numpy()
        description = "Synthetic template_id blocks; noise/style versions of one template stay together"
    else:
        if frame.ticker.isna().any() or frame.ticker.astype(str).eq("").any():
            raise ValueError("Real bootstrap requires nonmissing ticker identities")
        dates = pd.to_datetime(frame.end_date, errors="raise")
        if dates.isna().any():
            raise ValueError("Real bootstrap requires calendar dates; use --synthetic for generated examples")
        keys = (frame.ticker.astype(str) + ":" + dates.dt.to_period("Q").astype(str)).to_numpy()
        description = "IID ticker-calendar-quarter blocks, every row in the selected block retained"
    groups = [np.flatnonzero(keys == key) for key in sorted(set(keys))]
    groups = [g[np.argsort(frame.file.iloc[g].to_numpy(), kind="stable")] for g in groups]
    return groups, description


def bootstrap_indices(groups, *, resamples=2000, seed=0):
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    if len(groups) < 2:
        return []
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[i] for i in draw]) for draw in rng.integers(0, len(groups), (resamples, len(groups)))]


def _ci(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def _inference(groups, description, resamples, seed):
    return {"block_definition": description, "blocks": len(groups), "resamples": resamples, "seed": seed,
            "status": "Exploratory percentile intervals, conditional on fixed predictions; no refitting",
            "limitations": "Ticker-quarter blocks omit cross-ticker common shocks and serial dependence beyond a quarter. "
                           "Template blocks require independent generator templates. Intervals/tests are unadjusted for multiple comparisons.",
            "no_ci_reason": None if len(groups) >= 2 else (description if not groups else "Fewer than two independent blocks")}


def evaluate_frame(frame, options, *, synthetic=False, bootstrap_resamples=2000, seed=0):
    p = validate_frame(frame, options)
    result = evaluate_arrays(p, frame.label.to_numpy(), options)
    groups, description = bootstrap_groups(frame, synthetic=synthetic)
    samples = bootstrap_indices(groups, resamples=bootstrap_resamples, seed=seed)
    fields = ("accuracy", "macro_f1", "brier", "ece_15")
    draws = [evaluate_arrays(p[s], frame.label.to_numpy()[s], options) for s in samples]
    result["ci95"] = {key: _ci([draw[key] for draw in draws]) if draws else None for key in fields}
    result["inference"] = _inference(groups, description, bootstrap_resamples, seed)
    latency = pd.to_numeric(frame.latency_ms, errors="raise")
    result["latency_ms"] = {"mean": float(latency.mean()), "median": float(latency.median())}
    return result


def compare_arms(left, right, options, *, synthetic=False, bootstrap_resamples=2000, seed=0):
    """Positive differences mean left improves over right; require identical coverage."""
    validate_frame(left, options)
    validate_frame(right, options)
    if set(left.file) != set(right.file):
        raise ValueError("Paired arms must have identical file ID coverage")
    left = left.sort_values("file").reset_index(drop=True)
    right = right.set_index("file").loc[left.file].reset_index()
    for key in ("label", "ticker", "end_date", "template_id"):
        if key in left or key in right:
            if key not in left or key not in right or not left[key].fillna("").equals(right[key].fillna("")):
                raise ValueError(f"Paired arm identities/truth differ: {key}")
    lp, rp = _probabilities(left), _probabilities(right)
    labels = left.label.to_numpy()
    lhs, rhs = evaluate_arrays(lp, labels, options), evaluate_arrays(rp, labels, options)
    order = np.asarray(options)
    lh, rh = order[lp.argmax(1)] == labels, order[rp.argmax(1)] == labels
    wins, losses = int((lh & ~rh).sum()), int((rh & ~lh).sum())
    groups, description = bootstrap_groups(left, synthetic=synthetic)
    samples = bootstrap_indices(groups, resamples=bootstrap_resamples, seed=seed)
    signs = {"accuracy": 1, "macro_f1": 1, "brier": -1, "ece_15": -1}
    differences = {key: sign * (lhs[key] - rhs[key]) for key, sign in signs.items()}
    draws = {key: [] for key in signs}
    for rows in samples:
        lm, rm = evaluate_arrays(lp[rows], labels[rows], options), evaluate_arrays(rp[rows], labels[rows], options)
        for key, sign in signs.items():
            draws[key].append(sign * (lm[key] - rm[key]))
    return {"n": len(left), "direction": "Positive means left improves over right", "differences": differences,
            "ci95": {key: _ci(values) if values else None for key, values in draws.items()},
            "mcnemar": {"left_only_correct": wins, "right_only_correct": losses,
                        "ties": len(left) - wins - losses, "p_two_sided_exact": exact_mcnemar(wins, losses),
                        "assumption": "Independent discordant rows; clustered/dependent rows invalidate the nominal p-value"},
            "inference": _inference(groups, description, bootstrap_resamples, seed)}


def read_predictions(path, metadata_path=None):
    path = Path(path)
    meta_path = Path(metadata_path) if metadata_path else path.with_suffix(".meta.json")
    metadata = json.loads(meta_path.read_text())
    options = validate_options(metadata["options"])
    frame = pd.read_csv(path, keep_default_na=False)
    validate_frame(frame, options)
    return frame, options, metadata


def infer_synthetic_dataset(frame, metadata):
    """Dataset grouping is separate from whether predictions came from a fake server."""
    if "synthetic_dataset" in metadata:
        if not isinstance(metadata["synthetic_dataset"], bool):
            raise ValueError("synthetic_dataset metadata must be boolean")
        return metadata["synthetic_dataset"]
    return frame.end_date.fillna("").astype(str).str.strip().eq("").all().item()


def save_plots(metrics, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(metrics["options"])), metrics["options"], rotation=45, ha="right")
    ax.set_yticks(range(len(metrics["options"])), metrics["options"])
    ax.set(xlabel="Prediction", ylabel="Truth", title="Confusion matrix")
    for i, j in np.ndindex(matrix.shape):
        ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(output / "confusion.png", dpi=150); plt.close(fig)
    bins = [b for b in metrics["reliability"] if b["n"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.scatter([b["confidence"] for b in bins], [b["accuracy"] for b in bins], s=[20 + 100 * b["n"] / metrics["n"] for b in bins])
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean top-class confidence", ylabel="Accuracy", title="Reliability (15 equal-width bins)")
    fig.tight_layout(); fig.savefig(output / "reliability.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for threshold, point in metrics["risk_coverage"].items():
        if point["risk"] is not None:
            ax.scatter(point["coverage"], point["risk"])
            ax.annotate(f"p >= {threshold}", (point["coverage"], point["risk"]))
    ax.set(xlim=(0, 1.02), ylim=(0, 1), xlabel="Coverage", ylabel="Error rate among accepted predictions", title="Risk and coverage")
    fig.tight_layout(); fig.savefig(output / "risk_coverage.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    frame, options, metadata = read_predictions(args.predictions)
    synthetic = args.synthetic or infer_synthetic_dataset(frame, metadata)
    metrics = evaluate_frame(frame, options, synthetic=synthetic, bootstrap_resamples=args.resamples)
    metrics["input_metadata"] = metadata
    metrics["synthetic_dataset"] = synthetic
    fake = metadata.get("prediction_origin") == "fake_server" or (metadata.get("synthetic") is True and metadata.get("prediction_origin") != "baseline")
    metrics["evidence_status"] = "Infrastructure smoke-check predictions; no model-performance evidence" if fake else "Evaluation of recorded predictions"
    if args.compare:
        right, other_options, other_metadata = read_predictions(args.compare)
        if other_options != options:
            raise ValueError("Paired arms use different ordered options")
        if infer_synthetic_dataset(right, other_metadata) != infer_synthetic_dataset(frame, metadata):
            raise ValueError("Paired arms use different dataset grouping metadata")
        metrics["paired_comparison"] = compare_arms(frame, right, options, synthetic=synthetic, bootstrap_resamples=args.resamples)
        metrics["comparison_input_metadata"] = other_metadata
        if other_metadata.get("prediction_origin") == "fake_server" or (other_metadata.get("synthetic") is True and other_metadata.get("prediction_origin") != "baseline"):
            metrics["evidence_status"] = "Infrastructure smoke-check predictions; no model-performance evidence"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    save_plots(metrics, args.out)
    print(json.dumps({key: metrics[key] for key in ("evidence_status", "synthetic_dataset", "n", "accuracy", "macro_f1", "brier", "ece_15")}, indent=2))


if __name__ == "__main__":
    main()
