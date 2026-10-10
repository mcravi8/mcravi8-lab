"""Frozen numeric shortcut audit for noise-zero pattern versus None templates.

This is a regression guard for easy synthetic separation, not a proof that all
label shortcuts are absent. Features/models are fixed before revised generator
inspection. Independent audit parents are never part of model-scoring data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/synth.py"
CLASSES = ["Head and Shoulders", "Inverse Head and Shoulders", "Double Top", "Double Bottom", "Triangle", "Bull Flag", "None"]
PROTOCOL = {"frozen_at": "2026-10-10 before revised generator inspection", "audit_templates": 1400,
            "generator_seed": 20261017, "split_seed": 1701, "test_fraction": .4,
            "noise_level": 0., "shape_grid_positions": 32,
            "primary_model": {"kind": "StandardScaler + LogisticRegression", "C": 1., "class_weight": "balanced",
                              "max_iter": 1000, "random_state": 1701},
            "forest_diagnostic": {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 10,
                                  "class_weight": "balanced", "random_state": 1701, "n_jobs": 1},
            "primary_gate": "Heldout binary balanced accuracy must be strictly below 0.90",
            "balanced_accuracy_limit": .90,
            "forest_gate": "Heldout binary balanced accuracy must be strictly below 0.90 with the already-frozen depth4 forest",
            "diagnostic_trigger": "Either fixed classifier balanced accuracy >=0.90 blocks finalization",
            "split_unit": "Independent parent template_id and generator seed; no noise/style siblings",
            "stratification": "Seven source classes, each with 120 train and 80 heldout parents",
            "interpretation": "Conditional shortcut regression audit, not a general no-leakage proof. Separation can arise from easy legitimate geometry as well as nuisances."}

CONFIRMATION_PROTOCOL = {**PROTOCOL, "generator_seed": 20261023, "split_seed": 2301,
                         "frozen_at": "2026-10-10 after v2 failure, before v3 classifier evaluation"}
HISTORY = ROOT / "results/leakage_development_history.json"

SHAPE_FEATURES = [f"close_shape_{i:02d}" for i in range(32)] + [
    "turning_fraction", "path_efficiency", "total_variation_over_range", "step_std_over_range",
    "curvature_std_over_range", "step_lag1_correlation", "terminal_motion_over_range"]
NUISANCE_FEATURES = ["n_bars", "first_close", "range_over_first_close", "net_return", "return_mean", "return_std",
                     "mean_body_over_range", "body_std_over_range", "mean_wick_over_range", "wick_std_over_range",
                     "open_gap_over_range", "mean_volume", "volume_cv", "volume_lag1_correlation"]
FEATURES = SHAPE_FEATURES + NUISANCE_FEATURES


def _lag1(values):
    if len(values) < 3 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return 0.
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def numeric_features(frame):
    """Only visible OHLCV, never class/ID/seed/pivots/completion metadata."""
    columns = ["Open", "High", "Low", "Close", "Volume"]
    values = frame[columns].to_numpy(float)
    if len(values) < 6 or not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Audit requires at least six finite positive-price OHLCV bars")
    opening, high, low, close, volume = values.T
    span = max(float(np.ptp(close)), 1e-12)
    steps = np.diff(close)
    returns = steps / close[:-1]
    shape = (close - close.min()) / span
    sampled = np.interp(np.linspace(0, 1, 32), np.linspace(0, 1, len(close)), shape)
    total = max(float(np.abs(steps).sum()), 1e-12)
    body = np.abs(close - opening)
    wick = (high - np.maximum(opening, close)) + (np.minimum(opening, close) - low)
    signs = np.sign(steps)
    changes = float((signs[1:] * signs[:-1] < 0).mean())
    shape_statistics = [changes, abs(float(close[-1] - close[0])) / total, total / span,
                        float(steps.std()) / span, float(np.diff(steps).std()) / span, _lag1(steps),
                        float(close[-1] - close[-6]) / span]
    nuisance = [len(close), close[0], span / close[0], close[-1] / close[0] - 1,
                float(returns.mean()), float(returns.std(ddof=1)), float(body.mean()) / span,
                float(body.std()) / span, float(wick.mean()) / span, float(wick.std()) / span,
                float(np.abs(opening[1:] - close[:-1]).mean()) / span,
                float(volume.mean()), float(volume.std()) / max(float(volume.mean()), 1e-12), _lag1(volume)]
    features = np.r_[sampled, shape_statistics, nuisance]
    if features.shape != (len(FEATURES),) or not np.isfinite(features).all():
        raise ValueError("Invalid numeric audit feature vector")
    return features


def load_generator(path=SOURCE):
    path = Path(path)
    name = "exp06b_audited_generator_" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cohort(generator, protocol, audit_protocol=PROTOCOL):
    if protocol["classes"] != CLASSES:
        raise ValueError("Audit source classes differ from the frozen protocol")
    settings = {**protocol, "seed": audit_protocol["generator_seed"], "clean_templates": audit_protocol["audit_templates"]}
    specs = generator.template_specs(settings)
    if len(specs) != audit_protocol["audit_templates"]:
        raise ValueError("Generator returned the wrong number of audit parents")
    evaluation_specs = generator.template_specs(protocol)
    evaluation_seeds = {int(spec["seed"]) for spec in evaluation_specs}
    audit_seeds = {int(spec["seed"]) for spec in specs}
    if evaluation_seeds & audit_seeds:
        raise ValueError("Audit generator seeds overlap the200-template evaluation cohort")
    namespace = f"audit_seed_{audit_protocol['generator_seed']}"
    rows, features = [], []
    for spec in specs:
        clean, noise, info = generator.make_template(spec, settings)
        visible = generator.apply_noise(clean, noise, 0., info)
        features.append(numeric_features(visible))
        rows.append({"template_id": f"{namespace}:{spec['template_id']}", "seed": int(spec["seed"]), "label": spec["label"]})
    metadata = pd.DataFrame(rows)
    if metadata.template_id.duplicated().any() or metadata.seed.duplicated().any():
        raise ValueError("Audit parents require unique IDs and independently drawn unique seeds")
    metadata.attrs = {"audit_parent_namespace": namespace, "evaluation_generator_seed": int(protocol["seed"]),
                      "evaluation_parents": len(evaluation_specs), "evaluation_seed_overlap": 0,
                      "evaluation_seed_sha256": hashlib.sha256(json.dumps(sorted(evaluation_seeds)).encode()).hexdigest()}
    return np.stack(features), metadata


def split_parents(metadata, audit_protocol=PROTOCOL):
    if {"template_id", "seed", "label"} - set(metadata) or metadata.template_id.duplicated().any() or metadata.seed.duplicated().any():
        raise ValueError("Audit splits require unique parent IDs and seeds")
    from sklearn.model_selection import train_test_split
    positions = np.arange(len(metadata))
    train, test = train_test_split(positions, test_size=audit_protocol["test_fraction"], random_state=audit_protocol["split_seed"],
                                  stratify=metadata.label.to_numpy())
    if set(metadata.iloc[train].template_id) & set(metadata.iloc[test].template_id):
        raise ValueError("Parent-template overlap in audit split")
    if set(metadata.iloc[train].seed) & set(metadata.iloc[test].seed):
        raise ValueError("Parent-seed overlap in audit split")
    return train, test


def _metrics(model, x, y):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
    predicted = model.predict(x)
    probability = model.predict_proba(x)[:, list(model.classes_).index(1)]
    confusion = confusion_matrix(y, predicted, labels=[0, 1])
    return {"accuracy": float((predicted == y).mean()), "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "none_recall": float(confusion[0, 0] / confusion[0].sum()),
            "pattern_recall": float(confusion[1, 1] / confusion[1].sum()),
            "roc_auc": float(roc_auc_score(y, probability)), "confusion_matrix": confusion.tolist(),
            "confusion_order": ["None", "any declared pattern"]}


def audit_features(features, metadata, audit_protocol=PROTOCOL):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = np.asarray(features, dtype=float)
    if x.shape != (len(metadata), len(FEATURES)) or not np.isfinite(x).all():
        raise ValueError("Feature matrix does not match the frozen audit schema")
    y = metadata.label.ne("None").to_numpy(int)
    if set(y) != {0, 1}:
        raise ValueError("Binary audit requires None and pattern parents")
    train, test = split_parents(metadata, audit_protocol)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", max_iter=1000, random_state=1701))
    model.fit(x[train], y[train])
    forest = RandomForestClassifier(**audit_protocol["forest_diagnostic"])
    forest.fit(x[train], y[train])
    primary, diagnostic = _metrics(model, x[test], y[test]), _metrics(forest, x[test], y[test])
    coefficients = model.steps[-1][1].coef_[0]
    top = np.argsort(np.abs(coefficients))[::-1][:10]
    important = np.argsort(forest.feature_importances_)[::-1][:10]
    def identity_hash(positions):
        records = metadata.iloc[positions].sort_values("template_id").to_dict("records")
        return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    primary_passed = primary["balanced_accuracy"] < audit_protocol["balanced_accuracy_limit"]
    forest_passed = diagnostic["balanced_accuracy"] < audit_protocol["balanced_accuracy_limit"]
    return {"primary_gate_passed": primary_passed, "forest_gate_passed": forest_passed,
            "all_gates_passed": primary_passed and forest_passed,
            "diagnostic_near_perfect": diagnostic["balanced_accuracy"] >= audit_protocol["balanced_accuracy_limit"],
            "primary_logistic": primary, "forest_diagnostic": diagnostic,
            "train_counts": {"None": int((y[train] == 0).sum()), "patterns": int((y[train] == 1).sum())},
            "heldout_counts": {"None": int((y[test] == 0).sum()), "patterns": int((y[test] == 1).sum())},
            "source_class_counts": {"train": metadata.iloc[train].label.value_counts().to_dict(),
                                    "heldout": metadata.iloc[test].label.value_counts().to_dict()},
            "parent_template_overlap": 0, "parent_seed_overlap": 0,
            "train_parent_identity_sha256": identity_hash(train), "heldout_parent_identity_sha256": identity_hash(test),
            "all_pattern_dummy_accuracy": float(y[test].mean()), "all_pattern_dummy_balanced_accuracy": .5,
            "standardized_logistic_top_coefficients": [{"feature": FEATURES[i], "coefficient": float(coefficients[i])} for i in top],
            "forest_top_importances": [{"feature": FEATURES[i], "importance": float(forest.feature_importances_[i])} for i in important]}


def run_audit(source=SOURCE, protocol_path=None, audit_protocol=PROTOCOL):
    source = Path(source)
    protocol_path = Path(protocol_path) if protocol_path else ROOT / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    features, metadata = cohort(load_generator(source), protocol, audit_protocol)
    result = audit_features(features, metadata, audit_protocol)
    result.update(metadata.attrs)
    result.update({"generator_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "generator_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                   "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "frozen_audit_protocol": {**audit_protocol, "features": FEATURES},
                   "runtime": {"numpy": np.__version__, "pandas": pd.__version__}})
    import sklearn
    result["runtime"]["scikit_learn"] = sklearn.__version__
    return result


def run_current_audits():
    """Frozen regression cohort plus the predeclared fresh confirmation cohort."""
    result = run_audit()
    result["fresh_confirmation"] = run_audit(audit_protocol=CONFIRMATION_PROTOCOL)
    generator = load_generator()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    seeds = []
    for settings in (PROTOCOL, CONFIRMATION_PROTOCOL):
        specs = generator.template_specs({**protocol, "seed": settings["generator_seed"],
                                          "clean_templates": settings["audit_templates"]})
        seeds.append({item["seed"] for item in specs})
    if seeds[0] & seeds[1]:
        raise ValueError("Regression and confirmation cohorts share parent seeds")
    result["confirmation_regression_seed_overlap"] = 0
    result["regression_gates_passed"] = result["all_gates_passed"]
    result["all_gates_passed"] &= result["fresh_confirmation"]["all_gates_passed"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/leakage_audit.json")
    parser.add_argument("--legacy-source", type=Path)
    parser.add_argument("--legacy-protocol", type=Path)
    parser.add_argument("--legacy-git-ref", help="Optional historical repository reference for the preserved source")
    args = parser.parse_args()
    if bool(args.legacy_source) != bool(args.legacy_protocol):
        parser.error("Legacy source and protocol must both be supplied")
    result = run_current_audits()
    if args.legacy_source:
        result["legacy_comparison"] = run_audit(args.legacy_source, args.legacy_protocol)
        result["legacy_comparison"]["git_reference"] = args.legacy_git_ref
    if HISTORY.exists():
        result["development_history"] = {"file": HISTORY.name,
                                         "sha256": hashlib.sha256(HISTORY.read_bytes()).hexdigest(),
                                         "note": "v2 failure informed final negative-family coverage; both classifiers and gates stayed fixed."}
    result["interpretation"] = PROTOCOL["interpretation"]
    result["model_inference_run"] = False
    result["audit_only_fitting"] = "Fixed CPU shortcut classifiers; not a fitted chart-pattern baseline or a model-scoring split"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("all_gates_passed", "primary_gate_passed", "forest_gate_passed", "diagnostic_near_perfect", "primary_logistic", "forest_diagnostic", "fresh_confirmation")}, indent=2))
    if not result["all_gates_passed"]:
        raise SystemExit("Frozen numeric shortcut regression gate failed")


if __name__ == "__main__":
    main()
