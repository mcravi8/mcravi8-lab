"""Train-only constant, numeric logistic and explicit CPU CNN baselines.

Default CLI mode validates partitions without fitting. Synthetic collections are
only evaluation sets. Features use the visible bars, and the scaler is fitted
inside the logistic training pipeline. Hyperparameters are fixed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image

try:
    from .evaluate import validate_options
except ImportError:  # standalone CLI
    from evaluate import validate_options

EXP = Path(__file__).resolve().parents[2]
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
FEATURE_STEPS = 32
CNN_SIZE = (160, 120)
CNN_ARCH = "3 x [Conv(5x3)+BatchNorm+LeakyReLU+MaxPool(2,1)] -> AvgPool(5x4) -> Dropout(.5) -> Linear(K)"
CNN_EPOCHS, CNN_PATIENCE, CNN_BATCH, CNN_LR = 60, 8, 32, 1e-4


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / str(relative)).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Artifact path escapes the experiment root")
    return path


def training_rows(index):
    """Reject eval-only/synthetic fitting and enforce the frozen real date boundary."""
    required = {"file", "ticker", "end_date", "label", "split"}
    if required - set(index) or index.empty or index.file.duplicated().any():
        raise ValueError("Training requires nonempty unique real chart identities and split fields")
    if not index.split.isin(["train", "test", "gap", "eval"]).all():
        raise ValueError("Unknown train/test split")
    train = index[index.split.eq("train")].copy()
    if train.empty:
        raise ValueError("No training rows: synthetic/evaluation-only data cannot be fitted")
    for field in ("synthetic", "is_synthetic"):
        if field in train and train[field].astype(str).str.lower().isin(["true", "1"]).any():
            raise ValueError("Synthetic data is evaluation-only")
    if train.ticker.astype(str).str.upper().str.startswith("SYNTH").any():
        raise ValueError("Synthetic data is evaluation-only")
    dates = pd.to_datetime(train.end_date, errors="raise")
    if dates.isna().any() or (dates > pd.Timestamp("2018-12-31")).any():
        raise ValueError("Training dates must be known real dates through 2018")
    test = index[index.split.eq("test")]
    if not test.empty:
        test_dates = pd.to_datetime(test.end_date, errors="raise")
        if test_dates.isna().any() or (test_dates < pd.Timestamp("2019-04-01")).any():
            raise ValueError("Real test begins after the three-month gap, on 2019-04-01")
        if "window_start" in test:
            starts = pd.to_datetime(test.window_start, errors="raise")
            if starts.isna().any() or (starts < pd.Timestamp("2019-04-01")).any():
                raise ValueError("Test input windows must start after the three-month gap")
    if train.label.isna().any():
        raise ValueError("Missing training labels")
    return train.sort_values(["end_date", "ticker", "file"]).reset_index(drop=True)


def class_prior(labels, options):
    order = validate_options(options)
    labels = np.asarray(labels, dtype=str)
    if not len(labels) or not np.isin(labels, order).all():
        raise ValueError("Prior needs nonempty training labels from the declared options")
    return np.array([(labels == label).mean() for label in order])


def uniform_predictions(n, options):
    order = validate_options(options)
    if n < 1:
        raise ValueError("Cannot predict an empty evaluation collection")
    return np.full((n, len(order)), 1 / len(order))


def constant_predictions(train_labels, n, options):
    if n < 1:
        raise ValueError("Cannot predict an empty evaluation collection")
    prior = class_prior(train_labels, options)
    majority = np.zeros(len(prior)); majority[prior.argmax()] = 1.
    return {"train_prior": np.tile(prior, (n, 1)), "train_majority": np.tile(majority, (n, 1)),
            "uniform": uniform_predictions(n, options)}


def numeric_features(window, steps=FEATURE_STEPS):
    """Visible OHLC/first close minus one, volume/visible mean; fixed temporal grid."""
    if steps < 2 or window.empty or not set(OHLCV).issubset(window.columns):
        raise ValueError("Expected visible OHLCV bars and at least two feature positions")
    values = window[OHLCV].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("OHLCV must be finite, prices positive, volume nonnegative")
    tolerance = values[:, 1] * 1e-8  # match shared.data adjusted-price roundoff validation
    if (values[:, 1] + tolerance < np.maximum(values[:, 0], values[:, 3])).any() or (values[:, 2] > np.minimum(values[:, 0], values[:, 3]) + tolerance).any():
        raise ValueError("Invalid visible candle ordering")
    normalized = np.column_stack([values[:, :4] / values[0, 3] - 1,
                                  values[:, 4] / max(float(values[:, 4].mean()), 1.)])
    old, new = np.linspace(0, 1, len(values)), np.linspace(0, 1, steps)
    return np.column_stack([np.interp(new, old, normalized[:, j]) for j in range(5)]).ravel()


def load_visible_window(row, root, data_dir=None):
    root = Path(root)
    bars_file = row.get("bars_file", "")
    if isinstance(bars_file, str) and bars_file:
        path = safe_path(root, bars_file)
    else:
        if not row.get("end_date") or not row.get("window_start"):
            raise ValueError("Synthetic/nondated windows require an explicit visible bars_file")
        path = Path(data_dir or EXP / "shared/data") / f"{row['ticker']}.parquet"
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
        for date_column in ("Date", "date"):
            if date_column in frame:
                frame.index = pd.to_datetime(frame.pop(date_column), errors="raise")
                break
    else:
        raise ValueError("Visible bars_file must be CSV or parquet")
    rename = {column: column.capitalize() for column in frame.columns if column.lower() in [x.lower() for x in OHLCV]}
    frame = frame.rename(columns=rename)
    if row.get("end_date") and not row.get("window_start"):
        raise ValueError("Real visible windows require a declared window_start")
    if row.get("end_date") and row.get("window_start"):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("Real bars require unique chronological session dates")
        start, end = pd.Timestamp(row["window_start"]), pd.Timestamp(row["end_date"])
        frame = frame.loc[start:end]
        if frame.empty or frame.index[0].normalize() != start or frame.index[-1].normalize() != end:
            raise ValueError("Visible window does not match its declared start/end session")
    count = row.get("n_bars")
    if count is None or pd.isna(count) or float(count) < 1 or float(count) != int(float(count)) or len(frame) != int(float(count)):
        raise ValueError("Visible bar count differs from n_bars")
    numeric_features(frame)
    return frame[OHLCV]


def fit_logistic(train_features, train_labels, options):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    class_prior(train_labels, options)
    x = np.asarray(train_features, dtype=float)
    if x.ndim != 2 or len(x) != len(train_labels) or not np.isfinite(x).all():
        raise ValueError("Training features must be a finite matrix matching training labels")
    if len(np.unique(train_labels)) < 2:
        raise ValueError("Logistic regression needs at least two observed training classes")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=0))
    model.fit(x, np.asarray(train_labels))
    return model


def logistic_probabilities(model, features, options):
    order = validate_options(options)
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Prediction features must be a finite matrix")
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(order)))
    for i, label in enumerate(model.classes_):
        if label not in order:
            raise ValueError("Trained class is missing from the declared options")
        out[:, order.index(label)] = raw[:, i]
    return out


def split_train_validation(index):
    train = training_rows(index)
    if "window_start" not in train or train.window_start.isna().any() or train.window_start.astype(str).eq("").any():
        raise ValueError("CNN validation requires every training window_start to purge input overlap")
    starts = pd.to_datetime(train.window_start, errors="raise")
    if starts.isna().any() or (starts > pd.to_datetime(train.end_date)).any():
        raise ValueError("Invalid training input window chronology")
    dates = sorted(train.end_date.unique())
    if len(dates) < 3:
        raise ValueError("CNN validation needs at least three distinct training dates")
    n_val = max(1, round(.2 * len(dates)))
    val = train[train.end_date.isin(dates[-n_val:])].copy()
    fit = train[~train.end_date.isin(dates[-n_val:])].copy()
    boundary = pd.to_datetime(val.window_start).min()
    fit = fit[pd.to_datetime(fit.end_date) < boundary].copy()
    if fit.empty or val.empty:
        raise ValueError("Purging overlapping validation input windows leaves an empty partition")
    return fit.reset_index(drop=True), val.reset_index(drop=True)


def Net(n_classes):
    """Experiment 04 architecture, with arbitrary class count; lazy CPU construction."""
    if n_classes < 2:
        raise ValueError("CNN needs at least two classes")
    import torch.nn as nn
    def block(inputs, outputs):
        return nn.Sequential(nn.Conv2d(inputs, outputs, (5, 3), padding=(2, 1)), nn.BatchNorm2d(outputs),
                             nn.LeakyReLU(.01), nn.MaxPool2d((2, 1)))
    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d((5, 4)),
                         nn.Flatten(), nn.Dropout(.5), nn.Linear(128 * 20, n_classes))


def load_images(rows, root):
    images = []
    for file in rows.file:
        with Image.open(safe_path(root, file)) as source:
            image = source.convert("RGB").resize(CNN_SIZE, Image.Resampling.BILINEAR)
        images.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.)
    return np.stack(images)


def cnn_predict(model, images):
    import torch
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(torch.as_tensor(images[start:start + 128], device="cpu")), 1).numpy()
                               for start in range(0, len(images), 128)])


def train_cnn(index, options, root, output):
    """This function is called only by explicit --mode train --models ... cnn."""
    fit, val = split_train_validation(index)  # rejection occurs before importing torch
    order = validate_options(options)
    class_prior(fit.label, order); class_prior(val.label, order)
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = Net(len(order)).to("cpu")
    xf, xv = load_images(fit, root), load_images(val, root)
    yf = torch.tensor([order.index(label) for label in fit.label], device="cpu")
    yv = np.array([order.index(label) for label in val.label])
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR)
    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(CNN_EPOCHS):
        model.train()
        positions = torch.randperm(len(fit))
        for start in range(0, len(fit), CNN_BATCH):
            batch = positions[start:start + CNN_BATCH]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(torch.as_tensor(xf[batch.numpy()], device="cpu")), yf[batch])
            loss.backward(); optimizer.step()
        probabilities = cnn_predict(model, xv)
        loss = float(-np.log(np.clip(probabilities[np.arange(len(val)), yv], 1e-15, 1)).mean())
        history.append({"epoch": epoch + 1, "val_log_loss": loss})
        if loss < best - 1e-4:
            best, bad = loss, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CNN_PATIENCE:
                break
    if best_state is None:
        raise ValueError("CNN did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    before = cnn_predict(model, xv[:min(8, len(val))])
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "cnn.pt"
    torch.save({"state_dict": best_state, "options": order, "arch": CNN_ARCH, "size": list(CNN_SIZE)}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = Net(len(order)).to("cpu"); restored.load_state_dict(saved["state_dict"], strict=True)
    error = float(np.abs(before - cnn_predict(restored, xv[:min(8, len(val))])).max())
    if error > 1e-7:
        raise ValueError("CPU CNN checkpoint reload changes predictions")
    manifest = {"device": "cpu", "seed": 0, "arch": CNN_ARCH, "architecture_source": "04-jev-lora-and-cnn/src/cnn.py",
                "fit_rows": len(fit), "val_rows": len(val), "val_selected_from": "latest 20% of real training dates only; input-window overlap purged",
                "fit_last_date": fit.end_date.max(), "val_first_date": val.end_date.min(), "history": history,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "reload_max_abs_diff": error,
                "fixed_hyperparameters": {"epochs": CNN_EPOCHS, "patience": CNN_PATIENCE, "batch": CNN_BATCH, "lr": CNN_LR}}
    (output / "cnn_training.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return restored


def write_predictions(rows, probabilities, options, target, name, latency_ms=0., synthetic=False):
    order = validate_options(options)
    p = np.asarray(probabilities)
    columns = [column for column in ("file", "ticker", "end_date", "label", "template_id") if column in rows]
    frame = rows[columns].copy()
    frame["probabilities"] = [json.dumps(values.tolist(), separators=(",", ":")) for values in p]
    frame["prediction"] = np.asarray(order)[p.argmax(1)]
    frame["latency_ms"] = latency_ms
    try:
        from .evaluate import validate_frame
    except ImportError:
        from evaluate import validate_frame
    validate_frame(frame, order)
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    metadata = {"options": order, "baseline": name, "synthetic": False, "synthetic_dataset": bool(synthetic),
                "prediction_origin": "baseline", "trained_on": "None; uniform has no fitting" if name == "uniform" else "real split=train through 2018 only",
                "uniform_tie_rule": "First ordered option wins argmax ties; randomized uniform expected accuracy is 1/K",
                "uniform_random_expected_accuracy": 1 / len(order) if name == "uniform" else None,
                "numeric_features": "Visible prices / first visible close - 1; volume / visible mean; 32 evenly spaced positions",
                "logistic_fixed_hyperparameters": {"C": 1., "max_iter": 1000, "random_state": 0},
                "input_rows": len(frame), "predictions_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    target.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--eval-index", type=Path, help="Optional eval-only synthetic index; training still comes from --index")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    option_source = parser.add_mutually_exclusive_group(required=True)
    option_source.add_argument("--options", help="JSON list preserving scorer option order")
    option_source.add_argument("--options-file", type=Path, help="JSON file containing the ordered option list")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=("preflight", "train", "predict"), default="preflight",
                        help="predict only emits uniform probabilities; train explicitly fits requested models")
    parser.add_argument("--models", nargs="+", choices=("constants", "uniform", "logistic", "cnn"), default=["constants", "logistic"])
    args = parser.parse_args()
    options = validate_options(json.loads(args.options_file.read_text() if args.options_file else args.options))
    if args.mode == "predict" and args.models != ["uniform"]:
        parser.error("--mode predict supports --models uniform only; fitting requires --mode train")
    index = pd.read_csv(args.index, keep_default_na=False)
    train = training_rows(index) if any(name != "uniform" for name in args.models) else index.iloc[:0].copy()
    if not train.empty:
        class_prior(train.label, options)
    evaluation = pd.read_csv(args.eval_index, keep_default_na=False) if args.eval_index else index[index.split.isin(["test", "eval"])].copy()
    if evaluation.empty:
        raise ValueError("No evaluation rows")
    if not evaluation.label.isin(options).all():
        raise ValueError("Unknown evaluation labels")
    if args.eval_index and not evaluation.split.eq("eval").all():
        raise ValueError("--eval-index is reserved for eval-only synthetic collections")
    if args.mode == "preflight":
        print(json.dumps({"status": "CPU preflight only; no model fitting", "train_rows": len(train), "eval_rows": len(evaluation), "options": options}))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if list(args.out.glob("*_preds.csv")) or (args.out / "cnn.pt").exists():
        raise ValueError("Baseline artifacts exist; use a fresh output directory")
    root = args.eval_root or args.root
    predictions = constant_predictions(train.label, len(evaluation), options) if "constants" in args.models else {}
    if "uniform" in args.models:
        predictions["uniform"] = uniform_predictions(len(evaluation), options)
    if "logistic" in args.models:
        xtrain = np.stack([numeric_features(load_visible_window(row, args.root, args.data_dir)) for row in train.to_dict("records")])
        xtest = np.stack([numeric_features(load_visible_window(row, root, args.data_dir)) for row in evaluation.to_dict("records")])
        model = fit_logistic(xtrain, train.label, options)
        started = time.perf_counter()
        predictions["logistic"] = logistic_probabilities(model, xtest, options)
        logistic_ms = 1000 * (time.perf_counter() - started) / len(evaluation)
        import joblib
        joblib.dump(model, args.out / "logistic.joblib")
        np.testing.assert_allclose(predictions["logistic"], logistic_probabilities(joblib.load(args.out / "logistic.joblib"), xtest, options), atol=1e-12)
    if "cnn" in args.models:
        model = train_cnn(index, options, args.root, args.out)
        images = load_images(evaluation, root)
        started = time.perf_counter(); predictions["cnn"] = cnn_predict(model, images)
        cnn_ms = 1000 * (time.perf_counter() - started) / len(evaluation)
    for name, probabilities in predictions.items():
        latency = logistic_ms if name == "logistic" else (cnn_ms if name == "cnn" else 0.)
        write_predictions(evaluation, probabilities, options, args.out / f"{name}_preds.csv", name,
                          latency_ms=latency, synthetic=bool(args.eval_index) or bool(evaluation.split.eq("eval").all()))
    provenance = {"options": options, "train_rows": len(train), "eval_rows": len(evaluation),
                  "training_index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
                  "evaluation_index_sha256": hashlib.sha256((args.eval_index or args.index).read_bytes()).hexdigest(),
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "models": list(predictions)}
    (args.out / "baselines_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"models": list(predictions), "train_rows": len(train), "eval_rows": len(evaluation)}))


if __name__ == "__main__":
    main()
