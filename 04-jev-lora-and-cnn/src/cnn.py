"""CNN baseline with canonical splits, provenance and verified saved checkpoints.

Importing this module does not import PyTorch or select a GPU. Training is an
explicit CLI operation; CPU is the default device. Historical results are kept.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from splits import ROOT, load_index, representative_fixtures
from training_utils import (LABELS, PROBS, RELOAD_ATOL, hash_files, runtime, safe_path,
                            sha256, source_hashes, write_json)

SIZE = (160, 120)
SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS, PATIENCE, LR, BATCH = 60, 8, 1e-4, 32
ARCH = "3 x [Conv(5x3) + BatchNorm + LeakyReLU + MaxPool(2,1)] -> AvgPool(5x4) -> Dropout(0.5) -> Linear(3)"


def load_img(path, mirror=False, root=ROOT):
    with Image.open(safe_path(root, str(path))) as source:
        image = source.convert("RGB")
    if mirror:
        image = ImageOps.mirror(image)
    image = image.resize(SIZE, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.


def Net():
    """Construct the network lazily, allowing split/preparation tests without torch."""
    import torch.nn as nn

    class ChartNet(nn.Module):
        def __init__(self):
            super().__init__()

            def block(inputs, outputs):
                return nn.Sequential(nn.Conv2d(inputs, outputs, (5, 3), padding=(2, 1)),
                                     nn.BatchNorm2d(outputs), nn.LeakyReLU(.01), nn.MaxPool2d((2, 1)))

            self.f = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d((5, 4)))
            self.h = nn.Sequential(nn.Flatten(), nn.Dropout(.5), nn.Linear(128 * 20, 3))

        def forward(self, inputs):
            return self.h(self.f(inputs))

    return ChartNet()


def predict(model, images, device=None):
    import torch
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(images), 128):
            tensor = torch.as_tensor(images[start:start + 128], device=device)
            out.append(torch.softmax(model(tensor), 1).cpu().numpy())
    if not out:
        raise ValueError("Cannot score an empty image collection")
    return np.concatenate(out)


def save_checkpoint(model, target: Path, provenance: dict):
    import torch
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.pt")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save({"state_dict": state, "arch": ARCH, "image_size_wh": list(SIZE),
                "labels": LABELS, "provenance": provenance}, temporary)
    temporary.replace(target)
    return sha256(target)


def load_checkpoint(path: Path, device="cpu", expected_sha256=None):
    import torch
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("CNN checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["arch"] != ARCH or checkpoint["image_size_wh"] != list(SIZE) or checkpoint["labels"] != LABELS:
        raise ValueError("CNN checkpoint architecture or preprocessing mismatch")
    model = Net().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "results/revised_training")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--checkpoint", type=Path, help="Reload for inference only; do not train")
    parser.add_argument("--checkpoint-sha256", help="Required checksum for inference-only reload")
    args = parser.parse_args(argv)
    import torch
    environment = runtime()
    if (environment["packages"]["torch"] or "").split("+")[0] != "2.14.1":
        raise ValueError("Install requirements-training-cnn.txt; PyTorch 2.14.1 is required")
    idx = load_index(args.root / "charts/index.csv")
    input_hashes = hash_files(args.root, ["charts/index.csv", *idx.file])
    args.out.mkdir(parents=True, exist_ok=True)
    X = np.stack([load_img(file, root=args.root) for file in idx.file])
    if args.checkpoint:
        if not args.checkpoint_sha256:
            raise ValueError("--checkpoint-sha256 is required for reload")
        model = load_checkpoint(args.checkpoint, args.device, args.checkpoint_sha256)
        pd.DataFrame({"file": idx.file, **dict(zip(PROBS, predict(model, X).T))}).to_csv(
            args.out / "cnn_reloaded_preds.csv", index=False, float_format="%.17g")
        return
    if (args.out / "cnn_training.json").exists():
        raise ValueError("Training results already exist; use a fresh --out directory")
    y = idx.label.map(LABELS.index).to_numpy()
    test = idx.fit_split.eq("test").to_numpy()
    fit, val = idx.fit_split.eq("fit").to_numpy(), idx.fit_split.eq("val").to_numpy()
    Xpl = np.stack([load_img(file, mirror=True, root=args.root) for file in idx.file[test]])
    fixture_files = set(representative_fixtures(idx).file)
    fixture_mask = idx.file.isin(fixture_files).to_numpy()
    provenance = {"source_sha256": source_hashes(ROOT), "input_sha256": input_hashes,
                  "runtime": environment, "device": args.device, "preprocessing": "fixed RGB/255; bilinear160x120",
                  "fit_split": {str(k): int(v) for k, v in idx.fit_split.value_counts().items()}}
    rows, curves, placebo_rows = [], {}, []
    lossf = torch.nn.CrossEntropyLoss()
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = Net().to(args.device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        Xf, yf = torch.tensor(X[fit]), torch.tensor(y[fit])
        Xv, yv = X[val], y[val]
        best, best_state, best_ep, bad, history = np.inf, None, None, 0, []
        started = time.time()
        for epoch in range(MAX_EPOCHS):
            model.train()
            order = torch.randperm(len(Xf))
            running_loss = 0.
            for start in range(0, len(Xf), BATCH):
                batch = order[start:start + BATCH]
                opt.zero_grad()
                loss = lossf(model(Xf[batch].to(args.device)), yf[batch].to(args.device))
                loss.backward()
                opt.step()
                running_loss += loss.item() * len(batch)
            pv, pf = predict(model, Xv), predict(model, X[fit])
            vl = float(-np.log(np.clip(pv[np.arange(len(yv)), yv], 1e-9, 1)).mean())
            fl = float(-np.log(np.clip(pf[np.arange(fit.sum()), y[fit]], 1e-9, 1)).mean())
            history.append({"epoch": epoch + 1, "train_loss_running": running_loss / len(Xf),
                            "fit_logloss_eval": fl, "val_logloss": vl,
                            "val_acc": float((pv.argmax(1) == yv).mean())})
            if vl < best - 1e-4:
                best, best_ep, bad = vl, epoch + 1, 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        before_reload = predict(model, X[fixture_mask])
        checkpoint = args.out / "checkpoints" / f"cnn_seed_{seed}.pt"
        checksum = save_checkpoint(model, checkpoint, {**provenance, "seed": seed, "best_epoch": best_ep})
        model = load_checkpoint(checkpoint, args.device, checksum)
        reload_error = float(np.abs(before_reload - predict(model, X[fixture_mask])).max())
        if reload_error > RELOAD_ATOL:
            raise ValueError(f"CNN reload changes predictions by {reload_error}; outputs blocked")
        curves[seed] = {"best_epoch": best_ep, "best_val_logloss": best, "epochs_run": len(history),
                        "seconds": round(time.time() - started, 1), "history": history,
                        "checkpoint": str(checkpoint.relative_to(args.out)), "checkpoint_sha256": checksum,
                        "reload_max_abs_diff": reload_error}
        p_all, p_pl = predict(model, X), predict(model, Xpl)
        for position, row in enumerate(idx.itertuples()):
            rows.append({"seed": seed, "file": row.file, "ticker": row.ticker, "end_date": row.end_date,
                         "fit_split": row.fit_split, "label": row.label, **dict(zip(PROBS, p_all[position]))})
        placebo_rows.append(pd.DataFrame({"seed": seed, "file": idx.file[test],
                                          **dict(zip([p + "_pl" for p in PROBS], p_pl.T))}))
    df = pd.DataFrame(rows).merge(pd.concat(placebo_rows), on=["seed", "file"], how="left", validate="one_to_one")
    df.to_csv(args.out / "cnn_preds.csv", index=False, float_format="%.17g")
    config = {"image_size_wh": list(SIZE), "arch": ARCH, "optimizer": f"Adam lr={LR}", "batch": BATCH,
              "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "device": args.device, "seeds": SEEDS,
              "n_fit": int(fit.sum()), "n_val": int(val.sum())}
    write_json(args.out / "cnn_training.json", {"config": config, "seeds": curves, "provenance": provenance,
                                              "output_sha256": hash_files(args.out, ["cnn_preds.csv"])})


if __name__ == "__main__":
    main()
