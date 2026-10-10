"""Small CNN baseline in the spirit of Jiang, Kelly, Xiu (2023) on the same chart PNGs.

Train on 'fit' weeks, early-stop on 'val' weeks (last ~20% of train), never tune on
test. 5 seeds. Writes results/cnn_preds.csv (test real + mirrored placebo, plus
val/fit for reference) and results/cnn_training.json (per-epoch losses).
"""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from data import ROOT
from splits import load_index

LABELS = ["Up", "Flat", "Down"]
SIZE = (160, 120)  # W x H, downscaled from 640x480
SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS, PATIENCE, LR, BATCH = 60, 8, 1e-4, 32
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_img(path, mirror=False):
    im = Image.open(ROOT / path).convert("RGB")
    if mirror:
        im = ImageOps.mirror(im)
    im = im.resize(SIZE, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, (5, 3), padding=(2, 1)), nn.BatchNorm2d(o),
                                 nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1)))
        self.f = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d((5, 4)))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(128 * 20, 3))

    def forward(self, x):
        return self.h(self.f(x))


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out.append(torch.softmax(model(torch.tensor(X[i:i + 128], device=DEV)), 1).cpu().numpy())
    return np.concatenate(out)


def main():
    idx = load_index()
    y = idx.label.map(LABELS.index).to_numpy()
    X = np.stack([load_img(f) for f in idx.file])
    test = (idx.split == "test").to_numpy()
    Xpl = np.stack([load_img(f, mirror=True) for f in idx.file[test]])
    fit, val = (idx.fit_split == "fit").to_numpy(), (idx.fit_split == "val").to_numpy()
    print(f"fit {fit.sum()} val {val.sum()} test {test.sum()} device {DEV}")
    rows, curves = [], {}
    lossf = nn.CrossEntropyLoss()
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = Net().to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        Xf, yf = torch.tensor(X[fit]), torch.tensor(y[fit])
        Xv, yv = X[val], y[val]
        best, best_state, bad, hist = np.inf, None, 0, []
        t0 = time.time()
        for ep in range(MAX_EPOCHS):
            model.train()
            perm = torch.randperm(len(Xf))
            tl = 0.0
            for i in range(0, len(Xf), BATCH):
                b = perm[i:i + BATCH]
                opt.zero_grad()
                loss = lossf(model(Xf[b].to(DEV)), yf[b].to(DEV))
                loss.backward(); opt.step()
                tl += loss.item() * len(b)
            pv = predict(model, Xv)
            vl = float(-np.log(np.clip(pv[np.arange(len(yv)), yv], 1e-9, 1)).mean())
            pf = predict(model, X[fit])
            fl = float(-np.log(np.clip(pf[np.arange(fit.sum()), y[fit]], 1e-9, 1)).mean())
            hist.append({"epoch": ep + 1, "train_loss_running": tl / len(Xf), "fit_logloss_eval": fl, "val_logloss": vl,
                         "val_acc": float((pv.argmax(1) == yv).mean())})
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        best_ep = int(np.argmin([h["val_logloss"] for h in hist]) + 1)
        curves[seed] = {"best_epoch": best_ep, "best_val_logloss": best, "epochs_run": len(hist),
                        "seconds": round(time.time() - t0, 1), "history": hist}
        print(f"seed {seed}: best epoch {best_ep}/{len(hist)} val logloss {best:.4f} ({time.time()-t0:.0f}s)")
        p_all, p_pl = predict(model, X), predict(model, Xpl)
        for k, (i, r) in enumerate(idx.iterrows()):
            rows.append({"seed": seed, "file": r.file, "ticker": r.ticker, "end_date": r.end_date,
                         "fit_split": r.fit_split, "label": r.label,
                         "p_up": p_all[k, 0], "p_flat": p_all[k, 1], "p_down": p_all[k, 2]})
        tf = idx.file[test].tolist()
        pl = pd.DataFrame({"seed": seed, "file": tf, "p_up_pl": p_pl[:, 0], "p_flat_pl": p_pl[:, 1], "p_down_pl": p_pl[:, 2]})
        curves[seed]["_pl"] = pl
    df = pd.DataFrame(rows)
    pl = pd.concat([c.pop("_pl") for c in curves.values()])
    df = df.merge(pl, on=["seed", "file"], how="left")
    df.to_csv(ROOT / "results/cnn_preds.csv", index=False, float_format="%.6g")
    cfg = {"image_size_wh": SIZE, "arch": "3 x [Conv(5x3) + BatchNorm + LeakyReLU + MaxPool(2,1)] -> AvgPool(5x4) -> Dropout(0.5) -> Linear(3)",
           "optimizer": f"Adam lr={LR}", "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
           "device": DEV, "seeds": SEEDS, "n_fit": int(fit.sum()), "n_val": int(val.sum())}
    (ROOT / "results/cnn_training.json").write_text(json.dumps({"config": cfg, "seeds": curves}, indent=1) + "\n")


if __name__ == "__main__":
    main()
