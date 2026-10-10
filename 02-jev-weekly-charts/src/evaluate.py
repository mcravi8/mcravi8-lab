"""Holdout and full-sample metrics for zero-shot weekly chart predictions."""
import json
import numpy as np
import pandas as pd
from data import ROOT, TICKERS

LABELS = ["Up", "Flat", "Down"]
P = ["p_up", "p_flat", "p_down"]
RNG_SEED, B = 0, 2000


def onehot(labels):
    return np.array([[l == c for c in LABELS] for l in labels], dtype=float)


def evaluate(df, train, rng):
    y = df.label.to_numpy()
    probs = df[P].to_numpy()
    pred = np.array(LABELS)[probs.argmax(1)]
    hit = (pred == y).astype(float)
    maj_eval = df.label.value_counts().idxmax()
    maj_train = train.label.value_counts().reindex(LABELS).idxmax()
    base_eval = (y == maj_eval).astype(float)
    base_train = (y == maj_train).astype(float)
    freq = train.label.value_counts(normalize=True).reindex(LABELS).fillna(0).to_numpy()
    Y = onehot(y)
    brier_m = ((probs - Y) ** 2).sum(1)
    brier_r = ((freq - Y) ** 2).sum(1)
    pl = df[[c + "_pl" for c in P]].to_numpy()
    hit_pl = (np.array(LABELS)[pl.argmax(1)] == y).astype(float)
    n = len(y)
    idx = rng.integers(0, n, (B, n))

    def ci(base):
        d = hit[idx].mean(1) - base[idx].mean(1)
        return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    ci_eval, ci_train = ci(base_eval), ci(base_train)
    bm, br = float(brier_m.mean()), float(brier_r.mean())
    out = {"n": n, "label_counts": {l: int((y == l).sum()) for l in LABELS},
           "pred_counts": {l: int((pred == l).sum()) for l in LABELS},
           "hit_rate": float(hit.mean()),
           "base_rate_eval_majority": {"class": maj_eval, "rate": float(base_eval.mean())},
           "base_rate_train_majority": {"class": maj_train, "rate": float(base_train.mean())},
           "ci95_hit_minus_eval_base": ci_eval, "ci95_hit_minus_train_base": ci_train,
           "placebo_hit_rate": float(hit_pl.mean()),
           "train_class_freq": dict(zip(LABELS, map(float, freq))),
           "brier_model": bm, "brier_ref": br, "bss": 1 - bm / br,
           "mean_probs": dict(zip(LABELS, map(float, probs.mean(0))))}
    out["success"] = {"ci_excludes_zero_above_eval_base": ci_eval[0] > 0,
                      "ci_excludes_zero_above_train_base": ci_train[0] > 0,
                      "beats_placebo": out["hit_rate"] > out["placebo_hit_rate"],
                      "bss_positive": out["bss"] > 0}
    out["success"]["all_met_eval_base"] = all(out["success"][k] for k in ("ci_excludes_zero_above_eval_base", "beats_placebo", "bss_positive"))
    return out


def main():
    idx = pd.read_csv(ROOT / "charts/index.csv")
    pr = pd.read_csv(ROOT / "results/preds.csv")[["file"] + P]
    pl = pd.read_csv(ROOT / "results/preds_placebo.csv")[["file"] + P].rename(columns={c: c + "_pl" for c in P})
    df = idx.merge(pr, on="file", validate="1:1").merge(pl, on="file", validate="1:1")
    assert len(df) == len(idx)
    rng = np.random.default_rng(RNG_SEED)
    res = {"bootstrap_resamples": B, "seed": RNG_SEED, "tickers": {}}
    for t in TICKERS:
        d = df[df.ticker == t]
        tr = d[d.split == "train"]
        res["tickers"][t] = {"holdout": evaluate(d[d.split == "test"], tr, rng),
                             "full_sample": evaluate(d, tr, rng)}
    (ROOT / "results/metrics.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
