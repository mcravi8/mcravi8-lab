"""Exploratory weekly metrics with training-only references and paired inference."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from data import ROOT, TICKERS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.statistics import (BLOCK_LENGTHS, PRIMARY_BLOCK_LENGTH,
    block_bootstrap_means, exact_mcnemar, inference_metadata, paired_inference,
    percentile_ci, probability_metrics)

LABELS = ["Up", "Flat", "Down"]
P = ["p_up", "p_flat", "p_down"]
RNG_SEED, B = 0, 2000


def onehot(labels):
    return np.array([[label == c for c in LABELS] for label in labels], dtype=float)


def evaluate(df, train, rng, pooled_train=None):
    y, dates = df.label.to_numpy(), df.end_date
    model = probability_metrics(df[P].to_numpy(), y)
    placebo = probability_metrics(df[[c + "_pl" for c in P]].to_numpy(), y)
    hit, hp = model["hit_rows"], placebo["hit_rows"]
    freq = train.label.value_counts(normalize=True).reindex(LABELS).fillna(0).to_numpy()
    pooled_train = train if pooled_train is None else pooled_train
    pooled = pooled_train.label.value_counts(normalize=True).reindex(LABELS).fillna(0).to_numpy()
    maj_train = LABELS[int(freq.argmax())]
    counts = df.label.value_counts().reindex(LABELS).fillna(0)
    maj_eval = LABELS[int(counts.to_numpy().argmax())]
    base_train, base_eval = (y == maj_train).astype(float), (y == maj_eval).astype(float)
    ref = probability_metrics(np.tile(freq, (len(df), 1)), y)
    bm, br = model["brier_rows"], ref["brier_rows"]
    accuracy = paired_inference(hit - base_train, dates, resamples=B, seed=RNG_SEED)
    place = paired_inference(hit - hp, dates, resamples=B, seed=RNG_SEED)
    brier = paired_inference(br - bm, dates, resamples=B, seed=RNG_SEED)
    bss_ci = {}
    for length in BLOCK_LENGTHS:
        means = block_bootstrap_means(np.column_stack([bm, br]), dates,
            resamples=B, seed=RNG_SEED, block_length=length)
        valid = means[:, 1] > 0
        bss_ci[str(length)] = percentile_ci(1 - means[valid, 0] / means[valid, 1]) if valid.any() else None
    baselines = {}
    for name, q in (("ticker_train_frequency", freq), ("pooled_train_frequency", pooled),
                    ("uniform", np.full(3, 1/3)), ("always_up", np.array([1., 0., 0.]))):
        constant = probability_metrics(np.tile(q, (len(df), 1)), y)
        baselines[name] = {"hit_rate": constant["hit_rate"], "brier": constant["brier"],
            "log_loss": constant["log_loss"],
            "information": "ticker identity and training labels" if name.startswith("ticker") else "training labels only; no chart or ticker identity",
            "paired_accuracy": paired_inference(hit - constant["hit_rows"], dates, resamples=B, seed=RNG_SEED),
            "paired_brier_improvement": paired_inference(constant["brier_rows"] - bm, dates, resamples=B, seed=RNG_SEED)}
    primary = str(PRIMARY_BLOCK_LENGTH)
    wins, losses = int(((hit == 1) & (hp == 0)).sum()), int(((hit == 0) & (hp == 1)).sum())
    out = {"n": len(df), "weeks": int(df.end_date.nunique()),
        "label_counts": {label: int((y == label).sum()) for label in LABELS},
        "pred_counts": {label: int((np.array(LABELS)[df[P].to_numpy().argmax(1)] == label).sum()) for label in LABELS},
        "hit_rate": model["hit_rate"], "log_loss": model["log_loss"],
        "base_rate_eval_majority": {"class": maj_eval, "rate": float(base_eval.mean())},
        "base_rate_train_majority": {"class": maj_train, "rate": float(base_train.mean())},
        "ci95_hit_minus_eval_base": None,
        "oracle_ci_note": "Evaluation-selected majority is descriptive; no post-selection interval reported",
        "ci95_hit_minus_train_base": accuracy["ci95_by_block_length"][primary],
        "placebo_hit_rate": placebo["hit_rate"],
        "train_class_freq": dict(zip(LABELS, map(float, freq))),
        "brier_model": model["brier"], "brier_ref": ref["brier"], "bss": float(1 - bm.mean() / br.mean()) if br.mean() > 0 else None,
        "ci95_bss": bss_ci[primary], "ci95_bss_by_block_length": bss_ci,
        "ci95_hit_minus_placebo": place["ci95_by_block_length"][primary],
        "paired_inference": {"accuracy_vs_train_majority": accuracy,
                             "accuracy_vs_placebo": place, "brier_vs_train_frequency": brier},
        "placebo_discordant_pairs": {"model_wins": wins, "placebo_wins": losses,
            "ties": len(df) - wins - losses, "exact_mcnemar_p_if_independent_rows": exact_mcnemar(wins, losses)},
        "baselines": baselines, "mean_probs": dict(zip(LABELS, map(float, df[P].mean()))),
        "inference": inference_metadata()}
    out["success"] = {"ci_excludes_zero_above_eval_base": None,
        "ci_excludes_zero_above_train_base": out["ci95_hit_minus_train_base"][0] > 0,
        "beats_placebo": out["hit_rate"] > out["placebo_hit_rate"],
        "placebo_ci_above_zero": out["ci95_hit_minus_placebo"][0] > 0,
        "paired_placebo_p_below_0_05": place["week_signflip"]["p_two_sided"] < .05,
        "bss_positive": out["bss"] is not None and out["bss"] > 0,
        "bss_ci_above_zero": out["ci95_bss"] is not None and out["ci95_bss"][0] > 0,
        "confirmatory_claim_allowed": False}
    out["success"]["all_met_train_base"] = all(out["success"][k] for k in (
        "ci_excludes_zero_above_train_base", "placebo_ci_above_zero", "paired_placebo_p_below_0_05", "bss_ci_above_zero"))
    out["success"]["all_met_eval_base"] = None
    return out


def main():
    idx = pd.read_csv(ROOT / "charts/index.csv")
    pr = pd.read_csv(ROOT / "results/preds.csv")[["file"] + P]
    pl = pd.read_csv(ROOT / "results/preds_placebo.csv")[["file"] + P].rename(columns={c: c + "_pl" for c in P})
    df = idx.merge(pr, on="file", validate="1:1").merge(pl, on="file", validate="1:1")
    if len(df) != len(idx):
        raise ValueError("Missing chart predictions")
    rng = np.random.default_rng(RNG_SEED)
    pooled_train = df[df.split == "train"]
    res = {"bootstrap_resamples": B, "seed": RNG_SEED,
           "inference": inference_metadata(), "tickers": {}}
    for ticker in TICKERS:
        d = df[df.ticker == ticker]
        tr = d[d.split == "train"]
        res["tickers"][ticker] = {
            "holdout": evaluate(d[d.split == "test"], tr, rng, pooled_train),
            "full_sample": evaluate(d, tr, rng, pooled_train)}
    (ROOT / "results/metrics.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({ticker: {k: res["tickers"][ticker]["holdout"][k]
        for k in ("n", "hit_rate", "ci95_hit_minus_train_base", "bss", "ci95_bss", "success")}
        for ticker in TICKERS}, indent=2))


if __name__ == "__main__":
    main()
