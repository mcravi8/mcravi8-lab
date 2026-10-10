"""Experiment 04: compare zero-shot JEV, calibrated JEV (03), LoRA JEV and CNN
on the same 492 test charts with 03's metric code (src/evaluate.py)."""
import json
import os
import numpy as np
import pandas as pd
from data import ROOT, TICKERS
import evaluate as ev
from research.statistics import holm_adjust, inference_metadata, paired_inference, probability_metrics

os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P, PL = ev.P, ev.PL
O3 = ROOT.parent / "03-jev-12-tickers"


def base_frame():
    keep = ["file", "ticker", "split", "end_date", "label"]
    return pd.read_csv(ROOT / "results/predictions.csv")[keep]


def with_probs(probs):
    """probs: file + P + PL for test rows. Non-test rows get uniform placeholders
    (only their labels are used, for train-period base rates and frequencies)."""
    df = base_frame().merge(probs, on="file", how="left", validate="1:1")
    test = df.split == "test"
    if not df.loc[test, P + PL].notna().all().all():
        raise ValueError("Missing test predictions")
    df[P + PL] = df[P + PL].fillna(1 / 3)
    return ev.add_refs(df)


def summarize(df, rng):
    test = df[df.split == "test"]
    out = {"pooled": ev.metrics(test, rng, True, "maj_test"), "tickers": {}}
    for t in TICKERS:
        out["tickers"][t] = ev.metrics(test[test.ticker == t], rng, False, "maj_test")
    out["pooled"]["flat_share"] = out["pooled"]["pred_counts"]["Flat"] / out["pooled"]["n"]
    return out


def main():
    rng = np.random.default_rng(ev.SEED)
    pred3 = pd.read_csv(ROOT / "results/predictions.csv")
    models = {}
    models["JEV zero-shot (03)"] = pred3[["file"] + P + PL]
    m3 = json.loads((O3 / "results/metrics.json").read_text())["calibration"]
    W, b = np.array(m3["coef"]), np.array(m3["intercept"])
    def cal(p):
        z = np.log(np.clip(p, 1e-6, 1)) @ W.T + b
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)
    c = pred3[["file"]].copy()
    c[P] = cal(pred3[P].to_numpy())
    c[PL] = cal(pred3[PL].to_numpy())
    models["JEV calibrated (03)"] = c
    lora = pd.read_csv(ROOT / "results/lora_test_preds.csv")
    models["JEV LoRA (04)"] = lora[["file"] + P + PL]
    cnn = pd.read_csv(ROOT / "results/cnn_preds.csv")
    cnn_t = cnn[cnn.fit_split == "test"]
    ens = cnn_t.groupby("file")[P + PL].mean().reset_index()
    models["CNN 5-seed ensemble (04)"] = ens
    res = {"note": "Exploratory historical period (2025-10-17 to 2026-10-02), 492 charts, 41 dates. Train-selected references are primary. Paired moving retained-week blocks: 1/2/4/8; four weeks primary. Oracle test-majority is descriptive. Calibrated placebo uses train-fit calibration.", "inference": inference_metadata(),
           "models": {}}
    for name, probs in models.items():
        res["models"][name] = summarize(with_probs(probs), rng)
    # Paired head-to-head comparisons avoid comparing separate model intervals.
    scored = {}
    for name, probs in models.items():
        test = with_probs(probs).query("split == 'test'").sort_values("file")
        scored[name] = (test, probability_metrics(test[P].to_numpy(), test.label))
    comparisons = {}
    names = list(scored)
    for number, challenger in enumerate(names):
        d, model = scored[challenger]
        for reference in names[:number]:
            other_d, other = scored[reference]
            if not d.file.equals(other_d.file) or not d.label.equals(other_d.label):
                raise ValueError("Head-to-head comparisons require identical paired rows")
            comparisons[f"{challenger} minus {reference}"] = {
                "challenger": challenger, "reference": reference,
                "accuracy": paired_inference(model["hit_rows"] - other["hit_rows"], d.end_date),
                "brier_improvement": paired_inference(other["brier_rows"] - model["brier_rows"], d.end_date)}
    family = [value[kind] for value in comparisons.values() for kind in ("accuracy", "brier_improvement")]
    for result, adjusted in zip(family, holm_adjust([v["week_signflip"]["p_two_sided"] for v in family])):
        result["holm_p_12_model_metric_comparisons"] = adjusted
    res["paired_model_comparisons"] = comparisons
    res["paired_model_comparison_family"] = "Holm adjustment across 6 model pairs x 2 metrics; this declared family does not correct adaptive historical holdout reuse"
    seeds = {}
    for s, g in cnn_t.groupby("seed"):
        seeds[int(s)] = summarize(with_probs(g[["file"] + P + PL]), rng)["pooled"]
    keys = ["hit_rate", "edge_vs_train_base", "placebo_hit_rate", "brier_model", "bss", "flat_share"]
    res["cnn_per_seed_pooled"] = {s: {k: v[k] for k in keys + ["ci95_edge_vs_train_base"]} for s, v in seeds.items()}
    res["cnn_seed_mean_pooled"] = {k: float(np.mean([v[k] for v in seeds.values()])) for k in keys}
    res["cnn_seed_sd_pooled"] = {k: float(np.std([v[k] for v in seeds.values()], ddof=1)) for k in keys}
    (ROOT / "results/metrics.json").write_text(json.dumps(res, indent=2) + "\n")
    # Figure
    names = list(models)
    pooled = [res["models"][n]["pooled"] for n in names]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(names))
    e = np.array([p["edge_vs_train_base"] for p in pooled]); ci = np.array([p["ci95_edge_vs_train_base"] for p in pooled])
    ax[0].errorbar(x, e, yerr=[e - ci[:, 0], ci[:, 1] - e], fmt="o", capsize=4, label="real charts")
    pe = np.array([p["placebo_hit_rate"] - p["base_rate_train_majority"] for p in pooled])
    ax[0].scatter(x + 0.15, pe, marker="x", color="grey", label="mirrored placebo")
    ax[0].axhline(0, color="grey", lw=0.8); ax[0].legend(fontsize=8)
    ax[0].set_title("Pooled test hit rate minus train-majority base\n(95% moving 4-retained-week bootstrap CI)", fontsize=10)
    b_ = np.array([p["bss"] for p in pooled]); bci = np.array([p["ci95_bss"] for p in pooled])
    ax[1].bar(x, b_, color="tab:blue"); ax[1].errorbar(x, b_, yerr=[b_ - bci[:, 0], bci[:, 1] - b_], fmt="none", color="k", capsize=4)
    ax[1].axhline(0, color="grey", lw=0.8); ax[1].set_title("Pooled test Brier skill score vs train frequencies", fontsize=10)
    ax[2].bar(x, [p["flat_share"] for p in pooled], color="tab:orange")
    flat_true = (base_frame().query("split=='test'").label == "Flat").mean()
    ax[2].axhline(flat_true, color="k", ls="--", lw=1, label=f"actual Flat share {flat_true:.0%}")
    ax[2].legend(fontsize=8); ax[2].set_title("Share of test charts predicted Flat", fontsize=10)
    for a in ax:
        a.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "results/model_comparison.png", dpi=120)
    # Training curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ct = json.loads((ROOT / "results/cnn_training.json").read_text())
    for s, v in ct["seeds"].items():
        h = v["history"]
        ax[0].plot([r["epoch"] for r in h], [r["fit_logloss_eval"] for r in h], color="tab:blue", alpha=0.5, label="fit" if s == "0" else None)
        ax[0].plot([r["epoch"] for r in h], [r["val_logloss"] for r in h], color="tab:red", alpha=0.5, label="validation" if s == "0" else None)
    ax[0].axhline(np.log(3), color="grey", ls=":", label="ln 3 (uniform)"); ax[0].legend(); ax[0].set_title("CNN log loss (5 seeds)"); ax[0].set_xlabel("epoch")
    lt = json.loads((ROOT / "results/lora_training.json").read_text())
    h = lt["history"]
    ax[1].plot([r["epoch"] for r in h[1:]], [r["train_loss"] for r in h[1:]], "o-", label="train (running, since last eval)")
    ax[1].plot([r["epoch"] for r in h], [r["val_logloss"] for r in h], "s-", color="tab:red", label="validation")
    ax[1].axhline(np.log(3), color="grey", ls=":", label="ln 3 (uniform)"); ax[1].legend(); ax[1].set_title("JEV LoRA log loss"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(ROOT / "results/training_curves.png", dpi=120)
    for n in names:
        p = res["models"][n]["pooled"]
        print(f"{n}: hit {p['hit_rate']:.3f} base {p['base_rate_train_majority']:.3f} edge {p['edge_vs_train_base']:+.3f} {ev.ci and p['ci95_edge_vs_train_base']} pl {p['placebo_hit_rate']:.3f} Brier {p['brier_model']:.3f}/{p['brier_ref']:.3f} BSS {p['bss']:+.3f} {p['ci95_bss']} flat {p['flat_share']:.3f} all_met {p['success']['all_met']}")
    print("CNN seeds", json.dumps(res["cnn_per_seed_pooled"]), res["cnn_seed_mean_pooled"], res["cnn_seed_sd_pooled"])


if __name__ == "__main__":
    main()
