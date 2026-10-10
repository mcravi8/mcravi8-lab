"""Experiment 03 evaluation: per-ticker and pooled metrics, calibration,
and exploratory volatility analyses. Writes results/metrics.json,
results/predictions.csv, results/ticker_edge.png, results/vol_regimes.png."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from data import ROOT, TICKERS
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.statistics import (BLOCK_LENGTHS, PRIMARY_BLOCK_LENGTH,
    block_bootstrap_means, exact_mcnemar, holm_adjust, inference_metadata,
    paired_inference, percentile_ci, probability_metrics)

os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["Up", "Flat", "Down"]
P = ["p_up", "p_flat", "p_down"]
PL = [c + "_pl" for c in P]
B, SEED = 2000, 0
REUSED = ("TSLA", "NVDA")
O2 = ROOT.parent / "02-jev-weekly-charts"


def load():
    idx = pd.read_csv(ROOT / "charts/index.csv")
    new = pd.read_csv(ROOT / "results/preds.csv")[["file"] + P]
    newp = pd.read_csv(ROOT / "results/preds_placebo.csv")[["file"] + P].rename(columns=dict(zip(P, PL)))
    i2 = pd.read_csv(O2 / "charts/index.csv")[["file", "ticker", "end_date"]]
    old = i2.merge(pd.read_csv(O2 / "results/preds.csv")[["file"] + P], on="file")
    oldp = i2.merge(pd.read_csv(O2 / "results/preds_placebo.csv")[["file"] + P].rename(columns=dict(zip(P, PL))), on="file")
    old = old.merge(oldp.drop(columns="file"), on=["ticker", "end_date"]).drop(columns="file")
    a = idx[~idx.ticker.isin(REUSED)].merge(new, on="file", validate="1:1").merge(newp, on="file", validate="1:1")
    b = idx[idx.ticker.isin(REUSED)].merge(old, on=["ticker", "end_date"], validate="1:1")
    b["pred_source"] = "02 (byte-identical chart)"
    a["pred_source"] = "03"
    df = pd.concat([a, b]).sort_values(["ticker", "end_date"]).reset_index(drop=True)
    assert len(df) == len(idx) and df[P + PL].notna().all().all()
    return df


def add_refs(df):
    """Per-ticker reference quantities from that ticker's TRAIN weeks / test weeks."""
    df = df.copy()
    Y = np.array([[l == c for c in LABELS] for l in df.label], float)
    df[["y_up", "y_flat", "y_down"]] = Y
    for t in TICKERS:
        m = df.ticker == t
        tr = df[m & (df.split == "train")]
        freq = tr.label.value_counts(normalize=True).reindex(LABELS).fillna(0).to_numpy()
        df.loc[m, ["r_up", "r_flat", "r_down"]] = freq
        maj_tr = LABELS[int(np.argmax(freq))]  # ties broken in Up, Flat, Down order
        df.loc[m, "maj_train"] = maj_tr
        for s in ("test",):
            te = df[m & (df.split == s)]
            maj_te = te.label.value_counts().reindex(LABELS).idxmax()
            df.loc[m, "maj_test"] = maj_te
        full = df[m].label.value_counts().reindex(LABELS).idxmax()
        df.loc[m, "maj_full"] = full
    pooled = df.loc[df.split == "train", "label"].value_counts(normalize=True).reindex(LABELS).fillna(0).to_numpy()
    df[["r_pool_up", "r_pool_flat", "r_pool_down"]] = np.tile(pooled, (len(df), 1))
    Yv = df[["y_up", "y_flat", "y_down"]].to_numpy()
    df["hit"] = (np.array(LABELS)[df[P].to_numpy().argmax(1)] == df.label).astype(float)
    df["hit_pl"] = (np.array(LABELS)[df[PL].to_numpy().argmax(1)] == df.label).astype(float)
    df["base_train"] = (df.maj_train == df.label).astype(float)
    df["brier"] = ((df[P].to_numpy() - Yv) ** 2).sum(1)
    df["brier_ref"] = ((df[["r_up", "r_flat", "r_down"]].to_numpy() - Yv) ** 2).sum(1)
    return df



def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def baseline_report(d, model_probs):
    """Constants require no chart; ticker priors additionally know ticker identity."""
    n = len(d)
    forecasts = {
        "ticker_train_frequency": d[["r_up", "r_flat", "r_down"]].to_numpy(),
        "pooled_train_frequency": d[["r_pool_up", "r_pool_flat", "r_pool_down"]].to_numpy(),
        "uniform": np.full((n, 3), 1 / 3),
        "always_up": np.tile([1., 0., 0.], (n, 1)),
    }
    model = probability_metrics(model_probs, d.label)
    result = {}
    for name, p in forecasts.items():
        ref = probability_metrics(p, d.label)
        result[name] = {
            "hit_rate": ref["hit_rate"], "brier": ref["brier"], "log_loss": ref["log_loss"],
            "information": "ticker identity and training labels" if name.startswith("ticker") else "training labels only; no image or ticker identity",
            "paired_accuracy": paired_inference(model["hit_rows"] - ref["hit_rows"], d.end_date, resamples=B, seed=SEED),
            "paired_brier_improvement": paired_inference(ref["brier_rows"] - model["brier_rows"], d.end_date, resamples=B, seed=SEED),
        }
    return result


def metrics(d, rng, block, base_test_col, cal=None):
    """Historical exploratory metrics; four-week paired blocks are primary."""
    model = probability_metrics(d[P].to_numpy(), d.label)
    placebo = probability_metrics(d[PL].to_numpy(), d.label)
    hit, hp = model["hit_rows"], placebo["hit_rows"]
    bt = d.base_train.to_numpy()
    bte = (d[base_test_col] == d.label).astype(float).to_numpy()
    bm, br = model["brier_rows"], d.brier_ref.to_numpy()
    dates = d.end_date
    train_compare = paired_inference(hit - bt, dates, resamples=B, seed=SEED)
    placebo_compare = paired_inference(hit - hp, dates, resamples=B, seed=SEED)
    brier_compare = paired_inference(br - bm, dates, resamples=B, seed=SEED)
    bss_intervals = {}
    for length in BLOCK_LENGTHS:
        means = block_bootstrap_means(np.column_stack([bm, br]), dates,
                                     resamples=B, seed=SEED, block_length=length)
        valid = means[:, 1] > 0
        bss_intervals[str(length)] = percentile_ci(1 - means[valid, 0] / means[valid, 1]) if valid.any() else None
    primary = str(PRIMARY_BLOCK_LENGTH)
    res = {"n": int(len(d)), "weeks": int(d.end_date.nunique()),
           "label_counts": {l: int((d.label == l).sum()) for l in LABELS},
           "pred_counts": {l: int((np.array(LABELS)[d[P].to_numpy().argmax(1)] == l).sum()) for l in LABELS},
           "hit_rate": model["hit_rate"], "log_loss": model["log_loss"],
           "base_rate_train_majority": float(bt.mean()),
           "base_rate_test_majority": float(bte.mean()), "placebo_hit_rate": placebo["hit_rate"],
           "brier_model": model["brier"], "brier_ref": float(br.mean()),
           "bss": float(1 - bm.mean() / br.mean()) if br.mean() > 0 else None,
           "edge_vs_train_base": float((hit - bt).mean()),
           "edge_vs_test_base": float((hit - bte).mean()),
           "ci95_edge_vs_train_base": train_compare["ci95_by_block_length"][primary],
           "ci95_edge_vs_test_base": None,
           "oracle_ci_note": "Evaluation-selected majority is descriptive; no post-selection interval reported",
           "ci95_bss": bss_intervals[primary], "ci95_bss_by_block_length": bss_intervals,
           "ci95_hit_minus_placebo": placebo_compare["ci95_by_block_length"][primary],
           "paired_inference": {"accuracy_vs_train_majority": train_compare,
                                "accuracy_vs_placebo": placebo_compare,
                                "brier_vs_train_frequency": brier_compare},
           "baselines": baseline_report(d, d[P].to_numpy())}
    wins, losses = int(((hit == 1) & (hp == 0)).sum()), int(((hit == 0) & (hp == 1)).sum())
    res["placebo_discordant_pairs"] = {"model_wins": wins, "placebo_wins": losses,
        "ties": len(d) - wins - losses,
        "exact_mcnemar_p_if_independent_rows": exact_mcnemar(wins, losses),
        "note": "Row-level McNemar assumes independence and is descriptive for correlated stock rows; date sign flips are separately reported"}
    if cal is not None:
        cp = d[["q_up", "q_flat", "q_down"]].to_numpy()
        cm = probability_metrics(cp, d.label)
        cacc = paired_inference(cm["hit_rows"] - bt, dates, resamples=B, seed=SEED)
        cbrier = paired_inference(br - cm["brier_rows"], dates, resamples=B, seed=SEED)
        cbss = {}
        for length in BLOCK_LENGTHS:
            means = block_bootstrap_means(np.column_stack([cm["brier_rows"], br]), dates,
                                         resamples=B, seed=SEED, block_length=length)
            valid = means[:, 1] > 0
            cbss[str(length)] = percentile_ci(1 - means[valid, 0] / means[valid, 1]) if valid.any() else None
        res["calibrated"] = {"hit_rate": cm["hit_rate"], "brier": cm["brier"], "log_loss": cm["log_loss"],
            "bss": float(1 - cm["brier"] / br.mean()) if br.mean() > 0 else None,
            "ci95_edge_vs_train_base": cacc["ci95_by_block_length"][primary],
            "ci95_bss": cbss[primary], "ci95_bss_by_block_length": cbss,
            "paired_accuracy_vs_train_majority": cacc, "paired_brier_improvement": cbrier,
            "baselines": baseline_report(d, cp)}
    res["success"] = {
        "ci_above_test_base": None,
        "ci_above_train_base": res["ci95_edge_vs_train_base"][0] > 0,
        "beats_placebo": res["hit_rate"] > res["placebo_hit_rate"],
        "paired_placebo_p_below_0_05": placebo_compare["week_signflip"]["p_two_sided"] < .05,
        "placebo_ci_above_zero": res["ci95_hit_minus_placebo"][0] > 0,
        "bss_positive": res["bss"] is not None and res["bss"] > 0,
        "bss_ci_above_zero": res["ci95_bss"] is not None and res["ci95_bss"][0] > 0,
        "confirmatory_claim_allowed": False,
    }
    res["success"]["all_met"] = all(res["success"][key] for key in (
        "ci_above_train_base", "placebo_ci_above_zero", "paired_placebo_p_below_0_05", "bss_ci_above_zero"))
    res["inference"] = inference_metadata()
    return res


def calibrate(df):
    X = np.log(np.clip(df[P].to_numpy(), 1e-6, 1))
    tr = (df.split == "train").to_numpy()
    model = LogisticRegression(C=1e6, max_iter=5000).fit(X[tr], df.label[tr])
    order = [list(model.classes_).index(l) for l in LABELS]
    q = model.predict_proba(X)[:, order]
    Y = df[["y_up", "y_flat", "y_down"]].to_numpy()
    df["q_up"], df["q_flat"], df["q_down"] = q.T
    df["hit_cal"] = (np.array(LABELS)[q.argmax(1)] == df.label).astype(float)
    df["brier_cal"] = ((q - Y) ** 2).sum(1)
    return {"model": "multinomial logistic regression on log(p_up, p_flat, p_down), fit on pooled TRAIN rows, effectively unregularized (C=1e6)",
            "classes": LABELS, "coef": model.coef_[order].tolist(), "intercept": model.intercept_[order].tolist(),
            "test_pred_counts": {l: int((np.array(LABELS)[q.argmax(1)][(df.split == "test").to_numpy()] == l).sum()) for l in LABELS}}


def main():
    rng = np.random.default_rng(SEED)
    df = add_refs(load())
    cal = calibrate(df)
    test = df[df.split == "test"]
    out = {"bootstrap_resamples": B, "seed": SEED, "inference": inference_metadata(),
           "base_rate_definitions": "per ticker: train-majority class of that ticker applied to the evaluated rows (honest); test-majority (or full-sample majority for full rows) of that ticker (look-ahead). Pooled rates average these per-row indicators. Ties broken Up, Flat, Down.",
           "brier_reference": "per ticker train-period class frequencies",
           "calibration": cal, "tickers": {}, "pooled": {}}
    for t in TICKERS:
        d = df[df.ticker == t]
        out["tickers"][t] = {"holdout": metrics(d[d.split == "test"], rng, False, "maj_test", cal=True),
                             "full_sample": metrics(d, rng, False, "maj_full")}
    out["pooled"] = {"holdout": metrics(test, rng, True, "maj_test", cal=True),
                     "full_sample": metrics(df, rng, True, "maj_full")}
    # Exploratory 1: cross-ticker Spearman of train-period vol vs test edge / BSS
    vf = json.loads((ROOT / "results/vol_features.json").read_text())
    tv = np.array([vf["train_period_ann_vol"][t] for t in TICKERS])
    edge = np.array([out["tickers"][t]["holdout"]["edge_vs_train_base"] for t in TICKERS])
    bss = np.array([out["tickers"][t]["holdout"]["bss"] for t in TICKERS])
    sp = {}
    for name, y in (("edge_vs_train_base", edge), ("bss", bss)):
        rho = spearmanr(tv, y).statistic
        boots = []
        for _ in range(B):
            j = rng.integers(0, len(tv), len(tv))
            if len(set(j)) > 2:
                r = spearmanr(tv[j], y[j]).statistic
                if np.isfinite(r):
                    boots.append(r)
        sp[name] = {"spearman_rho": float(rho), "ci95_ticker_bootstrap": ci(boots)}
    out["exploratory_cross_ticker"] = {"note": "EXPLORATORY. n=12 tickers; volatility from train period only; outcome from test period.",
                                       "train_period_ann_vol": vf["train_period_ann_vol"], **sp}
    # Exploratory 2: test-period tercile analysis, moving retained-week block bootstrap
    reg = {}
    for f in ("rv20", "rv20_pct", "vix"):
        reg[f] = {}
        for lvl in ("low", "mid", "high"):
            d = test[test[f + "_regime"] == lvl]
            m = metrics(d, rng, True, "maj_test")
            reg[f][lvl] = {k: m[k] for k in ("n", "weeks", "hit_rate", "base_rate_train_majority",
                                             "edge_vs_train_base", "ci95_edge_vs_train_base", "bss", "ci95_bss")}
    out["exploratory_vol_regimes"] = {"note": "EXPLORATORY, test period, pooled across tickers, terciles from TRAIN cutoffs (results/vol_features.json). 3 measures x 3 terciles x 2 metrics = 18 intervals: expect about 1 to exclude zero by chance; no multiplicity correction applied.",
                                      **reg}
    # Exploratory 3: logistic regression of correct on vol features + ticker FE, test period
    feats = ["rv20", "rv20_pct", "vix", "spy_rv20"]
    Z = (test[feats] - df.loc[df.split == "train", feats].mean()) / df.loc[df.split == "train", feats].std()
    X = pd.concat([Z, pd.get_dummies(test.ticker, drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    groups = pd.factorize(test.end_date)[0]
    fit = sm.Logit(test.hit.to_numpy(), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    cis = fit.conf_int()
    out["exploratory_logit"] = {"note": "EXPLORATORY. Logit(correct) on test rows; features standardized with train mean/sd; ticker fixed effects; SEs clustered by week. rv20, spy_rv20 and vix are strongly correlated, so individual coefficients are unstable.",
                                "n": int(len(test)), "coef": {k: {"coef": float(fit.params[k]), "ci95": [float(cis.loc[k, 0]), float(cis.loc[k, 1])], "p": float(fit.pvalues[k])} for k in feats},
                                "feature_corr_test": Z.corr().round(3).to_dict()}
    adjusted = holm_adjust([out["exploratory_logit"]["coef"][k]["p"] for k in feats])
    for feature, value in zip(feats, adjusted):
        out["exploratory_logit"]["coef"][feature]["p_holm_4_features"] = value
    out["exploratory_logit"]["multiplicity"] = "Holm correction for these four coefficients only; all broader adaptive analyses remain exploratory"
    (ROOT / "results/metrics.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    keep = ["file", "ticker", "split", "week_start", "end_date", "target_date", "next_week_full", "label", "fwd_ret", "sigma",
            "rv20", "rv20_pct", "vix", "spy_rv20", "rv20_regime", "rv20_pct_regime", "vix_regime", "spy_rv20_regime",
            *P, *PL, "q_up", "q_flat", "q_down", "pred_source"]
    df[keep].to_csv(ROOT / "results/predictions.csv", index=False, float_format="%.6g")
    # Figure 1: per-ticker edge sorted by train vol, plus raw vs calibrated BSS
    order = sorted(TICKERS, key=lambda t: vf["train_period_ann_vol"][t])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    e = [out["tickers"][t]["holdout"]["edge_vs_train_base"] for t in order]
    c = np.array([out["tickers"][t]["holdout"]["ci95_edge_vs_train_base"] for t in order])
    x = np.arange(len(order))
    ax[0].errorbar(x, e, yerr=[np.array(e) - c[:, 0], c[:, 1] - np.array(e)], fmt="o", capsize=3)
    pe = out["pooled"]["holdout"]
    ax[0].errorbar([len(order) + 0.6], [pe["edge_vs_train_base"]], yerr=[[pe["edge_vs_train_base"] - pe["ci95_edge_vs_train_base"][0]], [pe["ci95_edge_vs_train_base"][1] - pe["edge_vs_train_base"]]], fmt="s", color="k", capsize=3)
    ax[0].axhline(0, color="grey", lw=0.8)
    ax[0].set_xticks(list(x) + [len(order) + 0.6], [f"{t}\n{vf['train_period_ann_vol'][t]:.0%}" for t in order] + ["Pooled"], fontsize=8)
    ax[0].set_title("Test hit rate minus train-majority base rate (95% CI)\nsorted by train-period annualized vol")
    raw = [out["tickers"][t]["holdout"]["bss"] for t in order] + [pe["bss"]]
    calb = [out["tickers"][t]["holdout"]["calibrated"]["bss"] for t in order] + [pe["calibrated"]["bss"]]
    x2 = np.arange(len(raw))
    ax[1].bar(x2 - 0.2, raw, 0.4, label="raw JEV")
    ax[1].bar(x2 + 0.2, calb, 0.4, label="calibrated (train-fit)")
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].set_xticks(x2, order + ["Pooled"], fontsize=8, rotation=45)
    ax[1].set_title("Test Brier skill score vs train class frequencies")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results/ticker_edge.png", dpi=120)
    # Figure 2: vol regimes
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for a, (f, title) in zip(ax, (("rv20", "Stock 20d realized vol"), ("rv20_pct", "Stock vol percentile (own history)"), ("vix", "VIX close"))):
        vals = [reg[f][l]["edge_vs_train_base"] for l in ("low", "mid", "high")]
        cc = np.array([reg[f][l]["ci95_edge_vs_train_base"] for l in ("low", "mid", "high")])
        a.errorbar(range(3), vals, yerr=[np.array(vals) - cc[:, 0], cc[:, 1] - np.array(vals)], fmt="o", capsize=4)
        a.axhline(0, color="grey", lw=0.8)
        a.set_xticks(range(3), [f"{l}\nn={reg[f][l]['n']}" for l in ("low", "mid", "high")])
        a.set_title(title, fontsize=10)
    ax[0].set_ylabel("Hit rate minus train-majority base")
    fig.suptitle("EXPLORATORY: test-period edge by volatility tercile (train cutoffs, moving retained-week block bootstrap 95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "results/vol_regimes.png", dpi=120)
    print(json.dumps({"pooled": out["pooled"], "sp": sp}, indent=1)[:3000])


if __name__ == "__main__":
    main()
