"""Per-chart volatility features, each known at that chart's Friday close.

Adds to charts/index.csv: rv20 (annualized std of the 20 daily close returns
ending Friday), rv20_pct (percentile of rv20 within the ticker's own rv20
history up to and including that Friday, expanding window), vix (^VIX close
that Friday), spy_rv20, and *_regime terciles (low/mid/high) using cutoffs
from TRAIN rows only. Also writes data/VIX.parquet and results/vol_features.json.
"""
import json
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from data import ROOT, TICKERS

FEATURES = ["rv20", "rv20_pct", "vix", "spy_rv20"]


def rv20(close):
    return close.pct_change(fill_method=None).rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)


def expanding_pct(series):
    vals = series.to_numpy()
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        hist = vals[:i + 1]
        hist = hist[np.isfinite(hist)]
        if np.isfinite(vals[i]) and len(hist):
            out[i] = (hist <= vals[i]).mean()
    return pd.Series(out, index=series.index)


def main():
    idx = pd.read_csv(ROOT / "charts/index.csv")
    yf.set_tz_cache_location(str(ROOT / ".cache" / "yfinance"))
    vix = yf.download("^VIX", start="2024-01-01", end=(date(2026, 10, 9) + timedelta(days=1)).isoformat(),
                      auto_adjust=False, progress=False, multi_level_index=False, ignore_tz=True)
    vix.index = pd.DatetimeIndex(vix.index).tz_localize(None).normalize()
    vix[["Close"]].to_parquet(ROOT / "data/VIX.parquet")
    feats = {}
    for t in TICKERS:
        c = pd.read_parquet(ROOT / f"data/{t}.parquet").Close
        r = rv20(c)
        feats[t] = pd.DataFrame({"rv20": r, "rv20_pct": expanding_pct(r)})
    spy = feats["SPY"].rv20.rename("spy_rv20")
    rows = []
    for row in idx.itertuples():
        d = pd.Timestamp(row.end_date)
        f = feats[row.ticker].loc[d]
        rows.append({"rv20": f.rv20, "rv20_pct": f.rv20_pct,
                     "vix": float(vix.Close.loc[d]), "spy_rv20": float(spy.loc[d])})
    idx = idx.drop(columns=[c for c in idx.columns if c in FEATURES or c.endswith("_regime")])
    idx = pd.concat([idx, pd.DataFrame(rows)], axis=1)
    assert idx[FEATURES].notna().all().all()
    train = idx[idx.split == "train"]
    cuts = {}
    for f in FEATURES:
        # market features are one value per week: use unique train weeks
        base = train.drop_duplicates("end_date")[f] if f in ("vix", "spy_rv20") else train[f]
        lo, hi = np.quantile(base, [1 / 3, 2 / 3])
        cuts[f] = [float(lo), float(hi)]
        idx[f + "_regime"] = np.where(idx[f] <= lo, "low", np.where(idx[f] <= hi, "mid", "high"))
    idx.to_csv(ROOT / "charts/index.csv", index=False, float_format="%.17g")
    # train-period realized vol per ticker (exploratory cross-ticker analysis)
    first = pd.Timestamp(train.week_start.min())
    tvol = {}
    for t in TICKERS:
        c = pd.read_parquet(ROOT / f"data/{t}.parquet").Close
        r = c.pct_change(fill_method=None)
        tvol[t] = float(r[(r.index >= first) & (r.index <= pd.Timestamp("2025-10-03"))].std(ddof=1) * np.sqrt(252))
    full = {t: float(pd.read_parquet(ROOT / f"data/{t}.parquet").Close.pct_change(fill_method=None).std(ddof=1) * np.sqrt(252)) for t in TICKERS}
    out = {"tercile_cutoffs_train": cuts, "train_period_ann_vol": tvol,
           "full_period_ann_vol": full, "train_vol_window": [str(first.date()), "2025-10-03"]}
    (ROOT / "results/vol_features.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
