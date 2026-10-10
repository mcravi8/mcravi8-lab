"""Causal weekly targets, chart-equivalent shape features, and a context ablation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from data import ROOT, validate_ohlcv

SHAPE = [f"{field.lower()}_{day}" for day in range(5) for field in ("Open", "High", "Low", "Close")] + [f"volume_{day}" for day in range(5)]
CONTEXT = ["week_return", "prior_week_return", "momentum_4w", "momentum_12w",
           "rv20", "sigma", "range_pct", "volume_ratio60"]


def labels(forward: float, sigma: float, flat_band: float) -> tuple[str, str | None]:
    if not np.isfinite(forward) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Forward return must be finite and volatility positive")
    if not np.isfinite(flat_band) or flat_band <= 0:
        raise ValueError("Flat band must be positive and finite")
    ternary = "Flat" if abs(forward) < flat_band * sigma else ("Up" if forward > 0 else "Down")
    binary = "Up" if forward > 0 else ("Down" if forward < 0 else None)
    return ternary, binary


def weekly_features(frame: pd.DataFrame, ticker: str, protocol: dict) -> pd.DataFrame:
    validate_ohlcv(frame)
    if frame.index[-1].date().isoformat() > protocol["development_cutoff"]:
        raise ValueError("Development features cannot include the future holdout")
    groups = [w for _, w in frame.groupby(frame.index.to_period("W-SUN"), sort=True)]
    closes = np.array([float(w.Close.iloc[-1]) for w in groups])
    returns = np.r_[np.nan, closes[1:] / closes[:-1] - 1]
    lookback = protocol["volatility_lookback_weeks"]
    daily_returns = frame.Close.pct_change(fill_method=None)
    rows = []
    # Context always includes twelve-week momentum, even with a shorter sigma
    # lookback. Negative array indices would otherwise read future closes.
    for k in range(max(lookback, 12), len(groups) - 1):
        w, nxt = groups[k], groups[k + 1]
        if [d.weekday() for d in w.index] != [0, 1, 2, 3, 4]:
            continue
        if k + 1 == len(groups) - 1 and nxt.index[-1].weekday() != 4:
            continue
        sigma = float(np.std(returns[k - lookback + 1:k + 1], ddof=1))
        forward = float(closes[k + 1] / closes[k] - 1)
        ternary, binary = labels(forward, sigma, protocol["flat_band"])
        low, high = float(w.Low.min()), float(w.High.max())
        span = max(high - low, np.finfo(float).eps * high)
        shape = ((w[["Open", "High", "Low", "Close"]].to_numpy() - low) / span).ravel().tolist()
        shape += (w.Volume.to_numpy() / max(float(w.Volume.max()), 1)).tolist()
        past = frame.loc[:w.index[-1]].tail(60)
        rv20 = float(daily_returns.loc[:w.index[-1]].tail(20).std(ddof=1) * np.sqrt(252))
        row = {"observation_id": f"{ticker}_{w.index[-1].date()}", "ticker": ticker,
               "week_start": str(w.index[0].date()), "end_date": str(w.index[-1].date()),
               "target_date": str(nxt.index[-1].date()), "fwd_ret": forward,
               "label_ternary": ternary, "label_binary": binary,
               "week_return": float(returns[k]), "prior_week_return": float(returns[k - 1]),
               "momentum_4w": float(closes[k] / closes[k - 4] - 1),
               "momentum_12w": float(closes[k] / closes[k - 12] - 1),
               "rv20": rv20, "sigma": sigma, "range_pct": (high - low) / closes[k],
               "volume_ratio60": float(w.Volume.mean() / max(past.Volume.mean(), 1)),
               **dict(zip(SHAPE, shape))}
        if not np.isfinite([row[c] for c in SHAPE + CONTEXT]).all():
            raise ValueError(f"Nonfinite features for {row['observation_id']}")
        rows.append(row)
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    protocol = json.loads((ROOT / "protocol.json").read_text())
    df = pd.concat([weekly_features(pd.read_parquet(ROOT / "data" / f"{t}.parquet"), t, protocol)
                    for t in protocol["tickers"]], ignore_index=True)
    df = df.sort_values(["end_date", "ticker"]).reset_index(drop=True)
    if df.observation_id.duplicated().any():
        raise ValueError("Duplicate observation identifiers")
    target = ROOT / "data/features.csv"
    df.to_csv(target, index=False, float_format="%.17g")
    manifest = {"rows": len(df), "weeks": int(df.end_date.nunique()),
                "first_chart_end": df.end_date.min(), "last_chart_end": df.end_date.max(),
                "per_ticker_rows": df.ticker.value_counts().sort_index().to_dict(),
                "shape_features": SHAPE, "context_features": CONTEXT,
                "zero_return_binary_rows_excluded": int(df.label_binary.isna().sum()),
                "features_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()}
    (ROOT / "results/features_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return df


if __name__ == "__main__":
    build()
