"""Render unlabeled Monday-Friday weekly candlesticks with next-week labels.

Reuses Experiment 01's drawing style (colors, layout, no axes or text).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data import ROOT, TICKERS, validate_ohlcv

os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

WINDOW = 5
VOL_WEEKS = 12
FLAT_BAND = 1 / 3
TRAIN_FRACTION = 0.6
GAP_WEEKS = 1
TRAIN_FIRST, TRAIN_LAST, GAP = "2024-04-05", "2025-10-03", {"2025-10-10"}
LABELS = ("Up", "Flat", "Down")
UP_COLOR = "#16856b"
DOWN_COLOR = "#d34b52"


def render_chart(window: pd.DataFrame, target: Path) -> None:
    """Same style as 01's render_chart, for a 5-session window."""
    validate_ohlcv(window)
    if len(window) != WINDOW:
        raise ValueError("A weekly chart requires exactly 5 sessions")
    figure = Figure(figsize=(6.4, 4.8), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    price = figure.add_axes((0.035, 0.275, 0.93, 0.69))
    volume = figure.add_axes((0.035, 0.045, 0.93, 0.185))
    low, high = float(window.Low.min()), float(window.High.max())
    padding = max((high - low) * 0.07, high * 0.0001)
    price.set_ylim(low - padding, high + padding)
    volume.set_ylim(0, max(float(window.Volume.max()) * 1.08, 1))
    for axis in (price, volume):
        axis.set_xlim(-0.9, WINDOW - 0.1)
        axis.set_axis_off()
        axis.set_facecolor("white")
    for position, bar in enumerate(window.itertuples()):
        color = UP_COLOR if bar.Close >= bar.Open else DOWN_COLOR
        price.vlines(position, bar.Low, bar.High, color=color, linewidth=1.3)
        bottom = min(bar.Open, bar.Close)
        height = abs(bar.Close - bar.Open)
        if height == 0:
            price.hlines(bar.Open, position - 0.31, position + 0.31, color=color, linewidth=1.3)
        else:
            price.add_patch(Rectangle((position - 0.31, bottom), 0.62, height,
                                      facecolor=color, edgecolor=color, linewidth=0.8))
        volume.bar(position, bar.Volume, width=0.62, color=color, linewidth=0)
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(target, format="PNG", optimize=False)
    figure.clear()


def weeks(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Calendar weeks (Mon-Fri) in order, each a slice of sessions."""
    key = frame.index.to_period("W-SUN")
    return [frame[key == p] for p in key.unique()]


def is_full(week: pd.DataFrame) -> bool:
    days = [d.weekday() for d in week.index]
    return days == [0, 1, 2, 3, 4]


def weekly_rows(frame: pd.DataFrame) -> list[dict]:
    validate_ohlcv(frame)
    ws = weeks(frame)
    last_close = [float(w.Close.iloc[-1]) for w in ws]
    # weekly return of calendar week k (all weeks, incl. shortened), known at its last session
    wret = [np.nan] + [last_close[k] / last_close[k - 1] - 1 for k in range(1, len(ws))]
    rows = []
    for k, w in enumerate(ws[:-1]):
        if not is_full(w):
            continue
        nxt = ws[k + 1]
        # the final week in data counts as complete only if it ends on Friday
        if k + 1 == len(ws) - 1 and nxt.index[-1].weekday() != 4:
            continue
        hist = wret[max(0, k - VOL_WEEKS + 1):k + 1]
        if k - VOL_WEEKS + 1 < 1 or len(hist) != VOL_WEEKS:
            continue  # warmup
        sigma = float(np.std(hist, ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Trailing weekly return volatility must be positive and finite")
        r = last_close[k + 1] / last_close[k] - 1
        if not np.isfinite(r):
            raise ValueError("Next-week return must be finite")
        label = "Flat" if abs(r) < FLAT_BAND * sigma else ("Up" if r > 0 else "Down")
        rows.append({"week_start": w.index[0].date().isoformat(),
                     "end_date": w.index[-1].date().isoformat(),
                     "target_date": nxt.index[-1].date().isoformat(),
                     "next_week_full": is_full(nxt), "fwd_ret": r, "sigma": sigma,
                     "label": label, "window": w})
    return rows


def main() -> None:
    per = {t: weekly_rows(pd.read_parquet(ROOT / "data" / f"{t}.parquet")) for t in TICKERS}
    dates = sorted({r["end_date"] for t in TICKERS for r in per[t]})
    train_dates = {d for d in dates if TRAIN_FIRST <= d <= TRAIN_LAST}
    gap_dates = {d for d in dates if d in GAP}
    n_train = len([d for d in dates if d <= TRAIN_LAST])
    out, n = [], 0
    for t in TICKERS:
        for r in per[t]:
            split = "train" if r["end_date"] in train_dates else ("gap" if r["end_date"] in gap_dates else "test")
            if split == "gap" or r["end_date"] < TRAIN_FIRST:
                continue
            n += 1
            rel = f"charts/chart_{n:06d}.png"
            render_chart(r["window"], ROOT / rel)
            out.append({"file": rel, "ticker": t, "split": split,
                        **{k: r[k] for k in ("week_start", "end_date", "target_date",
                                             "next_week_full", "label", "fwd_ret", "sigma")}})
    idx = pd.DataFrame(out)
    idx.to_csv(ROOT / "charts" / "index.csv", index=False, float_format="%.17g")
    counts = {t: {s: {**{l: int(((idx.ticker == t) & (idx.split == s) & (idx.label == l)).sum()) for l in LABELS},
                      "total": int(((idx.ticker == t) & (idx.split == s)).sum())} for s in ("train", "test")}
              for t in TICKERS}
    manifest = {"window_sessions": WINDOW, "vol_weeks": VOL_WEEKS, "flat_band": FLAT_BAND,
                "train_end": max(train_dates), "gap_dropped": sorted(gap_dates),
                "test_start": min(d for d in dates if d > max(GAP)), "test_end": dates[-1],
                "first_chart": dates[0],
                "holiday_short_next_weeks": int((~idx.next_week_full).sum()),
                "input_sha256": {t: hashlib.sha256((ROOT / "data" / f"{t}.parquet").read_bytes()).hexdigest() for t in TICKERS},
                "counts": counts}
    (ROOT / "results" / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
