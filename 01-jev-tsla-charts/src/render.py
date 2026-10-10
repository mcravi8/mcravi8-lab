"""Render unlabeled 20-session candlesticks and historical-volatility labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

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

WINDOW = 20
STEP = 5
HORIZON = 5
VOL_LOOKBACK = 60
FLAT_BAND = 1 / 3
LABELS = ("Up", "Flat", "Down")
IMAGE_SIZE = (640, 480)
UP_COLOR = "#16856b"
DOWN_COLOR = "#d34b52"


def trailing_volatility(close: pd.Series) -> pd.Series:
    """Sixty completed overlapping five-session returns, including today's."""
    return close.pct_change(HORIZON, fill_method=None).rolling(
        VOL_LOOKBACK, min_periods=VOL_LOOKBACK
    ).std(ddof=1)


def label_return(fwd_ret: float, sigma: float) -> str:
    """The Flat boundary is strict; degenerate volatility is undefined."""
    if not np.isfinite(fwd_ret) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Labels require a finite return and positive finite volatility")
    if abs(fwd_ret) < FLAT_BAND * sigma:
        return "Flat"
    return "Up" if fwd_ret > 0 else "Down"


def eligible_windows(frame: pd.DataFrame) -> list[dict]:
    """Keep the stride anchored at the first full chart, then omit warmup."""
    validate_ohlcv(frame)
    close = frame.Close
    volatility = trailing_volatility(close)
    windows = []
    first_warm_end = HORIZON + VOL_LOOKBACK - 1
    for end in range(WINDOW - 1, len(frame) - HORIZON, STEP):
        if end < first_warm_end:
            continue
        sigma = float(volatility.iloc[end])
        fwd_ret = float(close.iloc[end + HORIZON] / close.iloc[end] - 1)
        windows.append({
            "end_pos": end,
            "end_date": frame.index[end].date().isoformat(),
            "label": label_return(fwd_ret, sigma),
            "fwd_ret": fwd_ret,
        })
    return windows


def render_chart(window: pd.DataFrame, target: Path) -> None:
    """Only the visible window enters this function, including scale limits."""
    validate_ohlcv(window)
    if len(window) != WINDOW:
        raise ValueError(f"A chart requires exactly {WINDOW} trading sessions")
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
            price.hlines(bar.Open, position - 0.31, position + 0.31,
                         color=color, linewidth=1.3)
        else:
            price.add_patch(Rectangle((position - 0.31, bottom), 0.62, height,
                                      facecolor=color, edgecolor=color, linewidth=0.8))
        volume.bar(position, bar.Volume, width=0.62, color=color, linewidth=0)
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.png")
    Image.fromarray(pixels).save(temporary, format="PNG", optimize=False)
    temporary.replace(target)
    figure.clear()


def count_labels(rows: pd.DataFrame) -> dict:
    counts = {label: int((rows.label == label).sum()) for label in LABELS}
    return {**counts, "total": int(len(rows))}


def write_json(target: Path, payload: dict) -> None:
    temporary = target.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    frames = {}
    windows = {}
    input_hashes = {}
    for ticker in TICKERS:
        source = ROOT / "data" / f"{ticker}.parquet"
        frames[ticker] = pd.read_parquet(source, engine="pyarrow")
        windows[ticker] = eligible_windows(frames[ticker])
        if not windows[ticker]:
            raise ValueError(f"{ticker} has no eligible chart windows")
        input_hashes[ticker] = hashlib.sha256(source.read_bytes()).hexdigest()
    charts_dir = ROOT / "charts"
    samples_dir = ROOT / "results" / "samples"
    charts_dir.mkdir(exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ticker in TICKERS:
        for item in windows[ticker]:
            end = item["end_pos"]
            relative = Path("charts") / f"chart_{len(rows) + 1:06d}.png"
            render_chart(frames[ticker].iloc[end - WINDOW + 1:end + 1], ROOT / relative)
            rows.append({"file": relative.as_posix(), "ticker": ticker,
                         **{key: item[key] for key in ("end_date", "label", "fwd_ret")}})
    index = pd.DataFrame(rows, columns=("file", "ticker", "end_date", "label", "fwd_ret"))
    temporary = charts_dir / "index.tmp.csv"
    index.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(charts_dir / "index.csv")
    expected_names = {Path(name).name for name in index.file}
    for stale in charts_dir.glob("chart_*.png"):
        if stale.name not in expected_names:
            stale.unlink()
    counts = {ticker: count_labels(index[index.ticker == ticker]) for ticker in TICKERS}
    counts["combined"] = count_labels(index)
    write_json(ROOT / "results" / "label_counts.json", counts)
    samples = []
    for number, (ticker, fraction) in enumerate(
        (("TSLA", 0.25), ("NVDA", 0.5), ("TSLA", 0.75)), start=1
    ):
        ticker_rows = index[index.ticker == ticker].reset_index(drop=True)
        position = int((len(ticker_rows) - 1) * fraction)
        chosen = ticker_rows.iloc[position].to_dict()
        sample = Path("results") / "samples" / f"sample_{number:02d}.png"
        shutil.copyfile(ROOT / chosen["file"], ROOT / sample)
        samples.append({"sample_file": sample.as_posix(), "source_file": chosen["file"],
                        **{key: chosen[key] for key in ("ticker", "end_date", "label", "fwd_ret")},
                        "temporal_fraction": fraction,
                        "sha256": hashlib.sha256((ROOT / sample).read_bytes()).hexdigest()})
    pd.DataFrame(samples).to_csv(samples_dir / "manifest.csv", index=False,
                                 float_format="%.17g")
    manifest = {
        "window_sessions": WINDOW, "step_sessions": STEP, "horizon_sessions": HORIZON,
        "stride_first_end_position": WINDOW - 1,
        "first_warmed_end_position": HORIZON + VOL_LOOKBACK - 1,
        "volatility": "60 overlapping trailing 5-session close returns; sample std, ddof=1",
        "flat_band": FLAT_BAND,
        "flat_rule": "abs(fwd_ret) < flat_band * trailing_volatility",
        "image_size": list(IMAGE_SIZE), "image_mode": "RGB", "background": "white",
        "up_color": UP_COLOR, "down_color": DOWN_COLOR,
        "matplotlib_version": matplotlib.__version__,
        "input_sha256": input_hashes,
        "sample_selection": "TSLA quarter, NVDA midpoint, TSLA three-quarter; floor((n-1)*fraction)",
        "date_ranges": {
            ticker: {"first_end_date": windows[ticker][0]["end_date"],
                     "last_end_date": windows[ticker][-1]["end_date"]}
            for ticker in TICKERS
        },
        "counts": counts,
    }
    write_json(ROOT / "results" / "render_manifest.json", manifest)
    for ticker in TICKERS:
        dates = manifest["date_ranges"][ticker]
        print(f"{ticker}: {counts[ticker]}, chart end dates "
              f"{dates['first_end_date']} to {dates['last_end_date']}")
    print(f"Combined: {counts['combined']}")
    print("Index: charts/index.csv; 3 samples: results/samples/")


if __name__ == "__main__":
    main()
