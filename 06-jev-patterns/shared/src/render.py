"""Unlabelled chart renderer following Experiment 01's fixed no-text style."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.src.data import SHARED, validate_ohlcv
os.environ.setdefault("MPLCONFIGDIR", str(SHARED / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

UP_COLOR, DOWN_COLOR = "#16856b", "#d34b52"
STYLES = ("candle", "ohlc", "line")


def render_chart(frame: pd.DataFrame, path: Path, n_bars=20, size=(640, 480), style="candle", mirror=False):
    """Render exactly the supplied visible bars; horizontal mirror reverses pixels."""
    validate_ohlcv(frame)
    if isinstance(n_bars, bool) or not isinstance(n_bars, int) or n_bars < 2 or len(frame) != n_bars:
        raise ValueError("Supply exactly n_bars visible OHLCV rows, with n_bars >= 2")
    style = style.lower()
    if style not in STYLES or len(size) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 64 for x in size):
        raise ValueError("Expected candle/ohlc/line and integer width/height >= 64")
    width, height = size
    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    price = fig.add_axes((0.035, 0.275, 0.93, 0.69))
    volume = fig.add_axes((0.035, 0.045, 0.93, 0.185))
    low, high = float(frame.Low.min()), float(frame.High.max())
    padding = max((high - low) * .07, high * .0001)
    price.set_ylim(low - padding, high + padding)
    volume.set_ylim(0, max(float(frame.Volume.max()) * 1.08, 1))
    for ax in (price, volume):
        ax.set_xlim(-.9, n_bars - .1)
        ax.set_axis_off()
        ax.set_facecolor("white")
    if style == "line":
        price.plot(np.arange(n_bars), frame.Close.to_numpy(), color="#244a64", linewidth=1.3)
    for i, bar in enumerate(frame.itertuples()):
        color = UP_COLOR if bar.Close >= bar.Open else DOWN_COLOR
        if style != "line":
            price.vlines(i, bar.Low, bar.High, color=color, linewidth=1.3)
            if style == "ohlc":
                price.hlines(bar.Open, i - .31, i, color=color, linewidth=1.3)
                price.hlines(bar.Close, i, i + .31, color=color, linewidth=1.3)
            elif bar.Close == bar.Open:
                price.hlines(bar.Open, i - .31, i + .31, color=color, linewidth=1.3)
            else:
                price.add_patch(Rectangle((i - .31, min(bar.Open, bar.Close)), .62,
                    abs(bar.Close - bar.Open), facecolor=color, edgecolor=color, linewidth=.8))
        volume.bar(i, bar.Volume, width=.62, color=color, linewidth=0)
    canvas.draw()
    image = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())
    if mirror:
        image = ImageOps.mirror(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    image.save(tmp, format="PNG", optimize=False)
    tmp.replace(path)
    fig.clear()
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bars", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-bars", type=int, default=20)
    p.add_argument("--size", type=int, nargs=2, default=[640, 480])
    p.add_argument("--style", choices=STYLES, default="candle")
    p.add_argument("--mirror", action="store_true")
    args = p.parse_args()
    frame = pd.read_parquet(args.bars) if args.bars.suffix == ".parquet" else pd.read_csv(args.bars, index_col=0, parse_dates=True)
    render_chart(frame, args.out, args.n_bars, tuple(args.size), args.style, args.mirror)


if __name__ == "__main__":
    main()
