"""Frozen synthetic chart patterns with paired, increasing noise (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT.parent
sys.path.insert(0, str(EXPERIMENT))
PROTOCOL_PATH = ROOT / "protocol.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def template_specs(protocol: dict | None = None) -> list[dict]:
    """Round-robin classes; all variation is drawn before rendering/scoring."""
    protocol = protocol or read_protocol()
    rng = np.random.default_rng(protocol["seed"])
    specs = []
    for number in range(protocol["clean_templates"]):
        label = protocol["classes"][number % len(protocol["classes"])]
        specs.append({
            "template_id": f"template_{number:04d}", "label": label,
            "n_bars": int(rng.integers(protocol["bars_min"], protocol["bars_max"] + 1)),
            "seed": int(rng.integers(0, 2**31)),
            "subtype": (number // len(protocol["classes"])) % 3,
            "base": float(rng.uniform(60, 180)),
            "amplitude": float(rng.uniform(12, 30)),
            "shoulder_delta": float(rng.uniform(-0.015, 0.015)),
            "neckline_delta": float(rng.uniform(-0.01, 0.01)),
        })
    return specs


def geometry(spec: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return normalized defining pivots and explicit completion metadata."""
    name, subtype = spec["label"], spec["subtype"]
    d, q = spec["shoulder_delta"], spec["neckline_delta"]
    completion = {"breakout_direction": None, "breakout_level": None}
    if name in ("Head and Shoulders", "Inverse Head and Shoulders"):
        x = [0, .16, .32, .49, .65, .8, 1]
        y = np.array([0, .7 + d, .27 + q, 1, .27 - q, .7 - d, -.18])
        completion = {"breakout_direction": "down", "breakout_level": .27 - q}
    elif name in ("Double Top", "Double Bottom"):
        x = [0, .25, .5, .75, 1]
        y = np.array([0, 1 + d, .25, 1 - d, -.18])
        completion = {"breakout_direction": "down", "breakout_level": .25}
    elif name == "Triangle":
        x = [0, .16, .32, .48, .64, .8, .9, 1]
        if subtype == 0:  # Symmetric, upper and lower boundaries converge.
            y = np.array([0, 1, .18, .82, .36, .64, .45, .94])
            completion = {"breakout_direction": "up", "breakout_level": .64}
        else:  # Ascending / descending are exact price inversions.
            y = np.array([0, .9, .22, .9, .44, .9, .66, 1.2])
            completion = {"breakout_direction": "up", "breakout_level": .9}
            if subtype == 2:
                y = 1 - y
                completion = {"breakout_direction": "down", "breakout_level": .1}
    elif name == "Bull Flag":
        x = [0, .2, .3, .42, .54, .66, .78, .86, 1]
        y = np.array([0, 1, .94, .69, .82, .57, .7, .49, 1.2])
        completion = {"breakout_direction": "up", "breakout_level": 1.0}
    elif name == "None":
        x = np.linspace(0, 1, 9)
        y = x if subtype == 0 else (1 - x if subtype == 1 else x**1.5)
    else:
        raise ValueError(f"Unknown pattern: {name}")
    if name in ("Inverse Head and Shoulders", "Double Bottom"):
        y = 1 - y
        completion = {"breakout_direction": "up",
                      "breakout_level": 1 - completion["breakout_level"]}
    return np.asarray(x), np.asarray(y), completion


def make_template(spec: dict, protocol: dict | None = None) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Create a clean price path and one paired noise path for all severities."""
    protocol = protocol or read_protocol()
    n = spec["n_bars"]
    padding = int(np.floor(n * protocol["padding_fraction_each_side"]))
    start, stop = padding, n - padding - 1
    x, y, completion = geometry(spec)
    pivots = np.rint(start + x * (stop - start)).astype(int)
    if len(np.unique(pivots)) != len(pivots):
        raise ValueError("Insufficient bars to represent distinct geometry pivots")
    prefix_slope = (y[1] - y[0]) / (pivots[1] - pivots[0]) * .3
    suffix_slope = (y[-1] - y[-2]) / (pivots[-1] - pivots[-2]) * .3
    path = np.interp(np.arange(n), pivots, y)
    path[:start] = y[0] + (np.arange(start) - start) * prefix_slope
    path[stop + 1:] = y[-1] + (np.arange(stop + 1, n) - stop) * suffix_slope
    rng = np.random.default_rng(spec["seed"])
    noise = np.zeros(n)
    all_pivots = [0, *pivots] if start else list(pivots)
    for left, right in zip(all_pivots[:-1], all_pivots[1:]):
        increments = rng.normal(size=right - left)
        bridge = np.r_[0.0, np.cumsum(increments)]
        bridge -= np.linspace(0, bridge[-1], len(bridge))
        noise[left:right + 1] = bridge
    norm = np.max(np.abs(noise))
    if norm:
        noise /= norm
    close = spec["base"] + spec["amplitude"] * path
    # Same volume distribution for every class; independent of pattern geometry.
    volume = rng.integers(800_000, 1_200_000, n).astype(float)
    frame = make_ohlcv(close, volume, spec["amplitude"])
    details = {
        **spec, "pattern_start_bar": start, "pattern_end_bar": stop,
        "padding_left_bars": start, "padding_right_bars": n - 1 - stop,
        "pivot_bars": pivots.tolist(), "pivot_close": close[pivots].tolist(),
        "breakout_direction": completion["breakout_direction"],
        "breakout_level": (None if completion["breakout_level"] is None else
                           spec["base"] + spec["amplitude"] * completion["breakout_level"]),
        "clean_close_range": float(np.ptp(close)),
    }
    return frame, noise, details


def make_ohlcv(close: np.ndarray, volume: np.ndarray, amplitude: float) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    opening = np.r_[close[0], close[:-1]]
    wick = amplitude * .007
    return pd.DataFrame({"Open": opening, "High": np.maximum(opening, close) + wick,
                         "Low": np.minimum(opening, close) - wick, "Close": close,
                         "Volume": volume}, index=pd.bdate_range("2000-01-03", periods=len(close)))


def apply_noise(clean: pd.DataFrame, noise: np.ndarray, severity: float,
                details: dict) -> pd.DataFrame:
    if severity < 0 or not np.isfinite(severity):
        raise ValueError("Noise severity must be finite and nonnegative")
    close = clean.Close.to_numpy() + severity * details["clean_close_range"] * noise
    return make_ohlcv(close, clean.Volume.to_numpy(), details["amplitude"])


def fixed_frame(frame: pd.DataFrame, n_bars: int = 100) -> pd.DataFrame:
    """Interpolate the same realization onto a fixed time-axis sampling grid."""
    if n_bars != 100:
        raise ValueError("The frozen framing control uses exactly 100 bars")
    old, new = np.linspace(0, 1, len(frame)), np.linspace(0, 1, n_bars)
    values = {column: np.interp(new, old, frame[column]) for column in frame.columns}
    return pd.DataFrame(values, index=pd.bdate_range("2000-01-03", periods=n_bars))


def dataset(protocol: dict | None = None):
    protocol = protocol or read_protocol()
    for spec in template_specs(protocol):
        clean, noise, details = make_template(spec, protocol)
        for noise_number, severity in enumerate(protocol["noise_levels"]):
            row = {**details, "noise_level": severity, "noise_number": noise_number,
                   "is_clean": severity == 0, "split": "eval",
                   "replication_unit": spec["template_id"],
                   "id": f"synth_{int(spec['template_id'].split('_')[-1]):04d}_{noise_number}"}
            yield apply_noise(clean, noise, severity, details), row


def sample_grid(index: pd.DataFrame, out: Path, protocol: dict) -> Path:
    """Annotated review artifact only; scored PNGs never contain captions."""
    cell_w, cell_h, title_h, label_w = 384, 240, 36, 170
    levels = [0, 2, 5]
    grid = Image.new("RGB", (label_w + len(levels) * cell_w,
                             title_h + len(protocol["classes"]) * cell_h), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=16)
    for column, level in enumerate(levels):
        draw.text((label_w + column * cell_w + 8, 8),
                  f"Noise = {protocol['noise_levels'][level]:g}", fill="black", font=font)
    samples = out / "results" / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    for row_number, label in enumerate(protocol["classes"]):
        template = index[index.label == label].iloc[0].template_id
        text_label = label.replace(" and ", " & ").replace("Head & Shoulders", "H&S")
        draw.text((8, title_h + row_number * cell_h + 100), text_label, fill="black", font=font)
        for column, level in enumerate(levels):
            row = index[(index.template_id == template) & (index.noise_number == level)].iloc[0]
            with Image.open(out / row.file) as chart:
                grid.paste(chart.resize((cell_w, cell_h), Image.Resampling.LANCZOS),
                           (label_w + column * cell_w, title_h + row_number * cell_h))
                if level == 0:
                    chart.save(samples / f"sample_{row_number + 1:02d}.png")
    target = samples / "synthetic_grid.png"
    grid.save(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--fixed-bars", type=int, choices=[100])
    parser.add_argument("--no-render", action="store_true", help="Export bars/index only for CPU checks")
    args = parser.parse_args()
    if args.fixed_bars and args.output_dir.resolve() == ROOT.resolve():
        parser.error("Use a distinct --output-dir for the fixed-100 control")
    protocol = read_protocol()
    from shared.src.render import render_chart
    out = args.output_dir
    for child in ("charts", "data/bars", "results"):
        (out / child).mkdir(parents=True, exist_ok=True)
    rows, artifacts = [], []
    for number, (frame, info) in enumerate(dataset(protocol), start=1):
        source_bars = len(frame)
        if args.fixed_bars:
            frame = fixed_frame(frame, args.fixed_bars)
        chart = Path("charts") / f"{info['id']}.png"
        bars = Path("data/bars") / f"{info['id']}.csv"
        frame.to_csv(out / bars, index_label="Date", float_format="%.12g")
        if not args.no_render:
            render_chart(frame, out / chart, n_bars=len(frame),
                         size=tuple(protocol["image_size"]), style=protocol["style"], mirror=False)
        row = {"file": chart.as_posix(), "ticker": "SYNTH",
               "end_date": "", "label": info["label"],
               "template_id": info["template_id"], "noise_level": info["noise_level"],
               "noise_number": info["noise_number"], "is_clean": info["is_clean"],
               "split": info["split"], "replication_unit": info["replication_unit"],
               "n_bars": len(frame), "source_n_bars": source_bars,
               "framing": "fixed_100" if args.fixed_bars else "variable",
               "bars_file": bars.as_posix(), "label_source": "synthetic_generator"}
        rows.append(row)
        artifacts.append({**info, "file": chart.as_posix(), "bars_file": bars.as_posix(),
                          "rendered_n_bars": len(frame), "bars_sha256": digest(out / bars),
                          "png_sha256": digest(out / chart) if not args.no_render else None})
        if number % 100 == 0:
            print(f"Exported {number}/1200 synthetic examples", flush=True)
    index = pd.DataFrame(rows)
    index.to_csv(out / "charts/index.csv", index=False)
    index.to_csv(out / "results/synthetic_index.csv", index=False)
    grid = None if args.no_render else sample_grid(index, out, protocol)
    counts = {label: {"clean": int(((index.label == label) & index.is_clean).sum()),
                      "noisy": int(((index.label == label) & ~index.is_clean).sum()),
                      "total": int((index.label == label).sum())} for label in protocol["classes"]}
    counts["total"] = {"clean": int(index.is_clean.sum()), "noisy": int((~index.is_clean).sum()),
                       "total": len(index)}
    (out / "results/label_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    manifest = {"protocol_sha256": digest(PROTOCOL_PATH), "generator_sha256": digest(Path(__file__)),
                "renderer_sha256": digest(EXPERIMENT / "shared/src/render.py"),
                "seed": protocol["seed"], "n_templates": len(template_specs(protocol)),
                "n_charts": 0 if args.no_render else len(index), "noise_levels": protocol["noise_levels"],
                "fixed_bars": args.fixed_bars, "split": "evaluation_only",
                "inference_run": False, "replication_unit": "template_id",
                "sample_grid": None if grid is None else grid.relative_to(out).as_posix(),
                "runtime": {"numpy": np.__version__, "pandas": pd.__version__}, "artifacts": artifacts}
    (out / "results/render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
