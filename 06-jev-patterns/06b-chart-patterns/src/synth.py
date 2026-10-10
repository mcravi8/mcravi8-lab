"""Frozen synthetic chart patterns with matched nuisance variation (CPU only)."""
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
    """One common nuisance distribution, independent of the round-robin class."""
    protocol = protocol or read_protocol()
    rng = np.random.default_rng(protocol["seed"])
    specs = []
    for number in range(protocol["clean_templates"]):
        n = int(rng.integers(protocol["bars_min"], protocol["bars_max"] + 1))
        padding = [(a, b) for a in range(1, n // 4) for b in range(1, n // 4)
                   if .1 <= a / (n - a - b) <= .2 and .1 <= b / (n - a - b) <= .2]
        left, right = padding[int(rng.integers(len(padding)))]
        specs.append({
            "template_id": f"template_{number:04d}",
            "label": protocol["classes"][number % len(protocol["classes"])],
            "n_bars": n, "padding_left_bars": left, "padding_right_bars": right,
            "subtype": (number // len(protocol["classes"])) % 3,
            "none_subtype": (number // len(protocol["classes"])) % 6,
            "seed": int(rng.integers(0, 2**31)),
            "geometry_seed": int(rng.integers(0, 2**31)),
            "texture_seed": int(rng.integers(0, 2**31)),
            "margin_seed": int(rng.integers(0, 2**31)),
            "noise_seed": int(rng.integers(0, 2**31)),
            "bar_seed": int(rng.integers(0, 2**31)),
            "volume_seed": int(rng.integers(0, 2**31)),
            "base": float(np.exp(rng.uniform(np.log(40), np.log(220)))),
            "target_logvol": float(np.exp(rng.uniform(np.log(.004), np.log(.018)))),
            "microtexture": float(rng.uniform(.035, .11)),
            "gap_scale": float(rng.uniform(.08, .3)),
            "wick_scale": float(rng.uniform(.12, .4)),
            "volume_level": float(np.exp(rng.uniform(np.log(400_000), np.log(3_000_000)))),
            "volume_logstd": float(rng.uniform(.12, .35)),
        })
    return specs


def jitter_positions(base, rng):
    intervals = np.diff(base) * rng.uniform(.65, 1.35, len(base) - 1)
    return np.r_[0., np.cumsum(intervals) / intervals.sum()]


def obvious_completed_pattern(x, y):
    """Conservative source-control-point screen, not a real-market detector.

    Only a formation occupying most of the source window is excluded. The
    screen does not reject a None path merely for roughness or classifiability.
    """
    x, y = np.asarray(x), np.asarray(y)
    span = float(np.ptp(y))
    if span <= 0:
        return None
    y = (y - y.min()) / span
    sign = np.sign(np.diff(y))
    peaks = np.flatnonzero((sign[:-1] > 0) & (sign[1:] < 0)) + 1
    troughs = np.flatnonzero((sign[:-1] < 0) & (sign[1:] > 0)) + 1
    for inverted in (False, True):
        z = 1 - y if inverted else y
        tops, bottoms = (troughs, peaks) if inverted else (peaks, troughs)
        if len(tops) == 3:
            a, b, c = tops
            left = bottoms[(bottoms > a) & (bottoms < b)]
            right = bottoms[(bottoms > b) & (bottoms < c)]
            if len(left) == len(right) == 1 and x[c] - x[a] >= .4:
                lo, hi = z[left[0]], z[right[0]]
                if abs(z[a] - z[c]) <= .09 and z[b] - max(z[a], z[c]) >= .12 and abs(lo - hi) <= .12 and z[-1] < min(lo, hi) - .06:
                    return "Inverse Head and Shoulders" if inverted else "Head and Shoulders"
        if len(tops) == 2:
            a, b = tops
            middle = bottoms[(bottoms > a) & (bottoms < b)]
            if len(middle) == 1 and x[b] - x[a] >= .3:
                valley = z[middle[0]]
                if abs(z[a] - z[b]) <= .09 and min(z[a], z[b]) - valley >= .25 and z[-1] < valley - .06:
                    return "Double Bottom" if inverted else "Double Top"
    if len(peaks) >= 2 and len(troughs) >= 2:
        upper, lower = y[peaks], y[troughs]
        upper_slope = np.polyfit(x[peaks], upper, 1)[0]
        lower_slope = np.polyfit(x[troughs], lower, 1)[0]
        first_gap = upper[0] - lower[0]
        last_gap = upper[-1] - lower[-1]
        if (first_gap > .3 and 0 < last_gap < .55 * first_gap and
                upper_slope <= .12 and lower_slope >= -.12 and
                np.all(np.diff(upper) <= .03) and np.all(np.diff(lower) >= -.03) and
                (y[-1] > upper[-1] + .10 or y[-1] < lower[-1] - .10)):
            return "Triangle"
        # A flag needs an early strong pole, two declining highs/lows, a
        # reasonably parallel narrow channel, and a final break above the pole.
        first_peak = peaks[0]
        if (x[first_peak] <= .35 and y[first_peak] - y[0] >= .55 and
                upper_slope < -.15 and lower_slope < -.15 and
                abs(upper_slope - lower_slope) < .2 and
                np.all(np.diff(upper) < .02) and np.all(np.diff(lower) < .02) and
                y[first_peak] - lower.min() < .55 and y[-1] > y[first_peak] + .06):
            return "Bull Flag"
    return None


def none_geometry(spec, rng):
    families = ("random_walk", "choppy_range", "trend_with_pullbacks",
                "incomplete_formation", "broken_reversal", "broken_consolidation")
    family = families[spec["none_subtype"]]
    rejected = 0
    for _ in range(1000):
        count = int(rng.integers(7, 11))
        x = jitter_positions(np.linspace(0, 1, count), rng)
        if family in ("incomplete_formation", "broken_reversal", "broken_consolidation"):
            x, y, source_kind = hard_negative_geometry(spec, rng, family)
        elif family == "random_walk":
            y = np.r_[0., np.cumsum(rng.normal(size=count - 1) * np.sqrt(np.diff(x)))]
        elif family == "choppy_range":
            rho = rng.uniform(-.15, .35)
            y = np.empty(count); y[0] = rng.normal()
            for position in range(1, count):
                y[position] = rho * y[position - 1] + rng.normal()
        elif family == "trend_with_pullbacks":
            direction = rng.choice([-1., 1.])
            increments = direction * rng.uniform(.7, 1.3) * np.diff(x) + rng.uniform(.3, .7) * np.sqrt(np.diff(x)) * rng.normal(size=count - 1)
            if not (np.any(increments > 0) and np.any(increments < 0)):
                rejected += 1
                continue
            y = np.r_[0., np.cumsum(increments)]
        y = (y - y.min()) / np.ptp(y)
        if obvious_completed_pattern(x, y) is not None:
            rejected += 1
            continue
        return x, y, {"breakout_direction": None, "breakout_level": None,
                      "none_family": family, "none_screen_rejections": rejected,
                      "none_source_kind": source_kind if family in ("incomplete_formation", "broken_reversal", "broken_consolidation") else family}
    raise ValueError("None source rejection exceeded its frozen 1000-attempt bound")



def hard_negative_geometry(spec, rng, family):
    """Rule-defined nonpatterns; acceptance never uses classifier predictions."""
    if family == "incomplete_formation":
        source = rng.choice(["Head and Shoulders", "Inverse Head and Shoulders", "Double Top",
                             "Double Bottom", "Triangle", "Bull Flag"])
        local = {**spec, "label": source, "geometry_seed": int(rng.integers(0, 2**31))}
        x, y, completion = geometry(local)
        span = float(np.ptp(y[:-1]))
        if source == "Triangle":
            # The last envelope gap leaves room for a random margin in either
            # direction without completing the formation.
            if local["subtype"] == 0:
                y = np.array([0., 1., .12, .88, .23, .76, .30, .53])
            else:
                y = np.array([0., 1., .20, 1., .36, 1., .50, .75])
                if local["subtype"] == 2:
                    y = 1 - y
        elif source == "Bull Flag":
            y[-1] = 1. - .24 * span  # no confirmed break above the pole
        else:
            away = .23 * span
            y[-1] = completion["breakout_level"] + (away if completion["breakout_direction"] == "down" else -away)
        return x, y, "incomplete_" + source.lower().replace(" ", "_")
    if family == "broken_reversal":
        source = "Head and Shoulders" if rng.random() < .5 else "Double Top"
        local = {**spec, "label": source, "geometry_seed": int(rng.integers(0, 2**31))}
        x, y, _ = geometry(local)
        if source == "Head and Shoulders":
            y[3] = (y[1] + y[5]) / 2 + rng.uniform(-.01, .01)
            source_kind = "triple_top"
        else:
            y[3] = y[2] + (y[1] - y[2]) * rng.uniform(.45, .68)
            source_kind = "unequal_double_top"
        if rng.random() < .5:
            y = 1 - y
            source_kind = source_kind.replace("top", "bottom")
        return x, y, source_kind
    if family == "broken_consolidation":
        if rng.random() < .5:
            local = {**spec, "label": "Triangle", "geometry_seed": int(rng.integers(0, 2**31))}
            x, _, _ = geometry(local)
            expansion = rng.uniform(.12, .22)
            y = np.array([0., 1., -expansion, 1 + expansion, -2 * expansion,
                          1 + 2 * expansion, -3 * expansion, 1 + 2 * expansion + rng.uniform(.3, .45)])
            if rng.random() < .5:
                y = 1 - y
            return x, y, "diverging_bounds"
        local = {**spec, "label": "Bull Flag", "geometry_seed": int(rng.integers(0, 2**31))}
        x, y, _ = geometry(local)
        # A rising parallel consolidation violates the defined downward flag.
        # Merely reversing the initial pole would create a new upward pole
        # on the rebound, leaving a valid local flag in the negative class.
        fall = y[2] - y[6]
        y[2:8] += 2 * fall * np.array([0., 0., .5, .5, 1., 1.])
        return x, y, "rising_consolidation"
    raise ValueError("Unknown hard-negative family")


def geometry(spec: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Random valid pivots in latent log-price units; classes share nuisance draws."""
    rng = np.random.default_rng(spec["geometry_seed"])
    name, subtype = spec["label"], spec["subtype"]
    d = rng.uniform(-.015, .015)
    if name in ("Head and Shoulders", "Inverse Head and Shoulders"):
        neck, tilt = rng.uniform(.18, .35), rng.uniform(-.025, .025)
        shoulders = rng.uniform(.60, .83)
        x = jitter_positions([0, .16, .32, .49, .65, .8, 1], rng)
        y = np.array([neck - rng.uniform(.12, .30), shoulders + d, neck + tilt,
                      1., neck - tilt, shoulders - d, neck - tilt - rng.uniform(.34, .5)])
        completion = {"breakout_direction": "down", "breakout_level": neck - tilt}
    elif name in ("Double Top", "Double Bottom"):
        valley = rng.uniform(.18, .45)
        x = jitter_positions([0, .25, .5, .75, 1], rng)
        y = np.array([valley - rng.uniform(.1, .3), 1 + d, valley, 1 - d,
                      valley - rng.uniform(.34, .5)])
        completion = {"breakout_direction": "down", "breakout_level": valley}
    elif name == "Triangle":
        x = jitter_positions([0, .16, .32, .48, .64, .8, .9, 1], rng)
        if subtype == 0:
            last_upper = rng.uniform(.62, .72)
            y = np.array([0, 1, rng.uniform(.1, .22), rng.uniform(.80, .90),
                          rng.uniform(.30, .40), last_upper, rng.uniform(.44, .50),
                          last_upper + rng.uniform(.36, .52)])
            completion = {"breakout_direction": "up", "breakout_level": last_upper}
        else:
            ceiling = rng.uniform(.82, 1.)
            y = np.array([0, ceiling, rng.uniform(.20, .30), ceiling + rng.uniform(-.015, .015),
                          rng.uniform(.42, .50), ceiling + rng.uniform(-.015, .015),
                          rng.uniform(.61, .68), ceiling + rng.uniform(.36, .52)])
            completion = {"breakout_direction": "up", "breakout_level": max(y[1], y[3], y[5])}
            if subtype == 2:
                y = 1 - y
                completion = {"breakout_direction": "down", "breakout_level": 1 - completion["breakout_level"]}
    elif name == "Bull Flag":
        x = jitter_positions([0, .16, .25, .37, .49, .61, .73, .86, 1], rng)
        top, fall, width = rng.uniform(.92, .99), rng.uniform(.18, .28), rng.uniform(.17, .25)
        upper = top - fall * np.array([0., .5, 1.])
        lower = upper - width + rng.uniform(-.012, .012, 3)
        y = np.array([0., 1., upper[0], lower[0], upper[1], lower[1], upper[2], lower[2], 1 + rng.uniform(.36, .52)])
        completion = {"breakout_direction": "up", "breakout_level": 1.}
    elif name == "None":
        return none_geometry(spec, rng)
    else:
        raise ValueError(f"Unknown pattern: {name}")
    if name in ("Inverse Head and Shoulders", "Double Bottom"):
        y = 1 - y
        completion = {"breakout_direction": "up", "breakout_level": 1 - completion["breakout_level"]}
    return np.asarray(x), y, {**completion, "none_family": None, "none_screen_rejections": 0, "none_source_kind": None}


def bridge(n, rng):
    path = np.r_[0., np.cumsum(rng.normal(size=n - 1))]
    path -= np.linspace(0., path[-1], n)
    norm = np.max(np.abs(path))
    return path / norm if norm else path


def make_ohlcv(close, volume, amplitude, *, seed=0, sigma=None, gap_scale=.2, wick_scale=.2):
    """A class-independent, paired opening-gap and wick process."""
    close = np.asarray(close, dtype=float)
    sigma = float(sigma if sigma is not None else amplitude / np.median(close))
    rng = np.random.default_rng(seed)
    opening = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, sigma * gap_scale, len(close)))
    upper = sigma * wick_scale * rng.lognormal(-.5 * .3**2, .3, len(close))
    lower = sigma * wick_scale * rng.lognormal(-.5 * .3**2, .3, len(close))
    return pd.DataFrame({"Open": opening, "High": np.maximum(opening, close) * np.exp(upper),
                         "Low": np.minimum(opening, close) * np.exp(-lower), "Close": close,
                         "Volume": volume}, index=pd.bdate_range("2000-01-03", periods=len(close)))


def make_template(spec: dict, protocol: dict | None = None) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """One stochastic clean parent, a nested corruption bridge, and provenance."""
    n = spec["n_bars"]
    start, stop = spec["padding_left_bars"], n - spec["padding_right_bars"] - 1
    x, y, completion = geometry(spec)
    pivots = np.rint(start + x * (stop - start)).astype(int)
    if len(np.unique(pivots)) != len(pivots):
        raise ValueError("Insufficient bars to represent distinct geometry pivots")
    path = np.interp(np.arange(n), pivots, y)
    scale = float(np.ptp(y))
    texture_rng = np.random.default_rng(spec["texture_seed"])
    texture = texture_rng.normal(size=n)
    for position in range(1, n):
        texture[position] += .45 * texture[position - 1]
    texture -= np.interp(np.arange(n), pivots, texture[pivots])
    texture[:start] = 0.; texture[stop + 1:] = 0.
    texture /= max(float(np.max(np.abs(texture))), 1e-15)
    path += spec["microtexture"] * scale * texture
    margin_rng = np.random.default_rng(spec["margin_seed"])
    margins = []
    for count in (start, n - stop - 1):
        displacement = margin_rng.uniform(-.14, .14)
        margin = np.linspace(0, displacement, count + 1) + margin_rng.uniform(.015, .06) * bridge(count + 1, margin_rng)
        margin *= min(1., .16 / max(float(np.max(np.abs(margin))), 1e-15))
        margins.append(margin * scale)
    path[:start] = y[0] + margins[0][::-1][:-1]
    path[stop + 1:] = y[-1] + margins[1][1:]
    # The SAME independently drawn standard deviation of daily log returns is
    # imposed on every parent class; absolute price and volume are independent.
    factor = spec["target_logvol"] / np.std(np.diff(path), ddof=1)
    log_close = np.log(spec["base"]) + factor * (path - path[0])
    close = np.exp(log_close)
    vrng = np.random.default_rng(spec["volume_seed"])
    volume = spec["volume_level"] * vrng.lognormal(-.5 * spec["volume_logstd"]**2, spec["volume_logstd"], n)
    kwargs = {"seed": spec["bar_seed"], "sigma": spec["target_logvol"],
              "gap_scale": spec["gap_scale"], "wick_scale": spec["wick_scale"]}
    amplitude = spec["base"] * spec["target_logvol"]
    frame = make_ohlcv(close, volume, amplitude, **kwargs)
    noise = bridge(n, np.random.default_rng(spec["noise_seed"]))
    details = {
        **spec, "amplitude": amplitude, "pattern_start_bar": start, "pattern_end_bar": stop,
        "pivot_bars": pivots.tolist(), "pivot_close": close[pivots].tolist(),
        "breakout_direction": completion["breakout_direction"],
        "breakout_level": (None if completion["breakout_level"] is None else
                           float(np.exp(np.log(spec["base"]) + factor * (completion["breakout_level"] - path[0])))),
        "none_family": completion["none_family"], "none_screen_rejections": completion["none_screen_rejections"],
        "none_source_kind": completion["none_source_kind"],
        "clean_close_range": float(np.ptp(close)), "clean_log_range": float(np.ptp(log_close)),
        "realized_clean_logvol": float(np.std(np.diff(log_close), ddof=1)),
        "lead_in_log_return": float(log_close[start] - log_close[0]),
        "lead_out_log_return": float(log_close[-1] - log_close[stop]),
    }
    return frame, noise, details


def apply_noise(clean: pd.DataFrame, noise: np.ndarray, severity: float, details: dict) -> pd.DataFrame:
    if severity < 0 or not np.isfinite(severity):
        raise ValueError("Noise severity must be finite and nonnegative")
    close = np.exp(np.log(clean.Close.to_numpy()) + severity * details["clean_log_range"] * noise)
    return make_ohlcv(close, clean.Volume.to_numpy(), details["amplitude"], seed=details["bar_seed"],
                      sigma=details["target_logvol"], gap_scale=details["gap_scale"], wick_scale=details["wick_scale"])


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
                   "is_clean": severity == 0, "split": "eval", "replication_unit": spec["template_id"],
                   "id": f"synth_{int(spec['template_id'].split('_')[-1]):04d}_{noise_number}"}
            yield apply_noise(clean, noise, severity, details), row


def population_summary(protocol=None):
    protocol = protocol or read_protocol()
    rows = [info for _, info in dataset(protocol) if info["is_clean"]]
    summary = {"n_clean_parents": len(rows), "none_family_counts": {}, "classes": {}}
    for family in protocol["none_families"]:
        summary["none_family_counts"][family] = sum(row["none_family"] == family for row in rows)
    summary["none_source_screen_rejections"] = sum(row["none_screen_rejections"] for row in rows)
    for label in protocol["classes"]:
        selected = [row for row in rows if row["label"] == label]
        entry = {"n": len(selected)}
        for key in ("n_bars", "base", "realized_clean_logvol", "microtexture", "pattern_start_bar", "lead_in_log_return", "lead_out_log_return", "volume_logstd"):
            values = np.array([row[key] for row in selected])
            entry[key] = {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())}
        entry["lead_in_up_fraction"] = float(np.mean([row["lead_in_log_return"] > 0 for row in selected]))
        entry["lead_out_up_fraction"] = float(np.mean([row["lead_out_log_return"] > 0 for row in selected]))
        summary["classes"][label] = entry
    return summary


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



def diversity_grid(index: pd.DataFrame, out: Path, protocol: dict) -> Path:
    """First three positive parents per class; first six None parents in two rows."""
    cell_w, cell_h, label_w = 384, 240, 170
    grid = Image.new("RGB", (label_w + 3 * cell_w, 40 + (len(protocol["classes"]) + 1) * cell_h), "white")
    draw = ImageDraw.Draw(grid); font = ImageFont.load_default(size=16)
    for column in range(3):
        draw.text((label_w + column * cell_w + 8, 8), f"Independent clean parent {column + 1}", fill="black", font=font)
    for row_number, label in enumerate(protocol["classes"]):
        clean = index[index.label.eq(label) & index.is_clean].sort_values("template_id").head(6 if label == "None" else 3)
        caption = label.replace("Head and Shoulders", "H&S")
        draw.text((8, 40 + row_number * cell_h + 100), caption, fill="black", font=font)
        for number, (_, row) in enumerate(clean.iterrows()):
            column, display_row = number % 3, row_number + number // 3
            with Image.open(out / row.file) as chart:
                grid.paste(chart.resize((cell_w, cell_h), Image.Resampling.LANCZOS),
                           (label_w + column * cell_w, 40 + display_row * cell_h))
            if label == "None":
                draw.text((label_w + column * cell_w + 8, 40 + display_row * cell_h + 2),
                          row.none_family.replace("_", " "), fill="black", font=font)
    target = out / "results/samples/clean_diversity_grid.png"
    target.parent.mkdir(parents=True, exist_ok=True); grid.save(target)
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
                         size=tuple(protocol["image_size"]), style=protocol["style"], mirror=False,
                         volume_color=protocol["volume_color"])
        row = {"file": chart.as_posix(), "ticker": "SYNTH",
               "end_date": "", "label": info["label"],
               "template_id": info["template_id"], "noise_level": info["noise_level"],
               "noise_number": info["noise_number"], "is_clean": info["is_clean"],
               "split": info["split"], "replication_unit": info["replication_unit"],
               "n_bars": len(frame), "source_n_bars": source_bars,
               "framing": "fixed_100" if args.fixed_bars else "variable",
               "bars_file": bars.as_posix(), "label_source": "synthetic_generator",
               "none_family": info["none_family"] or "", "none_source_kind": info["none_source_kind"] or "",
               "volume_color": protocol["volume_color"]}
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
    diversity = None if args.no_render else diversity_grid(index, out, protocol)
    counts = {label: {"clean": int(((index.label == label) & index.is_clean).sum()),
                      "noisy": int(((index.label == label) & ~index.is_clean).sum()),
                      "total": int((index.label == label).sum())} for label in protocol["classes"]}
    counts["total"] = {"clean": int(index.is_clean.sum()), "noisy": int((~index.is_clean).sum()),
                       "total": len(index)}
    (out / "results/label_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    manifest = {"protocol_sha256": digest(PROTOCOL_PATH), "generator_sha256": digest(Path(__file__)),
                "renderer_sha256": digest(EXPERIMENT / "shared/src/render.py"),
                "context_sha256": digest(EXPERIMENT / "shared/context/chart_patterns.txt"),
                "protocol_version": protocol["schema_version"], "volume_color": protocol["volume_color"],
                "seed": protocol["seed"], "n_templates": len(template_specs(protocol)),
                "n_charts": 0 if args.no_render else len(index), "noise_levels": protocol["noise_levels"],
                "fixed_bars": args.fixed_bars, "split": "evaluation_only",
                "inference_run": False, "replication_unit": "template_id",
                "sample_grid": None if grid is None else grid.relative_to(out).as_posix(),
                "clean_diversity_grid": None if diversity is None else diversity.relative_to(out).as_posix(),
                "runtime": {"numpy": np.__version__, "pandas": pd.__version__}, "artifacts": artifacts}
    (out / "results/render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "results/population_stats.json").write_text(json.dumps(population_summary(protocol), indent=2) + "\n")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
