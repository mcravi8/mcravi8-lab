"""TA-Lib and independently calculated pandas candle labels; CPU only.

The pandas detector implements the same published definitions with rolling
Series operations. Agreement catches implementation differences, not the
subjectivity of the underlying definitions. All computations are causal.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import talib
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
PART = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.src.data import load, universe, validate_ohlcv

RULES_PATH = PART / "rules.json"
RULES = json.loads(RULES_PATH.read_text())
CLASSES = RULES["classes"]
PATTERNS = CLASSES[:-1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_trend(flags: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Five close-to-close changes ending immediately before pattern begins."""
    out = flags.copy()
    for label, span in RULES["prior_trend"]["span"].items():
        trend = close.shift(span) / close.shift(span + 5) - 1
        out[label] &= trend.lt(0) if label in RULES["prior_trend"]["negative"] else trend.gt(0)
    return out.astype(bool)


def talib_detector(frame: pd.DataFrame, trend: bool = True) -> pd.DataFrame:
    validate_ohlcv(frame)
    args = [frame[c].to_numpy(dtype=np.float64) for c in ["Open", "High", "Low", "Close"]]
    engulf = talib.CDLENGULFING(*args)
    flags = pd.DataFrame({
        "Doji": talib.CDLDOJI(*args) != 0,
        "Hammer": talib.CDLHAMMER(*args) != 0,
        "Shooting Star": talib.CDLSHOOTINGSTAR(*args) != 0,
        "Bullish Engulfing": engulf > 0,
        "Bearish Engulfing": engulf < 0,
        "Morning Star": talib.CDLMORNINGSTAR(*args, penetration=RULES["star_penetration"]) != 0,
        "Evening Star": talib.CDLEVENINGSTAR(*args, penetration=RULES["star_penetration"]) != 0,
    }, index=frame.index)
    return apply_trend(flags, frame.Close) if trend else flags


def pandas_detector(frame: pd.DataFrame, trend: bool = True) -> pd.DataFrame:
    """Independent vectorized implementation of TA-Lib default rule definitions."""
    validate_ohlcv(frame)
    o, h, l, c = [frame[x].astype(float) for x in ["Open", "High", "Low", "Close"]]
    body = (c - o).abs()
    body_low, body_high = frame[["Open", "Close"]].min(axis=1), frame[["Open", "Close"]].max(axis=1)
    upper, lower = h - body_high, body_low - l
    mean_body = body.shift(1).rolling(10).mean()
    tiny = (h - l).shift(1).rolling(10).mean() * 0.1
    near = (h - l).shift(2).rolling(5).mean() * 0.2
    up = c >= o
    engulf_up = up & ~up.shift(1, fill_value=True) & (
        ((c >= o.shift(1)) & (o < c.shift(1))) | ((c > o.shift(1)) & (o <= c.shift(1))))
    engulf_down = ~up & up.shift(1, fill_value=False) & (
        ((o >= c.shift(1)) & (c < o.shift(1))) | ((o > c.shift(1)) & (c <= o.shift(1))))
    first_long = body.shift(2) > mean_body.shift(2)
    middle_short = body.shift(1) <= mean_body.shift(1)
    last_nonshort = body > mean_body
    flags = pd.DataFrame({
        "Doji": body <= tiny,
        "Hammer": (body < mean_body) & (lower > body) & (upper < tiny) & (body_low <= l.shift(1) + near),
        "Shooting Star": (body < mean_body) & (upper > body) & (lower < tiny) & (body_low > body_high.shift(1)),
        "Bullish Engulfing": engulf_up,
        "Bearish Engulfing": engulf_down,
        "Morning Star": first_long & ~up.shift(2, fill_value=True) & middle_short & (body_high.shift(1) < body_low.shift(2)) & last_nonshort & up & (c > c.shift(2) + RULES["star_penetration"] * body.shift(2)),
        "Evening Star": first_long & up.shift(2, fill_value=False) & middle_short & (body_low.shift(1) > body_high.shift(2)) & last_nonshort & ~up & (c < c.shift(2) - RULES["star_penetration"] * body.shift(2)),
    }, index=frame.index).fillna(False)
    # Mirror function-specific TA-Lib minimum-history requirements.
    for label, lookback in {"Doji": 10, "Hammer": 11, "Shooting Star": 11,
                            "Bullish Engulfing": 2, "Bearish Engulfing": 2,
                            "Morning Star": 12, "Evening Star": 12}.items():
        flags.iloc[:lookback, flags.columns.get_loc(label)] = False
    return apply_trend(flags, c) if trend else flags.astype(bool)


def split_for_window(start: pd.Timestamp, end: pd.Timestamp) -> str | None:
    if start > end:
        raise ValueError("Window starts after it ends")
    if end <= pd.Timestamp(RULES["train_end"]):
        return "train"
    if start >= pd.Timestamp(RULES["test_start"]):
        return "test"
    return None


def agreement_table(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    if not a.index.equals(b.index) or list(a.columns) != list(b.columns):
        raise ValueError("Detector alignment mismatch")
    result = {}
    for label in PATTERNS:
        ta, independent = a[label], b[label]
        both, either = int((ta & independent).sum()), int((ta | independent).sum())
        result[label] = {"eligible_windows": len(a), "talib_positive": int(ta.sum()),
                         "pandas_positive": int(independent.sum()), "both_positive": both,
                         "either_positive": either,
                         "positive_jaccard": both / either if either else None,
                         "agreement_given_talib_positive": both / int(ta.sum()) if ta.any() else None,
                         "binary_agreement": float((ta == independent).mean()) if len(a) else None}
    an, bn = ~a.any(axis=1), ~b.any(axis=1)
    both, either = int((an & bn).sum()), int((an | bn).sum())
    result["None"] = {"eligible_windows": len(a), "talib_positive": int(an.sum()),
                      "pandas_positive": int(bn.sum()), "both_positive": both,
                      "either_positive": either, "positive_jaccard": both / either if either else None,
                      "agreement_given_talib_positive": both / int(an.sum()) if an.any() else None,
                      "binary_agreement": float((an == bn).mean()) if len(a) else None}
    return result


def candidates(frame: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    a, b = talib_detector(frame), pandas_detector(frame)
    rows, eligible, excluded = [], [], defaultdict(int)
    for i in range(RULES["n_bars"] - 1, len(frame)):
        end, start = frame.index[i], frame.index[i - RULES["n_bars"] + 1]
        split = split_for_window(start, end)
        if split is None:
            excluded["embargo_or_crossing_window"] += 1
            continue
        eligible.append(i)
        ta, independent = a.iloc[i].to_numpy(), b.iloc[i].to_numpy()
        if not np.array_equal(ta, independent):
            excluded["detector_disagreement"] += 1
            continue
        if ta.sum() > 1:
            excluded["multiple_labels"] += 1
            continue
        label = PATTERNS[int(np.flatnonzero(ta)[0])] if ta.any() else "None"
        obs = hashlib.sha256(f"06a:{ticker}:{end.date()}".encode()).hexdigest()[:20]
        rows.append({"id": obs, "file": f"charts/{obs}.png", "ticker": ticker,
                     "start_date": str(start.date()), "window_start": str(start.date()),
                     "bars_file": f"data/bars/{obs}.csv", "end_date": str(end.date()),
                     "year": int(end.year), "split": split, "label": label,
                     "bar_index": i, "n_bars": RULES["n_bars"]})
    return pd.DataFrame(rows), a.iloc[eligible], b.iloc[eligible], dict(excluded)


def stratified_sample(frame: pd.DataFrame, n: int, seed: int = RULES["seed"]) -> pd.DataFrame:
    """Round robin ticker/year cells; stable hashes avoid ticker/order preferences."""
    if len(frame) <= n:
        return frame.sort_values(["ticker", "end_date"]).copy()
    groups = []
    for key, group in frame.groupby(["ticker", "year"], sort=True):
        group = group.copy()
        group["_rank"] = [hashlib.sha256(f"{seed}:{x}".encode()).hexdigest() for x in group.id]
        group = group.sort_values("_rank").drop(columns="_rank")
        order = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
        groups.append((order, group))
    groups.sort(key=lambda item: item[0])
    chosen, depth = [], 0
    while len(chosen) < n:
        for _, group in groups:
            if depth < len(group):
                chosen.append(group.iloc[[depth]])
                if len(chosen) == n:
                    break
        depth += 1
    return pd.concat(chosen, ignore_index=True)


def diagnostic(frame: pd.DataFrame, label: str) -> dict:
    """Numerical aids for visual audit, never new inclusion criteria."""
    b = (frame.Close - frame.Open).abs()
    r = frame.High - frame.Low
    last = frame.iloc[-1]
    bl, bh = min(last.Open, last.Close), max(last.Open, last.Close)
    span = RULES["prior_trend"]["span"].get(label, 1)
    return {"last_body": float(b.iloc[-1]), "mean_prior_10_bodies": float(b.iloc[-11:-1].mean()),
            "tiny_wick_threshold": float(r.iloc[-11:-1].mean() * 0.1),
            "lower_wick": float(bl-last.Low), "upper_wick": float(last.High-bh),
            "prior_5_day_return": float(frame.Close.iloc[-span-1] / frame.Close.iloc[-span-6] - 1),
            "last_3_ohlc": frame[["Open", "High", "Low", "Close"]].tail(3).round(8).to_dict("records")}


def write_contact_sheets(review: pd.DataFrame, results: Path) -> None:
    # Audit captions live outside the model input PNGs.
    for page, offset in enumerate(range(0, len(review), 10), 1):
        sheet = Image.new("RGB", (1280, 2600), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, row in enumerate(review.iloc[offset:offset+10].itertuples()):
            x, y = (slot % 2) * 640, (slot // 2) * 520
            with Image.open(PART / row.file) as img:
                sheet.paste(img.convert("RGB"), (x, y+35))
            draw.text((x+8, y+4), f"{offset+slot+1:02d} {row.label} | {row.id} | {row.split}", fill="black")
        sheet.save(results / f"audit_contact_{page}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()
    results = PART / "results"
    results.mkdir(exist_ok=True, parents=True)
    frozen = {"rules_sha256": digest(RULES_PATH), "source_sha256": digest(Path(__file__)),
              "context_sha256": digest(ROOT / "shared/context/candles.txt")}
    all_candidates, all_a, all_b, raw, exclusions = [], [], [], {}, defaultdict(int)
    for ticker in universe()["tickers"]:
        frame = load(ticker)
        raw[ticker] = frame
        candidate, a, b, excluded = candidates(frame, ticker)
        all_candidates.append(candidate)
        all_a.append(a.reset_index(drop=True))
        all_b.append(b.reset_index(drop=True))
        for key, value in excluded.items():
            exclusions[key] += value
        print(f"{ticker}: {len(candidate)} single-label agreements", flush=True)
    pool = pd.concat(all_candidates, ignore_index=True)
    a, b = pd.concat(all_a, ignore_index=True), pd.concat(all_b, ignore_index=True)
    selected, counts = [], {}
    for label in CLASSES:
        counts[label] = {}
        for split in ["train", "test"]:
            subset = pool[(pool.label == label) & (pool.split == split)]
            target = RULES[f"per_class_{split}"]
            picked = stratified_sample(subset, target)
            selected.append(picked)
            counts[label][split] = {"available": len(subset), "selected": len(picked),
                                    "target": target, "deficit": max(0, target-len(picked)),
                                    "tickers": int(picked.ticker.nunique()), "years": int(picked.year.nunique())}
    index = pd.concat(selected, ignore_index=True).sort_values(["split", "label", "ticker", "end_date"]).reset_index(drop=True)
    (PART / "data").mkdir(exist_ok=True)
    pool.to_csv(PART / "data/candidates.csv", index=False)
    index.to_csv(results / "index.csv", index=False)
    (results / "label_counts.json").write_text(json.dumps({"total": len(index), "counts": counts}, indent=2)+"\n")
    stats = {"eligible_windows": len(a), "exact_detector_set_agreement": int((a == b).all(axis=1).sum()),
             "single_label_candidates": len(pool), "excluded": dict(exclusions), "per_class": agreement_table(a,b),
             "interpretation": "Independent implementations of the same rules; agreement is not independent expert validation."}
    (results / "agreement.json").write_text(json.dumps(stats, indent=2)+"\n")
    strata = index.groupby(["label", "split", "ticker", "year"]).size().rename("count").reset_index()
    strata.to_csv(results / "sampling_strata.csv", index=False)
    frozen.update({"talib_python_version": talib.__version__, "talib_c_version": str(talib.__ta_version__),
                   "numpy_version": np.__version__, "pandas_version": pd.__version__,
                   "data_manifest_sha256": digest(ROOT / "shared/results/data_manifest.json"),
                   "index_sha256": digest(results / "index.csv"), "selected": len(index),
                   "rendered": False, "no_gpu_or_model_calls": True})
    if not args.labels_only:
        from shared.src.render import render_chart
        for row in index.itertuples():
            window = raw[row.ticker].iloc[row.bar_index - RULES["n_bars"] + 1:row.bar_index+1]
            (PART / row.bars_file).parent.mkdir(parents=True, exist_ok=True)
            window.to_csv(PART / row.bars_file, index_label="Date", float_format="%.17g")
            render_chart(window, PART / row.file, n_bars=20, size=(640,480), style="candle", mirror=False)
        # Fixed selection of 8 class examples plus a second Doji. No clarity cherry-picking.
        sample_rows = [stratified_sample(index[index.label == label], 1, seed=607).iloc[0] for label in CLASSES if (index.label==label).any()]
        first_ids = {row.id for row in sample_rows}
        remaining = index[~index.id.isin(first_ids) & (index.label == "Doji")]
        sample_rows.append(stratified_sample(remaining, 1, seed=607).iloc[0])
        samples = []
        for number, row in enumerate(sample_rows, 1):
            dst = results / "samples" / f"sample_{number:02d}.png"
            shutil.copyfile(PART / row.file, dst)
            samples.append({"sample": str(dst.relative_to(PART)), "id": row.id, "label": row.label,
                            "ticker": row.ticker, "end_date": row.end_date, "split": row.split})
        (results / "samples/manifest.json").write_text(json.dumps(samples, indent=2)+"\n")
        review_parts = []
        for i, label in enumerate(CLASSES):
            count = 7 if i < 2 else 6  # 7+7+6*6 = 50
            for split, n in [("train", (count+1)//2), ("test", count//2)]:
                review_parts.append(stratified_sample(index[(index.label==label)&(index.split==split)], n, seed=608))
        review = pd.concat(review_parts, ignore_index=True)
        reviews = []
        for row in review.itertuples():
            window = raw[row.ticker].iloc[row.bar_index-19:row.bar_index+1]
            reviews.append({"id": row.id, "label": row.label, "split": row.split,
                            "ticker": row.ticker, "end_date": row.end_date,
                            "visual_status": "pending", "diagnostics": diagnostic(window,row.label)})
        (results / "hand_review.json").write_text(json.dumps({"reviewer": "pending", "count": len(reviews), "items": reviews}, indent=2)+"\n")
        write_contact_sheets(review, results)
        chart_hashes = {row.id: digest(PART / row.file) for row in index.itertuples()}
        (results / "chart_hashes.json").write_text(json.dumps(chart_hashes, indent=2)+"\n")
        frozen.update({"rendered": True, "renderer_sha256": digest(ROOT / "shared/src/render.py"),
                       "chart_hashes_sha256": digest(results / "chart_hashes.json"),
                       "size": [640,480], "n_bars": 20, "style": "candle", "mirror": False})
    (results / "render_manifest.json").write_text(json.dumps(frozen, indent=2)+"\n")
    print(json.dumps({"total": len(index), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
