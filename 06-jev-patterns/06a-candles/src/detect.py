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
import tempfile
from datetime import datetime, timezone

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



def clarity(frame: pd.DataFrame, label: str) -> tuple[bool, dict]:
    """Frozen geometric margins using only the twenty bars shown in the image."""
    config_path = PART / "clear_rules.json"
    cfg = json.loads(config_path.read_text())
    if cfg["original_label_rules_sha256"] != digest(RULES_PATH):
        raise ValueError("Clarity rules refer to different frozen label definitions")
    validate_ohlcv(frame)
    if len(frame) != cfg["visible_bars_only"] or label not in CLASSES:
        raise ValueError("Clarity requires a known label and exactly twenty visible bars")
    flags = pandas_detector(frame).iloc[-1]
    recognized = list(flags.index[flags])
    if recognized != ([] if label == "None" else [label]):
        raise ValueError("Clarity metadata disagrees with the original single-label rule")
    o, h, l, c = [frame[name].to_numpy(dtype=float) for name in ["Open", "High", "Low", "Close"]]
    body, ranges = np.abs(c-o), h-l
    body_low, body_high = np.minimum(c,o), np.maximum(c,o)
    upper, lower = h-body_high, body_low-l
    i, first, middle = len(frame)-1, len(frame)-3, len(frame)-2
    mean_body = lambda j: float(body[j-10:j].mean())
    mean_range = lambda j: float(ranges[j-10:j].mean())
    tiny = .1 * mean_range(i)
    near = .2 * float(ranges[i-6:i-1].mean())
    span = RULES["prior_trend"]["span"].get(label,1)
    before = i-span
    trend_move = float(c[before]-c[before-5])
    trend_range = float(ranges[before-9:before+1].mean())
    positive_scale = mean_range(i)>0 and mean_body(i)>0 and ranges[i]>0
    checks, near_patterns = {"nondegenerate_reference_and_candle": bool(positive_scale)}, []
    metrics = {"body":float(body[i]), "mean_prior_body":mean_body(i), "mean_prior_range":mean_range(i),
               "lower_wick":float(lower[i]), "upper_wick":float(upper[i]),
               "talib_tiny_threshold":tiny, "talib_near_allowance":near,
               "prior_five_session_move":trend_move, "prepattern_mean_range":trend_range}
    if label == "Doji":
        checks["doji_body_well_inside_threshold"] = body[i] <= cfg["doji"]["max_fraction_of_talib_body_threshold"] * tiny
    elif label in ["Hammer", "Shooting Star"]:
        rule=cfg["hammer_shooting_star"]
        hammer=label=="Hammer"
        long_wick, short_wick = (lower[i],upper[i]) if hammer else (upper[i],lower[i])
        checks.update({"body_clearly_short": body[i] <= rule["max_body_to_prior_mean_body"] * mean_body(i),
                       "long_wick_clearly_long": long_wick >= rule["min_long_wick_to_body"] * body[i],
                       "short_wick_clearly_tiny": short_wick <= rule["max_short_wick_to_talib_tiny_threshold"] * tiny,
                       "prior_trend_clearly_directional": (-trend_move if hammer else trend_move) >= rule["min_prior_trend_move_to_daily_range"] * trend_range})
        if hammer:
            checks["body_clearly_near_previous_low"] = body_low[i] <= l[i-1] + rule["hammer_max_fraction_of_near_allowance"] * near
        else:
            checks["body_gap_clearly_positive"] = body_low[i]-body_high[i-1] >= rule["shooting_min_body_gap_to_prior_mean_range"] * mean_range(i)
    elif label in ["Bullish Engulfing", "Bearish Engulfing"]:
        rule=cfg["engulfing"]
        low_extension=float(body_low[i-1]-body_low[i])
        high_extension=float(body_high[i]-body_high[i-1])
        checks.update({"previous_body_not_nearly_zero": body[i-1] >= rule["min_previous_body_to_its_prior_mean_body"] * mean_body(i-1),
                       "current_body_clearly_larger": body[i] >= rule["min_current_body_to_previous_body"] * body[i-1],
                       "lower_endpoint_clearly_extended": low_extension >= rule["min_extension_at_each_endpoint_to_previous_body"] * body[i-1],
                       "upper_endpoint_clearly_extended": high_extension >= rule["min_extension_at_each_endpoint_to_previous_body"] * body[i-1]})
        metrics.update({"previous_body":float(body[i-1]),"previous_mean_body":mean_body(i-1),
                        "lower_endpoint_extension":low_extension,"upper_endpoint_extension":high_extension})
    elif label in ["Morning Star", "Evening Star"]:
        rule=cfg["stars"]; morning=label=="Morning Star"
        gap=float(body_low[first]-body_high[middle]) if morning else float(body_low[middle]-body_high[first])
        penetration=float(c[i]-c[first]) if morning else float(c[first]-c[i])
        checks.update({"first_body_clearly_long": body[first] >= rule["min_first_body_to_its_prior_mean_body"] * mean_body(first),
                       "middle_body_clearly_short": body[middle] <= rule["max_middle_body_to_its_prior_mean_body"] * mean_body(middle),
                       "middle_body_clearly_smaller_than_first": body[middle] <= rule["max_middle_body_to_first_body"] * body[first],
                       "third_body_clearly_long": body[i] >= rule["min_third_body_to_its_prior_mean_body"] * mean_body(i),
                       "middle_body_gap_clearly_positive": gap >= rule["min_body_gap_to_middle_prior_mean_range"] * mean_range(middle),
                       "third_close_clearly_penetrates_first": penetration >= rule["min_penetration_fraction"] * body[first],
                       "prior_trend_clearly_directional": (-trend_move if morning else trend_move) >= rule["min_prior_trend_move_to_daily_range"] * trend_range})
        metrics.update({"first_body":float(body[first]),"middle_body":float(body[middle]),
                        "middle_body_gap":gap,"penetration_distance":penetration})
    else:
        rule=cfg["none"]
        up=c>=o
        allow=rule["engulf_endpoint_overlap_allowance_to_previous_body"] * body[i-1]
        near_engulf=(body_low[i] <= body_low[i-1]+allow) and (body_high[i] >= body_high[i-1]-allow)
        shared_star=(body[first] >= rule["star_min_first_and_third_body_to_prior_mean_body"]*mean_body(first)
                     and body[middle] <= rule["star_max_middle_body_to_prior_mean_body"]*mean_body(middle)
                     and body[i] >= rule["star_min_first_and_third_body_to_prior_mean_body"]*mean_body(i))
        gap_allow=rule["star_body_gap_overlap_allowance_to_middle_prior_mean_range"]*mean_range(middle)
        near_flags={
            "Doji": body[i] <= rule["doji_max_fraction_of_talib_body_threshold"]*tiny,
            "Hammer": body[i] <= rule["wick_max_body_to_prior_mean_body"]*mean_body(i) and lower[i] >= rule["wick_min_long_wick_to_body"]*body[i] and upper[i] <= rule["wick_max_short_wick_to_talib_tiny_threshold"]*tiny,
            "Shooting Star": body[i] <= rule["wick_max_body_to_prior_mean_body"]*mean_body(i) and upper[i] >= rule["wick_min_long_wick_to_body"]*body[i] and lower[i] <= rule["wick_max_short_wick_to_talib_tiny_threshold"]*tiny,
            "Bullish Engulfing": bool(up[i] and not up[i-1] and near_engulf),
            "Bearish Engulfing": bool(not up[i] and up[i-1] and near_engulf),
            "Morning Star": shared_star and not up[first] and up[i] and body_high[middle] <= body_low[first]+gap_allow and c[i]-c[first] >= rule["star_min_penetration_fraction"]*body[first],
            "Evening Star": shared_star and up[first] and not up[i] and body_low[middle] >= body_high[first]-gap_allow and c[first]-c[i] >= rule["star_min_penetration_fraction"]*body[first]}
        near_patterns=[name for name,value in near_flags.items() if value]
        checks.update({"no_near_"+name: not value for name,value in near_flags.items()})
    checks={name:bool(value) for name,value in checks.items()}
    return all(checks.values()), {"criteria":checks,"metrics":metrics,"near_patterns":near_patterns}


def annotate_clarity(index: pd.DataFrame, raw: dict, *, include_audit: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    out=index.copy()
    clear, reasons, diagnostics = [], [], {}
    for row in out.itertuples():
        frame=raw[row.ticker].iloc[row.bar_index-19:row.bar_index+1]
        if len(frame)!=20 or str(frame.index[0].date())!=row.window_start or str(frame.index[-1].date())!=row.end_date:
            raise ValueError("Visible bars differ from preserved index")
        flag, detail=clarity(frame,row.label)
        clear.append(flag)
        reasons.append(";".join(key for key,passed in detail["criteria"].items() if not passed))
        diagnostics[row.id]={"label":row.label,"clear":flag,**detail}
    out["clear"]=clear
    out["clarity_reason"]=reasons
    out["clarity_rule_version"]=1
    summary={"rules_sha256":digest(PART/"clear_rules.json"),"total":len(out),
             "clear":int(out.clear.sum()),"not_clear":int((~out.clear).sum()),
             "per_class":{},"scope":"Frozen heuristic margins for secondary analysis; original labels, splits and primary 1200-row sample unchanged. Not a proven error-rate estimate."}
    for label in CLASSES:
        subset=out[out.label==label]
        summary["per_class"][label]={split:{"total":len(part),"clear":int(part.clear.sum())}
                                    for split,part in subset.groupby("split",sort=True)}
    audit_path=PART/"results/hand_review.json"
    if include_audit and audit_path.exists():
        audit=json.loads(audit_path.read_text())
        reviewed=[item for item in audit["items"] if item.get("visual_status")!="pending"]
        subtle=sum(item.get("visual_status")=="subtle_or_ambiguous" for item in reviewed)
        summary["visual_audit"]={"reviewed":len(reviewed),"subtle_or_ambiguous":subtle,
                                  "observed_subtle_or_ambiguous_fraction":subtle/len(reviewed) if reviewed else None,
                                  "reviewer":"AI-assisted visual inspection; not independent human expert validation",
                                  "interpretation":"Empirical clarity observation in this 50-chart audit, not a proven label error rate or population accuracy estimate."}
    return out,diagnostics,summary


def augment_clarity() -> None:
    """Add fixed clarity flags, preserving labels, sampling and the completed audit."""
    from shared.src.render import render_chart
    results=PART/"results"
    index_path=results/"index.csv"
    before=pd.read_csv(index_path,keep_default_na=False)
    before_index_hash=digest(index_path)
    manifest=json.loads((results/"render_manifest.json").read_text())
    if before_index_hash != manifest["index_sha256"]:
        raise ValueError("Existing index differs from its pinned manifest")
    if digest(ROOT/"shared/results/data_manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Augmentation requires the original pinned input-data snapshot")
    chart_hashes=json.loads((results/"chart_hashes.json").read_text())
    audit_path=results/"hand_review.json"
    before_audit_hash=digest(audit_path)
    raw={ticker:load(ticker) for ticker in sorted(before.ticker.unique())}
    updated,diagnostics,summary=annotate_clarity(before,raw)
    original_columns=[name for name in before if name not in ["clear","clarity_reason","clarity_rule_version"]]
    pd.testing.assert_frame_equal(before[original_columns],updated[original_columns])
    # Re-render into temporary ignored storage. Verify all original image hashes
    # before replacing any chart, so a semantic renderer change fails closed.
    with tempfile.TemporaryDirectory(prefix="clarity-render-",dir=PART/"data") as directory:
        temporary=Path(directory)
        for row in updated.itertuples():
            if digest(PART/row.file)!=chart_hashes[row.id]:
                raise ValueError("Existing chart differs from its original checksum")
            frame=raw[row.ticker].iloc[row.bar_index-19:row.bar_index+1]
            output=temporary/f"{row.id}.png"
            render_chart(frame,output,n_bars=20,size=(640,480),style="candle",mirror=False)
            if digest(output)!=chart_hashes[row.id]:
                raise ValueError(f"Default renderer changed canonical chart {row.id}; audit cannot be reused")
        for row in updated.itertuples():
            (temporary/f"{row.id}.png").replace(PART/row.file)
    samples=json.loads((results/"samples/manifest.json").read_text())
    for sample in samples:
        row=updated[updated.id==sample["id"]].iloc[0]
        target=PART/sample["sample"]
        if digest(target)!=chart_hashes[row.id]:
            raise ValueError("Existing sample differs from its pinned source chart")
        shutil.copyfile(PART/row.file,target)
        if digest(target)!=chart_hashes[row.id]:
            raise ValueError("Regenerated sample differs from original")
    if digest(audit_path)!=before_audit_hash:
        raise ValueError("Completed visual audit changed during annotation")
    updated.to_csv(index_path,index=False)
    (results/"clarity_diagnostics.json").write_text(json.dumps({"rules_sha256":summary["rules_sha256"],"items":diagnostics},indent=2,allow_nan=False)+"\n")
    (results/"clarity_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    original=manifest.get("original_dataset_provenance",{key:manifest[key] for key in ["source_sha256","index_sha256","renderer_sha256","rules_sha256"]})
    manifest["original_dataset_provenance"]=original
    manifest.update({"index_sha256":digest(index_path),"source_sha256":digest(Path(__file__)),
                     "renderer_sha256":digest(ROOT/"shared/src/render.py"),
                     "clarity_augmentation":{"rules_sha256":summary["rules_sha256"],"prior_index_sha256":before_index_hash,
                         "diagnostics_sha256":digest(results/"clarity_diagnostics.json"),"summary_sha256":digest(results/"clarity_summary.json"),
                         "original_columns_unchanged":True,"chart_hashes_unchanged":True,"charts_rerendered":len(updated),
                         "samples_regenerated":len(samples),"preserved_audit_sha256":before_audit_hash,
                         "completed_at_utc":datetime.now(timezone.utc).isoformat(),"no_jev_results_used":True}})
    (results/"render_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--labels-only", action="store_true")
    mode.add_argument("--augment-clarity", action="store_true", help="Preserve the existing sample and visual audit; append fixed clarity flags and verify regenerated images")
    args = parser.parse_args()
    if args.augment_clarity:
        augment_clarity()
        return
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
    index, clarity_diagnostics, clarity_summary = annotate_clarity(index, raw, include_audit=False)
    (results / "clarity_diagnostics.json").write_text(json.dumps({"rules_sha256":clarity_summary["rules_sha256"],"items":clarity_diagnostics},indent=2,allow_nan=False)+"\n")
    (results / "clarity_summary.json").write_text(json.dumps(clarity_summary,indent=2,allow_nan=False)+"\n")
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
                   "clarity_rules_sha256": digest(PART/"clear_rules.json"),
                   "clarity_diagnostics_sha256": digest(results/"clarity_diagnostics.json"),
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
