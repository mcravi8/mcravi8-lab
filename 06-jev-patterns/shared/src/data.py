"""Download and load the Experiment 06 daily OHLCV universe (Yahoo via yfinance).

Raw parquet files go to shared/data/ (gitignored). A manifest with per-ticker row
counts, session ranges and SHA-256 hashes is copied to shared/results/.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time as clock

import numpy as np
import pandas as pd

SHARED = Path(__file__).resolve().parents[1]
EXP = SHARED.parent
DATA_DIR = SHARED / "data"
CONFIG = SHARED / "config" / "universe.json"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def universe(config: Path | None = None) -> dict:
    cfg = json.loads(Path(config or CONFIG).read_text())
    tickers = list(cfg["exp03_tickers"]) + list(cfg["additional_tickers"])
    if len(set(tickers)) != len(tickers):
        raise ValueError("Duplicate tickers in universe config")
    return {**cfg, "tickers": tickers}


def validate_ohlcv(frame: pd.DataFrame) -> None:
    """Fail on malformed input rather than silently filling sessions (as in Exp 05)."""
    if frame.empty or list(frame.columns) != OHLCV:
        raise ValueError("Expected nonempty Open, High, Low, Close, Volume data")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex")
    if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Session dates must be unique and increasing")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite OHLCV values")
    if (frame[OHLCV[:4]] <= 0).any().any() or (frame.Volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    tol = frame.High * 1e-8
    if ((frame.Low > frame[["Open", "Close"]].min(axis=1) + tol)
            | (frame.High < frame[["Open", "Close"]].max(axis=1) - tol)).any():
        raise ValueError("Invalid OHLC price ordering")


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Validate without filling, dropping or clipping bars used by pattern labels."""
    validate_ohlcv(frame)
    return frame.copy(), 0


def download_ticker(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, int]:
    import yfinance as yf
    last = None
    for attempt in range(3):
        try:
            frame = yf.download(ticker, start=start, end=(date.fromisoformat(end) + timedelta(days=1)).isoformat(),
                                interval="1d", auto_adjust=True, actions=False, prepost=False,
                                ignore_tz=True, multi_level_index=False, threads=False,
                                progress=False, timeout=30)
            if frame is None or frame.empty:
                raise ValueError("no data")
            frame = frame.loc[:, OHLCV].copy()
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
            frame.index.name = "Date"
            frame, dropped = clean(frame.sort_index())
            validate_ohlcv(frame)
            frame["Volume"] = frame.Volume.round().astype("int64")
            return frame, dropped
        except Exception as error:  # network or validation; retried
            last = error
            clock.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Could not download valid {ticker} data: {last}")


def _config_sha256() -> str:
    return hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def _validate_manifest(manifest: dict, cfg: dict) -> None:
    if (manifest.get("config_sha256") != _config_sha256()
            or manifest.get("start") != cfg["start"] or manifest.get("end") != cfg["end"]
            or manifest.get("auto_adjust") is not True):
        raise ValueError("Cached manifest does not match the frozen configuration; use --refresh explicitly")
    entries = manifest.get("tickers")
    if not isinstance(entries, dict) or not set(entries).issubset(cfg["tickers"]):
        raise ValueError("Cached manifest contains invalid or stale universe entries; use --refresh explicitly")


def _verified_frame(ticker: str, manifest: dict, cfg: dict) -> pd.DataFrame:
    """Check provenance before parsing bars, then validate values and declared coverage."""
    if ticker not in cfg["tickers"] or ticker not in manifest["tickers"]:
        raise ValueError(f"No verified frozen-universe manifest entry for {ticker}")
    entry = manifest["tickers"][ticker]
    target = DATA_DIR / f"{ticker}.parquet"
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != entry.get("sha256"):
        raise ValueError(f"Cached {ticker} checksum mismatch or file missing; inspect before --refresh")
    frame = pd.read_parquet(target)
    validate_ohlcv(frame)
    if (frame.index.min() < pd.Timestamp(cfg["start"]) or frame.index.max() > pd.Timestamp(cfg["end"])
            or len(frame) != entry.get("rows")
            or str(frame.index[0].date()) != entry.get("first_session")
            or str(frame.index[-1].date()) != entry.get("last_session")):
        raise ValueError(f"Cached {ticker} coverage differs from its manifest or frozen date range")
    return frame


def load(ticker: str) -> pd.DataFrame:
    cfg = universe()
    local_path, published_path = DATA_DIR / "manifest.json", SHARED / "results/data_manifest.json"
    if not local_path.exists() or not published_path.exists():
        raise ValueError("Verified local and published data manifests are required; run the downloader first")
    raw = local_path.read_bytes()
    if raw != published_path.read_bytes():
        raise ValueError("Local and published data manifests differ; finish or verify the downloader run")
    manifest = json.loads(raw)
    _validate_manifest(manifest, cfg)
    return _verified_frame(ticker, manifest, cfg)


def _write_if_changed(path: Path, raw: bytes) -> None:
    """Preserve exact bytes and timestamps for already-verified cached manifests."""
    if path.exists() and path.read_bytes() == raw:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_bytes(raw)
    temporary.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--refresh", action="store_true", help="Explicitly replace validated local snapshots")
    args = ap.parse_args()
    cfg = universe()
    selected = args.tickers or cfg["tickers"]
    if not set(selected).issubset(cfg["tickers"]) or len(set(selected)) != len(selected):
        raise ValueError("Requested tickers must be unique members of the frozen universe")
    previous_path = DATA_DIR / "manifest.json"
    raw_manifest = previous_path.read_bytes() if previous_path.exists() else None
    manifest = json.loads(raw_manifest) if raw_manifest is not None else None
    if manifest is not None:
        try:
            _validate_manifest(manifest, cfg)
        except ValueError:
            if not args.refresh:
                raise
            # A changed config creates a new acquisition. Never carry old-universe
            # entries into it, including on a requested subset refresh.
            manifest, raw_manifest = None, None
    import yfinance as yf
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(SHARED / ".cache" / "yfinance"))
    if manifest is None:
        manifest = {"source": "Yahoo Finance via yfinance", "yfinance_version": yf.__version__,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "start": cfg["start"], "end": cfg["end"], "auto_adjust": True, "tickers": {},
                    "config_sha256": _config_sha256(),
                    "validation": "No imputation, clipping or dropped bars; malformed OHLCV fails",
                    "survivorship_note": "Hand-selected present-day large caps and ETFs; retrospective adjusted Yahoo history"}
    for ticker in selected:
        target = DATA_DIR / f"{ticker}.parquet"
        saved = manifest["tickers"].get(ticker)
        if not args.refresh and saved and target.exists():
            frame = _verified_frame(ticker, manifest, cfg)
            print(ticker, len(frame), "verified local snapshot", flush=True)
            continue
        if not args.refresh and not saved and target.exists():
            raise ValueError(f"Existing {ticker} snapshot has no matching manifest entry; inspect before --refresh")
        frame, dropped = download_ticker(ticker, cfg["start"], cfg["end"])
        if frame.index.min() < pd.Timestamp(cfg["start"]) or frame.index.max() > pd.Timestamp(cfg["end"]):
            raise ValueError("Downloaded data outside frozen date range")
        temporary = target.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary, engine="pyarrow")
        temporary.replace(target)
        # Existing batch metadata and untouched ticker records remain unchanged.
        # New per-ticker metadata records the actual acquisition, never verification.
        manifest["tickers"][ticker] = {"rows": len(frame), "dropped_incomplete_rows": dropped,
                                       "first_session": str(frame.index[0].date()),
                                       "last_session": str(frame.index[-1].date()),
                                       "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                                       "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                                       "yfinance_version": yf.__version__}
        raw_manifest = (json.dumps(manifest, indent=2) + "\n").encode()
        _write_if_changed(previous_path, raw_manifest)
        print(ticker, len(frame), frame.index[0].date(), frame.index[-1].date(), flush=True)
    if raw_manifest is None:
        raise ValueError("No acquisition or verified manifest was produced")
    _write_if_changed(previous_path, raw_manifest)
    _write_if_changed(SHARED / "results/data_manifest.json", raw_manifest)


if __name__ == "__main__":
    main()
