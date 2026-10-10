"""Download the frozen 12-ticker development snapshot for CPU baselines."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time as clock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ("TSLA", "NVDA", "PLTR", "COIN", "AAPL", "MSFT", "AMZN", "META", "KO", "JNJ", "PG", "SPY")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
NEW_YORK = ZoneInfo("America/New_York")


def validate_ohlcv(frame: pd.DataFrame) -> None:
    """Fail on malformed input rather than silently filling trading sessions."""
    if frame.empty or list(frame.columns) != OHLCV:
        raise ValueError("Expected nonempty Open, High, Low, Close, Volume data")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex")
    if frame.index.hasnans or frame.index.has_duplicates:
        raise ValueError("Missing or duplicate session dates")
    if frame.index.tz is not None or not frame.index.equals(frame.index.normalize()):
        raise ValueError("Expected timezone-free daily session dates")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Session dates must be increasing")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite OHLCV values")
    if (frame[OHLCV[:4]] <= 0).any().any() or (frame.Volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if not np.equal(frame.Volume, np.floor(frame.Volume)).all():
        raise ValueError("Volume must be an integer share count")
    tolerance = frame.High * 1e-8
    if ((frame.Low > frame[["Open", "Close"]].min(axis=1) + tolerance)
            | (frame.High < frame[["Open", "Close"]].max(axis=1) - tolerance)
            | (frame.Low > frame.High + tolerance)).any():
        raise ValueError("Invalid OHLC price ordering")


def last_complete_date(as_of: date, now: datetime | None = None) -> date:
    """Conservatively omit today's daily bar until 16:15 New York time."""
    now = (now or datetime.now(NEW_YORK)).astimezone(NEW_YORK)
    if as_of > now.date():
        raise ValueError("--as-of cannot be in the future")
    if as_of == now.date() and now.time() < time(16, 15):
        return as_of - timedelta(days=1)
    return as_of


def download_ticker(ticker: str, start: date, cutoff: date) -> pd.DataFrame:
    """The inclusive cutoff becomes yfinance's exclusive next-day end."""
    last_error = None
    for attempt in range(3):
        try:
            frame = yf.download(
                ticker, start=start.isoformat(),
                end=(cutoff + timedelta(days=1)).isoformat(),
                interval="1d", auto_adjust=True, back_adjust=False,
                repair=False, actions=False, prepost=False,
                keepna=True, ignore_tz=True, multi_level_index=False,
                threads=False, progress=False, timeout=30,
            )
            if frame is None or frame.empty:
                raise ValueError(f"No data returned for {ticker}")
            frame = frame.loc[:, OHLCV].copy()
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
            frame.index.name = "Date"
            frame = frame.loc[(frame.index.date >= start) & (frame.index.date <= cutoff)]
            frame = frame.sort_index()
            validate_ohlcv(frame)
            frame[OHLCV[:4]] = frame[OHLCV[:4]].astype("float64")
            frame["Volume"] = frame.Volume.astype("int64")
            return frame
        except Exception as error:
            last_error = error
            if attempt < 2:
                clock.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Could not download valid {ticker} data: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2000, 1, 1))
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 10, 9),
                        help="Inclusive development cutoff; future holdout is never fetched here")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if args.as_of.isoformat() > protocol["development_cutoff"]:
        parser.error("Development downloads cannot include the reserved prospective period")
    cutoff = last_complete_date(args.as_of)
    if args.start > cutoff:
        parser.error("--start must be on or before the last complete date")
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(ROOT / ".cache" / "yfinance"))
    manifest = {
        "source": "Yahoo Finance via yfinance",
        "yfinance_version": yf.__version__,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": args.start.isoformat(),
        "requested_as_of": args.as_of.isoformat(),
        "last_complete_date": cutoff.isoformat(),
        "interval": "1d", "auto_adjust": True, "repair": False,
        "tickers": {},
    }
    for ticker in TICKERS:
        frame = download_ticker(ticker, args.start, cutoff)
        target = data_dir / f"{ticker}.parquet"
        temporary = target.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary, engine="pyarrow", index=True)
        temporary.replace(target)
        manifest["tickers"][ticker] = {
            "rows": len(frame), "first_session": frame.index[0].date().isoformat(),
            "last_session": frame.index[-1].date().isoformat(),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        print(f"{ticker}: {len(frame)} sessions, {frame.index[0].date()} to "
              f"{frame.index[-1].date()} -> {target.relative_to(ROOT)}")
    target = data_dir / "manifest.json"
    temporary = target.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(target)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/data_manifest.json").write_text(target.read_text())
    print(f"Manifest: {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
