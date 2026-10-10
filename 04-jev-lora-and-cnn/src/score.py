"""Batch/resume JEV System 1 HTTP scoring without provisioning any compute."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

from jev_api import (API_PATHS, DIRECTION_OPTIONS, DIRECTION_QUESTION,
                     TICKER_OPTIONS, TICKER_QUESTION, build_payload,
                     parse_probabilities)

ROOT = Path(__file__).resolve().parents[1]
PRED_FIELDS = ["file", "ticker", "end_date", "label", "p_up", "p_flat", "p_down", "latency_ms"]
PROBE_FIELDS = ["file", "ticker", "end_date", "predicted_ticker", "p_tsla", "p_nvda", "latency_ms"]
ERROR_FIELDS = ["file", "ticker", "end_date", "label", "mode", "api", "attempts", "error", "timestamp_utc"]


class RequestFailure(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def local_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Local inputs and outputs must stay in the experiment directory")
    return path


def endpoint(base_url: str, api: str, dry_run: bool) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("JEV_BASE_URL must be an http(s) server URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use a server URL without credentials, query parameters, or a fragment")
    if dry_run:
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.lower() == "localhost"
        if not loopback:
            raise ValueError("--dry-run only permits a loopback fake server")
    prefix = parsed.path.rstrip("/")
    if prefix.endswith("/v1"):
        prefix = prefix[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, prefix + API_PATHS[api], "", ""))


def load_index(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"file", "ticker", "end_date", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Index requires file,ticker,end_date,label columns")
        rows = list(reader)
    seen = set()
    for row in rows:
        if not row.get("file") or row["file"] in seen:
            raise ValueError("Index filenames must be nonempty and unique")
        if not row.get("ticker") or row.get("label") not in DIRECTION_OPTIONS:
            raise ValueError("Index has an unexpected ticker or direction label")
        datetime.strptime(row["end_date"], "%Y-%m-%d")
        image = local_path(row["file"])
        if image.suffix.lower() != ".png" or not image.is_file():
            raise ValueError(f"Missing PNG: {row['file']}")
        seen.add(row["file"])
    if not rows:
        raise ValueError("Index is empty")
    return rows


def probe_rows(rows: list[dict]) -> list[dict]:
    """Choose ten evenly spaced charts per ticker, independent of labels."""
    selected = []
    for ticker in TICKER_OPTIONS:
        candidates = sorted((row for row in rows if row["ticker"] == ticker),
                            key=lambda row: (row["end_date"], row["file"]))
        if len(candidates) < 10:
            raise ValueError("--ticker-probe needs at least ten charts per ticker")
        selected.extend(candidates[int((len(candidates) - 1) * step / 9)] for step in range(10))
    return selected


def image_url(row: dict, placebo: bool) -> str:
    path = local_path(row["file"])
    content = path.read_bytes()
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG":
            raise ValueError("Expected PNG bytes")
        image.load()
        if placebo:
            buffer = io.BytesIO()
            ImageOps.mirror(image.convert("RGB")).save(buffer, format="PNG")
            content = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(content).decode("ascii")


def post_json(url: str, payload: dict, timeout: float, api_key: str) -> dict:
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": "jev-scorer/1.0"}  # Runpod proxy 403s Python-urllib UA
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = Request(url, data=json.dumps(payload, allow_nan=False).encode("utf-8"),
                      headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(2_000_001)
        if len(body) > 2_000_000:
            raise RequestFailure("Server response exceeds 2 MB")
        result = json.loads(body)
        if not isinstance(result, dict):
            raise RequestFailure("Expected a JSON object response")
        return result
    except HTTPError as error:
        # Do not log server bodies: they can echo credentials or image payloads.
        raise RequestFailure(f"HTTP {error.code}", error.code == 429 or error.code >= 500) from error
    except (URLError, TimeoutError, OSError) as error:
        raise RequestFailure(f"Transport failure: {type(error).__name__}", True) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RequestFailure("Server returned invalid JSON") from error


def score_row(row: dict, args: argparse.Namespace, url: str, options: tuple,
              question: str, mode: str) -> tuple[dict | None, int, str]:
    started = time.perf_counter()
    attempts = 0
    try:
        payload = build_payload(image_url(row, mode == "placebo"), options, question,
                                api=args.api, model=args.model)
        while True:
            attempts += 1
            try:
                response = post_json(url, payload, args.timeout,
                                     "" if args.dry_run else os.environ.get("JEV_API_KEY", ""))
                if response.get("fake_server") and not args.dry_run:
                    raise RequestFailure("Fake server responses require --dry-run")
                probabilities = parse_probabilities(response, options, api=args.api)
                break
            except RequestFailure as error:
                if not error.retryable or attempts > args.retries:
                    raise
                time.sleep(min(2 ** (attempts - 1), 8))
        result = {key: row[key] for key in ("file", "ticker", "end_date")}
        if mode == "ticker-probe":
            result.update(predicted_ticker=options[max(range(len(options)), key=probabilities.__getitem__)],
                          p_tsla=probabilities[0], p_nvda=probabilities[1])
        else:
            result.update(label=row["label"], p_up=probabilities[0],
                          p_flat=probabilities[1], p_down=probabilities[2])
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return result, attempts, ""
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        key = os.environ.get("JEV_API_KEY", "")
        if key:
            message = message.replace(key, "[redacted]")
        return None, attempts, message[:500]


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_csv(path: Path, fields: list[str], row: dict | None = None) -> None:
    with path.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if handle.tell() == 0:
            writer.writeheader()
        if row is not None:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def saved_rows(path: Path, fields: list[str], index: dict, options: tuple,
               mode: str) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError("Existing predictions have a different CSV schema")
        records = list(reader)
    completed, valid = set(), []
    damaged = 0
    for record in records:
        # A process can stop midway through its last CSV append. Repair a
        # clearly torn row before checking the truth metadata on complete rows.
        if any(record.get(field) in (None, "") for field in fields) or None in record:
            damaged += 1
            continue
        filename = record.get("file")
        if filename not in index:
            raise ValueError("Existing predictions include an unknown index filename")
        for field in ("ticker", "end_date", *(() if mode == "ticker-probe" else ("label",))):
            if record.get(field) != index[filename][field]:
                raise ValueError("Existing prediction metadata differs from the current index")
        try:
            probability_fields = ("p_tsla", "p_nvda") if mode == "ticker-probe" else ("p_up", "p_flat", "p_down")
            probabilities = [float(record[field]) for field in probability_fields]
            parse_probabilities({"options": list(options), "probabilities": probabilities}, options)
            latency = float(record["latency_ms"])
            if not math.isfinite(latency) or latency < 0 or None in record:
                raise ValueError("Invalid latency or incomplete CSV row")
            if mode == "ticker-probe" and record["predicted_ticker"] != options[max(range(len(options)), key=probabilities.__getitem__)]:
                raise ValueError("Invalid predicted ticker")
        except (TypeError, ValueError, KeyError):
            damaged += 1
            continue
        if filename not in completed:
            valid.append(record)
            completed.add(filename)
        else:
            damaged += 1
    if damaged:
        temporary = path.with_suffix(".repair.csv")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(valid)
        temporary.replace(path)
        print(f"Repaired {damaged} incomplete/invalid/duplicate prediction rows; they will be retried", file=sys.stderr)
    return completed


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("JEV_BASE_URL", ""))
    parser.add_argument("--api", choices=tuple(API_PATHS), default=os.environ.get("JEV_API", "decide"))
    parser.add_argument("--model", default=os.environ.get("JEV_MODEL", "jev-decision"),
                        help="LoRA model name for chat-logprobs; native /decide selects its own model")
    parser.add_argument("--batch-size", type=int, default=os.environ.get("JEV_BATCH_SIZE", "16"))
    parser.add_argument("--concurrency", type=int, default=os.environ.get("JEV_CONCURRENCY", "4"))
    parser.add_argument("--timeout", type=float, default=os.environ.get("JEV_TIMEOUT", "60"))
    parser.add_argument("--retries", type=int, default=os.environ.get("JEV_RETRIES", "3"),
                        help="Additional attempts after transient transport, 429, or 5xx failures")
    parser.add_argument("--index", default="charts/index.csv")
    parser.add_argument("--results-dir", help="Experiment-relative or absolute local output directory")
    parser.add_argument("--limit", type=int, help="First N selected charts, applied before resume filtering")
    parser.add_argument("--dry-run", action="store_true", help="Loopback-only, synthetic output; default ten charts")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--placebo", action="store_true")
    modes.add_argument("--ticker-probe", action="store_true")
    args = parser.parse_args()
    if not args.base_url:
        parser.error("Set JEV_BASE_URL or --base-url")
    if args.api not in API_PATHS:
        parser.error("JEV_API must be decide or chat-logprobs")
    if args.batch_size < 1 or args.concurrency < 1 or args.retries < 0:
        parser.error("batch size/concurrency must be positive; retries must be nonnegative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.dry_run and args.limit is None and not args.ticker_probe:
        args.limit = 10
    return args


def run(args: argparse.Namespace) -> int:
    url = endpoint(args.base_url, args.api, args.dry_run)
    index_path = local_path(args.index)
    rows = load_index(index_path)
    mode = "ticker-probe" if args.ticker_probe else ("placebo" if args.placebo else "direction")
    options = TICKER_OPTIONS if args.ticker_probe else DIRECTION_OPTIONS
    question = TICKER_QUESTION if args.ticker_probe else DIRECTION_QUESTION
    fields = PROBE_FIELDS if args.ticker_probe else PRED_FIELDS
    filenames = {"direction": "preds.csv", "placebo": "preds_placebo.csv", "ticker-probe": "ticker_probe.csv"}
    default_dir = "results/dry_run" if args.dry_run else "results"
    results_dir = local_path(args.results_dir or default_dir)
    if args.dry_run and results_dir == ROOT / "results":
        raise ValueError("--dry-run must use a separate synthetic output directory")
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / filenames[mode]
    meta_path = output.with_suffix(".meta.json")
    identity = {"index_file": index_path.relative_to(ROOT).as_posix(),
                "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "image_sha256": {row["file"]: hashlib.sha256(local_path(row["file"]).read_bytes()).hexdigest() for row in rows},
                "mode": mode, "api": args.api, "model": args.model if args.api == "chat-logprobs" else None,
                "question": question, "options": list(options), "synthetic": args.dry_run,
                "api_implementation_sha256": hashlib.sha256((ROOT / "src/jev_api.py").read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    lock_path = cache / ("score-" + hashlib.sha256(str(output).encode()).hexdigest()[:16] + ".lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another scorer is already writing this prediction file") from error
        if meta_path.exists():
            existing = json.loads(meta_path.read_text())
            if existing.get("fingerprint") != fingerprint:
                raise ValueError("Resume input/API/mode mismatch; choose a fresh --results-dir")
        elif output.exists() and output.stat().st_size:
            raise ValueError("Existing predictions have no provenance; choose a fresh --results-dir")
        metadata = {"fingerprint": fingerprint, **identity,
                    "endpoint": url, "batch_size": args.batch_size, "concurrency": args.concurrency,
                    "timeout_seconds": args.timeout, "retries": args.retries,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(meta_path, metadata)
        by_file = {row["file"]: row for row in rows}
        completed = saved_rows(output, fields, by_file, options, mode)
        selected = probe_rows(rows) if args.ticker_probe else rows
        if args.limit is not None:
            selected = selected[:args.limit]
        pending = [row for row in selected if row["file"] not in completed]
        skipped = len(selected) - len(pending)
        append_csv(output, fields)
        errors_path = results_dir / "score_errors.csv"
        append_csv(errors_path, ERROR_FIELDS)
        succeeded = failed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                futures = {executor.submit(score_row, row, args, url, options, question, mode): row for row in batch}
                for future in as_completed(futures):
                    row = futures[future]
                    prediction, attempts, error = future.result()
                    if prediction is not None:
                        append_csv(output, fields, prediction)
                        succeeded += 1
                    else:
                        append_csv(errors_path, ERROR_FIELDS, {
                            **{key: row[key] for key in ("file", "ticker", "end_date", "label")},
                            "mode": mode, "api": args.api, "attempts": attempts, "error": error,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat()})
                        failed += 1
                print(f"{mode}: {succeeded} scored, {failed} failed, {skipped} already scored", flush=True)
        summary = {"mode": mode, "synthetic": args.dry_run, "selected": len(selected),
                   "already_scored": skipped, "newly_scored": succeeded, "failed": failed,
                   "output": output.relative_to(ROOT).as_posix(), "errors": errors_path.relative_to(ROOT).as_posix()}
        write_json(meta_path, {**metadata, "last_run": summary})
        print(json.dumps(summary, sort_keys=True))
    return int(failed > 0)


def main() -> int:
    try:
        return run(arguments())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Scoring stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
