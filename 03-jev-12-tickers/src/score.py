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

from jev_api import (API_PATHS, DEFAULT_HORIZON, DEFAULT_VOL_LOOKBACK,
                     DEFAULT_WINDOW_SESSIONS, DIRECTION_OPTIONS, TASKS,
                     TICKER_OPTIONS, TICKER_QUESTION, build_payload,
                     parse_probabilities, server_identity, task_label,
                     task_options, task_question, validate_native_identity)

ROOT = Path(__file__).resolve().parents[1]
PRED_FIELDS = ["file", "ticker", "end_date", "label", "p_up", "p_flat", "p_down", "latency_ms"]
PROBE_FIELDS = ["file", "ticker", "end_date", "predicted_ticker", "p_tsla", "p_nvda", "latency_ms"]
ERROR_FIELDS = ["file", "ticker", "end_date", "label", "mode", "task", "control", "api", "attempts", "error", "timestamp_utc"]


class RequestFailure(RuntimeError):
    def __init__(self, message: str, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


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


def image_url(row: dict, control: str) -> str:
    path = local_path(row["file"])
    content = path.read_bytes()
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG":
            raise ValueError("Expected PNG bytes")
        image.load()
        if control != "image":
            buffer = io.BytesIO()
            transformed = (ImageOps.mirror(image.convert("RGB")) if control == "mirror"
                           else Image.new("RGB", image.size, "white"))
            transformed.save(buffer, format="PNG")
            content = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(content).decode("ascii")


def http_json(url: str, payload: dict | None, timeout: float, api_key: str) -> dict:
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": "jev-scorer/1.0"}  # Runpod proxy 403s Python-urllib UA
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    data = json.dumps(payload, allow_nan=False).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
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
        raise RequestFailure(f"HTTP {error.code}", error.code == 429 or error.code >= 500, error.code) from error
    except (URLError, TimeoutError, OSError) as error:
        raise RequestFailure(f"Transport failure: {type(error).__name__}", True) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RequestFailure("Server returned invalid JSON") from error


def pin_server(args: argparse.Namespace) -> dict:
    """Fetch server identity before writes; never silently infer a weight revision."""
    info_url = endpoint(args.base_url, "decide", args.dry_run) + "/info"
    attempts = 0
    key = "" if args.dry_run else os.environ.get("JEV_API_KEY", "")
    while True:
        attempts += 1
        try:
            info = http_json(info_url, None, args.timeout, key)
            if info.get("fake_server") and not args.dry_run:
                raise ValueError("Fake server responses require --dry-run")
            if args.dry_run and info.get("fake_server") is not True:
                raise ValueError("--dry-run requires the synthetic fake server")
            return server_identity(info, args.model_revision)
        except RequestFailure as error:
            if args.api == "chat-logprobs" and error.status in (404, 405):
                if args.dry_run:
                    raise ValueError("--dry-run requires the synthetic fake server info endpoint") from error
                if not args.model_revision:
                    raise ValueError("Chat server lacks /v1/decide/info; pass --model-revision explicitly") from error
                return {"protocol": "jev27-bare-v1-chat-readout", "model": args.model,
                        "model_revision": args.model_revision, "revision_source": "caller-declared"}
            if not error.retryable or attempts > args.retries:
                raise
            time.sleep(min(2 ** (attempts - 1), 8))


def score_row(row: dict, args: argparse.Namespace, url: str, options: tuple,
              question: str, mode: str, control: str, identity: dict) -> tuple[dict | None, int, str]:
    started = time.perf_counter()
    attempts = 0
    try:
        payload = build_payload(image_url(row, control), options, question,
                                api=args.api, model=args.model)
        while True:
            attempts += 1
            try:
                response = http_json(url, payload, args.timeout,
                                     "" if args.dry_run else os.environ.get("JEV_API_KEY", ""))
                if response.get("fake_server") and not args.dry_run:
                    raise RequestFailure("Fake server responses require --dry-run")
                if args.dry_run and response.get("fake_server") is not True:
                    raise RequestFailure("--dry-run received a non-synthetic server response")
                if args.api == "decide":
                    validate_native_identity(response, identity)
                elif response.get("model") != args.model:
                    raise ValueError("Chat response model differs from requested decision LoRA")
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
            by_option = dict(zip(options, probabilities))
            result.update(label=row["label"], p_up=by_option["Up"],
                          p_flat=by_option.get("Flat", 0.0), p_down=by_option["Down"])
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
        handle.seek(0)
        existing = csv.reader(handle)
        header = next(existing, None)
        if header is not None and header != fields:
            raise ValueError(f"Existing CSV schema differs: {path.name}")
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
            parse_probabilities({"options": list(DIRECTION_OPTIONS) if mode != "ticker-probe" else list(options),
                                 "probabilities": probabilities},
                                DIRECTION_OPTIONS if mode != "ticker-probe" else options)
            if options == ("Up", "Down") and probabilities[1] != 0:
                raise ValueError("Binary direction predictions must have p_flat=0")
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
    parser.add_argument("--model-revision", default=os.environ.get("JEV_MODEL_REVISION", ""),
                        help="Immutable served weight revision/hash, required if server info omits it")
    parser.add_argument("--task", choices=TASKS, default="historical")
    parser.add_argument("--window-sessions", type=int, default=DEFAULT_WINDOW_SESSIONS)
    parser.add_argument("--horizon", choices=("next-5-trading-days", "next-calendar-week"), default=DEFAULT_HORIZON)
    parser.add_argument("--vol-lookback", type=int, default=DEFAULT_VOL_LOOKBACK)
    parser.add_argument("--flat-band", type=float, default=1 / 3)
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
    modes.add_argument("--blank", action="store_true", help="White PNG control with the same image dimensions")
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
    if args.window_sessions < 1 or args.vol_lookback < 2 or not math.isfinite(args.flat_band) or args.flat_band <= 0:
        parser.error("window must be positive, volatility lookback >=2, and flat band positive finite")
    if args.ticker_probe and args.task != "historical":
        parser.error("--ticker-probe must use the historical task")
    if args.dry_run and args.limit is None and not args.ticker_probe:
        args.limit = 10
    return args


def run(args: argparse.Namespace) -> int:
    url = endpoint(args.base_url, args.api, args.dry_run)
    index_path = local_path(args.index)
    rows = load_index(index_path)
    mode = "ticker-probe" if args.ticker_probe else ("placebo" if args.placebo else ("blank" if args.blank else "direction"))
    control = "mirror" if args.placebo else ("blank" if args.blank else "image")
    options = TICKER_OPTIONS if args.ticker_probe else task_options(args.task)
    questions = {}
    task_rows, excluded_zero = [], 0
    for row in rows:
        question = TICKER_QUESTION if args.ticker_probe else task_question(
            row, args.task, window_sessions=args.window_sessions, horizon=args.horizon,
            vol_lookback=args.vol_lookback, flat_band=args.flat_band)
        label = row["label"] if args.ticker_probe else task_label(row, args.task, args.flat_band)
        if label is None:
            excluded_zero += 1
            continue
        task_rows.append({**row, "label": label})
        questions[row["file"]] = question
    if not task_rows:
        raise ValueError("The selected task has no eligible index rows")
    fields = PROBE_FIELDS if args.ticker_probe else PRED_FIELDS
    stem = {"historical": "preds", "binary-direction": "preds_binary", "ternary-volatility": "preds_volatility"}[args.task]
    filename = "ticker_probe.csv" if args.ticker_probe else stem + {"direction": "", "placebo": "_placebo", "blank": "_blank"}[mode] + ".csv"
    default_dir = "results/dry_run" if args.dry_run else "results"
    results_dir = local_path(args.results_dir or default_dir)
    if args.dry_run and results_dir == ROOT / "results":
        raise ValueError("--dry-run must use a separate synthetic output directory")
    pinned = pin_server(args)
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / filename
    meta_path = output.with_suffix(".meta.json")
    identity = {"index_file": index_path.relative_to(ROOT).as_posix(),
                "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "image_sha256": {row["file"]: hashlib.sha256(local_path(row["file"]).read_bytes()).hexdigest() for row in rows},
                "mode": mode, "control": control, "task": args.task,
                "task_settings": {"window_sessions": args.window_sessions, "horizon": args.horizon,
                                  "vol_lookback": args.vol_lookback, "flat_band": args.flat_band},
                "api": args.api, "model": args.model if args.api == "chat-logprobs" else pinned["model"],
                "server_identity": pinned, "options": list(options), "synthetic": args.dry_run,
                "question_sha256": {file: hashlib.sha256(question.encode()).hexdigest() for file, question in questions.items()},
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
        by_file = {row["file"]: row for row in task_rows}
        completed = saved_rows(output, fields, by_file, options, mode)
        selected = probe_rows(task_rows) if args.ticker_probe else task_rows
        if args.limit is not None:
            selected = selected[:args.limit]
        pending = [row for row in selected if row["file"] not in completed]
        skipped = len(selected) - len(pending)
        append_csv(output, fields)
        errors_path = results_dir / ("score_errors.csv" if args.task == "historical" else stem + "_errors.csv")
        append_csv(errors_path, ERROR_FIELDS)
        succeeded = failed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                futures = {executor.submit(score_row, row, args, url, options, questions[row["file"]],
                                           mode, control, pinned): row for row in batch}
                for future in as_completed(futures):
                    row = futures[future]
                    prediction, attempts, error = future.result()
                    if prediction is not None:
                        append_csv(output, fields, prediction)
                        succeeded += 1
                    else:
                        append_csv(errors_path, ERROR_FIELDS, {
                            **{key: row[key] for key in ("file", "ticker", "end_date", "label")},
                            "mode": mode, "task": args.task, "control": control,
                            "api": args.api, "attempts": attempts, "error": error,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat()})
                        failed += 1
                print(f"{mode}: {succeeded} scored, {failed} failed, {skipped} already scored", flush=True)
        summary = {"mode": mode, "task": args.task, "control": control, "synthetic": args.dry_run,
                   "selected": len(selected), "excluded_zero_returns": excluded_zero,
                   "already_scored": skipped, "newly_scored": succeeded, "failed": failed,
                   "output": output.relative_to(ROOT).as_posix(), "errors": errors_path.relative_to(ROOT).as_posix()}
        write_json(meta_path, {**metadata, "last_run": summary})
        print(json.dumps(summary, sort_keys=True))
    return int(failed > 0)


def main() -> int:
    try:
        return run(arguments())
    except (OSError, ValueError, RequestFailure, json.JSONDecodeError) as error:
        print(f"Scoring stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
