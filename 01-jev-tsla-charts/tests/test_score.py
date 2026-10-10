"""Offline scorer contract, resumption, retry, and image-control checks."""

from __future__ import annotations

import contextlib
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fake_jev_server import decode_png, make_server
from jev_api import (API_PATHS, DIRECTION_OPTIONS, DIRECTION_QUESTION,
                     TICKER_OPTIONS, TICKER_QUESTION, build_payload,
                     parse_probabilities)

PRED_COLUMNS = ["file", "ticker", "end_date", "label", "p_up", "p_flat", "p_down", "latency_ms"]
PROBE_COLUMNS = ["file", "ticker", "end_date", "predicted_ticker", "p_tsla", "p_nvda", "latency_ms"]


@contextlib.contextmanager
def running_fake(*, fail_first: int = 0):
    server = make_server(fail_first=fail_first)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        cache = ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.results = self.work / "results"
        self.index = self.work / "index.csv"
        self.rows = []
        for number in range(20):
            image_path = self.work / f"chart_{number + 1:06d}.png"
            image = Image.new("RGB", (32, 24))
            image.putdata([(x * 7, y * 9, (x * 5 + y + number) % 256)
                           for y in range(24) for x in range(32)])
            image.save(image_path)
            self.rows.append({
                "file": str(image_path.relative_to(ROOT)),
                "ticker": "TSLA" if number < 10 else "NVDA",
                "end_date": f"2026-01-{number + 1:02d}",
                "label": ("Up", "Flat", "Down")[number % 3],
                "fwd_ret": 0.01 * (number % 3 - 1),
            })
        with self.index.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)

    def cli(self, url: str, *extra: str, success: bool = True):
        process = subprocess.run([
            sys.executable, str(ROOT / "src" / "score.py"),
            "--base-url", url, "--index", str(self.index),
            "--results-dir", str(self.results), "--dry-run", "--limit", "10",
            "--batch-size", "4", "--concurrency", "2", "--timeout", "2",
            "--retries", "0", *extra,
        ], cwd=ROOT, text=True, capture_output=True, timeout=30)
        if success:
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        else:
            self.assertNotEqual(process.returncode, 0, process.stderr + process.stdout)
        self.assertNotIn("data:image/png;base64,", process.stdout + process.stderr)
        return process

    def read_csv(self, name: str, expected: list[str] | None = None):
        with (self.results / name).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if expected is not None:
                self.assertEqual(reader.fieldnames, expected)
            return list(reader)

    def assert_probabilities(self, rows, names=("p_up", "p_flat", "p_down")) -> None:
        for row in rows:
            values = [float(row[name]) for name in names]
            self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1 for value in values))
            self.assertAlmostEqual(sum(values), 1, places=7)
            self.assertGreaterEqual(float(row["latency_ms"]), 0)

    def test_ten_chart_dry_run_and_resume(self) -> None:
        with running_fake() as (server, url):
            self.cli(url)
            rows = self.read_csv("preds.csv", PRED_COLUMNS)
            self.assertEqual(len(rows), 10)
            self.assertEqual({row["file"] for row in rows}, {row["file"] for row in self.rows[:10]})
            self.assert_probabilities(rows)
            self.assertEqual(len(server.requests), 10)
            for request in server.requests:
                self.assertEqual(request["path"], "/v1/decide")
                payload = request["body"]
                self.assertEqual(set(payload), {"kind", "state", "question", "options", "thinking"})
                self.assertEqual(payload["state"][0], "Chart: ")
                self.assertEqual(payload["question"], DIRECTION_QUESTION)
                self.assertEqual(payload["options"], list(DIRECTION_OPTIONS))
                self.assertEqual(payload["thinking"], "off")
                self.assertEqual(set(payload["state"][1]), {"image"})
            original = (self.results / "preds.csv").read_bytes()
            self.cli(url)
            self.assertEqual(len(server.requests), 10)
            self.assertEqual((self.results / "preds.csv").read_bytes(), original)

    def test_resume_repairs_torn_final_row(self) -> None:
        with running_fake() as (server, url):
            self.cli(url, "--limit", "2")
            first, last = self.read_csv("preds.csv", PRED_COLUMNS)
            with (self.results / "preds.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=PRED_COLUMNS)
                writer.writeheader()
                writer.writerow(first)
                # A killed process may leave only the filename of its final row.
                stream.write(last["file"] + "\n")
            self.cli(url, "--limit", "2")
            repaired = self.read_csv("preds.csv", PRED_COLUMNS)
            self.assertEqual(len(repaired), 2)
            self.assertIn(first, repaired)
            self.assertEqual({row["file"] for row in repaired}, {first["file"], last["file"]})
            self.assert_probabilities(repaired)
            self.assertEqual(len(server.requests), 3)
            sent = decode_png(server.requests[-1]["body"]["state"][1]["image"])
            with Image.open(ROOT / last["file"]) as original:
                self.assertEqual(sent.tobytes(), original.tobytes())

    def test_placebo_flips_image_pixels(self) -> None:
        with running_fake() as (server, url):
            self.cli(url, "--placebo", "--limit", "1")
            rows = self.read_csv("preds_placebo.csv", PRED_COLUMNS)
            self.assertEqual(len(rows), 1)
            self.assert_probabilities(rows)
            sent = decode_png(server.requests[0]["body"]["state"][1]["image"])
            with Image.open(ROOT / rows[0]["file"]) as original:
                expected = ImageOps.mirror(original)
                self.assertEqual(sent.size, expected.size)
                self.assertEqual(sent.tobytes(), expected.tobytes())

    def test_ticker_probe_twenty_balanced_charts(self) -> None:
        with running_fake() as (server, url):
            self.cli(url, "--ticker-probe", "--limit", "20")
            rows = self.read_csv("ticker_probe.csv", PROBE_COLUMNS)
            self.assertEqual(len(rows), 20)
            self.assertEqual(sum(row["ticker"] == "TSLA" for row in rows), 10)
            self.assertEqual(sum(row["ticker"] == "NVDA" for row in rows), 10)
            self.assert_probabilities(rows, ("p_tsla", "p_nvda"))
            self.assertTrue(all(row["predicted_ticker"] in TICKER_OPTIONS for row in rows))
            for request in server.requests:
                payload = request["body"]
                self.assertEqual(payload["question"], TICKER_QUESTION)
                self.assertEqual(payload["options"], list(TICKER_OPTIONS))
                self.assertEqual(set(payload), {"kind", "state", "question", "options", "thinking"})

    def test_chat_logprobs_three_and_two_options(self) -> None:
        with running_fake() as (server, url):
            self.cli(url, "--api", "chat-logprobs", "--limit", "3")
            self.assert_probabilities(self.read_csv("preds.csv", PRED_COLUMNS))
            self.assertEqual(len(server.requests), 3)
            for request in server.requests:
                self.assertEqual(request["path"], API_PATHS["chat-logprobs"])
                self.assertEqual(request["body"]["allowed_token_ids"], [32, 33, 34])
            # Independently exercise the same two-option request used by the probe.
            image_url = server.requests[0]["body"]["messages"][0]["content"][2]["image_url"]["url"]
            payload = build_payload(image_url, TICKER_OPTIONS, TICKER_QUESTION, "chat-logprobs")
            request = urllib.request.Request(url + API_PATHS["chat-logprobs"],
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=2) as response:
                probabilities = parse_probabilities(json.load(response), TICKER_OPTIONS, "chat-logprobs")
            self.assertEqual(len(probabilities), 2)
            self.assertAlmostEqual(sum(probabilities), 1)
            self.assertEqual(server.requests[-1]["body"]["allowed_token_ids"], [32, 33])

    def test_retry_and_permanent_failure_log(self) -> None:
        with running_fake(fail_first=1) as (server, url):
            self.cli(url, "--limit", "1", "--retries", "1")
            self.assertEqual(len(server.requests), 2)
            self.assertEqual(len(self.read_csv("preds.csv", PRED_COLUMNS)), 1)
        self.results = self.work / "failures"
        with running_fake(fail_first=100) as (server, url):
            self.cli(url, "--limit", "1", "--retries", "1", success=False)
            self.assertEqual(len(server.requests), 2)
            errors = self.read_csv("score_errors.csv")
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["file"], self.rows[0]["file"])
            self.assertEqual(errors[0]["mode"], "direction")
            self.assertEqual(errors[0]["api"], "decide")
            self.assertEqual(int(errors[0]["attempts"]), 2)
            self.assertIn("503", errors[0]["error"])
            with server.lock:
                server.fail_first = 0
            self.cli(url, "--limit", "1", "--retries", "1")
            self.assertEqual(len(server.requests), 3)
            self.assertEqual(len(self.read_csv("preds.csv", PRED_COLUMNS)), 1)
            self.assertEqual(len(self.read_csv("score_errors.csv")), 1)

    def test_resume_rejects_backend_or_image_change(self) -> None:
        with running_fake() as (server, url):
            self.cli(url, "--limit", "1")
            self.cli(url, "--limit", "1", "--api", "chat-logprobs", success=False)
            self.assertEqual(len(server.requests), 1)
            image_path = ROOT / self.rows[0]["file"]
            with Image.open(image_path) as original:
                changed = original.copy()
            changed.putpixel((0, 0), (255, 0, 255))
            changed.save(image_path)
            self.cli(url, "--limit", "1", success=False)
            self.assertEqual(len(server.requests), 1)

    def test_fake_service_requires_dry_run_and_loopback(self) -> None:
        with self.assertRaises(ValueError):
            make_server("0.0.0.0")
        with running_fake() as (server, url):
            process = subprocess.run([
                sys.executable, str(ROOT / "src" / "score.py"), "--base-url", url,
                "--index", str(self.index), "--results-dir", str(self.results), "--limit", "1",
                "--retries", "0", "--timeout", "2",
            ], cwd=ROOT, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(process.returncode, 0, process.stderr + process.stdout)
            self.assertEqual(len(server.requests), 1)
            self.assertFalse((self.results / "preds.csv").exists()
                             and len(self.read_csv("preds.csv")) > 0)


class ProbabilityContractTests(unittest.TestCase):
    def test_reordered_options_and_invalid_native_responses(self) -> None:
        response = {"options": ["Down", "Up", "Flat"], "probabilities": [0.2, 0.5, 0.3]}
        self.assertEqual(parse_probabilities(response, DIRECTION_OPTIONS), [0.5, 0.3, 0.2])
        invalid = [
            {"options": list(DIRECTION_OPTIONS)},
            {"options": ["Up", "Flat", "Flat"], "probabilities": [0.3, 0.3, 0.4]},
            {"options": list(DIRECTION_OPTIONS), "probabilities": [0.2, 0.3]},
            {"options": list(DIRECTION_OPTIONS), "probabilities": [0.2, 0.3, 0.6]},
            {"options": list(DIRECTION_OPTIONS), "probabilities": [float("nan"), 0.3, 0.7]},
            {"options": list(DIRECTION_OPTIONS), "probabilities": [-0.1, 0.4, 0.7]},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_probabilities(value, DIRECTION_OPTIONS)

    def test_chat_rejects_missing_option_logprob(self) -> None:
        response = {"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "token_id:32", "logprob": -0.5},
            {"token": "token_id:33", "logprob": -1.0},
        ]}]}}]}
        with self.assertRaisesRegex(ValueError, "missing"):
            parse_probabilities(response, DIRECTION_OPTIONS, "chat-logprobs")


if __name__ == "__main__":
    unittest.main()
