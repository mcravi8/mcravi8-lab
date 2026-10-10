"""Exercise both prospective tasks through every copied scorer CLI."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from fake_jev_server import decode_png
from test_score import running_fake


class CopiedScorerTests(unittest.TestCase):
    def test_prospective_tasks_and_blank_control_across_four_clis(self):
        with running_fake() as (server, url):
            for experiment in sorted(ROOT.parent.glob("0[1-4]-*")):
                cache = experiment / ".cache"
                cache.mkdir(exist_ok=True)
                with tempfile.TemporaryDirectory(dir=cache) as directory:
                    work = Path(directory)
                    image = work / "anonymous.png"
                    Image.new("RGB", (64, 48), (32, 76, 112)).save(image)
                    index = work / "index.csv"
                    with index.open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=["file", "ticker", "end_date", "label", "fwd_ret", "sigma"])
                        writer.writeheader()
                        writer.writerow({"file": str(image.relative_to(experiment)),
                                         "ticker": "TSLA" if experiment.name.startswith(("01-", "02-")) else "AAPL",
                                         "end_date": "2026-01-02", "label": "Flat", "fwd_ret": 0.04, "sigma": 0.06})
                    for task, api, filename in (("binary-direction", "chat-logprobs", "preds_binary.csv"),
                                               ("ternary-volatility", "decide", "preds_volatility_blank.csv")):
                        with self.subTest(experiment=experiment.name, task=task):
                            output = work / task
                            command = [sys.executable, str(experiment / "src/score.py"),
                                       "--base-url", url, "--api", api, "--task", task,
                                       "--dry-run", "--index", str(index), "--results-dir", str(output),
                                       "--limit", "1", "--retries", "0", "--timeout", "2"]
                            if task == "ternary-volatility":
                                command.append("--blank")
                            process = subprocess.run(command, cwd=experiment, text=True,
                                                     capture_output=True, timeout=10)
                            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
                            with (output / filename).open(newline="") as stream:
                                scored = list(csv.DictReader(stream))
                            self.assertEqual(len(scored), 1)
                            self.assertEqual(scored[0]["label"], "Up")
                            if task == "binary-direction":
                                self.assertEqual(float(scored[0]["p_flat"]), 0)
                            else:
                                payload = server.requests[-1]["body"]
                                sent = decode_png(payload["state"][1]["image"])
                                self.assertEqual(sent.getextrema(), ((255, 255),) * 3)
                                self.assertIn("times sigma = 0.02", payload["question"])
                                lookback = 60 if experiment.name.startswith("01-") else 12
                                self.assertIn(f"last {lookback} completed observations", payload["question"])
                            meta = json.loads((output / filename.replace(".csv", ".meta.json")).read_text())
                            self.assertEqual(meta["task"], task)
                            self.assertEqual(meta["server_identity"]["model_revision"], "synthetic-fixture-weights-v1")
            self.assertEqual(len(server.requests), 8)


if __name__ == "__main__":
    unittest.main()
