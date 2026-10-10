"""Independent numeric fixtures for System 1 and prospective task definitions."""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import unittest
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import jev_api
from test_score import running_fake


def chat_fixture(logits, model="jev-decision"):
    largest = max(logits)
    log_total = largest + math.log(sum(math.exp(value - largest) for value in logits))
    entries = [{"token": f"token_id:{32 + slot}", "logprob": value - log_total}
               for slot, value in enumerate(logits)]
    return {"model": model, "choices": [{"logprobs": {"content": [{"top_logprobs": entries[::-1]}]}}]}


def api_copies():
    for directory in sorted(ROOT.parent.glob("0[1-4]-*")):
        spec = importlib.util.spec_from_file_location("protocol_" + directory.name.replace("-", "_"), directory / "src/jev_api.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield directory.name, module


class System1NumericTests(unittest.TestCase):
    def test_pinned_bias_temperature_and_option_slot_mapping(self):
        # Independent expected values for raw logits [0.5, -1, 2], after the
        # published decision-head bias and T=1.0161280671450397. Both bias and
        # temperature materially change these values; sum-to-one alone is weak.
        expected_three = [0.17841123690119948, 0.04077082642778982, 0.7808179366710106]
        expected_two = [0.8139864831612915, 0.18601351683870856]
        for name, api in api_copies():
            with self.subTest(experiment=name):
                three = api.parse_probabilities(chat_fixture([0.5, -1, 2]), ["Up", "Flat", "Down"], "chat-logprobs")
                two = api.parse_probabilities(chat_fixture([0.5, -1]), ["Up", "Down"], "chat-logprobs")
                for actual, expected in zip(three + two, expected_three + expected_two):
                    self.assertAlmostEqual(actual, expected, places=13)
                reordered = {"options": ["Down", "Up", "Flat"],
                             "probabilities": [expected_three[2], expected_three[0], expected_three[1]]}
                mapped = api.parse_probabilities(reordered, ["Up", "Flat", "Down"])
                for actual, expected in zip(mapped, expected_three):
                    self.assertAlmostEqual(actual, expected, places=13)

    def test_native_and_chat_probability_parity_for_two_and_three_options(self):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 6), (21, 43, 65)).save(buffer, format="PNG")
        image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        with running_fake() as (server, url):
            for options in (["Up", "Flat", "Down"], ["Up", "Down"], ["TSLA", "NVDA"]):
                with self.subTest(options=options):
                    probabilities = []
                    for api in ("decide", "chat-logprobs"):
                        payload = jev_api.build_payload(image_url, options, "Fixture question", api)
                        request = urllib.request.Request(url + jev_api.API_PATHS[api],
                            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(request, timeout=2) as response:
                            result = json.load(response)
                        probabilities.append(jev_api.parse_probabilities(result, options, api))
                    for native, chat in zip(*probabilities):
                        self.assertAlmostEqual(native, chat, places=13)

    def test_server_revision_required_and_responses_bound_to_identity(self):
        info = {"protocol": "jev27-bare-v1", "model": "autotrust/JEV-27B-VL",
                "temperatures": {"choice": 1.0161280671450397}}
        with self.assertRaisesRegex(ValueError, "model-revision"):
            jev_api.server_identity(info)
        identity = jev_api.server_identity(info, "immutable-weights-hash")
        self.assertEqual(identity["revision_source"], "caller-declared")
        response = {"protocol": info["protocol"], "model": info["model"],
                    "kind": "choice", "effective_kind": "choice", "adaptation": "native"}
        jev_api.validate_native_identity(response, identity)
        for change in ({"protocol": "other"}, {"model": "another-model"},
                       {"thinking": {"used": True}}, {"system": 2},
                       {"model_revision": "changed-hash"}, {"adaptation": "permute"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                jev_api.validate_native_identity({**response, **change}, identity)
        with self.assertRaisesRegex(ValueError, "differs"):
            jev_api.server_identity({**info, "model_revision": "server-hash"}, "caller-hash")

    def test_chat_rejects_positive_and_duplicate_option_logprobs(self):
        for kind in ("positive", "duplicate"):
            response = chat_fixture([0.5, -1, 2])
            entries = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            if kind == "positive":
                entries[0]["logprob"] = 0.25
            else:
                entries.append(dict(entries[0]))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                jev_api.parse_probabilities(response, ["Up", "Flat", "Down"], "chat-logprobs")


class ProspectiveTaskTests(unittest.TestCase):
    def test_historical_prompt_preserved_across_all_copies(self):
        for name, api in api_copies():
            with self.subTest(experiment=name):
                expected = ("This is a 20-day daily stock chart. Over the next 5 trading days, will the price go Up, stay Flat, or go Down?"
                            if name.startswith("01-") else "This is a 5-day daily stock chart covering one trading week. Over the next week, will the price go Up, stay Flat, or go Down?")
                self.assertEqual(api.task_question({"sigma": "should-not-be-used"}, "historical"), expected)

    def test_binary_and_flat_boundaries_match_their_declared_targets(self):
        for name, api in api_copies():
            with self.subTest(experiment=name):
                for forward, expected in ((0, None), (0.001, "Up"), (-0.001, "Down")):
                    self.assertEqual(api.task_label({"fwd_ret": forward}, "binary-direction"), expected)
                for forward, expected in ((0, "Flat"), (0.24999, "Flat"), (0.25, "Up"), (-0.25, "Down")):
                    self.assertEqual(api.task_label({"fwd_ret": forward, "sigma": 0.75}, "ternary-volatility"), expected)
                for sigma in (0, -0.1, float("nan"), float("inf")):
                    with self.assertRaises(ValueError):
                        api.task_question({"sigma": sigma}, "ternary-volatility")

    def test_future_label_date_ticker_and_return_cannot_change_prompt(self):
        past = {"sigma": 0.06, "label": "Up", "fwd_ret": 0.3,
                "ticker": "TSLA", "end_date": "2026-01-02"}
        future_changed = {**past, "label": "Down", "fwd_ret": -0.8,
                          "ticker": "NVDA", "end_date": "2031-07-08"}
        for name, api in api_copies():
            for task in ("binary-direction", "ternary-volatility"):
                with self.subTest(experiment=name, task=task):
                    self.assertEqual(api.task_question(past, task), api.task_question(future_changed, task))
                    prompt = api.task_question(past, task)
                    if task == "ternary-volatility":
                        self.assertIn("sigma = 0.06", prompt)
                        self.assertIn("times sigma = 0.02", prompt)
                    for forbidden in ("TSLA", "NVDA", "2026-01-02", "2031-07-08", "fwd_ret"):
                        self.assertNotIn(forbidden, prompt)


if __name__ == "__main__":
    unittest.main()
