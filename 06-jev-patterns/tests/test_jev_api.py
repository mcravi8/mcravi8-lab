"""Native JEV contract fixtures, including extended choice and served aliases.

Reference: autotrust/JEV-27B-VL/serve_decide.py, DecideRequest.strategy and
response adaptation (native for <=16, strategy name for >16), checked 2026-10-10.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.src.jev_api import (
    MODEL, PROTOCOL, CHOICE_TEMPERATURE, build_payload, options_list,
    server_identity, parse_probabilities,
)

REVISION = "b" * 40


def info(model=MODEL):
    return {"model": model, "protocol": PROTOCOL, "model_revision": REVISION,
            "temperatures": {"choice": CHOICE_TEMPERATURE}}


def response(options, model=MODEL):
    return {"model": model, "protocol": PROTOCOL, "kind": "choice", "effective_kind": "choice",
            "adaptation": "native" if len(options) <= 16 else "single", "options": options,
            "probabilities": [1 / len(options)] * len(options)}


class JevApiTests(unittest.TestCase):
    def test_choice_boundaries_and_explicit_strategy(self):
        for n in (2, 7, 8, 16, 17, 256):
            options = [f"Class {number}" for number in range(n)]
            payload = build_payload("data:image/png;base64,AAA=", options, "Choose the shape.", "Definitions.")
            self.assertEqual(payload["strategy"], "single")
            self.assertEqual(payload["thinking"], "off")
            self.assertEqual(payload["options"], options)
            self.assertEqual(payload["question"], "Definitions.\n\nChoose the shape.")
            values = parse_probabilities(response(options), options, server_identity(info()))
            self.assertEqual(len(values), n)
            self.assertAlmostEqual(sum(values), 1)
            self.assertTrue(all(abs(p - 1 / n) < 1e-12 for p in values))
        for options in (["One"], [str(n) for n in range(257)], ["A", "A"], ["A", "B\nC"], "ABC"):
            with self.assertRaises(ValueError):
                options_list(options)

    def test_extended_response_requires_pinned_strategy(self):
        for n, wrong in ((16, "single"), (17, "native"), (17, "tournament"), (256, "permute")):
            options = [str(number) for number in range(n)]
            payload = response(options)
            payload["adaptation"] = wrong
            with self.assertRaises(ValueError):
                parse_probabilities(payload, options, server_identity(info()))

    def test_served_alias_is_explicit_and_identity_stays_pinned(self):
        identity = server_identity(info("jev"), expected_model="jev")
        self.assertEqual(identity["model"], "jev")
        self.assertEqual(identity["canonical_model"], MODEL)
        self.assertEqual(identity["choice_strategy"], "single")
        self.assertEqual(parse_probabilities(response(["A", "B"], "jev"), ["A", "B"], identity), [.5, .5])
        with self.assertRaises(ValueError):
            server_identity(info("jev"))
        with self.assertRaises(ValueError):
            parse_probabilities(response(["A", "B"], MODEL), ["A", "B"], identity)

    def test_literal_none_option_and_reordering(self):
        requested = ["Head and Shoulders", "Double Top", "None"]
        payload = response(["None", "Head and Shoulders", "Double Top"])
        payload["probabilities"] = [.2, .5, .3]
        self.assertEqual(parse_probabilities(payload, requested, server_identity(info())), [.5, .3, .2])

    def test_revisions_calibration_and_probability_validation(self):
        identity = server_identity(info())
        wrong = info()
        wrong["temperatures"] = None
        with self.assertRaises(ValueError):
            server_identity(wrong)
        wrong = info()
        wrong["temperatures"]["choice"] = 1
        with self.assertRaises(ValueError):
            server_identity(wrong)
        with self.assertRaises(ValueError):
            server_identity(info(), declared_revision="c" * 40)
        payload = response(["A", "B"])
        for probabilities in ([.3, .3], [True, 0], [float("nan"), .5], [.5], [-.2, 1.2]):
            payload["probabilities"] = probabilities
            with self.assertRaises(ValueError):
                parse_probabilities(payload, ["A", "B"], identity)
        payload = response(["A", "B"])
        payload["model_revision"] = "d" * 40
        with self.assertRaises(ValueError):
            parse_probabilities(payload, ["A", "B"], identity)
        payload = response(["A", "B"])
        payload["thinking"] = {"mode": "on", "used": True}
        with self.assertRaises(ValueError):
            parse_probabilities(payload, ["A", "B"], identity)


if __name__ == "__main__":
    unittest.main()
