"""Hand-calculated metrics and clustered inference invariants."""
import math
import json
import importlib.util
import types
from unittest.mock import patch
import pandas as pd
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.statistics import (block_bootstrap_indices, block_bootstrap_means,
    exact_mcnemar, holm_adjust, paired_inference, paired_week_signflip, probability_metrics)


class StatisticsTests(unittest.TestCase):
    def test_hand_calculated_probability_metrics(self):
        p = [[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [1/3, 1/3, 1/3]]
        out = probability_metrics(p, ["Up", "Flat", "Down"])
        self.assertAlmostEqual(out["hit_rate"], 1/3)
        self.assertAlmostEqual(out["brier"], (0.06 + 0.78 + 2/3) / 3)
        self.assertAlmostEqual(out["log_loss"], -(math.log(0.8) + math.log(0.3) + math.log(1/3))/3)

    def test_sparse_discordant_pairs_do_not_imply_significance(self):
        dates = ["2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23", "2026-01-30"]
        out = paired_week_signflip([-1] * 5, dates)
        self.assertEqual(out["p_two_sided"], 0.0625)
        self.assertTrue(out["exact"])
        self.assertEqual(exact_mcnemar(0, 5), 0.0625)
        self.assertEqual(exact_mcnemar(0, 0), 1)
        # Five discordant rows on one date are one sign-flip unit, not five.
        self.assertEqual(paired_week_signflip([-1] * 5, [dates[0]] * 5)["p_two_sided"], 1)

    def test_bootstrap_keeps_whole_dates_and_adjacent_blocks(self):
        dates = np.repeat(["2026-01-02", "2026-01-09", "2026-01-23", "2026-01-30"], 3)
        samples = block_bootstrap_indices(dates, np.random.default_rng(3), resamples=20, block_length=2)
        for selected in samples:
            self.assertEqual(len(selected), 12)
            for start in range(0, 12, 3):
                group = selected[start:start + 3]
                self.assertEqual(len(set(dates[group])), 1)
                self.assertEqual(group.tolist(), list(range(group[0], group[0] + 3)))
            groups = selected[::3] // 3
            for start in (0, 2):
                self.assertEqual(groups[start + 1], (groups[start] + 1) % 4)

    def test_row_order_invariance_and_holiday_gap_metadata(self):
        dates = np.repeat(["2026-01-02", "2026-01-09", "2026-01-23"], 2)
        values = np.array([1., 0., -1., 0., 1., -1.])
        order = [5, 3, 1, 4, 0, 2]
        a = paired_inference(values, dates, resamples=100, signflip_resamples=100)
        b = paired_inference(values[order], dates[order], resamples=100, signflip_resamples=100)
        self.assertEqual(a, b)
        self.assertEqual(a["date_clusters"], 3)
        self.assertEqual(a["calendar_gaps_over_7_days"], 1)
        self.assertEqual(a["largest_calendar_gap_days"], 14)

    def test_date_aggregation_is_equivalent_to_paired_rows(self):
        dates = ["2026-01-02", "2026-01-02", "2026-01-09"]
        values = np.array([[1., 2.], [3., 4.], [9., 6.]])
        samples = block_bootstrap_indices(dates, np.random.default_rng(7), resamples=20, block_length=1)
        expected = np.array([values[selected].mean(axis=0) for selected in samples])
        actual = block_bootstrap_means(values, dates, resamples=20, seed=7, block_length=1)
        np.testing.assert_allclose(actual, expected)

    def test_evaluator_uses_training_references_and_chart_blind_constants(self):
        source = ROOT / "03-jev-12-tickers" / "src"
        spec = importlib.util.spec_from_file_location("review_eval03", source / "evaluate.py")
        ev = importlib.util.module_from_spec(spec)
        stub = types.ModuleType("data")
        stub.ROOT, stub.TICKERS = source.parent, ["A", "B"]
        # Keep duplicate experiment-local module names isolated from other tests.
        with patch.dict(sys.modules, {"data": stub}):
            spec.loader.exec_module(ev)
        rows = []
        for split, dates in (("train", ["2026-01-02", "2026-01-09"]),
                             ("test", ["2026-01-16", "2026-01-23"])):
            for ticker in ("A", "B"):
                labels = (["Up", "Up"] if ticker == "A" else ["Down", "Down"]) if split == "train" else (["Flat", "Up"] if ticker == "A" else ["Up", "Down"])
                for day, label in zip(dates, labels):
                    rows.append({"ticker": ticker, "split": split, "end_date": day,
                                 "label": label, **dict.fromkeys(ev.P + ev.PL, 1/3)})
        with patch.object(ev, "TICKERS", ["A", "B"]), patch.object(ev, "B", 30):
            frame = ev.add_refs(pd.DataFrame(rows))
            out = ev.metrics(frame[frame.split == "test"], np.random.default_rng(0), True, "maj_test")
        json.dumps(out, allow_nan=False)
        self.assertAlmostEqual(out["base_rate_train_majority"], 0.5)
        self.assertAlmostEqual(out["baselines"]["pooled_train_frequency"]["brier"], 0.75)
        self.assertAlmostEqual(out["baselines"]["uniform"]["brier"], 2/3)
        self.assertIsNone(out["ci95_edge_vs_test_base"])
        self.assertFalse(out["success"]["confirmatory_claim_allowed"])
        self.assertEqual(out["inference"]["primary_block_length"], 4)

    def test_holm_declared_family(self):
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm_adjust([]), [])
        with self.assertRaises(ValueError):
            holm_adjust([1.1])

    def test_invalid_probabilities_and_empty_samples_fail(self):
        for p in ([[0.8, 0.1, 0.2]], [[float("nan"), 0, 1]], [[1.1, 0, -0.1]]):
            with self.assertRaises(ValueError):
                probability_metrics(p, ["Up"])
        with self.assertRaises(ValueError):
            paired_inference([], [])
        with self.assertRaises(ValueError):
            probability_metrics([[1, 0, 0]], ["Sideways"])


if __name__ == "__main__":
    unittest.main()
