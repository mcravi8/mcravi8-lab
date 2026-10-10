"""Weekly-window, horizon, and volatility regressions for Experiments 02-04."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def prices(length=125):
    positions = np.arange(length)
    close = 100 + positions * 0.3 + 2 * np.sin(positions * 0.3)
    return pd.DataFrame({"Open": close - 0.1, "High": close + 0.5,
                         "Low": close - 0.5, "Close": close,
                         "Volume": 1000 + positions},
                        index=pd.bdate_range("2024-01-08", periods=length, name="Date"))


def render_copies():
    for directory in sorted(ROOT.parent.glob("0[2-4]-*")):
        old_data = sys.modules.get("data")
        try:
            data_spec = importlib.util.spec_from_file_location("weekly_data", directory / "src/data.py")
            data = importlib.util.module_from_spec(data_spec)
            data_spec.loader.exec_module(data)
            sys.modules["data"] = data
            spec = importlib.util.spec_from_file_location("weekly_render_" + directory.name.replace("-", "_"), directory / "src/render.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if old_data is None:
                sys.modules.pop("data", None)
            else:
                sys.modules["data"] = old_data
        yield directory.name, module


class WeeklyValidationTests(unittest.TestCase):
    def test_zero_and_nonfinite_volatility_rejected(self):
        flat = prices()
        flat[["Open", "Close"]] = 100.0
        flat["High"], flat["Low"] = 101.0, 99.0
        for name, render in render_copies():
            with self.subTest(experiment=name):
                with self.assertRaisesRegex(ValueError, "volatility"):
                    render.weekly_rows(flat)
                for sigma in (float("nan"), float("inf")):
                    with patch.object(render.np, "std", return_value=sigma):
                        with self.assertRaisesRegex(ValueError, "volatility"):
                            render.weekly_rows(prices())

    def test_weekly_sigma_target_and_future_causality(self):
        original = prices()
        # Week 14 is visible; change the following week's prices only.
        end_position = 5 * 14 + 4
        changed = original.copy()
        changed.iloc[end_position + 1:, :4] *= 1.6
        for name, render in render_copies():
            with self.subTest(experiment=name):
                before = {row["end_date"]: row for row in render.weekly_rows(original)}
                after = {row["end_date"]: row for row in render.weekly_rows(changed)}
                end_date = original.index[end_position].date().isoformat()
                row, altered = before[end_date], after[end_date]
                past_ends = [float(original.Close.iloc[5 * week + 4]) for week in range(15)]
                past_returns = [past_ends[week] / past_ends[week - 1] - 1 for week in range(3, 15)]
                self.assertAlmostEqual(row["sigma"], float(np.std(past_returns, ddof=1)), places=14)
                self.assertAlmostEqual(row["fwd_ret"], original.Close.iloc[end_position + 5] / original.Close.iloc[end_position] - 1)
                self.assertEqual(row["target_date"], original.index[end_position + 5].date().isoformat())
                pd.testing.assert_frame_equal(row["window"], altered["window"])
                self.assertEqual(row["sigma"], altered["sigma"])
                self.assertNotEqual(row["fwd_ret"], altered["fwd_ret"])
                self.assertTrue(all(len(item["window"]) == 5 for item in before.values()))

    def test_holiday_target_and_incomplete_final_week(self):
        original = prices()
        # A shortened week remains a valid target but never a five-candle input.
        holiday = original.drop(original.index[5 * 15 + 2])
        source_end = original.index[5 * 14 + 4].date().isoformat()
        holiday_end = original.index[5 * 15 + 4].date().isoformat()
        for name, render in render_copies():
            with self.subTest(experiment=name):
                rows = {row["end_date"]: row for row in render.weekly_rows(holiday)}
                self.assertFalse(rows[source_end]["next_week_full"])
                self.assertEqual(rows[source_end]["target_date"], holiday_end)
                self.assertNotIn(holiday_end, rows)
                # End on Thursday: preceding full week has no complete target.
                incomplete = original.iloc[:5 * 17 + 4]
                previous_end = original.index[5 * 16 + 4].date().isoformat()
                self.assertNotIn(previous_end, {row["end_date"] for row in render.weekly_rows(incomplete)})


if __name__ == "__main__":
    unittest.main()
