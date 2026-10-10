"""Causality, label-boundary, window, image, and input-validation checks."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import validate_ohlcv
from render import eligible_windows, label_return, render_chart, trailing_volatility


def synthetic_prices(length: int = 90) -> pd.DataFrame:
    positions = np.arange(length)
    close = 100 + positions * 0.15 + 3 * np.sin(positions * 0.47)
    opening = close - 0.1 * np.cos(positions)
    return pd.DataFrame({
        "Open": opening, "High": np.maximum(opening, close) + 0.5,
        "Low": np.minimum(opening, close) - 0.5, "Close": close,
        "Volume": 1000 + positions * 11,
    }, index=pd.bdate_range("2024-01-02", periods=length, name="Date"))


class PipelineTests(unittest.TestCase):
    def test_hand_computed_trailing_volatility(self) -> None:
        close = synthetic_prices().Close
        returns = [close.iloc[end] / close.iloc[end - 5] - 1 for end in range(5, 65)]
        mean = sum(returns) / 60
        expected = math.sqrt(sum((value - mean) ** 2 for value in returns) / 59)
        result = trailing_volatility(close)
        self.assertTrue(result.iloc[:64].isna().all())
        self.assertAlmostEqual(float(result.iloc[64]), expected, places=14)

    def test_strict_flat_boundary(self) -> None:
        # A 0.75 std gives an exact 0.25 boundary at the one-third band.
        self.assertEqual(label_return(0, 0.75), "Flat")
        self.assertEqual(label_return(0.24999, 0.75), "Flat")
        self.assertEqual(label_return(-0.24999, 0.75), "Flat")
        self.assertEqual(label_return(0.25, 0.75), "Up")
        self.assertEqual(label_return(-0.25, 0.75), "Down")
        # These returns were Flat under the former half-std band.
        self.assertEqual(label_return(0.30, 0.75), "Up")
        self.assertEqual(label_return(-0.30, 0.75), "Down")
        for forward, sigma in ((0, 0), (0.01, -0.1), (0.01, float("nan")),
                               (float("inf"), 0.1)):
            with self.subTest(forward=forward, sigma=sigma), self.assertRaises(ValueError):
                label_return(forward, sigma)

    def test_window_stride_warmup_and_horizon(self) -> None:
        frame = synthetic_prices()
        windows = eligible_windows(frame)
        self.assertEqual([item["end_pos"] for item in windows], [64, 69, 74, 79, 84])
        self.assertEqual(eligible_windows(frame.iloc[:69]), [])
        self.assertEqual(len(eligible_windows(frame.iloc[:70])), 1)
        for item in windows:
            end = item["end_pos"]
            self.assertEqual(len(frame.iloc[end - 19:end + 1]), 20)
            self.assertLess(end + 5, len(frame))
            self.assertEqual(item["end_date"], frame.index[end].date().isoformat())
            self.assertAlmostEqual(item["fwd_ret"], frame.Close.iloc[end + 5] /
                                   frame.Close.iloc[end] - 1)

    def test_future_mutation_cannot_change_history_or_image(self) -> None:
        original = synthetic_prices()
        changed = original.copy()
        changed.iloc[70:, :4] *= 1.6
        changed.iloc[70:, 4] *= 2
        old_sigma = trailing_volatility(original.Close).iloc[:70]
        new_sigma = trailing_volatility(changed.Close).iloc[:70]
        pd.testing.assert_series_equal(old_sigma, new_sigma)
        old_outcome = eligible_windows(original)[1]
        new_outcome = eligible_windows(changed)[1]
        self.assertNotEqual(old_outcome["fwd_ret"], new_outcome["fwd_ret"])
        cache_dir = ROOT / ".cache"
        cache_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_dir) as directory:
            first, second = Path(directory) / "first.png", Path(directory) / "second.png"
            render_chart(original.iloc[50:70], first)
            render_chart(changed.iloc[50:70], second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with Image.open(first) as image:
                self.assertEqual(image.size, (640, 480))
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
                self.assertNotIn("ticker", image.info)
                self.assertNotIn("label", image.info)

    def test_invalid_input_fails(self) -> None:
        original = synthetic_prices()
        missing = original.copy()
        missing.iloc[10, 0] = float("nan")
        duplicate = original.copy()
        duplicate.index = pd.DatetimeIndex([*original.index[:10], original.index[9],
                                            *original.index[11:]])
        bad_order = original.copy()
        bad_order.iloc[10, 1] = bad_order.iloc[10, 2] - 1
        fractional_volume = original.copy().astype(float)
        fractional_volume.iloc[10, 4] = 1.5
        for bad in (missing, duplicate, bad_order, fractional_volume, original.iloc[::-1]):
            with self.subTest(index=bad.index), self.assertRaises(ValueError):
                validate_ohlcv(bad)
        flat = original.copy()
        flat[["Open", "Close"]] = 100.0
        flat["High"] = 100.5
        flat["Low"] = 99.5
        with self.assertRaises(ValueError):
            eligible_windows(flat)
        with self.assertRaises(ValueError):
            render_chart(original.iloc[:19], ROOT / ".cache" / "invalid.png")


if __name__ == "__main__":
    unittest.main()
