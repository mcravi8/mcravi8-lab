"""CPU fixtures for frozen synthetic geometry, noise pairing, and exports."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp06_synth", ROOT / "06b-chart-patterns/src/synth.py")
synth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth)


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.protocol = synth.read_protocol()
        self.specs = synth.template_specs(self.protocol)

    def test_frozen_counts_and_eval_only(self):
        rows = [info for _, info in synth.dataset(self.protocol)]
        self.assertEqual(len(rows), 1200)
        self.assertEqual(sum(row["is_clean"] for row in rows), 200)
        self.assertEqual(len({row["template_id"] for row in rows}), 200)
        self.assertTrue(all(row["split"] == "eval" for row in rows))
        for label in self.protocol["classes"]:
            clean = sum(row["label"] == label and row["is_clean"] for row in rows)
            self.assertIn(clean, (28, 29))
            self.assertEqual(sum(row["label"] == label for row in rows), clean * 6)

    def test_geometry_and_price_inversion(self):
        for top, inverse in ((0, 1), (2, 3)):
            first, second = self.specs[top].copy(), self.specs[top].copy()
            second["label"] = self.protocol["classes"][inverse]
            x1, y1, _ = synth.geometry(first)
            x2, y2, _ = synth.geometry(second)
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_allclose(y1 + y2, 1)
            self.assertFalse(np.array_equal(y1[::-1], y2))
        _, hs, _ = synth.geometry(self.specs[0])
        self.assertGreater(hs[3], max(hs[1], hs[5]))
        self.assertLessEqual(abs(hs[1] - hs[5]), .03)
        _, dt, _ = synth.geometry(self.specs[2])
        self.assertLessEqual(abs(dt[1] - dt[3]), .03)
        self.assertLess(dt[-1], dt[2])
        for subtype in range(3):
            triangle = {**self.specs[4], "subtype": subtype}
            _, points, info = synth.geometry(triangle)
            gaps = [abs(points[i+1] - points[i]) for i in (0, 2, 4)]
            self.assertTrue(gaps[0] > gaps[1] > gaps[2])
            self.assertIn(info["breakout_direction"], ("up", "down"))
        _, flag, _ = synth.geometry(self.specs[5])
        self.assertGreater(flag[1] - flag[0], .9)
        self.assertTrue(flag[2] > flag[4] > flag[6])
        self.assertTrue(flag[3] > flag[5] > flag[7])
        self.assertGreater(flag[-1], max(flag[:-1]))
        for subtype in range(3):
            _, none, _ = synth.geometry({**self.specs[6], "subtype": subtype})
            self.assertTrue(np.all(np.diff(none) >= 0) or np.all(np.diff(none) <= 0))

    def test_noise_is_nested_pivot_pinned_and_complete(self):
        for spec in self.specs[:7]:
            clean, bridge, info = synth.make_template(spec)
            prior = -1
            for level in self.protocol["noise_levels"]:
                frame = synth.apply_noise(clean, bridge, level, info)
                delta = frame.Close.to_numpy() - clean.Close.to_numpy()
                np.testing.assert_allclose(delta[info["pivot_bars"]], 0, atol=1e-12)
                magnitude = float(np.max(np.abs(delta)))
                self.assertGreater(magnitude, prior)
                self.assertAlmostEqual(magnitude, level * info["clean_close_range"], places=10)
                prior = magnitude
                tail = frame.Close.iloc[info["pattern_end_bar"]:]
                if info["breakout_direction"] == "up":
                    self.assertTrue((tail > info["breakout_level"]).all())
                elif info["breakout_direction"] == "down":
                    self.assertTrue((tail < info["breakout_level"]).all())

    def test_deterministic_valid_ohlcv_and_padding(self):
        self.assertEqual(self.specs, synth.template_specs())
        lengths = set()
        for frame, info in synth.dataset():
            lengths.add(len(frame))
            self.assertTrue(60 <= len(frame) <= 120)
            self.assertTrue(np.isfinite(frame.to_numpy()).all())
            self.assertTrue((frame.Low > 0).all())
            self.assertTrue((frame.High >= frame[["Open", "Close"]].max(axis=1)).all())
            self.assertTrue((frame.Low <= frame[["Open", "Close"]].min(axis=1)).all())
            self.assertTrue((frame.Volume > 0).all())
            self.assertLessEqual(info["padding_left_bars"] / len(frame), .2)
            self.assertLessEqual(info["padding_right_bars"] / len(frame), .2)
            pattern_bars = info["pattern_end_bar"] - info["pattern_start_bar"] + 1
            for key in ("padding_left_bars", "padding_right_bars"):
                self.assertTrue(.1 <= info[key] / pattern_bars <= .2)
        self.assertGreater(len(lengths), 40)
        a, noise_a, _ = synth.make_template(self.specs[0])
        b, noise_b, _ = synth.make_template(self.specs[0])
        pd.testing.assert_frame_equal(a, b)
        np.testing.assert_array_equal(noise_a, noise_b)

    def test_fixed_framing_retains_endpoints_and_valid_bars(self):
        original, _, _ = synth.make_template(self.specs[0])
        fixed = synth.fixed_frame(original)
        self.assertEqual(len(fixed), 100)
        np.testing.assert_allclose(fixed.iloc[[0, -1]], original.iloc[[0, -1]])
        self.assertTrue((fixed.High >= fixed[["Open", "Close"]].max(axis=1)).all())
        self.assertTrue((fixed.Low <= fixed[["Open", "Close"]].min(axis=1)).all())
        with self.assertRaises(ValueError):
            synth.fixed_frame(original, 99)

    def test_export_roundtrip_and_render_dimensions(self):
        from shared.src.render import render_chart
        from PIL import Image
        frame, _, _ = synth.make_template(self.specs[0])
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "bars.csv"
            png = Path(folder) / "chart.png"
            frame.to_csv(csv, index_label="Date")
            restored = pd.read_csv(csv, index_col="Date", parse_dates=True)
            np.testing.assert_allclose(restored.to_numpy(), frame.to_numpy())
            render_chart(frame, png, n_bars=len(frame), size=(1024, 640), style="candle", mirror=False)
            with Image.open(png) as image:
                self.assertEqual(image.size, (1024, 640))
                self.assertGreater(np.asarray(image).std(), 5)


if __name__ == "__main__":
    unittest.main()
