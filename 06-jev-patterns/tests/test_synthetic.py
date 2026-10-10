"""CPU fixtures for source geometry, nuisance matching, corruptions and export."""
import importlib.util
from pathlib import Path
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

    def test_geometry_and_inverse_constructions(self):
        for top, inverse in ((0, 1), (2, 3)):
            first, second = self.specs[top].copy(), self.specs[top].copy()
            second["label"] = self.protocol["classes"][inverse]
            x1, y1, _ = synth.geometry(first); x2, y2, _ = synth.geometry(second)
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_allclose(y1 + y2, 1)
            self.assertFalse(np.array_equal(y1[::-1], y2))
        for item in self.specs:
            x, y, info = synth.geometry(item)
            self.assertTrue((np.diff(x) > 0).all())
            if item["label"] == "Head and Shoulders":
                self.assertGreater(y[3], max(y[1], y[5]))
                self.assertLessEqual(abs(y[1] - y[5]), .03)
                self.assertLess(y[-1], min(y[2], y[4]))
            elif item["label"] == "Double Top":
                self.assertLessEqual(abs(y[1] - y[3]), .03)
                self.assertLess(y[-1], y[2])
            elif item["label"] == "Bull Flag":
                self.assertTrue(y[2] > y[4] > y[6])
                self.assertTrue(y[3] > y[5] > y[7])
                self.assertGreater(y[-1], max(y[:-1]))
            elif item["label"] == "None":
                self.assertIsNone(synth.obvious_completed_pattern(x, y))
                self.assertIsNone(info["breakout_direction"])

    def test_none_source_screen_rejects_obvious_completed_fixtures(self):
        examples = {
            "Head and Shoulders": ([0,.16,.32,.49,.65,.8,1], [0,.7,.25,1,.25,.7,-.2]),
            "Double Top": ([0,.25,.5,.75,1], [0,1,.25,1,-.2]),
            "Triangle": ([0,.16,.32,.48,.64,.8,.9,1], [0,1,.2,.8,.4,.6,.45,1.1]),
        }
        for label, (x, y) in examples.items():
            self.assertEqual(synth.obvious_completed_pattern(x, y), label)
        x, y = examples["Double Top"]
        self.assertEqual(synth.obvious_completed_pattern(x, 1 - np.asarray(y)), "Double Bottom")
        self.assertIsNone(synth.obvious_completed_pattern([0,.2,.4,.6,.8,1], [0,.8,.2,.7,.1,.5]))
        # Parallel falling bounds after a falling pole are not a converging triangle.
        self.assertIsNone(synth.obvious_completed_pattern(
            [0,.16,.25,.37,.49,.61,.73,.86,1], [1,0,.95,.72,.84,.58,.71,.48,1.4]))

    def test_parent_diversity_none_families_and_common_volatility(self):
        none_families = set()
        for label in self.protocol["classes"]:
            parents = [item for item in self.specs if item["label"] == label]
            shapes, lengths, starts, incoming, outgoing = set(), set(), set(), set(), set()
            for item in parents:
                frame, _, details = synth.make_template(item)
                x, y, _ = synth.geometry(item)
                shapes.add((tuple(x), tuple(y))); lengths.add(len(frame)); starts.add(details["pattern_start_bar"])
                incoming.add(np.sign(details["lead_in_log_return"])); outgoing.add(np.sign(details["lead_out_log_return"]))
                self.assertAlmostEqual(np.std(np.diff(np.log(frame.Close)), ddof=1), item["target_logvol"], places=12)
                self.assertTrue(.004 <= details["realized_clean_logvol"] <= .018)
                if label == "None":
                    none_families.add(details["none_family"])
                    self.assertGreater(np.sum(np.diff(np.sign(np.diff(frame.Close))) != 0), 1)
            self.assertEqual(len(shapes), len(parents))
            self.assertGreater(len(lengths), 12)
            self.assertGreater(len(starts), 5)
            self.assertEqual(incoming, {-1., 1.}); self.assertEqual(outgoing, {-1., 1.})
        self.assertEqual(none_families, {"random_walk", "choppy_range", "trend_with_pullbacks",
                                       "incomplete_formation", "broken_reversal", "broken_consolidation"})
        self.assertEqual(self.protocol["volume_color"], "neutral")

    def test_hard_negatives_are_frozen_families_without_target_completion(self):
        kinds = set()
        for item in self.specs:
            if item["label"] != "None":
                continue
            x, y, details = synth.geometry(item)
            kinds.add(details["none_source_kind"])
            self.assertIsNone(synth.obvious_completed_pattern(x, y))
            if details["none_source_kind"] == "triple_top":
                self.assertLess(max(y[1], y[3], y[5]) - min(y[1], y[3], y[5]), .06)
            if details["none_source_kind"] == "rising_consolidation":
                self.assertTrue(y[2] < y[4] < y[6])
                self.assertTrue(y[3] < y[5] < y[7])
            if details["none_source_kind"] == "unequal_double_top":
                self.assertGreater(y[1] - y[3], .15)
        self.assertTrue(any(kind.startswith("incomplete_") for kind in kinds))
        self.assertTrue(any(kind.startswith("triple_") for kind in kinds))
        self.assertIn("diverging_bounds", kinds)
        self.assertIn("rising_consolidation", kinds)
        summary = synth.population_summary()
        self.assertEqual(summary["none_family_counts"], {
            "random_walk": 5, "choppy_range": 5, "trend_with_pullbacks": 5,
            "incomplete_formation": 5, "broken_reversal": 4, "broken_consolidation": 4})

    def test_noise_is_nested_strong_and_not_pivot_pinned(self):
        self.assertEqual(self.protocol["noise_levels"], [0., .035, .08, .16, .30, .55])
        for item in self.specs[:7]:
            clean, noise, details = synth.make_template(item)
            prior = -1
            for level in self.protocol["noise_levels"]:
                frame = synth.apply_noise(clean, noise, level, details)
                delta = np.log(frame.Close.to_numpy()) - np.log(clean.Close.to_numpy())
                magnitude = float(np.max(np.abs(delta)))
                self.assertGreater(magnitude, prior)
                self.assertAlmostEqual(magnitude, level * details["clean_log_range"], places=10)
                np.testing.assert_allclose(delta[[0, -1]], 0, atol=1e-12)
                if level:
                    self.assertGreater(np.max(np.abs(delta[details["pivot_bars"]][1:-1])), 1e-5)
                prior = magnitude
            # Clean parents complete. Corruption deliberately has no such claim.
            tail = clean.Close.iloc[details["pattern_end_bar"]:]
            if details["breakout_direction"] == "up":
                self.assertTrue((tail > details["breakout_level"]).all())
            elif details["breakout_direction"] == "down":
                self.assertTrue((tail < details["breakout_level"]).all())

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
            pattern_bars = info["pattern_end_bar"] - info["pattern_start_bar"] + 1
            for key in ("padding_left_bars", "padding_right_bars"):
                self.assertTrue(.1 <= info[key] / pattern_bars <= .2)
        self.assertGreater(len(lengths), 40)
        a, noise_a, _ = synth.make_template(self.specs[0]); b, noise_b, _ = synth.make_template(self.specs[0])
        pd.testing.assert_frame_equal(a, b); np.testing.assert_array_equal(noise_a, noise_b)

    def test_fixed_framing_retains_endpoints_and_valid_bars(self):
        original, _, _ = synth.make_template(self.specs[0])
        fixed = synth.fixed_frame(original)
        self.assertEqual(len(fixed), 100)
        np.testing.assert_allclose(fixed.iloc[[0, -1]], original.iloc[[0, -1]])
        self.assertTrue((fixed.High >= fixed[["Open", "Close"]].max(axis=1)).all())
        self.assertTrue((fixed.Low <= fixed[["Open", "Close"]].min(axis=1)).all())
        with self.assertRaises(ValueError):
            synth.fixed_frame(original, 99)

    def test_export_roundtrip_and_neutral_volume(self):
        from shared.src.render import render_chart
        from PIL import Image
        frame, _, _ = synth.make_template(self.specs[0])
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "bars.csv"; png = Path(folder) / "chart.png"
            frame.to_csv(csv, index_label="Date")
            restored = pd.read_csv(csv, index_col="Date", parse_dates=True)
            np.testing.assert_allclose(restored.to_numpy(), frame.to_numpy())
            render_chart(frame, png, n_bars=len(frame), size=(1024, 640), style="candle", mirror=False, volume_color="neutral")
            with Image.open(png) as image:
                self.assertEqual(image.size, (1024, 640))
                pixels = np.asarray(image.convert("RGB"))
                self.assertGreater(pixels.std(), 5)
                volume = pixels[500:610]
                np.testing.assert_array_equal(volume[:,:,0], volume[:,:,1])
                np.testing.assert_array_equal(volume[:,:,1], volume[:,:,2])


if __name__ == "__main__":
    unittest.main()
