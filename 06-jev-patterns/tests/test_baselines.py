"""Train-only priors/scalers, visible-window causality and synthetic fit guards."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "shared/src"
sys.path.insert(0, str(SRC))
import baselines


def bars():
    return pd.DataFrame({"Open": [10., 11., 12.], "High": [12., 13., 14.], "Low": [9., 10., 11.],
                         "Close": [11., 12., 13.], "Volume": [100., 200., 300.]},
                        index=pd.date_range("2018-01-01", periods=3))


def real_index():
    dates = pd.date_range("2018-01-31", periods=10, freq="ME")
    return pd.DataFrame({"file": [f"charts/{i}.png" for i in range(10)], "ticker": "AAA",
                         "end_date": dates.strftime("%Y-%m-%d"), "window_start": (dates - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
                         "label": ["A", "B"] * 5, "split": "train", "n_bars": 20})


class BaselineTests(unittest.TestCase):
    def test_prior_and_majority_use_only_supplied_training_labels(self):
        p = baselines.constant_predictions(["A", "A", "B"], 2, ["B", "A", "None"])
        np.testing.assert_allclose(p["train_prior"], [[1 / 3, 2 / 3, 0.]] * 2)
        np.testing.assert_equal(p["train_majority"], [[0., 1., 0.]] * 2)
        np.testing.assert_allclose(p["uniform"], np.full((2, 3), 1 / 3))

    def test_numeric_features_are_visible_scale_invariant(self):
        original = bars()
        changed = original.copy(); changed.iloc[:, :4] *= 100; changed.Volume *= 100
        np.testing.assert_allclose(baselines.numeric_features(original), baselines.numeric_features(changed), atol=1e-14)
        known = baselines.numeric_features(original, steps=3).reshape(3, 5)
        np.testing.assert_allclose(known[:, :4], original.iloc[:, :4].to_numpy() / 11 - 1)
        np.testing.assert_allclose(known[:, 4], [.5, 1., 1.5])

    def test_adjusted_roundoff_is_accepted_but_material_ordering_rejected(self):
        adjusted = bars()
        adjusted.loc[adjusted.index[0], "Low"] = np.nextafter(adjusted.Close.iloc[0], np.inf)
        adjusted.loc[adjusted.index[0], "Open"] = adjusted.Close.iloc[0]
        self.assertTrue(np.isfinite(baselines.numeric_features(adjusted)).all())
        invalid = adjusted.copy(); invalid.loc[invalid.index[0], "Low"] += .01
        with self.assertRaisesRegex(ValueError, "candle ordering"):
            baselines.numeric_features(invalid)

    def test_synthetic_uniform_cli_predicts_without_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            idx = pd.DataFrame({"file": ["charts/synthetic.png"], "ticker": ["SYNTH"],
                                "end_date": [""], "label": ["None"], "split": ["eval"], "template_id": ["shape_0"]})
            path = root / "index.csv"; idx.to_csv(path, index=False)
            completed = subprocess.run([sys.executable, str(SRC / "baselines.py"), "--index", str(path), "--root", str(root),
                            "--options", '["A", "None"]', "--out", str(root / "predictions"), "--models", "uniform",
                            "--mode", "predict"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            metadata = json.loads((root / "predictions/uniform_preds.meta.json").read_text())
            self.assertFalse(metadata["synthetic"])
            self.assertTrue(metadata["synthetic_dataset"])
            self.assertEqual(metadata["prediction_origin"], "baseline")
            self.assertEqual(metadata["trained_on"], "None; uniform has no fitting")
            self.assertEqual(metadata["uniform_random_expected_accuracy"], .5)

    def test_future_bars_cannot_enter_numeric_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = bars()
            future = raw.iloc[-1:].copy(); future.index = [pd.Timestamp("2018-01-04")]
            full = pd.concat([raw, future]); full.to_parquet(root / "AAA.parquet")
            row = {"ticker": "AAA", "end_date": "2018-01-03", "window_start": "2018-01-01", "n_bars": 3}
            before = baselines.numeric_features(baselines.load_visible_window(row, root, root))
            full.loc["2018-01-04", ["Open", "High", "Low", "Close"]] *= 999
            full.to_parquet(root / "AAA.parquet")
            after = baselines.numeric_features(baselines.load_visible_window(row, root, root))
            np.testing.assert_array_equal(before, after)
            with self.assertRaises(ValueError):
                baselines.load_visible_window({**row, "n_bars": 4}, root, root)

    def test_logistic_scaler_is_not_refit_on_evaluation_and_absent_classes_zero(self):
        x = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.]])
        model = baselines.fit_logistic(x, ["A", "A", "B", "B"], ["B", "None", "A"])
        scaler = model.steps[0][1]
        np.testing.assert_allclose(scaler.mean_, x.mean(0))
        before = scaler.mean_.copy()
        p = baselines.logistic_probabilities(model, [[100000., -100000.]], ["B", "None", "A"])
        np.testing.assert_array_equal(scaler.mean_, before)
        self.assertEqual(p[0, 1], 0.)
        self.assertAlmostEqual(p.sum(), 1.)

    def test_eval_only_synthetic_cannot_fit_even_before_torch_import(self):
        synthetic = real_index(); synthetic["split"] = "eval"; synthetic["ticker"] = "SYNTH"
        with self.assertRaisesRegex(ValueError, "cannot be fitted"):
            baselines.training_rows(synthetic)
        with mock.patch.object(baselines, "Net", side_effect=AssertionError("model should not be constructed")):
            with self.assertRaises(ValueError):
                baselines.train_cnn(synthetic, ["A", "B"], Path("."), Path("."))
        mislabeled = synthetic.copy(); mislabeled["split"] = "train"
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            baselines.training_rows(mislabeled)

    def test_validation_is_latest_training_dates_and_purges_inputs(self):
        idx = real_index()
        fit, val = baselines.split_train_validation(idx)
        self.assertEqual(len(val), 2)
        self.assertLess(fit.end_date.max(), val.window_start.min())
        self.assertTrue(val.split.eq("train").all())
        with self.assertRaisesRegex(ValueError, "window_start"):
            baselines.split_train_validation(idx.drop(columns="window_start"))
        bad = idx.copy(); bad.loc[0, "end_date"] = "2020-01-01"
        with self.assertRaises(ValueError):
            baselines.training_rows(bad)

    def test_import_and_default_cli_need_no_torch(self):
        script = """
import builtins, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] == 'torch':
        raise AssertionError('Torch imported without explicit CNN training')
    return original(name, *args, **kwargs)
builtins.__import__ = guard
import baselines
"""
        subprocess.run([sys.executable, "-c", script, str(SRC)], check=True, capture_output=True, text=True)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional CPU torch is not installed")
    def test_cnn_has_arbitrary_output_classes_and_cpu_parameters(self):
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        model = baselines.Net(8)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 120, 160, device="cpu"))
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
