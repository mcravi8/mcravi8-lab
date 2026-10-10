"""CPU checks for split causality, reproducible bundles and fail-closed gates."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_training import prepare, verify_bundle
from splits import assign_fit_split, representative_fixtures
from training_utils import (accumulation_groups, parity_report, require_gpu_packages,
                            require_parity, runtime, sha256, source_hashes, verify_hashes)


def synthetic_index():
    fridays = pd.date_range("2024-01-05", periods=14, freq="W-FRI")
    rows = []
    for ticker in ("AAA", "BBB"):
        for i in [*range(10), 12, 13]:
            rows.append({"file": f"charts/{ticker}_{i}.png", "ticker": ticker,
                         "split": "train" if i < 10 else "test",
                         "week_start": (fridays[i] - pd.Timedelta(days=4)).date().isoformat(),
                         "end_date": fridays[i].date().isoformat(),
                         "target_date": (fridays[i] + pd.Timedelta(days=7)).date().isoformat(),
                         "label": ["Up", "Flat", "Down"][i % 3]})
    return pd.DataFrame(rows)


class SplitAndGateTests(unittest.TestCase):
    def test_purge_target_overlap_and_group_weeks(self):
        idx = assign_fit_split(synthetic_index())
        self.assertEqual(idx.fit_split.value_counts().to_dict(), {"fit": 14, "val": 4, "test": 4, "purged": 2})
        self.assertTrue((idx.groupby("end_date").fit_split.nunique() == 1).all())
        fit, val = idx[idx.fit_split == "fit"], idx[idx.fit_split == "val"]
        self.assertLess(fit.target_date.max(), val.week_start.min())
        self.assertLess(idx[idx.split == "train"].target_date.max(), idx[idx.split == "test"].week_start.min())

    def test_reject_overlap_duplicates_and_incomplete_input(self):
        idx = synthetic_index()
        bad = idx.copy()
        bad.loc[bad.split == "train", "target_date"] = "2025-01-01"
        with self.assertRaises(ValueError):
            assign_fit_split(bad)
        with self.assertRaises(ValueError):
            assign_fit_split(pd.concat([idx, idx.iloc[:1]]))
        with self.assertRaises(ValueError):
            assign_fit_split(idx.drop(columns="target_date"))

    def test_representative_parity_is_fit_only(self):
        idx = assign_fit_split(synthetic_index())
        fixtures = representative_fixtures(idx)
        self.assertTrue(fixtures.fit_split.eq("fit").all())
        self.assertEqual(set(fixtures.ticker), {"AAA", "BBB"})
        for ticker, rows in idx[idx.fit_split == "fit"].groupby("ticker"):
            chosen = fixtures[fixtures.ticker == ticker]
            self.assertEqual(set(chosen.label), set(rows.label))
            self.assertEqual(chosen.end_date.min(), rows.end_date.min())
            self.assertEqual(chosen.end_date.max(), rows.end_date.max())
        pd.testing.assert_frame_equal(fixtures, representative_fixtures(idx.sample(frac=1, random_state=3)))

    def test_historical_parity_discrepancy_blocks_training(self):
        baseline = np.array([[.4, .3, .3]])
        changed = baseline + [[.151832745, -.151832745, 0]]
        self.assertFalse(parity_report(changed, baseline)["passed"])
        with self.assertRaisesRegex(ValueError, "training blocked"):
            require_parity(changed, baseline)
        self.assertTrue(require_parity(baseline + [[.005, -.005, 0]], baseline)["passed"])
        with self.assertRaises(ValueError):
            parity_report(changed, baseline, tolerance=.2)
        with self.assertRaises(ValueError):
            require_parity([[float("nan"), 0, 1]], baseline)

    def test_accumulation_resets_each_epoch_and_averages_remainder(self):
        order = list(range(612))
        for _ in range(2):
            groups = list(accumulation_groups(order, 8))
            self.assertEqual(len(groups), 77)
            self.assertEqual([len(group) for group in groups[:2]], [8, 8])
            self.assertEqual(len(groups[-1]), 4)
            self.assertEqual([i for group in groups for i in group], order)
            self.assertEqual(sum(1 / len(groups[-1]) for _ in groups[-1]), 1.)

    def test_candidate_environment_fails_closed(self):
        environment = runtime()
        environment["packages"]["peft"] = None
        with self.assertRaisesRegex(ValueError, "Unsupported candidate"):
            require_gpu_packages(environment)

    def test_corrected_training_prompt_uses_only_known_sigma(self):
        from jev_training import chart_prompt
        original = {"file": "charts/example.png", "sigma": .03, "label": "Up", "fwd_ret": .07}
        changed = {**original, "label": "Down", "fwd_ret": -.99}
        prompt = chart_prompt(original, "ternary-volatility")
        self.assertIn("sigma = 0.03", prompt)
        self.assertEqual(prompt, chart_prompt(changed, "ternary-volatility"))

    def test_corrected_preparation_requires_fresh_matching_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "charts").mkdir()
            synthetic_index().to_csv(root / "charts/index.csv", index=False)
            with self.assertRaisesRegex(ValueError, "fresh matching reference metadata"):
                prepare(root, root / "bundle", root / "reference.csv", "ternary-volatility")

    def test_corrected_reference_prompt_and_model_identity_are_verified(self):
        from jev_api import task_question
        from prepare_training import TASK_SETTINGS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "charts").mkdir()
            (root / "src").mkdir()
            idx = synthetic_index()
            idx["sigma"] = .03
            idx["fwd_ret"] = idx.label.map({"Up": .02, "Flat": 0., "Down": -.02})
            idx.to_csv(root / "charts/index.csv", index=False)
            for file in idx.file:
                Image.new("RGB", (32, 24), "white").save(root / file)
            refs = idx[["file", "ticker", "end_date", "label"]].copy()
            refs[["p_up", "p_flat", "p_down"]] = [.4, .3, .3]
            refs.to_csv(root / "reference.csv", index=False)
            metadata = {"task": "ternary-volatility", "task_settings": TASK_SETTINGS,
                        "synthetic": False, "control": "image", "server_identity": {"model_revision": "fixture-revision"},
                        "image_sha256": {file: sha256(root / file) for file in idx.file},
                        "question_sha256": {row["file"]: hashlib.sha256(task_question(
                            row, "ternary-volatility", **TASK_SETTINGS).encode()).hexdigest() for row in idx.to_dict("records")}}
            meta_path = root / "reference.meta.json"
            meta_path.write_text(json.dumps(metadata))
            prepare(root, root / "bundle", root / "reference.csv", "ternary-volatility", meta_path)
            _, fixtures, manifest = verify_bundle(root / "bundle")
            self.assertIn("sigma", fixtures.columns)
            self.assertEqual(manifest["task"], "ternary-volatility")
            metadata["question_sha256"][idx.file.iloc[0]] = "wrong"
            meta_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "question hash"):
                prepare(root, root / "bundle", root / "reference.csv", "ternary-volatility", meta_path)

    def test_module_imports_allocate_no_gpu_and_need_no_torch(self):
        script = """
import builtins, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'peft', 'safetensors'}:
        raise AssertionError('GPU package imported eagerly: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
import cnn, lora_train, lora_reload, jev_training, splits, prepare_training
"""
        subprocess.run([sys.executable, "-c", script, str(ROOT / "src")], check=True, capture_output=True, text=True)

    def test_preparation_identity_export_and_hash_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            (root / "charts").mkdir(parents=True)
            (root / "src").mkdir()
            (root / "src/example.py").write_text("# fixture\n")
            idx = synthetic_index()
            idx.to_csv(root / "charts/index.csv", index=False)
            for file in idx.file:
                Image.new("RGB", (640, 480), "white").save(root / file)
            refs = idx[["file", "ticker", "end_date", "label"]].copy()
            refs[["p_up", "p_flat", "p_down"]] = [.4, .3, .3]
            reference = root / "reference.csv"
            refs.to_csv(reference, index=False)
            output = Path(directory) / "bundle"
            first = prepare(root, output, reference)
            exported, fixtures, manifest = verify_bundle(output)
            self.assertEqual(first["counts"], {"fit": 14, "val": 4, "test": 4, "purged": 2})
            self.assertTrue(fixtures.fit_split.eq("fit").all())
            second = prepare(root, output, reference)
            self.assertEqual(first["files_sha256"], second["files_sha256"])
            (output / exported.file.iloc[0]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_bundle(output)
            refs.loc[0, "ticker"] = "WRONG"
            refs.to_csv(reference, index=False)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                prepare(root, output, reference)

    def test_lora_reload_rejects_a_different_valid_bundle_before_gpu_import(self):
        import builtins
        import jev_training
        from prepare_training import TASK_SETTINGS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            (root / "charts").mkdir(parents=True)
            (root / "src").mkdir()
            idx = synthetic_index()
            idx.to_csv(root / "charts/index.csv", index=False)
            for file in idx.file:
                Image.new("RGB", (32, 24), "white").save(root / file)
            refs = idx[["file", "ticker", "end_date", "label"]].copy()
            refs[["p_up", "p_flat", "p_down"]] = [.4, .3, .3]
            reference = root / "reference.csv"
            refs.to_csv(reference, index=False)
            first, second = Path(directory) / "first", Path(directory) / "second"
            prepare(root, first, reference)
            Image.new("RGB", (32, 24), "red").save(root / idx.file.iloc[0])
            prepare(root, second, reference)
            verify_bundle(second)  # Internally valid, but not this checkpoint's inputs.
            checkpoint = Path(directory) / "checkpoint"
            checkpoint.mkdir()
            adapter = checkpoint / "adapter_model.safetensors"
            adapter.write_bytes(b"CPU fixture, never loaded")
            provenance = {"bundle_manifest_sha256": {"manifest.json": sha256(first / "manifest.json")},
                          "base": {"task": "historical", "task_settings": TASK_SETTINGS},
                          "adapter_sha256": {adapter.name: sha256(adapter)}, "source_sha256": source_hashes(ROOT)}
            manifest_path = checkpoint / "provenance.json"
            manifest_path.write_text(json.dumps(provenance))
            validated, _ = jev_training.validate_reload(checkpoint, first)
            self.assertEqual(len(validated), len(idx))
            original_import = builtins.__import__

            def guard(name, *args, **kwargs):
                if name.split(".")[0] in {"torch", "peft", "transformers", "safetensors"}:
                    raise AssertionError("GPU import occurred before provenance validation")
                return original_import(name, *args, **kwargs)

            with patch.object(jev_training, "load_system1") as loader, patch("builtins.__import__", guard):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    jev_training.load_trained(root / "missing_model", second, checkpoint)
                provenance["base"]["task"] = "ternary-volatility"
                manifest_path.write_text(json.dumps(provenance))
                with self.assertRaisesRegex(ValueError, "task/prompt settings"):
                    jev_training.load_trained(root / "missing_model", first, checkpoint)
                provenance["base"]["task"] = "historical"
                provenance["source_sha256"]["src/jev_training.py"] = "0" * 64
                manifest_path.write_text(json.dumps(provenance))
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    jev_training.load_trained(root / "missing_model", first, checkpoint)
                loader.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional for preparation tests")
class CNNCheckpointTests(unittest.TestCase):
    def test_cpu_checkpoint_reload_preserves_predictions_and_bn_state(self):
        import torch
        from cnn import Net, load_checkpoint, predict, save_checkpoint
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = Net().to("cpu")
        images = np.random.default_rng(0).uniform(size=(2, 3, 120, 160)).astype("float32")
        before = predict(model, images, "cpu")
        bn = model.f[0][1].running_mean.clone()
        predict(model, images, "cpu")
        torch.testing.assert_close(bn, model.f[0][1].running_mean, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            checksum = save_checkpoint(model, path, {"fixture": "random CPU weights; no training"})
            restored = load_checkpoint(path, "cpu", checksum)
            np.testing.assert_array_equal(before, predict(restored, images, "cpu"))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_checkpoint(path, "cpu", "0" * 64)


if __name__ == "__main__":
    unittest.main()
