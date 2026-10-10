"""Two frozen CPU shortcut guards, independent parents and visible-feature fixtures."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp06b_leakage_audit", ROOT / "06b-chart-patterns/src/leakage_audit.py")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


class LeakageUnitTests(unittest.TestCase):
    def test_parent_and_seed_split_is_disjoint_and_stratified(self):
        labels = [label for label in audit.CLASSES for _ in range(10)]
        rows = pd.DataFrame({"template_id": [f"parent_{i}" for i in range(70)], "seed": np.arange(70), "label": labels})
        train, test = audit.split_parents(rows)
        self.assertFalse(set(rows.iloc[train].template_id) & set(rows.iloc[test].template_id))
        self.assertFalse(set(rows.iloc[train].seed) & set(rows.iloc[test].seed))
        self.assertEqual(rows.iloc[test].label.value_counts().to_dict(), dict.fromkeys(audit.CLASSES, 4))
        repeated = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique parent"):
            audit.split_parents(repeated)

    def test_features_ignore_label_seed_and_future_metadata(self):
        close = np.array([100., 101., 100.5, 102., 101., 103., 102., 104.])
        opening = np.r_[close[0], close[:-1]]
        bars = pd.DataFrame({"Open": opening, "High": np.maximum(opening, close) + .1,
                             "Low": np.minimum(opening, close) - .1, "Close": close,
                             "Volume": np.full(len(close), 1e6)})
        original = audit.numeric_features(bars)
        bars["label"], bars["seed"], bars["fwd_ret"] = "None", 99999, -99999.
        np.testing.assert_array_equal(original, audit.numeric_features(bars))
        self.assertEqual(len(original), len(audit.FEATURES))
        self.assertTrue(np.isfinite(original).all())

    def test_balanced_gate_catches_easy_negative_shortcut(self):
        labels = [label for label in audit.CLASSES for _ in range(28)]
        n = len(labels)
        rows = pd.DataFrame({"template_id": [f"parent_{i}" for i in range(n)], "seed": np.arange(n), "label": labels})
        features = np.zeros((n, len(audit.FEATURES)))
        features[:, audit.FEATURES.index("path_efficiency")] = np.where(rows.label.eq("None"), 1., .25)
        result = audit.audit_features(features, rows)
        self.assertFalse(result["all_gates_passed"])
        self.assertEqual(result["primary_logistic"]["balanced_accuracy"], 1.)
        self.assertEqual(result["forest_diagnostic"]["balanced_accuracy"], 1.)
        self.assertGreater(result["all_pattern_dummy_accuracy"], .8)
        self.assertEqual(result["all_pattern_dummy_balanced_accuracy"], .5)

    def test_frozen_models_features_and_gate_are_explicit(self):
        self.assertEqual(audit.PROTOCOL["audit_templates"], 1400)
        self.assertEqual(audit.PROTOCOL["generator_seed"], 20261017)
        self.assertEqual(audit.PROTOCOL["split_seed"], 1701)
        self.assertEqual(audit.PROTOCOL["balanced_accuracy_limit"], .90)
        self.assertEqual(audit.PROTOCOL["primary_model"]["C"], 1.)
        self.assertEqual(audit.PROTOCOL["forest_diagnostic"]["max_depth"], 4)
        self.assertNotIn("seed", audit.FEATURES)
        self.assertNotIn("label", audit.FEATURES)
        self.assertNotIn("template_id", audit.FEATURES)


class RevisedGeneratorLeakageRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = audit.run_current_audits()

    def test_noise_zero_none_cannot_be_near_perfectly_separated_by_fixed_logistic(self):
        self.assertTrue(self.result["primary_gate_passed"], self.result["primary_logistic"])

    def test_noise_zero_none_cannot_be_near_perfectly_separated_by_fixed_depth4_forest(self):
        self.assertTrue(self.result["forest_gate_passed"], self.result["forest_diagnostic"])

    def test_predeclared_fresh_confirmation_passes_both_unchanged_gates(self):
        result = self.result["fresh_confirmation"]
        self.assertTrue(result["all_gates_passed"], result)
        self.assertEqual(result["frozen_audit_protocol"]["generator_seed"], 20261023)
        self.assertEqual(result["frozen_audit_protocol"]["split_seed"], 2301)
        self.assertEqual(result["frozen_audit_protocol"]["primary_model"], audit.PROTOCOL["primary_model"])
        self.assertEqual(result["frozen_audit_protocol"]["forest_diagnostic"], audit.PROTOCOL["forest_diagnostic"])
        self.assertEqual(self.result["confirmation_regression_seed_overlap"], 0)
        self.assertEqual(result["evaluation_seed_overlap"], 0)
        self.assertEqual(result["train_counts"], {"None": 120, "patterns": 720})
        self.assertEqual(result["heldout_counts"], {"None": 80, "patterns": 480})

    def test_parent_identity_and_binary_denominators_are_fixed(self):
        self.assertEqual(self.result["train_counts"], {"None": 120, "patterns": 720})
        self.assertEqual(self.result["heldout_counts"], {"None": 80, "patterns": 480})
        self.assertEqual(self.result["parent_template_overlap"], 0)
        self.assertEqual(self.result["parent_seed_overlap"], 0)
        self.assertEqual(self.result["evaluation_seed_overlap"], 0)
        self.assertEqual(self.result["evaluation_parents"], 200)
        self.assertEqual(self.result["audit_parent_namespace"], "audit_seed_20261017")
        self.assertNotEqual(self.result["train_parent_identity_sha256"], self.result["heldout_parent_identity_sha256"])


if __name__ == "__main__":
    unittest.main()
