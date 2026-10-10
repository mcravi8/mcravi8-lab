"""Hand-computed multiclass metrics, clustered samples and paired identity guards."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared/src"))
import evaluate

ORDER = ["A", "B", "None"]
P = np.array([[.8, .1, .1], [.1, .8, .1], [.4, .35, .25], [.05, .05, .9]])
LABELS = ["A", "A", "None", "None"]


def frame(probabilities=P, labels=LABELS):
    n = len(labels)
    return pd.DataFrame({"file": [f"chart_{i:03d}.png" for i in range(n)], "ticker": ["AAA"] * n,
                         "end_date": ["2020-01-02" if i < n // 2 else "2020-04-02" for i in range(n)],
                         "label": labels, "probabilities": [json.dumps(row.tolist()) for row in probabilities],
                         "prediction": np.asarray(ORDER)[np.asarray(probabilities).argmax(1)], "latency_ms": [10.] * n})


class EvaluationTests(unittest.TestCase):
    def test_clear_subset_uses_only_explicit_flags_and_retains_class_order(self):
        rows=frame();rows['clear']=[True,False,False,True]
        subset=evaluate.select_clear_subset(rows)
        self.assertEqual(subset.file.tolist(),[rows.file.iloc[0],rows.file.iloc[3]])
        result=evaluate.evaluate_frame(subset,ORDER,bootstrap_resamples=5)
        self.assertEqual(result['n'],2);self.assertEqual(result['accuracy'],1.)
        self.assertEqual(result['options'],ORDER)
        with self.assertRaisesRegex(ValueError,'explicit clear'):
            evaluate.select_clear_subset(frame())
        for flags in [['False']*4,['True','','False','False'],[1,0,0,1]]:
            with self.assertRaises(ValueError):
                evaluate.select_clear_subset(rows.assign(clear=flags))
        altered=rows.copy();altered['clear']=[False,True,False,True]
        with self.assertRaisesRegex(ValueError,'coverage'):
            evaluate.compare_arms(subset,evaluate.select_clear_subset(altered),ORDER,bootstrap_resamples=5)

    def test_hand_computed_metrics(self):
        m = evaluate.evaluate_arrays(P, LABELS, ORDER)
        self.assertEqual(m["confusion_matrix"], [[1, 1, 0], [0, 0, 0], [1, 0, 1]])
        self.assertEqual(m["accuracy"], .5)
        self.assertAlmostEqual(m["macro_f1"], 7 / 18)
        self.assertAlmostEqual(m["brier"], .595)
        self.assertAlmostEqual(m["ece_15"], .275)
        self.assertEqual(m["risk_coverage"]["0.5"]["coverage"], .75)
        self.assertAlmostEqual(m["risk_coverage"]["0.7"]["risk"], 1 / 3)
        self.assertEqual(m["risk_coverage"]["0.9"]["risk"], 0.)
        self.assertEqual(sum(b["n"] for b in m["reliability"]), 4)

    def test_empty_coverage_and_confidence_one(self):
        m = evaluate.evaluate_arrays([[1 / 3] * 3], ["None"], ORDER)
        for value in m["risk_coverage"].values():
            self.assertEqual(value["n"], 0)
            self.assertIsNone(value["risk"])
            self.assertIsNone(value["accuracy"])
        perfect = evaluate.evaluate_arrays([[0., 0., 1.]], ["None"], ORDER)
        self.assertEqual(perfect["reliability"][-1]["n"], 1)
        self.assertEqual(perfect["ece_15"], 0.)

    def test_groups_retain_every_ticker_quarter_row(self):
        rows = frame()
        groups, _ = evaluate.bootstrap_groups(rows)
        self.assertEqual([g.tolist() for g in groups], [[0, 1], [2, 3]])
        for sample in evaluate.bootstrap_indices(groups, resamples=30):
            counts = np.bincount(sample, minlength=4)
            self.assertEqual(counts[0], counts[1])
            self.assertEqual(counts[2], counts[3])
        first = evaluate.evaluate_frame(rows, ORDER, bootstrap_resamples=50)
        second = evaluate.evaluate_frame(rows.sample(frac=1, random_state=7), ORDER, bootstrap_resamples=50)
        self.assertEqual(first["ci95"], second["ci95"])

    def test_synthetic_templates_and_explicit_no_ci(self):
        rows = frame(); rows["end_date"] = ""; rows["ticker"] = "SYNTH"
        absent = evaluate.evaluate_frame(rows, ORDER, synthetic=True, bootstrap_resamples=10)
        self.assertTrue(all(value is None for value in absent["ci95"].values()))
        self.assertIn("template_id", absent["inference"]["no_ci_reason"])
        rows["template_id"] = ["template_a", "template_a", "template_b", "template_b"]
        supplied = evaluate.evaluate_frame(rows, ORDER, synthetic=True, bootstrap_resamples=10)
        self.assertEqual(supplied["inference"]["blocks"], 2)
        self.assertIsNotNone(supplied["ci95"]["accuracy"])
        with self.assertRaises(ValueError):
            evaluate.evaluate_frame(rows, ORDER, synthetic=False, bootstrap_resamples=10)

    def test_paired_alignment_and_sparse_exact_test(self):
        n = 10
        right = frame(np.tile([1., 0., 0.], (n, 1)), ["A"] * n)
        left = right.copy()
        left.loc[:4, "probabilities"] = json.dumps([0., 1., 0.])
        left.loc[:4, "prediction"] = "B"
        comparison = evaluate.compare_arms(left, right.sample(frac=1, random_state=8), ORDER, bootstrap_resamples=25)
        self.assertEqual(comparison["mcnemar"]["left_only_correct"], 0)
        self.assertEqual(comparison["mcnemar"]["right_only_correct"], 5)
        self.assertEqual(comparison["mcnemar"]["p_two_sided_exact"], .0625)
        self.assertEqual(comparison["differences"]["accuracy"], -.5)
        changed = right.copy(); changed.loc[0, "label"] = "B"
        with self.assertRaisesRegex(ValueError, "truth differ"):
            evaluate.compare_arms(left, changed, ORDER, bootstrap_resamples=5)
        with self.assertRaisesRegex(ValueError, "coverage"):
            evaluate.compare_arms(left, right.iloc[:-1], ORDER, bootstrap_resamples=5)
        with self.assertRaises(ValueError):
            evaluate.compare_arms(pd.concat([left, left.iloc[:1]]), right, ORDER, bootstrap_resamples=5)

    def test_fake_prediction_flag_does_not_change_dataset_grouping(self):
        self.assertFalse(evaluate.infer_synthetic_dataset(frame(), {"synthetic": True, "prediction_origin": "fake_server"}))
        synthetic = frame(); synthetic["end_date"] = ""
        self.assertTrue(evaluate.infer_synthetic_dataset(synthetic, {"synthetic": False, "prediction_origin": "jev"}))
        self.assertTrue(evaluate.infer_synthetic_dataset(frame(), {"synthetic_dataset": True}))

    def test_literal_none_label_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preds.csv"
            frame().to_csv(path, index=False)
            path.with_suffix(".meta.json").write_text(json.dumps({"options": ORDER}))
            loaded, options, _ = evaluate.read_predictions(path)
            self.assertEqual(loaded.label.iloc[-1], "None")
            self.assertEqual(options, ORDER)

    def test_reject_invalid_probabilities_options_and_prediction(self):
        for p in ([[.2, .3, .2]], [[np.nan, .5, .5]], [[-.1, .5, .6]]):
            with self.assertRaises(ValueError):
                evaluate.evaluate_arrays(p, ["A"], ORDER)
        with self.assertRaises(ValueError):
            evaluate.evaluate_arrays([[.5, .5]], ["A"], ["A", "A"])
        wrong = frame(); wrong.loc[0, "prediction"] = "B"
        with self.assertRaisesRegex(ValueError, "argmax"):
            evaluate.validate_frame(wrong, ORDER)


if __name__ == "__main__":
    unittest.main()
