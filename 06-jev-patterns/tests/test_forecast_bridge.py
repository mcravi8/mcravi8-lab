"""Leakage, target semantics, provenance and paired forecast tests; synthetic data only."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "shared/src"))
import forecast_bridge as bridge

PROTOCOL = json.loads((EXP / "config/forecast_bridge_v1.json").read_text())
PATTERNS = json.loads((EXP / "06a-candles/options.json").read_text())


def bars(n=110):
    rng = np.random.default_rng(22)
    close = 100*np.exp(np.cumsum(rng.normal(0, .015, n)))
    return pd.DataFrame({"Open": close*.999, "High": close*1.01, "Low": close*.99,
                         "Close": close, "Volume": rng.integers(100, 1000, len(close))},
                        index=pd.bdate_range("2018-01-01", periods=len(close), name="Date"))


def learning_rows(n, test=False):
    rng = np.random.default_rng(23 if test else 17)
    names = bridge.feature_sets(PATTERNS)["B_detector"] + [f"jev_{i}" for i in range(8)]
    f = pd.DataFrame(rng.normal(size=(n, len(names))), columns=names)
    dates = pd.bdate_range("2019-04-01" if test else "2017-01-01", periods=n)
    f["file"] = [f"{'test' if test else 'train'}_{i}" for i in range(n)]
    f["end_date"] = dates.strftime("%Y-%m-%d")
    f["target_date"] = (dates+pd.offsets.BDay(5)).strftime("%Y-%m-%d")
    f["information_start"] = (dates-pd.offsets.BDay(64)).strftime("%Y-%m-%d")
    f["label"] = np.resize(["Up", "Flat", "Down"], n)
    f["split"] = "test" if test else "train"
    return f


class ForecastBridgeTests(unittest.TestCase):
    def test_target_is_five_sessions_and_sigma_matches_trailing_rule(self):
        frame = bars()
        result, reason = bridge.target_at(frame, 80, PROTOCOL)
        expected = frame.Close.pct_change(5, fill_method=None).rolling(60, min_periods=60).std(ddof=1).iloc[80]
        self.assertIsNone(reason)
        self.assertAlmostEqual(result["sigma"], expected, places=14)
        self.assertEqual(result["target_date"], str(frame.index[85].date()))
        self.assertAlmostEqual(result["forward_return"], frame.Close.iloc[85]/frame.Close.iloc[80]-1)
        self.assertEqual(result["information_start"], str(frame.index[16].date()))

    def test_future_changes_cannot_change_features_scale_or_detector_label(self):
        frame = bars()
        detector = bridge.detector_module()
        flags = detector.talib_detector(frame)
        position = next(i for i in range(64, 100) if flags.iloc[i].sum() <= 1)
        labels = flags.columns[flags.iloc[position]].tolist()
        label = labels[0] if labels else "None"
        idx = pd.DataFrame([{"file": "chart.png", "ticker": "AAA", "window_start": str(frame.index[position-19].date()),
                             "end_date": str(frame.index[position].date()), "split": "train", "label": label, "n_bars": 20}])
        p = np.full((1, 8), 1/8)
        before, _ = bridge.build_cohort(idx, p, PATTERNS, {"AAA": frame}, PROTOCOL, verify_visible_files=False)
        changed = frame.copy()
        changed.iloc[position+1:, :4] *= 1.5
        after, _ = bridge.build_cohort(idx, p, PATTERNS, {"AAA": changed}, PROTOCOL, verify_visible_files=False)
        columns = bridge.feature_sets(PATTERNS)["C_jev"] + [f"detector_{j}" for j in range(8)]
        np.testing.assert_array_equal(before[columns].to_numpy(), after[columns].to_numpy())
        self.assertEqual(before.sigma.iloc[0], after.sigma.iloc[0])
        self.assertEqual(before.pattern_label.iloc[0], after.pattern_label.iloc[0])
        self.assertNotEqual(before.forward_return.iloc[0], after.forward_return.iloc[0])

    def test_history_and_missing_forward_exclusions(self):
        frame = bars()
        self.assertEqual(bridge.target_at(frame, 63, PROTOCOL)[1], "insufficient_trailing_history")
        self.assertIsNone(bridge.target_at(frame, 64, PROTOCOL)[1])
        self.assertEqual(bridge.target_at(frame, len(frame)-5, PROTOCOL)[1], "missing_forward_target")
        flat = frame.copy(); flat.loc[:, "Close"] = 100
        self.assertEqual(bridge.target_at(flat, 80, PROTOCOL)[1], "nonpositive_trailing_sigma")

    def test_training_target_crossing_2018_boundary_is_excluded(self):
        frame = bars(150)
        frame.index = pd.bdate_range("2018-08-01", periods=len(frame), name="Date")
        end = pd.Timestamp("2018-12-31")
        position = int(frame.index.get_loc(end))
        flags = bridge.detector_module().talib_detector(frame)
        labels = flags.columns[flags.iloc[position]].tolist()
        self.assertLessEqual(len(labels), 1)
        idx = pd.DataFrame([{"file": "late.png", "ticker": "AAA", "window_start": str(frame.index[position-19].date()),
                             "end_date": str(end.date()), "split": "train", "label": labels[0] if labels else "None", "n_bars": 20}])
        earlier = next(i for i in range(64, 100) if flags.iloc[i].sum() <= 1)
        early_labels = flags.columns[flags.iloc[earlier]].tolist()
        early = {"file":"early.png","ticker":"AAA","window_start":str(frame.index[earlier-19].date()),
                 "end_date":str(frame.index[earlier].date()),"split":"train",
                 "label":early_labels[0] if early_labels else "None","n_bars":20}
        idx = pd.concat([idx,pd.DataFrame([early])],ignore_index=True)
        retained, excluded = bridge.build_cohort(idx,np.full((2,8),1/8),PATTERNS,{"AAA":frame},PROTOCOL,verify_visible_files=False)
        self.assertEqual(retained.file.tolist(),["early.png"])
        self.assertEqual(excluded[0]["reason"],"train_target_after_boundary")

    def test_scaling_checkpoint_and_forecasts_do_not_use_test_labels(self):
        train, test = learning_rows(120), learning_rows(15, test=True)
        with tempfile.TemporaryDirectory() as folder:
            before, checks = bridge.forecast_fold(train, test, PATTERNS, PROTOCOL, folder)
            changed = test.copy()
            changed["label"] = "Down"
            after, _ = bridge.forecast_fold(train, changed, PATTERNS, PROTOCOL)
            for arm in bridge.ARMS:
                np.testing.assert_array_equal(before[arm], after[arm])
                np.testing.assert_allclose(before[arm].sum(1), 1., atol=1e-12)
            self.assertTrue(all(check["reload_max_abs_diff"] == 0 for check in checks.values()))
            self.assertEqual(checks["A_numeric"]["feature_count"], 168)
            self.assertEqual(checks["C_jev"]["feature_count"], 176)

    def test_purge_rejects_equal_information_boundary_and_shared_files(self):
        train, test = learning_rows(120), learning_rows(15, test=True)
        bridge.check_partition(train, test, PROTOCOL)
        changed = train.copy(); changed.loc[0, "target_date"] = test.information_start.min()
        with self.assertRaisesRegex(ValueError, "overlap"):
            bridge.check_partition(changed, test, PROTOCOL)
        changed = train.copy(); changed.loc[0, "file"] = test.file.iloc[0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            bridge.check_partition(changed, test, PROTOCOL)

    def test_frozen_input_hash_change_is_rejected_before_loading_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"input.csv"; path.write_text("original")
            protocol = {"input_sha256": {"input.csv": bridge.digest(path)}}
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                bridge.verify_protocol(protocol, folder)

    def test_arm_features_exclude_future_and_identity_metadata(self):
        forbidden = {"label", "pattern_label", "forward_return", "target_date", "sigma", "end_date", "file", "ticker", "split", "clear"}
        for columns in bridge.feature_sets(PATTERNS).values():
            self.assertFalse(forbidden.intersection(columns))
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            bridge.visible_features(bars().tail(19))

    def test_alignment_rejects_partial_identity_and_fake_results(self):
        f = pd.DataFrame([{"file":"a","ticker":"AAA","window_start":"2017-01-01","end_date":"2017-02-01",
                           "label":"None","split":"train","bars_file":"bars.csv","n_bars":20}])
        pred = f.copy()
        pred["probabilities"] = json.dumps([1/8]*8)
        pred["data_origin"], pred["source_collection"], pred["evidence_role"], pred["accuracy_valid"] = "market","candles","benchmark",True
        meta = {"synthetic":False,"prediction_origin":"jev"}
        bridge.align_inputs(f, pred, PATTERNS, meta)
        bad = pred.copy(); bad.loc[0,"end_date"]="2020-01-01"
        with self.assertRaisesRegex(ValueError,"identity differs"):
            bridge.align_inputs(f,bad,PATTERNS,meta)
        with self.assertRaisesRegex(ValueError,"identical unique file coverage"):
            bridge.align_inputs(f,pred.iloc[:0],PATTERNS,meta)
        with self.assertRaisesRegex(ValueError,"real JEV"):
            bridge.align_inputs(f,pred,PATTERNS,{"synthetic":True,"prediction_origin":"jev"})

    def test_paired_brier_direction_and_constant_interval(self):
        order = PROTOCOL["target"]["options"]
        labels = np.array(order*2)
        perfect = np.eye(3)[np.arange(6)%3]
        uniform = np.full((6,3),1/3)
        result = bridge.paired_difference(perfect,uniform,labels,["2020-01-03"]*3+["2020-01-10"]*3,order,PROTOCOL)
        self.assertAlmostEqual(result["brier_improvement"],2/3)
        np.testing.assert_allclose(result["ci95"]["brier"]["4"],[2/3,2/3])

    def test_entrypoint_does_not_import_gpu_or_network_clients(self):
        script = (EXP/"shared/src/forecast_bridge.py").read_text()
        for text in ["import torch", "import requests", "import httpx", "download_ticker(", "run_gpu"]:
            self.assertNotIn(text,script)


if __name__ == "__main__":
    unittest.main()
