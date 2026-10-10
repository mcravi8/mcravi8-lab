"""Causality, chronological folds, negative controls, and frozen-model recovery."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import SHAPE, CONTEXT, labels, weekly_features
from baselines import folds, shuffled_labels, fit_logistic, export_model, predict_frozen, predict_ordered


def prices(n=550):
    x = np.arange(n)
    close = 100 + .03 * x + 3 * np.sin(.17 * x)
    opening = close + .2 * np.cos(.3 * x)
    return pd.DataFrame({"Open": opening, "High": np.maximum(close, opening) + .5,
                         "Low": np.minimum(close, opening) - .5, "Close": close,
                         "Volume": 1000 + 13 * x}, index=pd.bdate_range("2020-01-06", periods=n, name="Date"))


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "protocol.json").read_text())

    def test_explicit_targets_and_degenerate_volatility(self):
        self.assertEqual(labels(.25, .75, 1 / 3), ("Up", "Up"))
        self.assertEqual(labels(-.25, .75, 1 / 3), ("Down", "Down"))
        self.assertEqual(labels(0, .75, 1 / 3), ("Flat", None))
        self.assertEqual(labels(.1, .75, 1 / 3), ("Flat", "Up"))
        with self.assertRaises(ValueError):
            labels(0, 0, 1 / 3)
        for flat_band in (0, -1 / 3, float("nan"), float("inf")):
            with self.subTest(flat_band=flat_band), self.assertRaisesRegex(ValueError, "Flat band"):
                labels(0.1, 0.75, flat_band)

    def test_future_changes_cannot_change_past_features(self):
        original = prices()
        altered = original.copy()
        altered.iloc[400:, :4] *= 1.8
        altered.iloc[400:, 4] *= 2
        a = weekly_features(original, "AAA", self.protocol)
        b = weekly_features(altered, "AAA", self.protocol)
        cutoff = str(original.index[399].date())
        columns = ["observation_id", *SHAPE, *CONTEXT]
        pd.testing.assert_frame_equal(a[a.end_date <= cutoff][columns], b[b.end_date <= cutoff][columns])
        self.assertFalse(a[a.end_date <= cutoff].fwd_ret.equals(b[b.end_date <= cutoff].fwd_ret))

    def test_short_volatility_lookback_still_waits_for_momentum_history(self):
        protocol = dict(self.protocol, volatility_lookback_weeks=4)
        original = prices(250)
        altered = original.copy()
        altered.iloc[125:, :4] *= 1.8
        before = weekly_features(original, "AAA", protocol)
        after = weekly_features(altered, "AAA", protocol)
        # Twelve-week momentum needs week k=12, regardless of sigma's lookback.
        self.assertEqual(before.end_date.min(), str(original.index[64].date()))
        cutoff = str(original.index[124].date())
        columns = ["observation_id", *SHAPE, *CONTEXT]
        pd.testing.assert_frame_equal(before[before.end_date <= cutoff][columns],
                                      after[after.end_date <= cutoff][columns])

    def test_numeric_shapes_are_price_and_volume_scale_invariant(self):
        original = prices()
        scaled = original.copy()
        scaled.iloc[:, :4] = scaled.iloc[:, :4] * 7 + 300
        scaled.Volume *= 19
        a = weekly_features(original, "AAA", self.protocol)
        b = weekly_features(scaled, "AAA", self.protocol)
        np.testing.assert_allclose(a[SHAPE], b[SHAPE], atol=1e-12)

    def test_holiday_chart_excluded_but_target_uses_last_session(self):
        frame = prices()
        target_friday = pd.Timestamp("2020-07-03")
        frame = frame.drop(target_friday)
        rows = weekly_features(frame, "AAA", self.protocol)
        self.assertNotIn("2020-07-02", rows.end_date.tolist())
        preceding = rows[rows.end_date == "2020-06-26"].iloc[0]
        self.assertEqual(preceding.target_date, "2020-07-02")
        self.assertAlmostEqual(preceding.fwd_ret, frame.loc['2020-07-02', 'Close'] / frame.loc['2020-06-26', 'Close'] - 1)

    def test_folds_purge_labels_and_do_not_admit_future_holdout(self):
        end = pd.date_range("2018-01-05", "2021-12-31", freq="W-FRI")
        frame = pd.DataFrame({"end_date": end.strftime("%Y-%m-%d"),
                              "week_start": (end - pd.Timedelta(days=4)).strftime("%Y-%m-%d"),
                              "target_date": (end + pd.Timedelta(days=7)).strftime("%Y-%m-%d")})
        p = dict(self.protocol, first_development_test_year=2020, last_development_test_year=2021, minimum_train_weeks=50)
        out = list(folds(frame, p))
        self.assertEqual(len(out), 2)
        for _, train, test in out:
            self.assertLess(train.target_date.max(), test.week_start.min())
        bad = frame.copy(); bad.loc[0, "end_date"] = self.protocol["prospective_first_chart_end"]
        with self.assertRaises(ValueError):
            list(folds(bad, p))

    def test_shuffling_preserves_complete_date_vectors(self):
        rows = []
        for date in pd.date_range("2020-01-03", periods=12, freq="W-FRI"):
            for ticker in ("AAA", "BBB"):
                rows.append({"end_date": str(date.date()), "ticker": ticker, "label": f"{date.date()}-{ticker}"})
        frame = pd.DataFrame(rows)
        out = shuffled_labels(frame, "label", 17)
        self.assertNotEqual(out.tolist(), frame.label.tolist())
        for _, ix in frame.groupby("end_date").groups.items():
            self.assertEqual(len({x.rsplit('-', 1)[0] for x in out.loc[ix]}), 1)
        self.assertEqual(sorted(out), sorted(frame.label))

    def test_saved_model_roundtrip_and_training_only_scaler(self):
        train = pd.DataFrame({"x": [-4., -3., -2., -1., 1., 2., 3., 4.],
                              "y": ["Down"] * 4 + ["Up"] * 4})
        model = fit_logistic(train, "y", ["x"], self.protocol)
        self.assertEqual(float(model.steps[0][1].mean_[0]), 0.)
        test = pd.DataFrame({"x": [-100., 0., 100.]})
        payload = json.loads(json.dumps(export_model(model, ["x"], ["Up", "Down"])))
        np.testing.assert_allclose(predict_ordered(model, test, ["x"], ["Up", "Down"]), predict_frozen(payload, test), atol=1e-12)
        self.assertEqual(float(model.steps[0][1].mean_[0]), 0.)


if __name__ == "__main__":
    unittest.main()
