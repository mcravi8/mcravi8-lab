"""Golden geometry, causality, detector independence and date split tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("candles_detect", EXP / "06a-candles/src/detect.py")
detect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detect)


def fixture(label: str) -> pd.DataFrame:
    n = 30
    uptrend = label in {"Shooting Star", "Evening Star"}
    close = 100 + np.arange(n) * (0.5 if uptrend else -0.5)
    open_ = close - 1 if uptrend else close + 1
    frame = pd.DataFrame({"Open":open_, "High":np.maximum(open_,close)+.5,
                          "Low":np.minimum(open_,close)-.5,"Close":close,"Volume":1000},
                         index=pd.bdate_range("2017-01-02",periods=n))
    def bar(position, o, h, l, c):
        frame.iloc[position, :4] = [o,h,l,c]
    if label == "Doji":
        bar(-1, 85,86,84,85.05)
    elif label == "Hammer":
        o = frame.Low.iloc[-2]-.1
        bar(-1,o,o+.40,o-1,o+.35)
    elif label == "Shooting Star":
        o = frame.Close.iloc[-2]+1
        bar(-1,o,o+1.35,o-.05,o+.35)
    elif label == "Bullish Engulfing":
        bar(-2,100,100.5,98.5,99)
        bar(-1,98.8,100.7,98.3,100.2)
    elif label == "Bearish Engulfing":
        bar(-2,99,100.5,98.5,100)
        bar(-1,100.2,100.7,98.3,98.8)
    elif label == "Morning Star":
        bar(-3,100,100.5,97.5,98)
        bar(-2,97,97.6,96.7,97.3)
        bar(-1,97.2,99.5,96.9,99.2)
    elif label == "Evening Star":
        bar(-3,100,102.5,99.5,102)
        bar(-2,103,103.6,102.7,103.3)
        bar(-1,103.2,103.5,100.9,101.2)
    elif label != "None":
        raise ValueError(label)
    return frame


class CandleTests(unittest.TestCase):
    def test_all_eight_golden_labels(self):
        for label in detect.CLASSES:
            with self.subTest(label=label):
                frame = fixture(label)
                expected = [] if label == "None" else [label]
                for fn in [detect.talib_detector,detect.pandas_detector]:
                    flags = fn(frame).iloc[-1]
                    self.assertEqual(list(flags.index[flags]),expected)

    def test_trend_is_strict_and_before_pattern(self):
        for label,span in detect.RULES["prior_trend"]["span"].items():
            with self.subTest(label=label):
                frame = fixture(label)
                # Make preceding five-session close return exactly zero. OHLC shape stays intact.
                endpoint = len(frame)-span-1
                earlier = endpoint-5
                delta = frame.Close.iloc[endpoint]-frame.Close.iloc[earlier]
                frame.iloc[earlier,:4] += delta
                for fn in [detect.talib_detector,detect.pandas_detector]:
                    self.assertTrue(fn(frame,trend=False)[label].iloc[-1])
                    self.assertFalse(fn(frame,trend=True)[label].iloc[-1])

    def test_reject_missing_gap_and_insufficient_star_penetration(self):
        star = fixture("Shooting Star")
        # Slide entire final candle down until its body touches the prior body.
        star.iloc[-1,:4] -= star.Open.iloc[-1]-star.Close.iloc[-2]
        morning = fixture("Morning Star")
        morning.iloc[-1,:4] = [95.5,98.6,95,98.5]  # long green, but below 98+.3*2.
        for fn in [detect.talib_detector,detect.pandas_detector]:
            self.assertFalse(fn(star)["Shooting Star"].iloc[-1])
            self.assertFalse(fn(morning)["Morning Star"].iloc[-1])

    def test_engulf_accepts_one_equal_endpoint_not_both(self):
        frame = fixture("Bullish Engulfing")
        frame.loc[frame.index[-1],"Open"] = 99
        for fn in [detect.talib_detector,detect.pandas_detector]:
            self.assertTrue(fn(frame)["Bullish Engulfing"].iloc[-1])
        frame.loc[frame.index[-1],"Close"] = 100
        for fn in [detect.talib_detector,detect.pandas_detector]:
            self.assertFalse(fn(frame)["Bullish Engulfing"].iloc[-1])

    def test_overlapping_patterns_are_excluded_not_none(self):
        frame = fixture("Hammer")
        frame.loc[frame.index[-1],"Close"] = frame.Open.iloc[-1]+.1
        frame.loc[frame.index[-1],"High"] = frame.Close.iloc[-1]+.05
        flags=detect.pandas_detector(frame).iloc[-1]
        self.assertTrue(flags.Doji and flags.Hammer)
        rows,_,_,excluded = detect.candidates(frame,"XYZ")
        self.assertFalse((rows.end_date == str(frame.index[-1].date())).any())
        self.assertGreaterEqual(excluded["multiple_labels"],1)

    def test_independent_detector_randomized_parity_and_no_talib_calls(self):
        rng=np.random.default_rng(6)
        n=5000
        close=100+np.cumsum(rng.normal(0,.4,n))
        open_=close+rng.normal(0,.7,n)
        frame=pd.DataFrame({"Open":open_,"High":np.maximum(open_,close)+rng.uniform(0,1,n),
                            "Low":np.minimum(open_,close)-rng.uniform(0,1,n),"Close":close,"Volume":1000},
                            index=pd.bdate_range("2005-01-03",periods=n))
        ta=detect.talib_detector(frame)
        with patch.object(detect.talib,"CDLDOJI",side_effect=AssertionError("independence")), \
             patch.object(detect.talib,"CDLENGULFING",side_effect=AssertionError("independence")):
            independent=detect.pandas_detector(frame)
        pd.testing.assert_frame_equal(ta,independent)

    def test_future_ohlcv_cannot_change_past_labels(self):
        frame=fixture("Morning Star")
        tail=fixture("Evening Star")
        tail.index=pd.bdate_range(frame.index[-1]+pd.offsets.BDay(1),periods=len(tail))
        joined=pd.concat([frame,tail])
        for fn in [detect.talib_detector,detect.pandas_detector]:
            before=fn(joined).iloc[:len(frame)]
            joined.iloc[len(frame):,:4] *= 10
            pd.testing.assert_frame_equal(before,fn(joined).iloc[:len(frame)])

    def test_split_gap_and_complete_window_boundaries(self):
        d=pd.Timestamp
        self.assertEqual(detect.split_for_window(d("2018-12-01"),d("2018-12-31")),"train")
        self.assertIsNone(detect.split_for_window(d("2018-12-31"),d("2019-01-02")))
        self.assertIsNone(detect.split_for_window(d("2019-03-29"),d("2019-04-29")))
        self.assertEqual(detect.split_for_window(d("2019-04-01"),d("2019-04-30")),"test")
        with self.assertRaises(ValueError):
            detect.split_for_window(d("2019-05-01"),d("2019-04-30"))

    def test_sampler_is_order_invariant_and_balances_ticker_year_cells(self):
        rows=[]
        for ticker in ["A","B","C"]:
            for year in [2010,2011]:
                for i in range(10):
                    rows.append({"id":f"{ticker}{year}{i}","ticker":ticker,"year":year,"end_date":f"{year}-01-{i+1:02d}"})
        frame=pd.DataFrame(rows)
        a=detect.stratified_sample(frame,13)
        b=detect.stratified_sample(frame.sample(frac=1,random_state=3),13)
        self.assertEqual(list(a.id),list(b.id))
        sizes=a.groupby(["ticker","year"]).size()
        self.assertLessEqual(sizes.max()-sizes.min(),1)
        self.assertEqual(len(a.id.unique()),13)

    def test_agreement_rates_use_positive_denominator(self):
        a=pd.DataFrame(False,index=range(100),columns=detect.PATTERNS)
        b=a.copy()
        a.loc[0,"Hammer"]=True
        b.loc[1,"Hammer"]=True
        stats=detect.agreement_table(a,b)["Hammer"]
        self.assertEqual(stats["binary_agreement"],.98)
        self.assertEqual(stats["positive_jaccard"],0)
        self.assertEqual(stats["either_positive"],2)


class ClarityTests(unittest.TestCase):
    def test_well_inside_rules_is_clear_for_every_class(self):
        for label in detect.CLASSES:
            with self.subTest(label=label):
                frame=fixture(label)
                if label=="Evening Star":
                    frame.loc[frame.index[-1],["Low","Close"]]=[100.4,100.7]
                clear,diagnostics=detect.clarity(frame.tail(20),label)
                self.assertTrue(clear,diagnostics)

    def test_near_original_boundaries_are_not_clear(self):
        for label in detect.PATTERNS:
            frame=fixture(label)
            final=frame.index[-1]
            if label=="Doji":
                frame.loc[final,"Close"]=85.15
            elif label=="Hammer":
                frame.loc[final,"Low"]=frame.Open.iloc[-1]-.36
            elif label=="Shooting Star":
                frame.loc[final,"High"]=frame.Close.iloc[-1]+.36
            elif label=="Bullish Engulfing":
                frame.loc[final,"Open"]=98.95
            elif label=="Bearish Engulfing":
                frame.loc[final,"Close"]=98.95
            elif label=="Morning Star":
                frame.loc[final,"Close"]=98.7
            # Original Evening Star fixture penetrates40%, above the30% label
            # threshold but below the frozen50% clarity threshold.
            with self.subTest(label=label):
                flags=detect.pandas_detector(frame).iloc[-1]
                self.assertEqual(list(flags.index[flags]),[label])
                clear,diagnostics=detect.clarity(frame.tail(20),label)
                self.assertFalse(clear,diagnostics)
                self.assertTrue(any(not passed for passed in diagnostics["criteria"].values()))

    def test_none_rejects_hammer_like_shape_failing_only_placement(self):
        frame=fixture("Hammer")
        final=frame.index[-1]
        o=frame.Low.iloc[-2]+1
        frame.loc[final,["Open","High","Low","Close"]]=[o,o+.4,o-1,o+.35]
        self.assertFalse(detect.pandas_detector(frame).iloc[-1].any())
        clear,diagnostics=detect.clarity(frame.tail(20),"None")
        self.assertFalse(clear)
        self.assertIn("Hammer",diagnostics["near_patterns"])

    def test_clarity_is_causal_and_preserves_original_index_fields(self):
        frame=fixture("Hammer")
        end=frame.index[-1]
        before=pd.DataFrame([{"id":"fixture","ticker":"XYZ","file":"charts/fixture.png",
                              "label":"Hammer","split":"train","bar_index":len(frame)-1,
                              "window_start":str(frame.index[-20].date()),"end_date":str(end.date())}])
        a,diagnostics,_=detect.annotate_clarity(before,{"XYZ":frame})
        future=fixture("None")
        future.index=pd.bdate_range(end+pd.offsets.BDay(1),periods=len(future))
        future.iloc[:,:4]*=100
        b,new_diagnostics,_=detect.annotate_clarity(before,{"XYZ":pd.concat([frame,future])})
        pd.testing.assert_frame_equal(a,b)
        pd.testing.assert_frame_equal(a[before.columns],before)
        self.assertEqual(diagnostics,new_diagnostics)
        self.assertEqual(a.label.tolist(),["Hammer"])
        self.assertEqual(a.split.tolist(),["train"])

    def test_wrong_label_and_hidden_history_are_rejected(self):
        frame=fixture("Hammer")
        with self.assertRaisesRegex(ValueError,"twenty"):
            detect.clarity(frame,"Hammer")
        with self.assertRaisesRegex(ValueError,"original single-label"):
            detect.clarity(frame.tail(20),"None")


if __name__ == "__main__":
    unittest.main()
