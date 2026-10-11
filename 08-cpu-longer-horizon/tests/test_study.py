"""Causality, matching, purging and end-to-end CPU forecast verification."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('exp08_study',EXP/'src/study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def prices(n=1100,start='2016-01-01',seed=31):
    rng=np.random.default_rng(seed)
    c=100*np.exp(np.cumsum(rng.normal(.0003,.018,n)))
    o=c*np.exp(rng.normal(0,.005,n))
    return pd.DataFrame({'Open':o,'High':np.maximum(o,c)*1.012,
                         'Low':np.minimum(o,c)*.988,'Close':c,
                         'Volume':rng.integers(1000,100000,n)},
                        index=pd.bdate_range(start,periods=n))


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((EXP/'protocol.json').read_text())

    def test_future_mutations_leave_views_and_target_scale_unchanged(self):
        frame=prices(); modified=frame.copy()
        modified.iloc[601:,:4]*=2; modified.iloc[601:,4]*=3
        a,b=study.views_at(frame,600),study.views_at(modified,600)
        self.assertEqual(a,b)
        for h in (5,20):
            cfg={'target':dict(self.protocol['target'],horizon_sessions=h)}
            ta,_=study.target_at(frame,600,cfg);tb,_=study.target_at(modified,600,cfg)
            self.assertEqual(ta['sigma'],tb['sigma'])
            self.assertNotEqual(ta['forward_return'],tb['forward_return'])
            self.assertEqual(ta['target_date'],str(frame.index[600+h].date()))

    def test_horizon_scale_is_computed_from_past_returns_of_that_horizon(self):
        frame=prices()
        for h in (5,20):
            cfg={'target':dict(self.protocol['target'],horizon_sessions=h)}
            target,reason=study.target_at(frame,600,cfg)
            self.assertIsNone(reason)
            c=frame.Close.to_numpy()[600-60-h+1:601]
            expected=np.std(c[h:]/c[:-h]-1,ddof=1)
            self.assertAlmostEqual(target['sigma'],expected,places=14)
            self.assertAlmostEqual(target['forward_return'],frame.Close.iloc[600+h]/frame.Close.iloc[600]-1)

    def test_current_views_do_not_use_older_year_bars(self):
        frame=prices(); changed=frame.copy()
        changed.iloc[:501,:4]*=3
        current,year=study.views_at(frame,600)
        other,other_year=study.views_at(changed,600)
        self.assertEqual(current,other)
        self.assertNotEqual(year,other_year)
        self.assertFalse(any('year' in k for k in current))

    def test_weekly_partial_periods_only_use_visible_data(self):
        frame=prices(20,start='2020-01-01')
        cut=frame.loc[:'2020-01-08']
        weekly=study.weekly_bars(cut)
        self.assertEqual(weekly.Close.iloc[-1],cut.Close.iloc[-1])
        self.assertEqual(weekly.Volume.sum(),cut.Volume.sum())
        self.assertEqual(weekly.Open.iloc[0],cut.Open.iloc[0])
        self.assertEqual(weekly.High.iloc[-1],cut.loc['2020-01-06':].High.max())

    def test_feature_price_and_volume_scale_invariance(self):
        frame=prices();scaled=frame.copy()
        scaled.iloc[:,:4]*=7;scaled.Volume*=11
        a=study.views_at(frame,600);b=study.views_at(scaled,600)
        for first,second in zip(a,b):
            np.testing.assert_allclose(list(first.values()),list(second.values()),rtol=1e-10,atol=1e-10)

    def test_common_folds_purge_longest_target_and_year_history(self):
        ends=pd.date_range('2012-01-06','2021-12-31',freq='W-FRI')
        rows=pd.DataFrame({'observation_id':ends.strftime('%Y-%m-%d'),
            'end_date':ends.strftime('%Y-%m-%d'),
            'information_start':(ends-pd.Timedelta(days=365)).strftime('%Y-%m-%d'),
            'target_date_5':(ends+pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
            'target_date_20':(ends+pd.Timedelta(days=28)).strftime('%Y-%m-%d')})
        p=dict(self.protocol,test_years=[2020,2021])
        folds=list(study.folds(rows,p));self.assertEqual(len(folds),2)
        for _,train,test in folds:
            self.assertLess(train.target_date_20.max(),test.information_start.min())
            self.assertGreaterEqual(test.information_start.min(),p['test_information_start'])
            self.assertEqual(len(set(train.observation_id)&set(test.observation_id)),0)
            self.assertGreaterEqual(train.end_date.min(),f"{int(test.end_date.iloc[0][:4])-5}-01-01")

    def test_prospective_dates_are_rejected(self):
        rows=pd.DataFrame({'end_date':['2026-10-16']})
        with self.assertRaisesRegex(ValueError,'future holdout'):
            list(study.folds(rows,self.protocol))

    def test_scaler_uses_training_only_and_reload_preserves_probabilities(self):
        train=pd.DataFrame({'x':[-4,-3,-2,-1,0,1,2,3,4],
                            'label':['Down']*3+['Flat']*3+['Up']*3})
        test=pd.DataFrame({'x':[-1000,0,1000]})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.joblib'
            p,audit=study.fit_predictions(train,test,['x'],self.protocol['model'],path)
            model=study.joblib.load(path)
            self.assertEqual(model.steps[0][1].mean_[0],0.)
            self.assertEqual(audit['reload_max_abs_diff'],0.)
            np.testing.assert_allclose(p.sum(1),1.)
            self.assertEqual(p.shape,(3,3))

    def test_shuffling_preserves_date_vectors(self):
        train=pd.DataFrame([{'end_date':str(day),'ticker':ticker,'label':f'{day}:{ticker}'}
                            for day in range(12) for ticker in ['A','B']])
        out=study.shuffled_training_labels(train,'label',83)
        self.assertNotEqual(out.tolist(),train.label.tolist())
        self.assertEqual(sorted(out),sorted(train.label))
        for _,rows in train.groupby('end_date'):
            self.assertEqual(len({x.split(':')[0] for x in out.loc[rows.index]}),1)

    def test_cpu_end_to_end_fixture_and_pairing_guards(self):
        p=deepcopy(self.protocol)
        p.update(test_years=[2019],minimum_train_dates=20,test_information_start='2018-01-01')
        p['decision']['minimum_positive_years']=1;p['uncertainty']['resamples']=30
        # Real run uses verified cached files; this test supplies synthetic OHLCV.
        raw={'SPY':prices(900),'AAA':prices(900,seed=17)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);protocol=root/'protocol.json'
            study.write_json(protocol,p)
            with patch.object(study,'EXP',root),patch.object(study,'verify_protocol',return_value={}), \
                 patch.object(study,'load',side_effect=lambda t:raw[t]), \
                 patch.object(study,'universe',return_value={'tickers':list(raw)}):
                result=study.run(protocol,root/'results')
            self.assertEqual(set(result['horizons']),{'5','20'})
            self.assertGreater(result['horizons']['20']['n'],0)
            manifest=json.loads((root/'results/manifest.json').read_text())
            self.assertFalse(manifest['gpu_started'])
            self.assertTrue(all(m['reload_max_abs_diff']==0 for fold in json.loads((root/'results/folds.json').read_text()) for m in fold['models'].values()))
            predictions=pd.read_csv(root/'results/predictions.csv.gz')
            altered=predictions.drop(predictions.index[0])
            with self.assertRaisesRegex(ValueError,'identity/truth'):
                study.evaluate_predictions(altered,p)
            changed=predictions.copy();changed.loc[changed.index[0],'label']='UNKNOWN'
            with self.assertRaises(ValueError):
                study.evaluate_predictions(changed,p)

    def test_protocol_freezes_only_two_decision_tests_without_search(self):
        self.assertEqual(self.protocol['horizons'],[5,20])
        self.assertFalse(self.protocol['model']['hyperparameter_search'])
        self.assertEqual(self.protocol['uncertainty']['primary_block'],8)
        self.assertEqual(self.protocol['decision']['marginal_confidence'],.975)
        self.assertEqual(len(self.protocol['decision']['comparisons']),2)
        self.assertIn('05-cpu-walkforward/protocol.json',self.protocol['input_sha256'])


if __name__=='__main__':
    unittest.main()
