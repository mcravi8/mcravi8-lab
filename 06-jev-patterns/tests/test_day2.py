"""CPU-only tests of frozen jobs, reference transport, and future-run fail-closed gates."""
from contextlib import redirect_stdout
from io import StringIO
import importlib.util
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

EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP));sys.path.insert(0,str(Path(__file__).parent))
from shared.src import prepare_day2 as prepare, evaluate, score, jev_api
from test_score import server
import test_score
spec=importlib.util.spec_from_file_location('future_gpu_runbook',EXP/'scripts/run_gpu.py')
runbook=importlib.util.module_from_spec(spec);spec.loader.exec_module(runbook)


class Day2Tests(unittest.TestCase):
    def fixture(self):
        base={'root':'06-jev-patterns/06a-candles','index':'results/index.csv','out':'results/future.csv',
              'options_file':'06-jev-patterns/06a-candles/options.json','question':'Frozen question',
              'context_file':None,'reference_image':None,'control':'image','optional_reference':False,'n_rows':20}
        jobs=[dict(base,id='candles_zero_shot_p1',kind='candles'),
              dict(base,id='synthetic_variable_zero_shot_p1',kind='patterns'),
              dict(base,id='optional',kind='patterns',optional_reference=True)]
        return {'jobs':jobs,'runtime':{'port':8000,'concurrency':4,'batch_size':16,'timeout':120,'retries':3}}

    def test_global_parity_gates_precede_all_prediction_jobs(self):
        config=self.fixture();calls=[]
        with redirect_stdout(StringIO()):
            runbook.execute_jobs(config,{'model':jev_api.MODEL,'revision':'a'*40},'python',
                                 runner=lambda command,check:calls.append(command))
        self.assertEqual(len(calls),4)
        self.assertTrue(all('--parity-only' in c for c in calls[:2]))
        self.assertTrue(all('--parity-only' not in c for c in calls[2:]))
        self.assertTrue(all(c[c.index('--concurrency')+1]=='4' for c in calls))
        self.assertEqual(len(runbook.selected_jobs(config,True)),3)

    def test_failed_global_gate_stops_before_any_prediction(self):
        calls=[]
        def fail(command,check):
            calls.append(command)
            raise subprocess.CalledProcessError(1,command)
        with self.assertRaises(subprocess.CalledProcessError):
            runbook.execute_jobs(self.fixture(),{'model':jev_api.MODEL,'revision':'a'*40},'python',runner=fail)
        self.assertEqual(len(calls),1);self.assertIn('--parity-only',calls[0])

    def test_default_arguments_do_not_authorize_execution(self):
        self.assertFalse(runbook.arguments([]).execute)
        self.assertFalse(runbook.arguments([]).include_references)
        with redirect_stdout(StringIO()),patch('sys.stderr',StringIO()),self.assertRaises(SystemExit):
            runbook.arguments(['--execute'])

    def test_asset_and_model_hash_mismatch_block_before_serving(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'fixture';path.write_bytes(b'original')
            with patch.object(runbook,'REPO',root):
                config={'assets':{'fixture':runbook.sha(path)}}
                runbook.validate_assets(config);path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'changed'):runbook.validate_assets(config)
                with self.assertRaises(ValueError):runbook.checked_path('../escape')
            contract={'weight_files':{'fixture':{'size':7,'sha256':'0'*64}},'files':{}}
            with self.assertRaisesRegex(ValueError,'weight differs'):runbook.verify_model(root,contract)

    def test_runtime_version_changes_reject_resume(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(runbook.subprocess,'check_output',return_value='{"vllm":"pinned"}'):
                runbook.runtime_lock('server','scorer',Path(d))
                runbook.runtime_lock('server','scorer',Path(d))
            with patch.object(runbook.subprocess,'check_output',return_value='{"vllm":"different"}'):
                with self.assertRaisesRegex(ValueError,'runtime changed'):
                    runbook.runtime_lock('server','scorer',Path(d))

    def test_price_inversion_swaps_high_low_preserves_volume_and_is_involution(self):
        frame=pd.DataFrame({'Open':[10.,13.],'High':[14.,15.],'Low':[9.,10.],
                            'Close':[13.,11.],'Volume':[123,456]})
        inverted=prepare.invert_prices(frame)
        pd.testing.assert_frame_equal(prepare.invert_prices(inverted),frame)
        np.testing.assert_array_equal(inverted.Volume,frame.Volume)
        self.assertTrue((inverted.High>=inverted[['Open','Close']].max(axis=1)).all())
        self.assertTrue((inverted.Low<=inverted[['Open','Close']].min(axis=1)).all())
        self.assertNotIn('Bull Flag',prepare.MAP);self.assertNotIn('None',prepare.MAP)
        for a,b in prepare.MAP.items():self.assertEqual(prepare.MAP[b],a)

    def test_call_budget_counts_predictions_parity_optional_and_global(self):
        jobs=self.fixture()['jobs']
        report=prepare.estimate(jobs)
        self.assertEqual(report['prediction_calls'],40)
        self.assertEqual(report['total_decide_calls'],200)
        self.assertEqual(prepare.estimate(jobs,True)['total_decide_calls'],260)

    def test_style_selection_is_unique_and_order_independent(self):
        index=pd.DataFrame([{'file':f'{i}.png','label':str(i%7),'split':'test' if i%2 else 'train'}
                            for i in range(240)])
        subset=prepare.style_subset(index)
        self.assertEqual(len(subset),200);self.assertTrue(subset.file.is_unique)
        self.assertEqual(subset.file.tolist(),prepare.style_subset(index.iloc[::-1]).file.tolist())

    def test_mixed_origin_metrics_require_explicit_cohort(self):
        frame=pd.DataFrame({'data_origin':['market','synthetic','market'],'split':['train','eval','test'],
                            'end_date':['2015-01-02','','2020-01-02']})
        with self.assertRaisesRegex(ValueError,'mixed origins'):evaluate.infer_synthetic_dataset(frame,{})
        synth=evaluate.select_cohort(frame,'synthetic','eval')
        self.assertTrue(evaluate.infer_synthetic_dataset(synth,{'synthetic_dataset':False}))
        market=evaluate.select_cohort(frame,'market','test')
        self.assertEqual(len(market),1);self.assertFalse(evaluate.infer_synthetic_dataset(market,{}))
        with self.assertRaisesRegex(ValueError,'no rows'):evaluate.select_cohort(frame,'synthetic','test')

    def test_reference_payload_limit_order_and_plain_arm_unchanged(self):
        url='data:image/png;base64,AAAA'
        bare=jev_api.build_payload(url,['A','B'],'Question')
        self.assertEqual(bare['state'],['Chart: ',{'image':url}])
        ref={'image':url,'caption':'Reference only'}
        payload=jev_api.build_payload(url,['A','B'],'Question','Definitions',[ref])
        self.assertEqual(payload['state'],['Reference only',{'image':url},'Target chart to classify: ',{'image':url}])
        with self.assertRaises(ValueError):jev_api.build_payload(url,['A','B'],'Q',reference_images=[ref]*8)

    def test_reference_http_gate_resume_and_changed_board_rejected(self):
        helper=test_score.ScorerTests()
        with tempfile.TemporaryDirectory() as d,server() as fixture,redirect_stdout(StringIO()):
            root=Path(d);options=helper.prepare(root)
            reference=root/'reference.png';Image.new('RGB',(64,64),'white').save(reference)
            extra=['--reference-image',str(reference),'--limit','1']
            self.assertEqual(score.run(helper.args(root,fixture,options,*extra)),0)
            self.assertEqual(len(fixture.requests),41)
            for request in fixture.requests:
                self.assertEqual(sum(isinstance(item,dict) and 'image' in item for item in request['state']),2)
                self.assertNotIn('PRIVATE_TICKER',json.dumps(request))
            Image.new('RGB',(64,64),'black').save(reference)
            with self.assertRaisesRegex(ValueError,'provenance changed'):
                score.run(helper.args(root,fixture,options,*extra))
            self.assertEqual(len(fixture.requests),41)



class FrozenArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((EXP/'config/inference_arms.json').read_text())

    def test_frozen_assets_and_published_config_hash(self):
        config=self.config
        report=json.loads((EXP/'shared/results/day2_preparation.json').read_text())
        self.assertEqual(runbook.sha(EXP/'config/inference_arms.json'),report['config_sha256'])
        for relative,expected in config['assets'].items():
            path=runbook.checked_path(relative)
            if path.is_file():
                self.assertEqual(runbook.sha(path),expected,relative)
            else:
                # A fresh clone omits raw chart PNGs, but must retain all config,
                # indexes, source and reference boards required for preparation.
                self.assertIn('/charts/',relative)
        self.assertEqual(config['execution_status'],'not_run')

    def test_declared_jobs_cover_all_arm_indexes_and_option_labels(self):
        config=self.config
        for job in runbook.selected_jobs(config,True):
            root=runbook.REPO/job['root']
            rows=pd.read_csv(root/job['index'],keep_default_na=False)
            self.assertEqual(len(rows),job['n_rows'])
            self.assertTrue(rows.file.is_unique)
            options=json.loads((runbook.REPO/job['options_file']).read_text())
            self.assertTrue(rows.label.isin(options).all())
            self.assertIn(job['paraphrase'],[1,2])
            self.assertEqual(job['question'],config['questions'][job['kind']][job['paraphrase']-1])
            if job['role']=='time_placebo':
                self.assertFalse(job['accuracy_valid']);self.assertFalse(job['expected_class_map'])
            if job['optional_reference']:
                self.assertIsNotNone(job['reference_image']);self.assertIsNotNone(job['context_file'])
        self.assertEqual(config['call_budget']['default'],prepare.estimate(config['jobs']))
        self.assertEqual(config['call_budget']['with_references'],prepare.estimate(config['jobs'],True))

    def test_style_parents_and_market_synthetic_separation(self):
        a,b=self.config['styles']
        self.assertEqual(a['origin_counts'],{'market':200})
        self.assertEqual(b['origin_counts'],{'market':150,'synthetic':50})
        for group in [a,b]:
            self.assertEqual(len(set(group['ids'])),200)
            self.assertEqual(group['styles'],['candle','ohlc','line'])
        refs=json.loads((EXP/'shared/references/provenance.json').read_text())
        self.assertTrue(all(row['split']=='train' for row in refs['candles']))
        self.assertEqual(len(refs['patterns']),7)


if __name__=='__main__':unittest.main()
