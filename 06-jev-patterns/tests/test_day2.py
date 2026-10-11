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
from types import SimpleNamespace
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
              dict(base,id='synthetic_variable_zero_shot_p1',kind='patterns')]
        return {'jobs':jobs,'runtime':{'port':8000,'concurrency':1,'batch_size':1,'timeout':120,'retries':3,'max_model_len':16384}}

    def test_global_parity_gates_precede_all_prediction_jobs(self):
        config=self.fixture();calls=[]
        with redirect_stdout(StringIO()):
            runbook.execute_jobs(config,{'model':jev_api.MODEL,'revision':'a'*40},'python',
                                 runner=lambda command,check:calls.append(command))
        self.assertEqual(len(calls),4)
        self.assertTrue(all('--parity-only' in c for c in calls[:2]))
        self.assertTrue(all('--parity-only' not in c for c in calls[2:]))
        self.assertTrue(all(c[c.index('--concurrency')+1]=='1' for c in calls))
        self.assertTrue(all(c[c.index('--batch-size')+1]=='1' for c in calls))
        self.assertEqual(len(runbook.selected_jobs(config)),2)

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
        with patch('sys.stderr',StringIO()),self.assertRaises(SystemExit):
            runbook.arguments(['--include-references'])
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

    def runtime_fixture(self):
        pins=json.loads((EXP/'config/gpu_runtime.json').read_text())
        versions={'server':dict(pins['server_packages']),'scorer':dict(pins['scorer_packages']),
                  'server_build':{'torch_version':'2.13.0+cu130','torch_cuda':'13.0',
                                  'batch_invariance_supported':True}}
        return pins,versions

    def test_runtime_version_changes_reject_resume(self):
        pins,versions=self.runtime_fixture()
        def responses(extra=None):
            server=dict(versions['server'])
            if extra:server['unrelated-package']=extra
            return [json.dumps(server),json.dumps(versions['scorer']),json.dumps(versions['server_build'])]
        with tempfile.TemporaryDirectory() as d:
            for _ in range(2):
                with patch.object(runbook.subprocess,'check_output',side_effect=responses()):
                    runbook.runtime_lock('server','scorer',Path(d),pins)
            self.assertEqual(json.loads((Path(d)/'runtime_lock.json').read_text())['server'],versions['server'])
            with patch.object(runbook.subprocess,'check_output',side_effect=responses('changed')):
                with self.assertRaisesRegex(ValueError,'runtime changed'):
                    runbook.runtime_lock('server','scorer',Path(d),pins)

    def test_runtime_rejects_nightly_torch_cuda_and_scorer_drift(self):
        pins,versions=self.runtime_fixture()
        runbook.validate_runtime(versions,pins)
        for role,name in [('server','vllm'),('server','torch'),('scorer','numpy')]:
            changed=json.loads(json.dumps(versions));changed[role][name]='different'
            with self.assertRaisesRegex(ValueError,'runtime pin differs'):runbook.validate_runtime(changed,pins)
        changed=json.loads(json.dumps(versions));changed['server_build']['torch_cuda']='12.9'
        with self.assertRaisesRegex(ValueError,'Torch CUDA build'):runbook.validate_runtime(changed,pins)
        versions['server_build']['batch_invariance_supported']=False
        with self.assertRaisesRegex(ValueError,'BATCH_INVARIANT'):runbook.validate_runtime(versions,pins)

    def test_deterministic_native_server_keeps_system1_contract_and_supported_flags(self):
        command=runbook.server_command('/root/vllm-venv/bin/python','/weights',self.fixture(),
            {'model':jev_api.MODEL})
        for flag in ['--enable-lora','--trust-request-chat-template','--enforce-eager',
                     '--no-enable-prefix-caching','--no-enable-chunked-prefill','--no-async-scheduling']:
            self.assertIn(flag,command)
        for flag,value in [('--max-num-seqs','8'),('--seed','0'),('--logprobs-mode','processed_logprobs'),
                           ('--gdn-prefill-backend','triton'),('--max-num-batched-tokens','16384')]:
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertNotIn('--enable-prefix-caching',command)
        help_text=' '.join(part for part in command if part.startswith('--'))
        with patch.object(runbook.subprocess,'check_output',return_value=help_text):
            runbook.validate_server_flags('python','/weights',command,{})
        with patch.object(runbook.subprocess,'check_output',return_value=help_text.replace('--enforce-eager','')):
            with self.assertRaisesRegex(ValueError,'required execution flag'):
                runbook.validate_server_flags('python','/weights',command,{})

    def test_execution_policy_and_explicit_linux_interpreters(self):
        pins,_=self.runtime_fixture();config=self.fixture()
        runbook.validate_execution(config,pins)
        config['runtime']['concurrency']=4
        with self.assertRaisesRegex(ValueError,'concurrency1'):runbook.validate_execution(config,pins)
        with patch('sys.stderr',StringIO()),self.assertRaises(SystemExit):
            runbook.arguments(['--execute','--server-python','/root/vllm-venv/bin/python'])
        args=runbook.arguments(['--execute','--server-python','/root/vllm-venv/bin/python',
                               '--scorer-python','/root/svenv/bin/python'])
        self.assertEqual(str(args.scorer_python),'/root/svenv/bin/python')
        self.assertIsNone(runbook.arguments([]).scorer_python)

    def test_runtime_fingerprint_reaches_every_gate_and_job(self):
        calls=[]
        with redirect_stdout(StringIO()):
            runbook.execute_jobs(self.fixture(),{'model':jev_api.MODEL,'revision':'a'*40},'python',
                                 runner=lambda command,check:calls.append(command),runtime_fingerprint='c'*64)
        self.assertEqual(len(calls),4)
        self.assertTrue(all(c[c.index('--execution-fingerprint')+1]=='c'*64 for c in calls))

    def test_build_preflight_failure_cannot_launch_gpu_server(self):
        config=self.fixture();config['gpu_runtime']='06-jev-patterns/config/gpu_runtime.json'
        args=SimpleNamespace(server_python='/fake/server/python',scorer_python='/root/svenv/bin/python',
                             model_dir='/fake/model')
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(runbook,'EXP',Path(d)),
            patch.object(runbook,'validate_assets'),
            patch.object(runbook,'verify_model'),
            patch.object(runbook,'runtime_lock',side_effect=ValueError('Torch CUDA build differs')),
            patch.object(runbook.subprocess,'Popen') as launch,
        ):
            with self.assertRaisesRegex(ValueError,'Torch CUDA build'):
                runbook.run_session(config,{},args)
            launch.assert_not_called()

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

    def test_call_budget_counts_predictions_parity_and_global(self):
        jobs=self.fixture()['jobs']
        report=prepare.estimate(jobs)
        self.assertEqual(report['prediction_calls'],40)
        self.assertEqual(report['total_decide_calls'],200)
        self.assertEqual(report['jobs'],2)
        self.assertEqual(report['job_parity_calls'],80)
        self.assertEqual(report['global_parity_calls'],80)

    def test_style_selection_is_unique_and_order_independent(self):
        index=pd.DataFrame([{'file':f'{i}.png','label':str(i%7),'split':'test' if i%2 else 'train'}
                            for i in range(240)])
        subset=prepare.style_subset(index)
        self.assertEqual(len(subset),100);self.assertTrue(subset.file.is_unique)
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

    def test_runbook_rejects_reference_arm_over_budget_and_accuracy_enabled_showcase(self):
        config=self.fixture()
        config['jobs'][0]['reference_image']='reference.png'
        with self.assertRaisesRegex(ValueError,'Reference-image'):runbook.selected_jobs(config)
        config=self.fixture();config['jobs'][0]['n_rows']=10000
        with self.assertRaisesRegex(ValueError,'below 10,000'):runbook.selected_jobs(config)
        config=self.fixture();config['jobs'][0].update(name='real_variable',accuracy_valid=True)
        with self.assertRaisesRegex(ValueError,'showcase-only'):runbook.selected_jobs(config)
        config['jobs'][0].update(evidence_role='showcase_only',accuracy_valid=False)
        self.assertEqual(len(runbook.selected_jobs(config)),2)

    def test_showcase_and_time_placebo_cannot_produce_accuracy_or_arm_comparisons(self):
        benchmark=pd.DataFrame({'file':['x.png'],'ticker':['A'],'end_date':['2020-01-02'],
                               'label':['A'],'prediction':['A'],'probabilities':['[0.8,0.2]'],
                               'latency_ms':[1],'evidence_role':['benchmark'],'accuracy_valid':[True]})
        self.assertEqual(evaluate.evaluate_frame(benchmark,['A','B'],bootstrap_resamples=1)['accuracy'],1)
        showcase=benchmark.assign(evidence_role='showcase_only')
        for frame in [showcase,benchmark.assign(source_collection='real_variable')]:
            with self.assertRaisesRegex(ValueError,'[Ss]howcase'):
                evaluate.evaluate_frame(frame,['A','B'])
            with self.assertRaisesRegex(ValueError,'[Ss]howcase'):
                evaluate.compare_arms(benchmark,frame,['A','B'])
            with self.assertRaisesRegex(ValueError,'[Ss]howcase'):
                evaluate.compare_arms(frame,benchmark,['A','B'])
        with self.assertRaisesRegex(ValueError,'accuracy is invalid'):
            evaluate.evaluate_frame(benchmark.assign(accuracy_valid=False),['A','B'])

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
        self.assertEqual(config['runtime']['concurrency'],1)
        self.assertEqual(config['runtime']['batch_size'],1)
        self.assertEqual(config['prior_attempt']['prediction_calls'],0)
        pins=json.loads((runbook.REPO/config['gpu_runtime']).read_text())
        runbook.validate_execution(config,pins)
        observed=json.loads((runbook.REPO/pins['observed_inventory']).read_text())
        for role in ['server','scorer']:
            for name,value in pins[role+'_packages'].items():
                self.assertEqual(value,observed[role][name])
        self.assertTrue(all('/frozen_v3/' in j['out'] for j in config['jobs']))

    def test_archived_diagnostics_and_requirements_match_runtime_contract(self):
        folder=EXP/'shared/results/diagnostics/2026-10-10'
        diagnosis=json.loads((folder/'diagnosis.json').read_text())
        for name,expected in diagnosis['source_sha256'].items():
            self.assertEqual(runbook.sha(folder/name),expected)
        self.assertEqual(diagnosis['rows_over_0_01'],3)
        self.assertEqual(diagnosis['argmax_agreement_count'],20)
        pins=json.loads((EXP/'config/gpu_runtime.json').read_text())
        for filename,key in [('requirements-scoring.txt','scorer_packages'),
                             ('requirements-jev-server.txt','server_packages')]:
            requirements=dict(line.split('==') for line in (EXP/filename).read_text().splitlines()
                              if line and not line.startswith('#'))
            self.assertEqual(requirements,pins[key])
        self.assertFalse(pins['correction_gpu_verified'])

    def test_declared_jobs_cover_all_arm_indexes_and_option_labels(self):
        config=self.config
        for job in runbook.selected_jobs(config):
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
            self.assertNotEqual(job['arm'],'references')
            self.assertFalse(job.get('reference_image'))
            if job['role']=='showcase':
                self.assertEqual(job['evidence_role'],'showcase_only')
                self.assertFalse(job['accuracy_valid'])
                self.assertTrue(rows.evidence_role.eq('showcase_only').all())
                self.assertTrue(rows.accuracy_valid.eq(False).all())
        self.assertEqual(config['call_budget']['default'],prepare.estimate(config['jobs']))
        budget=config['call_budget']['default']
        self.assertEqual(budget['jobs'],21)
        self.assertEqual(budget['prediction_calls'],8900)
        self.assertEqual(budget['job_parity_calls']+budget['global_parity_calls'],920)
        self.assertEqual(budget['total_decide_calls'],9820)
        self.assertLess(budget['total_decide_calls'],config['fresh_call_cap'])
        self.assertEqual(set(config['call_budget']),{'default'})

    def test_style_parents_and_market_synthetic_separation(self):
        a,b=self.config['styles']
        self.assertEqual(a['origin_counts'],{'market':100})
        self.assertEqual(b['origin_counts'],{'synthetic':100})
        for group in [a,b]:
            self.assertEqual(len(set(group['ids'])),100)
            self.assertEqual(group['styles'],['candle','ohlc','line'])

        for name in ['candles','patterns']:
            rows=pd.read_csv(runbook.REPO/next(c['root']+'/'+c['index'] for c in self.config['collections']
                             if c['name']==name+'_style_candle'),keep_default_na=False)
            if name=='candles':
                self.assertTrue(rows.split.eq('test').all())
            else:
                self.assertTrue(rows.is_clean.eq(True).all())
                self.assertTrue(rows.template_id.is_unique)

    def test_secondary_jobs_use_only_frozen_hundred_chart_subsets(self):
        config=self.config
        for job in config['jobs']:
            if job['role'] in ['primary','showcase']:
                self.assertEqual(job['paraphrase'],1)
            elif job['role']=='paraphrase':
                self.assertEqual(job['n_rows'],100)
                self.assertEqual(job['paraphrase'],2)
                self.assertIn(job['arm'],['zero_shot','definitions'])
            else:
                self.assertEqual(job['n_rows'],100)
                self.assertEqual(job['paraphrase'],1)
                self.assertEqual(job['arm'],'zero_shot')
        for collection in config['collections']:
            if collection['role'] not in ['paraphrase','alias','time_placebo']:
                continue
            rows=pd.read_csv(runbook.REPO/collection['root']/collection['index'],keep_default_na=False)
            main=next(c for c in config['collections'] if c['name']==rows.source_collection.iloc[0])
            originals=pd.read_csv(runbook.REPO/main['root']/main['index'],keep_default_na=False)
            self.assertTrue(set(rows.id)<=set(originals.id))
            self.assertEqual(len(rows.id.unique()),100)
        for subset in config['paraphrase_subsets']+config['control_subsets']:
            self.assertEqual(subset['n_rows'],100)
            self.assertEqual(len(set(subset['ids'])),100)

    def test_real_showcase_and_reported_detector_agreement(self):
        config=self.config
        self.assertEqual(config['real_showcase']['n_parents'],150)
        self.assertFalse(config['real_showcase']['accuracy_valid'])
        real=[c for c in config['collections'] if c['name'].startswith('real_')]
        self.assertEqual(len(real),2)
        ids=[]
        for collection in real:
            self.assertEqual(collection['role'],'showcase')
            self.assertEqual(collection['n_rows'],150)
            rows=pd.read_csv(runbook.REPO/collection['root']/collection['index'],keep_default_na=False)
            ids.append(set(rows.id))
            with self.assertRaisesRegex(ValueError,'Showcase-only'):
                evaluate.evaluate_frame(rows,[])
        self.assertEqual(ids[0],ids[1])
        result=config['reported_results']['detector_agreement']
        table=pd.read_csv(runbook.REPO/result['table'],keep_default_na=False)
        raw=json.loads((runbook.REPO/result['details']).read_text())
        self.assertEqual(len(table),7)
        self.assertTrue(table.evidence_type.eq('detector_overlap_not_model_accuracy').all())
        for row in table.itertuples():
            self.assertAlmostEqual(row.agreement_jaccard,row.matched/(row.external+row.independent-row.matched))
            self.assertAlmostEqual(row.agreement_jaccard,raw['classes'][row.label]['positive_jaccard'])
        self.assertEqual(table[table.label=='None'].decision_unit.iloc[0],'absence-screen window')
        self.assertEqual(config['reference_arm']['status'],'removed')
        self.assertFalse(any('/references/' in name for name in config['assets']))



if __name__=='__main__':unittest.main()
