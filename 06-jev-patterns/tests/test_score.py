import base64
from contextlib import contextmanager, redirect_stdout
import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import time
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared.src import score, jev_api
from fake_jev_server import FakeServer


@contextmanager
def server(**kwargs):
    fixture=FakeServer(**kwargs);t=threading.Thread(target=fixture.serve_forever,daemon=True);t.start()
    try:yield fixture
    finally:fixture.shutdown();fixture.server_close();t.join()


class ScorerTests(unittest.TestCase):
    def prepare(self, root, count=24):
        (root/'charts').mkdir()
        rows=[]
        options=['Doji','Hammer','Shooting Star','Bullish Engulfing','Bearish Engulfing','Morning Star','Evening Star','None']
        for i in range(count):
            file=f'charts/chart_{i:03d}.png';Image.new('RGB',(64,64),(i*7%256,10,230)).save(root/file)
            rows.append({'file':file,'ticker':'PRIVATE_TICKER','end_date':'2018-12-31','label':options[i%8]})
        with (root/'charts/index.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return options

    def args(self,root,fixture,options,*extra):
        return score.arguments(['--root',str(root),'--base-url',f'http://127.0.0.1:{fixture.server_port}',
          '--dry-run','--question','Which pattern is at the right edge?','--options',*options,
          '--timeout','2','--retries','1','--out','results/preds.csv',*extra])

    def test_twenty_repeat_gate_ten_score_resume_and_no_metadata_leak(self):
        with tempfile.TemporaryDirectory() as d, server(reverse=True) as fixture, redirect_stdout(StringIO()):
            root=Path(d);options=self.prepare(root)
            args=self.args(root,fixture,options,'--limit','10');self.assertEqual(score.run(args),0)
            self.assertEqual(len(fixture.requests),50)
            report=json.loads((root/'results/preds.parity.json').read_text())
            self.assertEqual(len(report['files']),20);self.assertEqual(report['max_abs_diff'],0)
            summary=json.loads((root/'results/preds.summary.json').read_text())
            for field in ['mean_abs_diff','max_abs_diff','p90_abs_diff','p90_chart_max_abs_diff','flagged_charts',
                          'near_tie_first_count','near_tie_second_count','near_tie_chart_count','near_tie_charts',
                          'allowed_argmax_mismatch_count','disallowed_argmax_mismatch_count']:
                self.assertEqual(summary['parity_'+field],report[field])
            self.assertEqual(score.run(self.args(root,fixture,options)),0)
            self.assertEqual(len(fixture.requests),104)
            self.assertEqual(score.run(self.args(root,fixture,options)),0)
            self.assertEqual(len(fixture.requests),104)
            rows=list(csv.DictReader(StringIO((root/'results/preds.csv').read_text())))
            self.assertEqual(len(rows),24);self.assertIn('None',[r['label'] for r in rows])
            for r in rows:self.assertAlmostEqual(sum(json.loads(r['probabilities'])),1)
            for request in fixture.requests:
                self.assertNotIn('PRIVATE_TICKER',json.dumps(request));self.assertNotIn('2018-12-31',json.dumps(request))
                self.assertEqual(request['options'],options);self.assertEqual(request['thinking'],'off')

    def test_drifting_repeat_gate_blocks_predictions(self):
        with tempfile.TemporaryDirectory() as d, server(drift=True) as fixture, redirect_stdout(StringIO()):
            root=Path(d);options=self.prepare(root)
            with self.assertRaisesRegex(ValueError,'Parity failed'):score.run(self.args(root,fixture,options))
            self.assertFalse((root/'results/preds.csv').exists())
            self.assertFalse(json.loads((root/'results/preds.parity.json').read_text())['passed'])

    def test_resume_rejects_changed_context_image_and_revision(self):
        with tempfile.TemporaryDirectory() as d, server() as fixture, redirect_stdout(StringIO()):
            root=Path(d);options=self.prepare(root)
            score.run(self.args(root,fixture,options,'--limit','1'))
            (root/'context.txt').write_text('New definitions')
            with self.assertRaisesRegex(ValueError,'provenance changed'):
                score.run(self.args(root,fixture,options,'--context-file',str(root/'context.txt')))
            fixture.revision='b'*40
            with self.assertRaisesRegex(ValueError,'provenance changed'):score.run(self.args(root,fixture,options))
            fixture.revision='a'*40;Image.new('RGB',(64,64),'white').save(root/'charts/chart_000.png')
            with self.assertRaisesRegex(ValueError,'provenance changed'):score.run(self.args(root,fixture,options))

    def test_retry_blank_context_and_explicit_parity_mode(self):
        with tempfile.TemporaryDirectory() as d, server(fail_first=1) as fixture, redirect_stdout(StringIO()) as output:
            root=Path(d);options=self.prepare(root)
            (root/'context.txt').write_text('Definitions only.')
            score.run(self.args(root,fixture,options,'--control','blank','--context-file',str(root/'context.txt'),'--parity-only'))
            console=json.loads(output.getvalue())
            self.assertEqual(console['mean_abs_diff'],0);self.assertEqual(console['p90_abs_diff'],0)
            self.assertEqual(console['flagged_charts'],[])
            self.assertIn('near_tie_chart_count',console);self.assertIn('allowed_argmax_mismatch_count',console)
            self.assertEqual(console['parity_policy']['version'],7)
            self.assertEqual(len(fixture.requests),41)
            self.assertFalse((root/'results/preds.csv').exists())
            payload=fixture.requests[-1]
            self.assertTrue(payload['question'].startswith('Definitions only.\n\n'))
            b64=payload['state'][1]['image'].split(',')[1]
            self.assertEqual(Image.open(BytesIO(base64.b64decode(b64))).getextrema(),((255,255),)*3)

    def test_corrupt_final_probabilities_recomputed_and_fake_requires_flag(self):
        with tempfile.TemporaryDirectory() as d, server() as fixture, redirect_stdout(StringIO()):
            root=Path(d);options=self.prepare(root)
            score.run(self.args(root,fixture,options))
            path=root/'results/preds.csv'
            rows=list(csv.DictReader(StringIO(path.read_text())));rows[-1]['probabilities']='broken'
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            score.run(self.args(root,fixture,options));self.assertEqual(len(fixture.requests),105)
            args=self.args(root,fixture,options);args.dry_run=False
            with self.assertRaisesRegex(ValueError,'Fake server'):score.run(args)

    def test_resume_reruns_gate_after_server_restart(self):
        with tempfile.TemporaryDirectory() as d, server() as fixture, redirect_stdout(StringIO()):
            root=Path(d);options=self.prepare(root)
            score.run(self.args(root,fixture,options,'--limit','1'))
            fixture.counts.clear();fixture.drift=True
            with self.assertRaisesRegex(ValueError,'Parity failed'):
                score.run(self.args(root,fixture,options))
            self.assertEqual(len(list(csv.DictReader(StringIO((root/'results/preds.csv').read_text())))),1)

    def test_torn_first_field_is_repaired_but_interior_damage_fails(self):
        with tempfile.TemporaryDirectory() as d, server() as fixture, redirect_stdout(StringIO()):
            root=Path(d);options=self.prepare(root)
            score.run(self.args(root,fixture,options))
            path=root/'results/preds.csv';lines=path.read_text().splitlines()
            path.write_text('\n'.join(lines[:-1])+'\ncharts/cha')
            score.run(self.args(root,fixture,options))
            summary=json.loads(path.with_suffix('.summary.json').read_text())
            self.assertEqual((summary['repaired'],summary['scored'],summary['skipped']),(1,1,23))
            lines=path.read_text().splitlines();lines.insert(2,'charts/cha')
            path.write_text('\n'.join(lines)+'\n')
            with self.assertRaisesRegex(ValueError,'interior prediction'):
                score.run(self.args(root,fixture,options))

    def test_served_alias_and_more_than_sixteen_options(self):
        with tempfile.TemporaryDirectory() as d, server(model='jev') as fixture, redirect_stdout(StringIO()):
            root=Path(d);options=self.prepare(root)+[f'Extra {i}' for i in range(9)]
            score.run(self.args(root,fixture,options,'--served-model-name','jev','--limit','1'))
            meta=json.loads((root/'results/preds.meta.json').read_text())
            self.assertEqual(meta['server_identity']['model'],'jev')
            self.assertEqual(meta['server_identity']['canonical_model'],jev_api.MODEL)
            self.assertEqual(fixture.requests[-1]['strategy'],'single')

    def test_original_outlier_is_included_and_flagged_under_mean_policy(self):
        evidence=json.loads((Path(__file__).resolve().parents[1]/
            'shared/results/diagnostics/2026-10-10/candles.parity.json').read_text())
        client,rows=self.parity_fixture(evidence['first'],evidence['second'])
        rows=[{'file':file} for file in evidence['files']]
        with tempfile.TemporaryDirectory() as d:
            report=score.parity_check(client,rows,Path(d)/'gate.json','fixture',force=True)
        self.assertTrue(report['passed']);self.assertEqual(report['atol'],0.01)
        self.assertEqual(report['rows_over_flag_threshold'],1)
        self.assertEqual(report['flagged_files'],[evidence['files'][14]])
        self.assertEqual(report['flagged_charts'][0]['chart_number'],15)
        self.assertEqual(report['argmax_agreement_count'],20)
        self.assertEqual(report['execution']['parity_concurrency'],1)
        self.assertAlmostEqual(report['max_abs_diff'],0.03334436881129002)
        self.assertAlmostEqual(report['mean_abs_diff'],0.0019776217887860653)
        self.assertEqual(report['probability_comparisons'],160)

    def parity_fixture(self,first,second):
        client=SimpleNamespace(args=SimpleNamespace(options=[str(i) for i in range(len(first[0]))],
            dry_run=False,concurrency=1,batch_size=1,execution_fingerprint='a'*64),
            score=Mock(side_effect=[(p,0,1) for p in first+second]))
        rows=[{'file':f'{i:02}.png'} for i in range(20)]
        return client,rows

    def test_saved_v4_pairs_pass_new_rule_without_rewriting_original_failure(self):
        source=Path(__file__).resolve().parents[1]/'shared/results/diagnostics/2026-10-10-v4-parity/candles.parity.json'
        before=source.read_bytes();evidence=json.loads(before)
        self.assertFalse(evidence['passed']);self.assertEqual(evidence['atol'],0.01)
        client,rows=self.parity_fixture(evidence['first'],evidence['second'])
        with tempfile.TemporaryDirectory() as d:
            report=score.parity_check(client,rows,Path(d)/'gate.json','fixture',force=True)
        self.assertTrue(report['passed']);self.assertEqual(report['atol'],0.01)
        self.assertEqual(report['argmax_agreement_count'],20)
        self.assertTrue(report['top_predictions_matched'])
        self.assertEqual(report['rows_over_flag_threshold'],0)
        self.assertAlmostEqual(report['max_abs_diff'],0.018410737935632693)
        self.assertEqual(source.read_bytes(),before)

    def test_mean_gate_includes_0_01_boundary_and_rejects_larger_mean(self):
        first=[[0,.02,.48,.5]]*20
        with tempfile.TemporaryDirectory() as d:
            client,rows=self.parity_fixture(first,[[.02,0,.48,.5]]*20)
            report=score.parity_check(client,rows,Path(d)/'boundary.json','boundary',force=True)
            self.assertEqual(report['mean_abs_diff'],0.01);self.assertTrue(report['passed'])
            client,rows=self.parity_fixture(first,[[.020004,0,.479996,.5]]*20)
            path=Path(d)/'above.json'
            with self.assertRaisesRegex(ValueError,'mean absolute probability difference.*> 0.01'):
                score.parity_check(client,rows,path,'above',force=True)
            report=json.loads(path.read_text())
            self.assertFalse(report['passed']);self.assertTrue(report['top_predictions_matched'])

    def test_mean_uses_all_option_entries_and_reports_linear_percentiles(self):
        first=[[0,.2,.3,.5]]*20;second=[[.016,.184,.3,.5]]*20
        report=score.parity_statistics(first,second,[f'{i}.png' for i in range(20)],4)
        self.assertAlmostEqual(report['mean_abs_diff'],.008)
        self.assertAlmostEqual(report['p90_abs_diff'],.016)
        self.assertAlmostEqual(report['p90_chart_max_abs_diff'],.016)
        self.assertEqual(report['probability_comparisons'],80)
        self.assertTrue(report['passed'])  # mean of chart maxima would exceed0.01
        # Sparse tail:78 zeros and one row with .04,.04. P90 option drift is0;
        # chart-max p90 is0; every tail entry must still enter the mean.
        sparse=score.parity_statistics([[0,.04,.46,.5]]*20,
            [[.04,0,.46,.5]]+[[0,.04,.46,.5]]*19,[str(i) for i in range(20)],4)
        self.assertAlmostEqual(sparse['mean_abs_diff'],.001)
        self.assertEqual(sparse['p90_abs_diff'],0)
        self.assertEqual(sparse['rows_over_flag_threshold'],1)
        # Chart-max quantile interpolates between sorted rows18 and19.
        varied=[[v,.1-v,.3,.6] for v in [.0]*18+[.02,.04]]
        interpolated=score.parity_statistics([[0,.1,.3,.6]]*20,varied,[str(i) for i in range(20)],4)
        self.assertAlmostEqual(interpolated['p90_chart_max_abs_diff'],.002)

    def test_flag_threshold_is_strict_and_flagged_charts_still_require_argmax(self):
        first=[[0,.04,.46,.5]]*20
        exact=score.parity_statistics(first,[[.03,.01,.46,.5]]+first[1:],[str(i) for i in range(20)],4)
        self.assertEqual(exact['rows_over_flag_threshold'],0);self.assertTrue(exact['passed'])
        above=score.parity_statistics(first,[[.030001,.009999,.46,.5]]+first[1:],[str(i) for i in range(20)],4)
        self.assertEqual(above['rows_over_flag_threshold'],1);self.assertTrue(above['passed'])
        flipped=score.parity_statistics([[.49,.51]]*20,[[.53,.47]]+[[.49,.51]]*19,[str(i) for i in range(20)],2)
        self.assertTrue(flipped['probability_drift_passed']);self.assertFalse(flipped['passed'])
        self.assertEqual(flipped['argmax_mismatch_files'],['0'])
        self.assertEqual(flipped['flagged_charts'][0]['argmax_matched'],False)

    def test_saved_v5_failure_passes_cpu_replay_without_mutating_source(self):
        source=Path(__file__).resolve().parents[1]/'shared/results/diagnostics/2026-10-10-v5-parity/candles.parity.json'
        before=source.read_bytes();evidence=json.loads(before)
        self.assertFalse(evidence['passed']);self.assertEqual(evidence['atol'],.03)
        client,rows=self.parity_fixture(evidence['first'],evidence['second'])
        rows=[{'file':file} for file in evidence['files']]
        with tempfile.TemporaryDirectory() as d:
            report=score.parity_check(client,rows,Path(d)/'gate.json','fixture',force=True)
        self.assertTrue(report['passed']);self.assertEqual(report['argmax_agreement_count'],20)
        self.assertEqual(report['flagged_files'],[evidence['files'][14]])
        self.assertAlmostEqual(report['mean_abs_diff'],.002458124933190978)
        self.assertAlmostEqual(report['p90_abs_diff'],.005723228075901335)
        self.assertAlmostEqual(report['max_abs_diff'],.04121730474518304)
        self.assertEqual(source.read_bytes(),before)

    def test_non_near_tie_argmax_flip_blocks_predictions_and_logs_failure(self):
        first=[.50,.47]+[.005]*6;flipped=[.47,.50]+[.005]*6
        pairs=[first]*20+[flipped]+[first]*19
        with tempfile.TemporaryDirectory() as d,server() as fixture,redirect_stdout(StringIO()):
            root=Path(d);options=self.prepare(root)
            with patch.object(score.Client,'score',side_effect=[(p,0,1) for p in pairs]) as calls:
                with self.assertRaisesRegex(ValueError,'top predictions differ outside near ties on 1/20'):
                    score.run(self.args(root,fixture,options,'--limit','1'))
            self.assertEqual(calls.call_count,40)
            self.assertFalse((root/'results/preds.csv').exists())
            report=json.loads((root/'results/preds.parity.json').read_text())
            self.assertTrue(report['probability_drift_passed'])
            self.assertFalse(report['top_predictions_matched']);self.assertFalse(report['passed'])
            self.assertEqual(report['argmax_agreement_count'],19)
            self.assertEqual(report['argmax_mismatch_files'],[report['files'][0]])
            errors=list(csv.DictReader(StringIO((root/'results/preds_errors.csv').read_text())))
            self.assertEqual(errors[-1]['phase'],'parity')

    def test_cached_gate_rejects_changed_policy(self):
        with tempfile.TemporaryDirectory() as d:
            client,rows=self.parity_fixture([[.1,.9]]*20,[[.1,.9]]*20)
            path=Path(d)/'gate.json'
            score.parity_check(client,rows,path,'fixture',force=True)
            cached=json.loads(path.read_text());cached['parity_policy']['version']=6
            path.write_text(json.dumps(cached))
            with self.assertRaisesRegex(ValueError,'Parity policy differs'):
                score.parity_check(client,rows,path,'fixture')
            self.assertEqual(client.score.call_count,40)

    def test_tied_top_probability_uses_declared_option_order(self):
        first=[[.45,.45,.1]]*20
        with tempfile.TemporaryDirectory() as d:
            client,rows=self.parity_fixture(first,first)
            report=score.parity_check(client,rows,Path(d)/'tie.json','tie',force=True)
            self.assertTrue(report['passed'])
            self.assertEqual(report['parity_policy']['argmax_tie_break'],'first option in declared order')
            client,rows=self.parity_fixture(first,[[.449,.451,.1]]*20)
            report=score.parity_check(client,rows,Path(d)/'tie_flip.json','tie_flip',force=True)
            self.assertTrue(report['passed']);self.assertFalse(report['top_predictions_matched'])
            self.assertEqual(report['argmax_agreement_count'],0)
            self.assertEqual(report['near_tie_chart_count'],20)
            self.assertEqual(report['allowed_argmax_mismatch_count'],20)
            self.assertEqual(report['disallowed_argmax_mismatch_count'],0)

    def test_near_tie_exception_accepts_either_pass_and_keeps_raw_agreement(self):
        for a,b,first_count,second_count in [
            ([.35,.345,.305],[.33,.365,.305],1,0),
            ([.33,.365,.305],[.35,.345,.305],0,1)]:
            first=[a]+[[.7,.2,.1]]*19;second=[b]+first[1:]
            stats=score.parity_statistics(first,second,[str(i) for i in range(20)],3)
            self.assertTrue(stats['passed']);self.assertFalse(stats['top_predictions_matched'])
            self.assertEqual(stats['argmax_agreement_count'],19)
            self.assertEqual(stats['near_tie_first_count'],first_count)
            self.assertEqual(stats['near_tie_second_count'],second_count)
            self.assertEqual(stats['near_tie_chart_count'],1)
            self.assertEqual(stats['allowed_argmax_mismatch_files'],['0'])
            self.assertEqual(stats['disallowed_argmax_mismatch_count'],0)
        # Near-tied charts with stable tops are counted, but not as allowed mismatches.
        stats=score.parity_statistics([[.45,.449,.101]]*20,[[.45,.449,.101]]*20,[str(i) for i in range(20)],3)
        self.assertEqual(stats['near_tie_chart_count'],20);self.assertEqual(stats['allowed_argmax_mismatch_count'],0)

    def test_near_tie_margin_boundary_and_non_near_tie_mismatches(self):
        cases=[([.35,.34,.31],[.34,.35,.31],True),
               ([.3500005,.3399995,.31],[.3399995,.3500005,.31],False)]
        for a,b,accepted in cases:
            stats=score.parity_statistics([a]+[[.7,.2,.1]]*19,[b]+[[.7,.2,.1]]*19,[str(i) for i in range(20)],3)
            self.assertEqual(stats['passed'],accepted)
            self.assertTrue(stats['probability_drift_passed'])
            self.assertEqual(stats['allowed_argmax_mismatch_count'],int(accepted))
            self.assertEqual(stats['disallowed_argmax_mismatch_count'],int(not accepted))
        self.assertAlmostEqual(score.PARITY_NEAR_TIE_MARGIN,.01)

    def test_near_tie_exceptions_do_not_override_mean_drift_failure(self):
        client,rows=self.parity_fixture([[.4,.395,.205]]*20,[[.3,.45,.25]]*20)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'gate.json'
            with self.assertRaisesRegex(ValueError,'mean absolute probability difference'):
                score.parity_check(client,rows,path,'fixture',force=True)
            stats=json.loads(path.read_text())
        self.assertTrue(stats['argmax_condition_passed']);self.assertFalse(stats['passed'])
        self.assertEqual(stats['allowed_argmax_mismatch_count'],20)
        self.assertEqual(stats['disallowed_argmax_mismatch_count'],0)

    def test_v6_near_tie_failure_cpu_replay_scores_and_logs_exception_counts(self):
        source=Path(__file__).resolve().parents[1]/'shared/results/diagnostics/2026-10-11-v6-parity/candles.parity.json'
        before=source.read_bytes();evidence=json.loads(before)
        self.assertFalse(evidence['passed']);self.assertEqual(evidence['parity_policy']['version'],6)
        pairs=evidence['first']+evidence['second']+[[1]+[0]*7]
        with tempfile.TemporaryDirectory() as d,server() as fixture,redirect_stdout(StringIO()):
            root=Path(d);options=self.prepare(root,20)
            # Use original filenames/order so archived chart#11 remains identifiable.
            with (root/'charts/index.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            for row,file in zip(rows,evidence['files']):
                (root/row['file']).rename(root/file);row['file']=file
            with (root/'charts/index.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            with patch.object(score.Client,'score',side_effect=[(p,0,1) for p in pairs]) as calls:
                self.assertEqual(score.run(self.args(root,fixture,options,'--limit','1')),0)
            self.assertEqual(calls.call_count,41)
            gate=json.loads((root/'results/preds.parity.json').read_text())
            summary=json.loads((root/'results/preds.summary.json').read_text())
            self.assertTrue(gate['passed']);self.assertFalse(gate['top_predictions_matched'])
            self.assertEqual(gate['argmax_agreement_count'],19)
            self.assertEqual(gate['near_tie_first_count'],0);self.assertEqual(gate['near_tie_second_count'],1)
            self.assertEqual(gate['allowed_argmax_mismatch_files'],['charts/7d763e0bb9783f355926.png'])
            self.assertEqual(gate['near_tie_charts'][0]['chart_number'],11)
            self.assertAlmostEqual(gate['mean_abs_diff'],.0026098296378638183)
            self.assertAlmostEqual(gate['max_abs_diff'],.022243287166081593)
            self.assertEqual(summary['parity_near_tie_chart_count'],1)
            self.assertEqual(summary['parity_allowed_argmax_mismatch_count'],1)
            self.assertEqual(summary['parity_disallowed_argmax_mismatch_count'],0)
            self.assertEqual(len(list(csv.DictReader(StringIO((root/'results/preds.csv').read_text())))),1)
            self.assertFalse((root/'results/preds_errors.csv').exists())
        self.assertEqual(source.read_bytes(),before)

    def test_serial_defaults_and_changed_execution_cannot_resume(self):
        with tempfile.TemporaryDirectory() as d,server() as fixture,redirect_stdout(StringIO()):
            root=Path(d);options=self.prepare(root)
            args=self.args(root,fixture,options,'--limit','2','--execution-fingerprint','a'*64)
            self.assertEqual((args.concurrency,args.batch_size),(1,1))
            original=score.Client.score;active=0;maximum=0;lock=threading.Lock()
            def tracked(client,row):
                nonlocal active,maximum
                with lock:active+=1;maximum=max(maximum,active)
                try:
                    time.sleep(.002)
                    return original(client,row)
                finally:
                    with lock:active-=1
            with patch.object(score.Client,'score',tracked):
                score.run(args)
            self.assertEqual(maximum,1);self.assertEqual(len(fixture.requests),42)
            for extra in [('--concurrency','4','--execution-fingerprint','a'*64),
                          ('--batch-size','16','--execution-fingerprint','a'*64),
                          ('--execution-fingerprint','b'*64)]:
                with self.assertRaisesRegex(ValueError,'provenance changed'):
                    score.run(self.args(root,fixture,options,*extra))
            self.assertEqual(len(fixture.requests),42)

    def test_option_contract_and_limits(self):
        options=[f'Option {i}' for i in range(256)]
        p=jev_api.build_payload('data:image/png;base64,abc',options,'Q','Context')
        self.assertEqual(len(p['options']),256)
        for bad in [['A'],['A','A'],['A',''],['A','B\nC'],['A']*257]:
            with self.assertRaises(ValueError):jev_api.options_list(bad)
        for probs in [[float('nan'),1],[.2,.3],[True,0],[1.1,-.1]]:
            with self.assertRaises(ValueError):jev_api.probabilities(probs,2)
        with self.assertRaisesRegex(ValueError,'20 indexed'):score.selected_parity([{'file':str(i)} for i in range(19)])
        with self.assertRaises(ValueError):score.base_endpoint('http://remote.test',True)


if __name__=='__main__':unittest.main()
