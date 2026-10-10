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
        with tempfile.TemporaryDirectory() as d, server(fail_first=1) as fixture, redirect_stdout(StringIO()):
            root=Path(d);options=self.prepare(root)
            (root/'context.txt').write_text('Definitions only.')
            score.run(self.args(root,fixture,options,'--control','blank','--context-file',str(root/'context.txt'),'--parity-only'))
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
