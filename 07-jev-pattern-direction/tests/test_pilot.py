"""CPU-only target/selection/provenance and loopback scoring checks."""
import base64
from contextlib import redirect_stdout, redirect_stderr
import csv
from io import BytesIO, StringIO
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parent
sys.path.insert(0, str(EXP/'src'))
sys.path.insert(0, str(REPO/'06-jev-patterns/tests'))
from common import OPTIONS, ARMS, context, digest, question, verify_bundle, write_json
from fake_jev_server import FakeServer


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, EXP/relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


scorer = module('exp07_scorer', 'src/score.py')
prep = module('exp07_prepare', 'src/prepare.py')
evaluation = module('exp07_evaluation', 'src/evaluate.py')
gpu = module('exp07_gpu', 'scripts/run_gpu.py')


def fixture(root, count=24):
    protocol = json.loads((EXP/'protocol.json').read_text())
    protocol['input_sha256'] = {}
    protocol['evaluation']['resamples'] = 50
    protocol['call_budget'] = 4*count+20+200
    write_json(root/'protocol.json', protocol)
    (root/'context').mkdir()
    for source in (EXP/'context').glob('*.txt'):
        (root/'context'/source.name).write_bytes(source.read_bytes())
    (root/'charts').mkdir()
    rows = []
    for i in range(count):
        file = f'charts/chart_{i:03d}.png'
        Image.new('RGB', (64, 64), (i*7 % 256, 12, 213)).save(root/file)
        rows.append({'file': file, 'ticker': 'PRIVATE_TICKER', 'end_date': f'2021-04-{i+1:02d}',
            'target_date': '2021-05-01', 'window_start': '2020-11-01', 'label': OPTIONS[i%3],
            'forward_return': .01234, 'sigma': .02+i*.0001})
    (root/'results').mkdir()
    pd.DataFrame(rows).to_csv(root/'results/index.csv', index=False)
    pd.DataFrame(rows[:20]).to_csv(root/'results/blank_index.csv', index=False)
    write_json(root/'results/prior.json', {'probabilities': [.4, .25, .35]})
    files = ['results/index.csv', 'results/blank_index.csv', 'results/prior.json', *[r['file'] for r in rows]]
    manifest = {'protocol_sha256': digest(root/'protocol.json'), 'n': count, 'n_dates': count,
        'n_decide_calls': protocol['call_budget'], 'asset_sha256': {p: digest(root/p) for p in files}}
    write_json(root/'results/preparation.json', manifest)
    return protocol, manifest, rows


def settings(server, limit=None):
    return SimpleNamespace(base_url=f'http://127.0.0.1:{server.server_port}', dry_run=True,
        timeout=2, retries=0, execution_fingerprint='', limit=limit)


class PilotTests(unittest.TestCase):
    def test_prompt_uses_only_causal_sigma_and_combines_exact_text(self):
        protocol = json.loads((EXP/'protocol.json').read_text())
        row = {'sigma': .03, 'label': 'Up', 'forward_return': 100, 'ticker': 'SECRET'}
        first = question(row, protocol)
        row.update(label='Down', forward_return=-100, ticker='DIFFERENT')
        self.assertEqual(first, question(row, protocol))
        self.assertIn('threshold=0.01', first)
        self.assertNotIn('SECRET', first)
        self.assertEqual(context('combined'), '\n\n'.join([(EXP/'context'/name).read_text().strip()
            for name in ('candles.txt', 'formations.txt', 'shared.txt')]))
        self.assertEqual(context('combined'), context('combined_blank'))
        for bad in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                question({'sigma': bad}, protocol)

    def test_selection_does_not_use_future_prices_or_labels(self):
        dates = pd.bdate_range('2018-01-01', '2020-12-31')
        frame = pd.DataFrame({'Close': np.arange(len(dates))+100}, index=dates)
        raw = {ticker: frame.copy() for ticker in ('SPY', 'A', 'B', 'C')}
        cfg = {'sample': {'seed': 71, 'n_bars': 100, 'tickers_per_date': 3,
               'earliest_test_input': '2019-04-01', 'date_quotas': {'2019': 2, '2020': 2}},
               'target': {'horizon_sessions': 5}}
        first = prep.select_observations(raw, cfg)
        for value in raw.values():
            value.Close = np.linspace(1e6, 1, len(value))
        self.assertEqual(first, prep.select_observations(raw, cfg))
        self.assertEqual(len(first), 12)
        self.assertEqual(len(set(first)), 12)

    def test_render_has_two_causal_panels_no_identifying_metadata(self):
        protocol = json.loads((EXP/'protocol.json').read_text())
        dates = pd.bdate_range('2020-01-01', periods=100)
        close = 100 + np.sin(np.arange(100)/4) + np.arange(100)*.05
        frame = pd.DataFrame({'Open': close-.1, 'High': close+.3, 'Low': close-.4,
                              'Close': close, 'Volume': np.arange(100)+1000}, index=dates)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'chart.png'
            prep.render_panels(frame, path, protocol)
            with Image.open(path) as image:
                self.assertEqual(image.size, (1024, 1024))
                self.assertEqual(image.info, {})
                self.assertGreater(np.asarray(image)[:640].std(), 0)
                self.assertGreater(np.asarray(image)[640:].std(), 0)
            with self.assertRaises(ValueError):
                prep.render_panels(frame.iloc[:-1], path, protocol)

    def test_full_fake_pipeline_resume_and_blank_control(self):
        server = FakeServer(reverse=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d, redirect_stdout(StringIO()):
                root = Path(d)
                protocol, manifest, rows = fixture(root)
                for arm in ARMS:
                    scorer.run_arm(root, protocol, manifest, arm, settings(server))
                self.assertEqual(len(server.requests), protocol['call_budget'])
                before = len(server.requests)
                for arm in ARMS:
                    scorer.run_arm(root, protocol, manifest, arm, settings(server))
                self.assertEqual(len(server.requests), before)
                for request in server.requests:
                    self.assertEqual(request['options'], OPTIONS)
                    self.assertEqual(request['thinking'], 'off')
                    self.assertNotIn('PRIVATE_TICKER', json.dumps(request))
                    self.assertNotIn('2021-04-', json.dumps(request))
                    self.assertNotIn('0.01234', request['question'])
                data = base64.b64decode(server.requests[-1]['state'][1]['image'].split(',')[1])
                self.assertEqual(Image.open(BytesIO(data)).getextrema(), ((255, 255),)*3)
                result = evaluation.evaluate(root, synthetic=True)
                self.assertFalse(result['scientific_result'])
                self.assertFalse(result['continuation_criteria_met'])
                self.assertEqual(result['n'], 24)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    evaluation.evaluate(root, synthetic=False)
                path = root/'results/dry_run/combined.csv'
                path.write_text(path.read_text()+'\n')
                with self.assertRaisesRegex(ValueError, 'altered prediction'):
                    evaluation.evaluate(root, synthetic=True)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_failed_gate_blocks_prediction_rows(self):
        server = FakeServer(drift=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d); protocol, manifest, _ = fixture(root)
                with self.assertRaisesRegex(ValueError, 'Parity failed'):
                    scorer.run_arm(root, protocol, manifest, 'combined', settings(server))
                self.assertFalse((root/'results/dry_run/combined.csv').exists())
                self.assertTrue((root/'results/dry_run/combined_errors.csv').exists())
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_bundle_rejects_image_or_protocol_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); protocol, manifest, rows = fixture(root)
            verify_bundle(root)
            (root/rows[0]['file']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                verify_bundle(root)
            protocol['target']['flat_band'] = .5
            write_json(root/'protocol.json', protocol)
            with self.assertRaisesRegex(ValueError, 'protocol changed'):
                verify_bundle(root)

    def test_paired_brier_direction_and_constant_interval(self):
        cfg = {'evaluation': {'resamples': 50, 'seed': 71, 'block_lengths': [1, 2, 4, 8]}}
        good = np.tile([1., 0., 0.], (24, 1)); bad = np.tile([0., 0., 1.], (24, 1))
        result = evaluation.paired(good, bad, ['Up']*24, [str(i) for i in range(24)], cfg)
        self.assertEqual(result['brier_reduction'], 2)
        self.assertEqual(result['ci95']['4'], [2, 2])

    def test_plan_never_starts_server(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(StringIO()):
            protocol, manifest, _ = fixture(Path(d))
            with patch.object(gpu, 'verify_bundle', return_value=(protocol, manifest)), patch.object(gpu.subprocess, 'Popen') as launch:
                gpu.execute(SimpleNamespace(execute=False))
                launch.assert_not_called()

    def test_live_limit_and_missing_runtime_fingerprint_are_rejected(self):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                scorer.arguments(['--base-url', 'https://example.com', '--limit', '10'])
            with self.assertRaises(SystemExit):
                scorer.arguments(['--base-url', 'https://example.com'])


if __name__ == '__main__':
    unittest.main()
