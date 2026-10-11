"""Native System 1 direction calls with row-specific, causal Flat thresholds."""
from __future__ import annotations
import argparse
import base64
import csv
from datetime import datetime, timezone
import fcntl
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, OPTIONS, ARMS, context, digest, question, verify_bundle, write_json
from shared.src.score import (Client, OUTPUT_FIELDS, ERROR_FIELDS, append_csv, canonical_hash,
                              load_index, local_path, parity_check, resume_rows)
from shared.src.jev_api import build_payload, parse_probabilities, parity_policy, MODEL


class ForecastClient(Client):
    def score(self, row):
        started = time.perf_counter()
        with Image.open(local_path(self.args.root, row['file'])) as source:
            image = source.convert('RGB')
        if self.args.control == 'blank':
            image = Image.new('RGB', image.size, 'white')
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        payload = build_payload(url, OPTIONS, question(row, self.args.protocol), self.args.context)
        response, attempts = self.http('/v1/decide', payload)
        return parse_probabilities(response, OPTIONS, self.identity), round((time.perf_counter()-started)*1000, 3), attempts


def run_arm(root, protocol, manifest, arm, settings):
    root = Path(root)
    name = 'blank_index.csv' if arm == 'combined_blank' else 'index.csv'
    fields, rows = load_index(root/'results'/name, root, OPTIONS)
    origin = 'dry_run' if settings.dry_run else 'inference/v1'
    out = root/f'results/{origin}/{arm}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    fields += OUTPUT_FIELDS
    args = SimpleNamespace(root=root, options=OPTIONS, base_url=settings.base_url, dry_run=settings.dry_run,
        model_revision='a'*40 if settings.dry_run else protocol['model_revision'], served_model_name=MODEL,
        context=context(arm, root), protocol=protocol, control='blank' if arm == 'combined_blank' else 'image',
        reference_image=[], reference_caption='', timeout=settings.timeout, retries=settings.retries,
        concurrency=1, batch_size=1, execution_fingerprint=settings.execution_fingerprint)
    (root/'.cache').mkdir(exist_ok=True)
    with (root/f'.cache/{arm}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client = ForecastClient(args)
        identity = {'protocol_sha256': digest(root/'protocol.json'), 'preparation_sha256': digest(root/'results/preparation.json'),
            'index_sha256': digest(root/'results'/name), 'context': args.context,
            'question_template': protocol['question_template'], 'arm': arm, 'options': OPTIONS,
            'server_identity': client.identity, 'endpoint': client.base, 'synthetic': settings.dry_run,
            'prediction_origin': 'fake_server' if settings.dry_run else 'jev', 'parity_policy': parity_policy(),
            'execution': {'concurrency': 1, 'batch_size': 1, 'runtime_fingerprint': settings.execution_fingerprint},
            'source_sha256': digest(Path(__file__)), 'control': args.control}
        fingerprint = canonical_hash(identity)
        meta_path = out.with_suffix('.meta.json')
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            if old != {**identity, 'fingerprint': fingerprint}:
                raise ValueError('Resume provenance changed; use a new frozen version')
        elif out.exists():
            raise ValueError('Predictions lack provenance')
        else:
            write_json(meta_path, {**identity, 'fingerprint': fingerprint})
        original = {row['file']: row for row in rows}
        done, repaired = resume_rows(out, fields, original, OPTIONS)
        pending = [row for row in rows if row['file'] not in done]
        if settings.limit is not None:
            pending = pending[:settings.limit]
        errors = out.with_name(out.stem+'_errors.csv')
        if not pending:
            return {'arm': arm, 'scored': 0, 'skipped': len(done), 'synthetic': settings.dry_run}
        try:
            gate = parity_check(client, rows, out.with_suffix('.parity.json'), fingerprint, force=True)
        except Exception as error:
            append_csv(errors, ERROR_FIELDS, {'file': '', 'phase': 'parity', 'attempts': getattr(error, 'attempts', 1),
                'error': str(error), 'timestamp_utc': datetime.now(timezone.utc).isoformat()})
            raise
        scored = failed = 0
        for row in pending:
            try:
                probabilities, latency, _ = client.score(row)
                append_csv(out, fields, {**row, 'probabilities': json.dumps(probabilities),
                    'prediction': OPTIONS[max(range(3), key=probabilities.__getitem__)], 'latency_ms': latency})
                scored += 1
            except Exception as error:
                failed += 1
                append_csv(errors, ERROR_FIELDS, {'file': row['file'], 'phase': 'score',
                    'attempts': getattr(error, 'attempts', 1), 'error': str(error),
                    'timestamp_utc': datetime.now(timezone.utc).isoformat()})
        summary = {'arm': arm, 'scored': scored, 'failed': failed, 'skipped': len(done), 'repaired': repaired,
            'output_sha256': digest(out) if out.exists() else None,
            'synthetic': settings.dry_run, 'repeatability': {key: gate[key] for key in
                ('passed', 'mean_abs_diff', 'max_abs_diff', 'p90_abs_diff', 'flagged_files',
                 'argmax_agreement_count', 'near_tie_chart_count', 'allowed_argmax_mismatch_count')}}
        write_json(out.with_suffix('.summary.json'), summary)
        if failed:
            raise RuntimeError(f'{failed} scoring failures in {arm}; resume after addressing errors')
        return summary


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=EXP)
    p.add_argument('--base-url', default=os.getenv('JEV_BASE_URL', ''))
    p.add_argument('--execution-fingerprint', default='')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--retries', type=int, default=3)
    args = p.parse_args(argv)
    if not args.base_url:
        p.error('Supply the existing JEV_BASE_URL or --base-url')
    if args.limit is not None and (args.limit < 1 or not args.dry_run):
        p.error('--limit is allowed only for a synthetic dry run')
    if args.timeout <= 0 or args.retries < 0:
        p.error('Invalid timeout/retries')
    if not args.dry_run and (len(args.execution_fingerprint) != 64 or
                            any(c not in '0123456789abcdef' for c in args.execution_fingerprint)):
        p.error('Live scoring requires the runtime fingerprint from scripts/run_gpu.py')
    return args


def main(argv=None):
    settings = arguments(argv)
    protocol, manifest = verify_bundle(settings.root)
    for arm in ARMS:
        print(json.dumps(run_arm(settings.root, protocol, manifest, arm, settings)), flush=True)


if __name__ == '__main__':
    main()
