"""Resumable arbitrary-option JEV scoring with a mandatory 20-image repeatability gate."""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from PIL import Image, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.src.jev_api import build_payload, options_list, probabilities, server_identity, parse_probabilities, PARITY_ATOL, MODEL

OUTPUT_FIELDS = ["probabilities", "prediction", "latency_ms"]


class RequestFailure(RuntimeError):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts

ERROR_FIELDS = ["file", "phase", "attempts", "error", "timestamp_utc"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def local_path(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("File must remain inside the experiment root")
    return path


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def append_csv(path, fields, row):
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush(); os.fsync(stream.fileno())


def load_index(path, root, options):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not {'file', 'label'}.issubset(fields) or set(fields) & set(OUTPUT_FIELDS) or not rows:
        raise ValueError("Index needs file/label rows and no reserved output columns")
    if len(fields) != len(set(fields)) or any(None in row or None in row.values() for row in rows):
        raise ValueError("Malformed index CSV")
    if len({r['file'] for r in rows}) != len(rows):
        raise ValueError("Duplicate chart file identifiers")
    for row in rows:
        if row['label'] not in options:
            raise ValueError("Index label is absent from requested options")
        with Image.open(local_path(root, row['file'])) as im:
            if im.format != 'PNG':
                raise ValueError("Index files must be PNG")
            im.verify()
    return fields, rows


def base_endpoint(base, dry_run):
    parsed = urlsplit(base)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use an HTTP(S) URL without credentials, query or fragment")
    loopback = parsed.hostname in ('localhost', '127.0.0.1', '::1')
    if dry_run and not loopback:
        raise ValueError("--dry-run only permits a loopback fake server")
    if parsed.scheme != 'https' and not loopback:
        raise ValueError("Remote inference requires HTTPS")
    if parsed.path.rstrip('/') not in ('', '/v1', '/v1/decide'):
        raise ValueError("Expected a base URL or /v1/decide endpoint")
    return f'{parsed.scheme}://{parsed.netloc}'


class Client:
    def __init__(self, args):
        self.args = args
        self.base = base_endpoint(args.base_url, args.dry_run)
        self.api_key = '' if args.dry_run else os.environ.get('JEV_API_KEY', '')
        self.references = []
        for path in args.reference_image:
            with Image.open(path) as reference:
                if reference.format != 'PNG':
                    raise ValueError('Reference images must be PNG')
                buffer = BytesIO(); reference.convert('RGB').save(buffer, format='PNG')
            self.references.append({'image': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(),
                                    'caption': args.reference_caption})
        info, _ = self.http('/v1/decide/info')
        if info.get('synthetic') and not args.dry_run:
            raise ValueError("Fake server requires --dry-run")
        self.identity = server_identity(info, args.model_revision, expected_model=args.served_model_name)

    def http(self, path, payload=None):
        for attempt in range(self.args.retries + 1):
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = 'Bearer ' + self.api_key
            request = Request(self.base + path, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
            try:
                with urlopen(request, timeout=self.args.timeout) as response:
                    value = json.loads(response.read())
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object")
                return value, attempt + 1
            except HTTPError as error:
                retry = error.code in (408, 429) or error.code >= 500
                message = f'HTTP {error.code}'
            except (URLError, TimeoutError, OSError) as error:
                retry, message = True, type(error).__name__
            if not retry or attempt == self.args.retries:
                raise RequestFailure(f'{message}; attempts={attempt + 1}', attempt + 1)
            time.sleep(min(.2 * 2 ** attempt, 2))
        raise AssertionError('unreachable')

    def score(self, row):
        started = time.perf_counter()
        with Image.open(local_path(self.args.root, row['file'])) as source:
            image = source.convert('RGB')
        if self.args.control == 'mirror':
            image = ImageOps.mirror(image)
        elif self.args.control == 'blank':
            image = Image.new('RGB', image.size, 'white')
        buffer = BytesIO(); image.save(buffer, format='PNG')
        url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        payload = build_payload(url, self.args.options, self.args.question, self.args.context, self.references)
        response, attempts = self.http('/v1/decide', payload)
        p = parse_probabilities(response, self.args.options, self.identity)
        return p, round((time.perf_counter() - started) * 1000, 3), attempts


def selected_parity(rows):
    if len(rows) < 20:
        raise ValueError("The parity gate requires at least 20 indexed charts")
    ordered = sorted(rows, key=lambda r: r['file'])
    return [ordered[(len(rows) - 1) * i // 19] for i in range(20)]


def parity_check(client, rows, path, fingerprint, force=False):
    fixtures = selected_parity(rows)
    ids = [r['file'] for r in fixtures]
    if path.exists() and not force:
        report = json.loads(path.read_text())
        if report.get('fingerprint') != fingerprint or report.get('files') != ids:
            raise ValueError("Parity provenance differs; use a fresh output")
    else:
        # Finish all 20 first-pass calls before the second pass.
        first = [client.score(row)[0] for row in fixtures]
        second = [client.score(row)[0] for row in fixtures]
        report = {'fingerprint': fingerprint, 'files': ids, 'first': first, 'second': second,
                  'atol': PARITY_ATOL, 'synthetic': client.args.dry_run,
                  'note': 'Repeatability only; this does not establish HF-vLLM parity or model accuracy'}
    if len(report.get('first', [])) != 20 or len(report.get('second', [])) != 20:
        raise ValueError("Incomplete parity result")
    first = [probabilities(p, len(client.args.options)) for p in report['first']]
    second = [probabilities(p, len(client.args.options)) for p in report['second']]
    maximum = max(abs(x-y) for a,b in zip(first,second) for x,y in zip(a,b))
    report.update(max_abs_diff=maximum, passed=maximum <= PARITY_ATOL, atol=PARITY_ATOL)
    write_json(path, report)
    if maximum > PARITY_ATOL:
        raise ValueError(f'Parity failed: max probability difference {maximum:.6f} > {PARITY_ATOL}; scoring blocked')
    return report


def resume_rows(path, fields, original, options):
    if not path.exists():
        return set(), 0
    kept, dropped, seen = [], 0, set()
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError("Saved prediction columns differ")
        saved = list(reader)
        for position, row in enumerate(saved):
            # A crash can truncate the first field too, before it identifies a file.
            if None in row or None in row.values():
                if position != len(saved) - 1:
                    raise ValueError('Malformed interior prediction record')
                dropped += 1; continue
            file = row.get('file')
            if file not in original or file in seen:
                raise ValueError("Saved file missing from index or duplicated")
            if any(row[k] != v for k,v in original[file].items()):
                raise ValueError("Saved observation identity differs from index")
            try:
                p = probabilities(json.loads(row['probabilities']), len(options))
                if row['prediction'] != options[max(range(len(p)), key=p.__getitem__)] or not math.isfinite(float(row['latency_ms'])) or float(row['latency_ms']) < 0:
                    raise ValueError('Invalid saved forecast')
            except (ValueError, TypeError, json.JSONDecodeError):
                dropped += 1; continue
            seen.add(file); kept.append(row)
    if dropped:
        tmp = path.with_suffix('.repair.csv')
        with tmp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(kept)
        tmp.replace(path)
    return seen, dropped


def run(args):
    args.root = Path(args.root).resolve()
    index = local_path(args.root, args.index)
    out = local_path(args.root, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    args.options = options_list(args.options)
    fields, rows = load_index(index, args.root, args.options)
    fields += OUTPUT_FIELDS
    original = {r['file']: r for r in rows}
    meta_path, parity_path = out.with_suffix('.meta.json'), out.with_suffix('.parity.json')
    errors_path = out.with_name(out.stem + '_errors.csv')
    lockdir = args.root / '.cache'; lockdir.mkdir(exist_ok=True)
    with (lockdir / (hashlib.sha256(str(out).encode()).hexdigest() + '.lock')).open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another scorer owns this output") from error
        client = Client(args)
        identity = {'index_sha256': digest(index), 'image_sha256': {r['file']:digest(local_path(args.root,r['file'])) for r in rows},
            'options': args.options, 'question': args.question, 'context': args.context, 'control': args.control,
            'reference_images': [{'sha256':digest(p),'caption':args.reference_caption} for p in args.reference_image],
            'server_identity': client.identity, 'endpoint': client.base, 'synthetic': args.dry_run,
            'prediction_origin': 'fake_server' if args.dry_run else 'jev',
            'synthetic_dataset': all(bool(r.get('template_id')) and not r.get('end_date') for r in rows),
            'source_sha256': {p.name:digest(p) for p in (Path(__file__), Path(__file__).with_name('jev_api.py'))}}
        fingerprint = canonical_hash(identity)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get('fingerprint') != fingerprint or canonical_hash({k:v for k,v in meta.items() if k != 'fingerprint'}) != fingerprint:
                raise ValueError("Resume provenance changed; use a fresh output path")
        elif out.exists():
            raise ValueError("Existing predictions have no provenance metadata")
        else:
            write_json(meta_path, {**identity, 'fingerprint': fingerprint})
        done, repaired = resume_rows(out, fields, original, args.options)
        pending = [r for r in rows if r['file'] not in done]
        if args.limit is not None:
            pending = pending[:args.limit]
        try:
            # A URL/weight revision cannot identify a running server process.
            # Recheck on every invocation that will score, including a resume.
            parity = parity_check(client, rows, parity_path, fingerprint, force=args.parity_only or bool(pending))
        except Exception as error:
            append_csv(errors_path, ERROR_FIELDS, {'file':'', 'phase':'parity', 'attempts':getattr(error,'attempts',1), 'error':str(error), 'timestamp_utc':datetime.now(timezone.utc).isoformat()})
            raise
        if args.parity_only:
            print(json.dumps({'parity_passed':True, 'charts':20, 'max_abs_diff':parity['max_abs_diff'], 'synthetic':args.dry_run})); return 0
        succeeded, failed = 0, 0
        def process(row):
            try: return row, client.score(row), None
            except Exception as error: return row, None, error
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for start in range(0,len(pending),args.batch_size):
                for row, scored, error in executor.map(process, pending[start:start+args.batch_size]):
                    if error is not None:
                        failed += 1
                        append_csv(errors_path, ERROR_FIELDS, {'file':row['file'],'phase':'score','attempts':getattr(error,'attempts',1),'error':str(error),'timestamp_utc':datetime.now(timezone.utc).isoformat()})
                    else:
                        p, latency, attempts = scored
                        append_csv(out, fields, {**row, 'probabilities':json.dumps(p),
                            'prediction':args.options[max(range(len(p)),key=p.__getitem__)], 'latency_ms':latency})
                        succeeded += 1
        summary = {'scored':succeeded,'failed':failed,'skipped':len(done),'repaired':repaired,'parity_max_abs_diff':parity['max_abs_diff'], 'synthetic':args.dry_run}
        write_json(out.with_suffix('.summary.json'), summary)
        print(json.dumps(summary))
        return int(failed > 0)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--index', default='charts/index.csv')
    p.add_argument('--out', help='Defaults to results/preds.csv, or results/dry_run/preds.csv for synthetic calls')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--options', nargs='+')
    group.add_argument('--options-file', type=Path)
    p.add_argument('--question', required=True)
    p.add_argument('--context-file', type=Path)
    p.add_argument('--reference-image', type=Path, action='append', default=[])
    p.add_argument('--reference-caption', default='Labelled reference examples; use these only as definitions. Classify the separate target chart.')
    p.add_argument('--control', choices=('image','mirror','blank'), default='image')
    p.add_argument('--base-url', default=os.getenv('JEV_BASE_URL',''))
    p.add_argument('--served-model-name', default=os.getenv('JEV_SERVED_MODEL',MODEL))
    p.add_argument('--model-revision', default=os.getenv('JEV_MODEL_REVISION',''))
    p.add_argument('--batch-size', type=int, default=int(os.getenv('JEV_BATCH_SIZE','16')))
    p.add_argument('--concurrency', type=int, default=int(os.getenv('JEV_CONCURRENCY','4')))
    p.add_argument('--timeout', type=float, default=float(os.getenv('JEV_TIMEOUT','60')))
    p.add_argument('--retries', type=int, default=int(os.getenv('JEV_RETRIES','3')))
    p.add_argument('--limit', type=int)
    p.add_argument('--parity-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    args.out = args.out or ('results/dry_run/preds.csv' if args.dry_run else 'results/preds.csv')
    if args.options_file:
        args.options = json.loads(args.options_file.read_text())
    args.context = args.context_file.read_text() if args.context_file else ''
    if len(args.reference_image) > 7:
        p.error('At most seven reference images plus one target are supported')
    if min(args.batch_size,args.concurrency) < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or args.retries < 0 or (args.limit is not None and args.limit < 1):
        p.error('Invalid batch, concurrency, timeout, retries or limit')
    if not args.base_url:
        p.error('JEV_BASE_URL or --base-url is required')
    return args


def main():
    try:
        return run(arguments())
    except (ValueError, RuntimeError, OSError) as error:
        print(str(error),file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
