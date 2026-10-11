"""Frozen assets and prompts for the direct pattern-guided forecast pilot."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parent
EXP06 = REPO / '06-jev-patterns'
sys.path.insert(0, str(EXP06))
sys.path.insert(0, str(REPO))
OPTIONS = ['Up', 'Flat', 'Down']
ARMS = ['zero_shot', 'candles', 'formations', 'combined', 'combined_blank']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def checked(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Asset escapes experiment root')
    return path


def context(arm, root=EXP):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    parts = []
    if arm in ('candles', 'combined', 'combined_blank'):
        parts.append((Path(root) / 'context/candles.txt').read_text().strip())
    if arm in ('formations', 'combined', 'combined_blank'):
        parts.append((Path(root) / 'context/formations.txt').read_text().strip())
    if parts:
        parts.append((Path(root) / 'context/shared.txt').read_text().strip())
    return '\n\n'.join(parts)


def question(row, protocol):
    # This allowlist is deliberate: truth, ticker, date and pattern labels never
    # enter the request. Sigma was computed using closes available at chart end.
    sigma = float(row['sigma'])
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('Positive finite trailing sigma required')
    band = protocol['target']['flat_band'] * sigma
    return protocol['question_template'].format(sigma=sigma, threshold=band)


def verify_inputs(protocol, repo=REPO):
    for relative, expected in protocol['input_sha256'].items():
        if digest(checked(repo, relative)) != expected:
            raise ValueError(f'Frozen input changed: {relative}')


def verify_bundle(root=EXP):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    verify_inputs(protocol, root.parent)
    manifest = json.loads((root / 'results/preparation.json').read_text())
    if manifest['protocol_sha256'] != digest(root / 'protocol.json'):
        raise ValueError('Prepared protocol changed')
    for relative, expected in manifest['asset_sha256'].items():
        if digest(checked(root, relative)) != expected:
            raise ValueError(f'Prepared asset changed: {relative}')
    return protocol, manifest
