"""CPU preparation from verified, cached real data; no downloads or inference."""
from __future__ import annotations
from collections import Counter
import json
import importlib.metadata
import platform
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, digest, verify_bundle, verify_inputs, write_json
from shared.src.data import load, universe
from shared.src.forecast_bridge import target_at
from shared.src.render import render_chart


def ranked(values, seed, salt):
    import hashlib
    return sorted(values, key=lambda x: hashlib.sha256(f'{seed}:{salt}:{x}'.encode()).hexdigest())


def weekly_positions(frame):
    # Holiday-shortened weeks still use their actual last observed session.
    return frame.groupby(frame.index.to_period('W-FRI')).tail(1).index


def select_observations(raw, protocol):
    """Choose dates/tickers without reading future price values or class labels."""
    cfg = protocol['sample']
    reference = raw['SPY']
    dates = []
    for stamp in weekly_positions(reference):
        i = reference.index.get_loc(stamp)
        if i < cfg['n_bars'] - 1 or i + protocol['target']['horizon_sessions'] >= len(reference):
            continue
        if str(reference.index[i - cfg['n_bars'] + 1].date()) < cfg['earliest_test_input']:
            continue
        dates.append(str(stamp.date()))
    selected = []
    for year, quota in cfg['date_quotas'].items():
        pool = [date for date in dates if date.startswith(year)]
        if len(pool) < quota:
            raise ValueError(f'Insufficient forecast dates for {year}')
        selected.extend(ranked(pool, cfg['seed'], f'dates:{year}')[:quota])
    observations = []
    for date in sorted(selected):
        eligible = []
        for ticker, frame in raw.items():
            if pd.Timestamp(date) not in frame.index:
                continue
            i = frame.index.get_loc(pd.Timestamp(date))
            if (i >= cfg['n_bars'] - 1 and i + protocol['target']['horizon_sessions'] < len(frame)
                    and str(frame.index[i-cfg['n_bars']+1].date()) >= cfg['earliest_test_input']):
                eligible.append(ticker)
        if len(eligible) < cfg['tickers_per_date']:
            raise ValueError(f'Insufficient tickers on {date}')
        for ticker in ranked(eligible, cfg['seed'], f'tickers:{date}')[:cfg['tickers_per_date']]:
            observations.append((ticker, date))
    return observations


def render_panels(frame, output, protocol):
    cfg = protocol['sample']
    if len(frame) != cfg['n_bars']:
        raise ValueError('Expected exactly 100 visible bars')
    with tempfile.TemporaryDirectory() as temporary:
        top, bottom = Path(temporary)/'top.png', Path(temporary)/'bottom.png'
        render_chart(frame, top, n_bars=cfg['n_bars'], size=(1024, 640), volume_color='neutral')
        render_chart(frame.tail(cfg['zoom_bars']), bottom, n_bars=cfg['zoom_bars'], size=(1024, 384), volume_color='neutral')
        image = Image.new('RGB', (1024, 1024), 'white')
        with Image.open(top) as pane:
            image.paste(pane, (0, 0))
        with Image.open(bottom) as pane:
            image.paste(pane, (0, 640))
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format='PNG')


def prior_counts(raw, protocol):
    counts = Counter()
    latest_target = ''
    for ticker, frame in raw.items():
        for stamp in weekly_positions(frame):
            if str(stamp.date()) > protocol['prior']['end_date']:
                continue
            target, reason = target_at(frame, frame.index.get_loc(stamp), protocol)
            if reason or target['target_date'] > protocol['prior']['end_date']:
                continue
            counts[target['label']] += 1
            latest_target = max(latest_target, target['target_date'])
    if not counts or not all(counts[option] for option in protocol['target']['options']):
        raise ValueError('Training prior must have all classes')
    return {'counts': dict(counts), 'n': sum(counts.values()), 'latest_target': latest_target,
            'probabilities': [counts[o]/sum(counts.values()) for o in protocol['target']['options']]}


def prepare(root=EXP):
    root = Path(root)
    if (root/'results/preparation.json').exists():
        _, manifest = verify_bundle(root)
        print('Verified existing frozen bundle; no rerendering or outcome recomputation')
        return manifest
    protocol = json.loads((root/'protocol.json').read_text())
    verify_inputs(protocol, root.parent)
    cfg = protocol['sample']
    raw = {ticker: load(ticker) for ticker in universe()['tickers']}
    observations = select_observations(raw, protocol)
    if len(observations) != cfg['n_rows']:
        raise ValueError('Unexpected cohort size')
    # Select first; only now derive held-out outcomes.
    rows = []
    for ticker, date in observations:
        frame = raw[ticker]
        i = frame.index.get_loc(pd.Timestamp(date))
        target, reason = target_at(frame, i, protocol)
        if reason:
            raise ValueError(f'Frozen observation is ineligible: {ticker} {date}: {reason}')
        visible = frame.iloc[i-cfg['n_bars']+1:i+1]
        identifier = __import__('hashlib').sha256(f'{ticker}:{date}'.encode()).hexdigest()[:20]
        file = f'charts/{identifier}.png'
        render_panels(visible, root/file, protocol)
        rows.append({'file': file, 'ticker': ticker, 'end_date': date,
                     'window_start': str(visible.index[0].date()), 'split': 'test',
                     'n_bars': cfg['n_bars'], 'data_origin': 'market',
                     'evidence_role': 'exploratory_forecast', 'accuracy_valid': True, **target})
    index = pd.DataFrame(rows).sort_values(['end_date', 'ticker']).reset_index(drop=True)
    blank = []
    for date, group in index.groupby('end_date', sort=True):
        ids = ranked(group.file.tolist(), cfg['seed'], f'blank:{date}')[:cfg['blank_per_date']]
        blank.extend(ids)
    blank_index = index[index.file.isin(blank)].copy()
    root.joinpath('results').mkdir(exist_ok=True)
    index.to_csv(root/'results/index.csv', index=False)
    blank_index.to_csv(root/'results/blank_index.csv', index=False)
    prior = prior_counts(raw, protocol)
    if prior['latest_target'] >= index.window_start.min():
        raise ValueError('Training prior targets overlap test information')
    write_json(root/'results/prior.json', prior)
    samples = ranked(index.file.tolist(), cfg['seed'], 'samples')[:3]
    for number, file in enumerate(samples, 1):
        destination = root/f'results/samples/sample_{number}.png'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/file, destination)
    assets = ['results/index.csv', 'results/blank_index.csv', 'results/prior.json', *index.file.tolist(),
              *[f'results/samples/sample_{i}.png' for i in range(1, 4)]]
    manifest = {'protocol_sha256': digest(root/'protocol.json'), 'n': len(index),
                'n_dates': index.end_date.nunique(), 'n_tickers': index.ticker.nunique(),
                'date_range': [index.end_date.min(), index.end_date.max()],
                'label_counts': index.label.value_counts().to_dict(), 'blank_n': len(blank_index),
                'n_decide_calls': 4*len(index)+len(blank_index)+5*40,
                'asset_sha256': {p: digest(root/p) for p in assets},
                'sample_sources': samples, 'inference_run': False,
                'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name)
                    for name in ('numpy', 'pandas', 'matplotlib', 'pillow', 'pyarrow')},
                'selection': 'Fixed seed/year quotas; dates/tickers chosen before outcome derivation; no pattern or target balancing'}
    write_json(root/'results/preparation.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'asset_sha256'}, indent=2))
    return manifest


if __name__ == '__main__':
    prepare()
