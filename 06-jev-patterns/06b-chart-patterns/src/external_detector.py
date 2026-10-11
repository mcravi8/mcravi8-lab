"""Pinned unmodified stock-pattern functions, cached outside version control.

The upstream source is GPL-3.0. This adapter downloads and verifies it; no
upstream source is copied into this repository or into the second detector.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULES = json.loads((ROOT / 'config/real_rules.json').read_text())
CACHE = ROOT.parent / '.cache/stock-pattern'
FUNCTIONS = {'Head and Shoulders': 'find_hns', 'Inverse Head and Shoulders': 'find_reverse_hns',
             'Double Top': 'find_double_top', 'Double Bottom': 'find_double_bottom',
             'Triangle': 'find_triangles', 'Bull Flag': 'find_bullish_flag'}


def load_upstream(fetch=False):
    expected = RULES['external']
    for name, key in [('src/utils.py', 'source_sha256'), ('LICENSE', 'license_sha256')]:
        path = CACHE / name
        if not path.exists() and fetch:
            url = f"https://raw.githubusercontent.com/{expected['repository']}/{expected['revision']}/{name}"
            data = urlopen(url, timeout=60).read()
            if hashlib.sha256(data).hexdigest() != expected[key]:
                raise ValueError('Downloaded upstream file checksum mismatch')
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected[key]:
            raise ValueError(f'Pinned upstream source missing or changed: {path}; use --fetch')
    name = 'exp06_stock_pattern_pinned'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, CACHE / 'src/utils.py')
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def cached_pivots(frame):
    """Exact vectorized equivalent of pinned get_max_min(6,6).

    Upstream's center is index7 of13 (not6). A pivot is only available after
    its five right-hand observations. Window slicing enforces both edges.
    """
    n = len(frame); high = frame.High.to_numpy(); low = frame.Low.to_numpy()
    records = []
    if n >= 13:
        for values, kind in [(high, 'high'), (low, 'low')]:
            windows = np.lib.stride_tricks.sliding_window_view(values, 13)
            offset = windows.argmax(1) if kind == 'high' else windows.argmin(1)
            positions = np.flatnonzero(offset == 7) + 7
            records.extend((int(i), kind, float(values[i]), float(frame.Volume.iloc[i])) for i in positions)
    return sorted(records, key=lambda row: (row[0], row[1] != 'high'))


def pivot_frame(records, frame, start, end, only_low=False):
    selected = [r for r in records if start + 7 <= r[0] <= end - 5 and (not only_low or r[1] == 'low')]
    return pd.DataFrame({'P': [r[2] for r in selected], 'V': [r[3] for r in selected]},
                        index=pd.DatetimeIndex([frame.index[r[0]] for r in selected]))


def atr(frame, window=20):
    previous = frame.Close.shift()
    tr = pd.concat([frame.High-frame.Low, (frame.High-previous).abs(), (frame.Low-previous).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean().to_numpy()


def _line(points, first, second, index):
    p, q = points[first], points[second]
    i, j = int(index.get_loc(p[0])), int(index.get_loc(q[0]))
    slope = (q[1]-p[1])/(j-i)
    return float(slope), float(p[1]-slope*i)


def boundaries(result, label, index):
    p = result['points']
    if label in ('Head and Shoulders', 'Inverse Head and Shoulders'):
        m, b = _line(p, 'B', 'D', index)
        return [(-1 if label == 'Head and Shoulders' else 1, m, b)]
    if label in ('Double Top', 'Double Bottom', 'Bull Flag'):
        return [(-1 if label == 'Double Top' else 1, 0., float(p['B'][1]))]
    return [(1, *_line(p, 'A', 'C', index)), (-1, *_line(p, 'B', 'D', index))]


def scan(frame, ticker):
    """Scan every historical prefix and then confirm within available final bars."""
    upstream = load_upstream(); config = RULES['external']
    pivots = cached_pivots(frame); volatility = atr(frame)
    confirmed, errors = {}, []
    for end in range(config['lookback'] - 1, len(frame)):
        start = end-config['lookback']+1
        visible = frame.iloc[start:end+1]
        both = pivot_frame(pivots, frame, start, end)
        low = pivot_frame(pivots, frame, start, end, True)
        for label, function in FUNCTIONS.items():
            selected = low if label == 'Bull Flag' else both
            if len(selected) < (1 if label == 'Bull Flag' else 5):
                continue
            try:
                result = getattr(upstream, function)(ticker, visible, selected, {'FLAG_MAX_BARS': 5})
                if result is None:
                    continue
                origin = int(frame.index.get_loc(result['start']))
                point_ids = tuple(str(v[0]) for k,v in result['points'].items() if k != list(result['points'])[-1])
                key = (label, point_ids)
                # A candidate confirmed further in the future must not suppress
                # a later proposal that confirms earlier under its causal ATR.
                if key in confirmed and confirmed[key]['completion']<=end+1:
                    continue
                limits = boundaries(result, label, frame.index)
                for q in range(end+1, min(len(frame), end+1+config['completion_search_bars'])):
                    matched = [(direction,m,b) for direction,m,b in limits
                               if direction*(frame.Close.iloc[q]-(m*q+b)) > config['breakout_atr_buffer']*volatility[end]]
                    if matched:
                        direction,m,b = matched[0]
                        event={'label':label,'start':origin,'completion':q,'detected_at':end,
                                       'input_start':start,'direction':direction,'slope':m,'intercept':b,
                                       'margin_atr':float(direction*(frame.Close.iloc[q]-(m*q+b))/volatility[end]),
                                       'height_atr':float((visible.High.max()-visible.Low.min())/volatility[end]),
                                       'source_points':{k:[int(frame.index.get_loc(v[0])),float(v[1])] for k,v in result['points'].items()}}
                        if key not in confirmed or q<confirmed[key]['completion']:
                            confirmed[key]=event
                        break
            except (IndexError, KeyError, ValueError, AssertionError, ZeroDivisionError) as exc:
                # Errors are reported and affected windows are excluded, never None.
                errors.append({'ticker':ticker,'end':end,'label':label,'error':type(exc).__name__+': '+str(exc)})
    return sorted(confirmed.values(),key=lambda e:(e['completion'],e['label'],e['start'])), errors


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--fetch', action='store_true');args=p.parse_args()
    module=load_upstream(args.fetch)
    print(json.dumps({'revision':RULES['external']['revision'],'license':RULES['external']['license'],
                      'functions':{k:hasattr(module,v) for k,v in FUNCTIONS.items()}}))
