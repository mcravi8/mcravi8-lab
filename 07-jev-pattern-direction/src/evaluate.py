"""Paired direct-forecast evaluation; refuses incomplete or synthetic live results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, OPTIONS, ARMS, context, digest, verify_bundle, write_json
from shared.src.evaluate import evaluate_arrays
from shared.src.jev_api import probabilities, parity_policy
from shared.src.score import canonical_hash, parity_statistics
from research.statistics import block_bootstrap_means, percentile_ci


def paired(left, right, truth, dates, protocol):
    y = np.eye(3)[[OPTIONS.index(v) for v in truth]]
    delta = ((right-y)**2).sum(1) - ((left-y)**2).sum(1)
    cfg = protocol['evaluation']
    return {'brier_reduction': float(delta.mean()), 'direction': 'positive means left improves',
            'ci95': {str(block): percentile_ci(block_bootstrap_means(delta, dates,
                resamples=cfg['resamples'], seed=cfg['seed'], block_length=block)[:, 0])
                for block in cfg['block_lengths']}}


def aligned(path, expected, root, protocol, arm, synthetic):
    summary = json.loads(path.with_suffix('.summary.json').read_text())
    if summary['failed'] or summary['output_sha256'] != digest(path):
        raise ValueError('Failed or altered prediction artifact')
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    if canonical_hash({k: v for k, v in meta.items() if k != 'fingerprint'}) != meta['fingerprint']:
        raise ValueError('Prediction metadata fingerprint differs')
    if meta['synthetic'] is not synthetic or meta['prediction_origin'] != ('fake_server' if synthetic else 'jev'):
        raise ValueError('Fake/live result provenance differs')
    if meta['protocol_sha256'] != digest(root/'protocol.json') or meta['preparation_sha256'] != digest(root/'results/preparation.json'):
        raise ValueError('Prediction freeze differs')
    if (meta['arm'] != arm or meta['options'] != OPTIONS or meta['context'] != context(arm, root)
            or meta['question_template'] != protocol['question_template']):
        raise ValueError('Prediction arm/prompt differs')
    gate = json.loads(path.with_suffix('.parity.json').read_text())
    if not gate['passed'] or gate['parity_policy'] != parity_policy() or gate['fingerprint'] != meta['fingerprint']:
        raise ValueError('Missing or failed repeatability gate')
    if not parity_statistics(gate['first'], gate['second'], gate['files'], 3)['passed']:
        raise ValueError('Stored repeatability passes do not satisfy policy')
    frame = pd.read_csv(path)
    if frame.file.duplicated().any() or set(frame.file) != set(expected.file):
        raise ValueError('Incomplete or duplicated paired coverage')
    frame = frame.set_index('file').loc[expected.file].reset_index()
    expected = expected.reset_index(drop=True)
    for field in ('label', 'ticker', 'end_date', 'target_date', 'window_start'):
        if not frame[field].astype(str).equals(expected[field].astype(str)):
            raise ValueError(f'Prediction identity/truth changed: {field}')
    p = np.array([probabilities(json.loads(value), 3) for value in frame.probabilities])
    if not np.array_equal(np.array(OPTIONS)[p.argmax(1)], frame.prediction.to_numpy()):
        raise ValueError('Prediction class does not match option probabilities')
    return p, meta, gate


def evaluate(root=EXP, synthetic=False):
    root = Path(root)
    protocol, manifest = verify_bundle(root)
    index = pd.read_csv(root/'results/index.csv')
    blank = pd.read_csv(root/'results/blank_index.csv')
    folder = root/('results/dry_run' if synthetic else 'results/inference/v1')
    arrays, metas, gates = {}, {}, {}
    for arm in ARMS:
        arrays[arm], metas[arm], gates[arm] = aligned(folder/f'{arm}.csv', blank if arm == 'combined_blank' else index, root, protocol, arm, synthetic)
    if len({json.dumps(meta['server_identity'], sort_keys=True) for meta in metas.values()}) != 1:
        raise ValueError('Arms use different model identities')
    if len({meta['execution']['runtime_fingerprint'] for meta in metas.values()}) != 1:
        raise ValueError('Arms use different execution runtimes')
    prior = json.loads((root/'results/prior.json').read_text())
    prior_p = np.tile(prior['probabilities'], (len(index), 1))
    result = {'synthetic': synthetic, 'scientific_result': not synthetic,
        'n': len(index), 'n_dates': index.end_date.nunique(), 'metrics': {}, 'paired': {},
        'repeatability': {arm: {k: gate[k] for k in ('mean_abs_diff', 'max_abs_diff', 'p90_abs_diff',
            'argmax_agreement_count', 'near_tie_chart_count', 'allowed_argmax_mismatch_count')} for arm, gate in gates.items()},
        'inference_note': 'Exploratory, historical adaptive research. Four-week intervals count retained dates, carrying all tickers together; gaps span additional calendar time. No refit uncertainty/multiplicity correction or independent confirmation.'}
    for arm in ARMS[:-1]:
        result['metrics'][arm] = evaluate_arrays(arrays[arm], index.label, OPTIONS)
    result['metrics']['training_prior'] = evaluate_arrays(prior_p, index.label, OPTIONS)
    result['metrics']['uniform'] = evaluate_arrays(np.full((len(index), 3), 1/3), index.label, OPTIONS)
    for other in ('zero_shot', 'candles', 'formations'):
        result['paired'][f'combined_vs_{other}'] = paired(arrays['combined'], arrays[other], index.label, index.end_date, protocol)
    result['paired']['combined_vs_prior'] = paired(arrays['combined'], prior_p, index.label, index.end_date, protocol)
    subset = index.set_index('file').index.get_indexer(blank.file)
    result['metrics']['combined_blank'] = evaluate_arrays(arrays['combined_blank'], blank.label, OPTIONS)
    result['metrics']['combined_control_subset'] = evaluate_arrays(arrays['combined'][subset], blank.label, OPTIONS)
    result['paired']['combined_vs_blank'] = paired(arrays['combined'][subset], arrays['combined_blank'], blank.label, blank.end_date, protocol)
    checks = {
        'primary_combined_vs_zero_shot_ci_positive': result['paired']['combined_vs_zero_shot']['ci95']['4'][0] > 0,
        'combined_brier_below_prior': result['metrics']['combined']['brier'] < result['metrics']['training_prior']['brier'],
        'combined_point_brier_below_both_single_guidance_arms': all(result['metrics']['combined']['brier'] < result['metrics'][arm]['brier'] for arm in ('candles', 'formations')),
        'image_control_ci_positive': result['paired']['combined_vs_blank']['ci95']['4'][0] > 0}
    result['continuation_checks'] = checks
    result['continuation_criteria_met'] = all(checks.values()) and not synthetic
    result['artifact_sha256'] = {str(path.relative_to(root)): digest(path) for path in folder.glob('*') if path.is_file()}
    output = root/('results/dry_run/evaluation' if synthetic else 'results/evaluation_v1')
    write_json(output/'metrics.json', result)
    lines = ['# Direct pattern-guided five-session forecast pilot', '',
        '**SYNTHETIC FIXTURE ONLY — no JEV forecasting result.**' if synthetic else 'Retrospective exploratory evaluation; prompting, not fine-tuning.', '',
        '| Arm | Accuracy | Macro-F1 | Brier ↓ | Log loss ↓ |', '| --- | ---: | ---: | ---: | ---: |']
    for arm, metrics in result['metrics'].items():
        lines.append(f"| {arm} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.3f} | {metrics['brier']:.6f} | {metrics['log_loss']:.6f} |")
    lines += ['', 'Primary comparison: combined versus zero-shot, paired Brier. See metrics.json for intervals, single-guidance comparisons, image control and repeatability.', '', result['inference_note'], '',
              'Continuation criteria: ' + json.dumps(checks), '', 'No execution costs, turnover or trading P&L were evaluated.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic', action='store_true')
    args = p.parse_args()
    result = evaluate(synthetic=args.synthetic)
    print(json.dumps({'n': result['n'], 'synthetic': result['synthetic'], 'continuation_checks': result['continuation_checks']}, indent=2))
