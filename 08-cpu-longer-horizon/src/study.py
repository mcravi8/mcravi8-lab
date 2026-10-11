"""Frozen CPU horizon/history comparison; no downloads, inference or RL."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parent
sys.path.insert(0, str(REPO/'06-jev-patterns'))
sys.path.insert(0, str(REPO))
from shared.src.data import load, universe, validate_ohlcv
from shared.src.baselines import numeric_features, logistic_probabilities
from shared.src.evaluate import evaluate_arrays
from shared.src.forecast_bridge import target_at
from research.statistics import block_bootstrap_means, probability_metrics

OPTIONS = ['Up', 'Flat', 'Down']
ARMS = ['train_prior', 'uniform', 'current_history', 'multiscale_history']
P_COLS = ['p_up', 'p_flat', 'p_down']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def weekly_bars(visible):
    """Only supplied bars enter partial boundary weeks; never fetch a future Friday."""
    return visible.groupby(visible.index.to_period('W-FRI'), sort=True).agg(
        {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})


def window_features(window, prefix, steps):
    grid = numeric_features(window, steps=steps)
    close = window.Close.to_numpy(float)
    changes = close[1:]/close[:-1]-1
    last = window.iloc[-1]
    span = max(float(last.High-last.Low), np.finfo(float).eps*float(last.Close))
    summaries = {
        'return':close[-1]/close[0]-1, 'last_return':changes[-1],
        'bar_return_std':changes.std(ddof=1),
        'range_pct':(window.High.max()-window.Low.min())/close[-1],
        'last_volume_ratio':float(last.Volume)/max(float(window.Volume.mean()),1.),
        'drawdown':float((close/np.maximum.accumulate(close)-1).min()),
        'last_body':(last.Close-last.Open)/span,
        'last_upper_wick':(last.High-max(last.Close,last.Open))/span,
        'last_lower_wick':(min(last.Close,last.Open)-last.Low)/span,
    }
    return {**{f'{prefix}_grid_{i}':float(x) for i,x in enumerate(grid)},
            **{f'{prefix}_{key}':float(x) for key,x in summaries.items()}}


def views_at(frame, position):
    """Current features use <=100 bars; year features add only causal weekly bars."""
    if position < 251 or position >= len(frame):
        raise ValueError('All arms require 252 available sessions on the same row')
    visible = frame.iloc[position-251:position+1]
    short = {}
    for bars, steps in [(5,5), (20,20), (100,16)]:
        short.update(window_features(visible.tail(bars), f'daily{bars}', steps))
    long = window_features(weekly_bars(visible), 'year_weekly', 16)
    return short, long


def market_features(frame, position):
    current, long = views_at(frame, position)
    def select(values, prefix):
        return {f'market_{key}':values[f'{prefix}_{key}']
                for key in ('return', 'bar_return_std', 'range_pct')}
    # Separate names prevent duplicate columns when the year view is appended.
    return ({f'market100_{key}':v for key,v in select(current,'daily100').items()},
            {f'market_year_{key}':v for key,v in select(long,'year_weekly').items()})


def verify_protocol(protocol, repo=REPO):
    root = Path(repo).resolve()
    for relative, expected in protocol['input_sha256'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError(f'Frozen input/source changed: {relative}')
    if sys.version_info[:2] != (3,11):
        raise ValueError('Use the pinned Python 3.11 research environment')
    versions = {}
    for line in (root/'06-jev-patterns/requirements.txt').read_text().splitlines():
        if '==' in line:
            name, expected = line.split('==',1)
            versions[name] = importlib.metadata.version(name)
            if versions[name] != expected:
                raise ValueError(f'Pinned dependency differs: {name}')
    return versions


def build_cohort(raw, protocol):
    """Naturally sampled actual week ends, common rows for both horizons/views."""
    market = raw['SPY']
    for frame in raw.values():
        validate_ohlcv(frame)
        if str(frame.index[-1].date()) > protocol['data_cutoff']:
            raise ValueError('Input exceeds frozen data cutoff')
    dates = market.groupby(market.index.to_period('W-FRI'),sort=True).tail(1).index
    market_lookup = {stamp:i for i,stamp in enumerate(market.index)}
    market_cache = {}
    rows, excluded = [], Counter()
    base_columns, added_columns = None, None
    for ticker, frame in sorted(raw.items()):
        lookup = {stamp:i for i,stamp in enumerate(frame.index)}
        for stamp in dates:
            i, mi = lookup.get(stamp), market_lookup[stamp]
            if i is None:
                excluded['missing_reference_session'] += 1
                continue
            if min(i,mi) < 251:
                excluded['short_history'] += 1
                continue
            if i+max(protocol['horizons']) >= len(frame):
                excluded['missing_longest_target'] += 1
                continue
            short,long = views_at(frame,i)
            if stamp not in market_cache:
                market_cache[stamp] = market_features(market,mi)
            mshort,mlong = market_cache[stamp]
            short.update(mshort); long.update(mlong)
            targets = {}
            for horizon in protocol['horizons']:
                cfg = {'target':dict(protocol['target'],horizon_sessions=horizon)}
                target, reason = target_at(frame,i,cfg)
                if reason:
                    break
                targets[horizon] = target
            if len(targets) != len(protocol['horizons']):
                excluded[reason] += 1
                continue
            base_columns = list(short)+['target_sigma']
            added_columns = list(long)
            row = {'observation_id':f'{ticker}_{stamp.date()}', 'ticker':ticker,
                   'end_date':str(stamp.date()), 'current_window_start':str(frame.index[i-99].date()),
                   'information_start':str(min(frame.index[i-251],market.index[mi-251]).date()),
                   **short, **long}
            for horizon, target in targets.items():
                row.update({f'{key}_{horizon}':value for key,value in target.items()})
            rows.append(row)
        print(f'Prepared causal features: {ticker}',flush=True)
    if not rows:
        raise ValueError('No eligible common observations')
    cohort = pd.DataFrame(rows).sort_values(['end_date','ticker']).reset_index(drop=True)
    if cohort.observation_id.duplicated().any():
        raise ValueError('Duplicate forecast identities')
    if not np.isfinite(cohort[[c for c in base_columns if c!='target_sigma']+added_columns].to_numpy()).all():
        raise ValueError('Nonfinite causal feature matrix')
    return cohort, {'current_history':base_columns,
                    'multiscale_history':base_columns+added_columns}, dict(excluded)


def folds(cohort, protocol):
    if cohort.end_date.max() >= protocol['protected_holdout_start']:
        raise ValueError('Experiment 05 future holdout cannot enter this study')
    last_target = cohort[[f'target_date_{h}' for h in protocol['horizons']]].max(axis=1)
    for year in protocol['test_years']:
        test = cohort[(cohort.end_date.str[:4]==str(year))
                      & (cohort.information_start>=protocol['test_information_start'])].copy()
        if test.empty:
            continue
        train = cohort[(cohort.end_date>=f"{year-protocol['train_years']}-01-01")
                       & (cohort.end_date<f'{year}-01-01')
                       & (last_target<test.information_start.min())].copy()
        if train.end_date.nunique()<protocol['minimum_train_dates']:
            raise ValueError(f'Insufficient common training dates in {year}')
        if set(train.observation_id)&set(test.observation_id):
            raise ValueError('Overlapping train/test identities')
        if last_target.loc[train.index].max()>=test.information_start.min():
            raise ValueError('Training targets cross test information history')
        yield year,train,test


def shuffled_training_labels(train, column, seed):
    """Preserve full ticker/date vectors by permuting dates of identical coverage."""
    rng = np.random.default_rng(seed)
    output = train[column].copy()
    cells = {}
    for date, rows in train.groupby('end_date',sort=True):
        cells.setdefault(tuple(sorted(rows.ticker)),[]).append(date)
    lookup = train.set_index(['end_date','ticker'])[column]
    for dates in cells.values():
        partners = dict(zip(dates,rng.permutation(dates)))
        for date in dates:
            rows = train[train.end_date==date]
            output.loc[rows.index] = [lookup.loc[partners[date],ticker] for ticker in rows.ticker]
    return output


def fit_predictions(train, test, columns, cfg, cache=None, *, shuffled=False):
    y = shuffled_training_labels(train,'label',cfg['seed']) if shuffled else train.label
    if set(y)!=set(OPTIONS):
        raise ValueError('All classes required in each training fold')
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=cfg['C'],max_iter=cfg['max_iter'],random_state=cfg['seed']))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(train[columns].to_numpy(float),y.to_numpy())
    p = logistic_probabilities(model,test[columns].to_numpy(float),OPTIONS)
    np.testing.assert_allclose(model.steps[0][1].mean_,train[columns].mean().to_numpy(),atol=1e-12,rtol=1e-12)
    audit = {'features':len(columns),'iterations':model.steps[1][1].n_iter_.tolist()}
    if cache is not None:
        joblib.dump(model,cache)
        reloaded = logistic_probabilities(joblib.load(cache),test[columns].to_numpy(float),OPTIONS)
        difference = float(abs(p-reloaded).max())
        if difference>1e-12:
            raise ValueError('Checkpoint reload changes probabilities')
        audit.update(checkpoint_sha256=digest(cache),reload_max_abs_diff=difference)
    return p,audit


def evaluate_predictions(predictions, protocol):
    results = {'horizons':{},'comparisons':{}}
    aligned = {}
    for horizon in protocol['horizons']:
        frames = {}
        for arm in ARMS+['shuffled_multiscale']:
            rows = predictions[(predictions.horizon==horizon)&(predictions.arm==arm)].sort_values('observation_id')
            if rows.empty or rows.observation_id.duplicated().any():
                raise ValueError('Incomplete or duplicated prediction coverage')
            if frames:
                reference = next(iter(frames.values()))
                if (not rows.observation_id.tolist()==reference.observation_id.tolist()
                    or not rows.label.tolist()==reference.label.tolist()):
                    raise ValueError('Paired prediction identity/truth differs')
            frames[arm] = rows
        metrics = {}
        for arm,rows in frames.items():
            m = evaluate_arrays(rows[P_COLS].to_numpy(),rows.label,OPTIONS)
            prior_rows = frames['train_prior']
            m['brier_skill_vs_prior'] = 1-m['brier']/evaluate_arrays(prior_rows[P_COLS].to_numpy(),prior_rows.label,OPTIONS)['brier']
            m['per_year'] = {str(year):{k:v for k,v in evaluate_arrays(group[P_COLS].to_numpy(),group.label,OPTIONS).items()
                                     if k in ('n','accuracy','brier')}
                             for year,group in rows.groupby('fold')}
            metrics[arm] = m
        reference = frames['train_prior']
        aligned[horizon] = reference.observation_id.tolist()
        results['horizons'][str(horizon)] = {
            'n':len(reference),'dates':reference.end_date.nunique(),
            'label_counts':reference.label.value_counts().reindex(OPTIONS,fill_value=0).to_dict(),
            'test_majority_base_rate':float(reference.label.value_counts().max()/len(reference)),
            'arms':metrics}
        for left,right in [('current_history','train_prior'),('multiscale_history','current_history'),('multiscale_history','train_prior')]:
            lm = probability_metrics(frames[left][P_COLS].to_numpy(),reference.label,OPTIONS)
            rm = probability_metrics(frames[right][P_COLS].to_numpy(),reference.label,OPTIONS)
            delta = rm['brier_rows']-lm['brier_rows']
            hit = lm['hit_rows']-rm['hit_rows']
            ci, accuracy_ci, adjusted_ci = {},{},{}
            cfg = protocol['uncertainty']
            for block in cfg['blocks']:
                draws = block_bootstrap_means(np.column_stack([delta,hit]),reference.end_date,
                    resamples=cfg['resamples'],seed=cfg['seed'],block_length=block)
                ci[str(block)] = np.percentile(draws[:,0],[2.5,97.5]).tolist()
                accuracy_ci[str(block)] = np.percentile(draws[:,1],[2.5,97.5]).tolist()
                adjusted_ci[str(block)] = np.percentile(draws[:,0],[1.25,98.75]).tolist()
            annual = {year:metrics[right]['per_year'][year]['brier']-metrics[left]['per_year'][year]['brier']
                      for year in metrics[left]['per_year']}
            results['comparisons'][f'{horizon}:{left}_vs_{right}'] = {
                'brier_reduction':float(delta.mean()),'accuracy_change':float(hit.mean()),
                'ci95':ci,'accuracy_ci95':accuracy_ci,'ci97_5':adjusted_ci,
                'positive_brier_years':sum(v>0 for v in annual.values()),'per_year_brier_reduction':annual}
    if len({tuple(ids) for ids in aligned.values()})!=1:
        raise ValueError('Horizon comparisons must use identical observations')
    block = str(protocol['uncertainty']['primary_block'])
    horizon_gain = results['comparisons']['20:current_history_vs_train_prior']
    view_gain = results['comparisons']['20:multiscale_history_vs_current_history']
    minimum = protocol['decision']['minimum_positive_years']
    results['continuation'] = {
        'longer_horizon_gate':horizon_gain['ci97_5'][block][0]>0 and horizon_gain['positive_brier_years']>=minimum,
        'year_view_gate':view_gain['ci97_5'][block][0]>0 and view_gain['positive_brier_years']>=minimum
             and results['horizons']['20']['arms']['multiscale_history']['brier']<results['horizons']['20']['arms']['train_prior']['brier'],
        'independent_confirmation':False,
        'meaning':'A passed CPU gate motivates a separately frozen confirmation study; it does not authorize GPU inference or establish a trading edge.'}
    return results


def report_markdown(result):
    lines = ['# CPU longer-horizon results v1','',
             'Retrospective exploratory numerical forecasts, with a protocol/source freeze pushed before deriving these outcomes. No JEV inference or RL training.', '',
             '| Horizon | Model | N | Accuracy | Brier | Brier skill vs prior |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for horizon,part in result['horizons'].items():
        for arm,m in part['arms'].items():
            lines.append(f"| {horizon} sessions | {arm} | {m['n']} | {m['accuracy']:.2%} | {m['brier']:.6f} | {m['brier_skill_vs_prior']:+.2%} |")
        lines += ['',f"{horizon}-session labels: {part['label_counts']}; {part['dates']} forecast-date clusters. Test-majority accuracy {part['test_majority_base_rate']:.2%} is descriptive only.",'']
    lines += ['Lower Brier is better. Compare models within a horizon; raw accuracy/Brier across different targets is not a matched forecast comparison.', '',
              '| Paired comparison | Brier reduction | 95% eight-date CI | 97.5% decision CI | Positive years |',
              '| --- | ---: | --- | --- | ---: |']
    for name,m in result['comparisons'].items():
        lo,hi=m['ci95']['8']; alo,ahi=m['ci97_5']['8']
        lines.append(f"| {name} | {m['brier_reduction']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | [{alo:+.6f}, {ahi:+.6f}] | {m['positive_brier_years']}/{len(m['per_year_brier_reduction'])} |")
    lines += ['',f"Longer-horizon CPU gate: **{'passed' if result['continuation']['longer_horizon_gate'] else 'failed'}**. Year-view CPU gate: **{'passed' if result['continuation']['year_view_gate'] else 'failed'}**.", '',
              'The two decision comparisons use Bonferroni 97.5% marginal intervals for a 5% family across the two gates, and require at least five positive annual comparisons. The year-view gate also requires beating the 20-session prior. These are approximate bootstrap intervals, not guaranteed familywise coverage.', '',
              'Circular moving blocks carry all tickers at each date together; eight retained weekly dates are primary, with 4/12/52-date sensitivities in metrics.json. Twenty-session outcomes overlap. Intervals condition on fitted predictions and assume approximate stationarity; no refit uncertainty is included.', '',
              'All arms share eligible rows, five-year annual training periods and the longest-history purge. Training targets for both horizons precede the first test information window. The year view uses 252 daily sessions aggregated into weekly OHLCV, including partial boundary weeks only through the chart end. Current features include 5/20/100-session views. A common SPY context and the causal target sigma are supplied.', '',
              'Limitations: fixed linear model, hand-selected surviving universe, adjusted close-to-close targets, adaptively reused historical years, overlapping histories, partial 2026. This is not a direct replication of the 300-chart JEV pilot, a pattern-detector ablation, or evidence about every possible model. No costs, executable entry, risk-adjusted P&L, GPU, HTTP inference or market-data download. Experiment 05 future holdout/model remain unchanged.', '']
    return '\n'.join(lines)


def run(protocol_path=EXP/'protocol.json', out=EXP/'results/evaluation_v1'):
    protocol = json.loads(Path(protocol_path).read_text())
    versions = verify_protocol(protocol)
    out = Path(out)
    if out.exists():
        raise ValueError('Preserve existing results; use a separately versioned output directory')
    started = time.monotonic()
    raw = {ticker:load(ticker) for ticker in universe()['tickers']}
    cohort,features,excluded = build_cohort(raw,protocol)
    cache = EXP/'.cache/v1'; cache.mkdir(parents=True,exist_ok=True)
    tables,records = [],[]
    for year,train,test in folds(cohort,protocol):
        print(f'Annual fold {year}: {len(train)} train / {len(test)} test',flush=True)
        record = {'year':year,'train_rows':len(train),'train_dates':train.end_date.nunique(),'test_rows':len(test),
                  'test_dates':test.end_date.nunique(),'train_target_date_max':train[[f'target_date_{h}' for h in protocol['horizons']]].max().max(),
                  'test_information_start_min':test.information_start.min(),'models':{}}
        for horizon in protocol['horizons']:
            a,b = train.copy(),test.copy()
            for frame in (a,b):
                frame['label']=frame[f'label_{horizon}']; frame['target_sigma']=frame[f'sigma_{horizon}']
            prior = a.label.value_counts(normalize=True).reindex(OPTIONS,fill_value=0).to_numpy()
            predictions = {'train_prior':np.tile(prior,(len(b),1)),'uniform':np.full((len(b),3),1/3)}
            record[f'train_counts_{horizon}'] = a.label.value_counts().reindex(OPTIONS,fill_value=0).to_dict()
            for arm in ['current_history','multiscale_history','shuffled_multiscale']:
                columns = features['current_history' if arm=='current_history' else 'multiscale_history']
                p,audit = fit_predictions(a,b,columns,protocol['model'],cache/f'{year}_{horizon}_{arm}.joblib',
                    shuffled=arm=='shuffled_multiscale')
                predictions[arm]=p
                record['models'][f'{horizon}:{arm}']=audit
            for arm,p in predictions.items():
                saved = b[['observation_id','ticker','end_date','information_start']].copy()
                saved['target_date']=b[f'target_date_{horizon}'];saved['label']=b.label
                saved['horizon']=horizon;saved['fold']=year;saved['arm']=arm
                saved['forward_return']=b[f'forward_return_{horizon}'];saved['sigma']=b.target_sigma
                saved[P_COLS]=p; saved['prediction']=np.array(OPTIONS)[p.argmax(1)]
                tables.append(saved)
        records.append(record)
    if len(records)!=len(protocol['test_years']):
        raise ValueError('Every frozen annual test fold must be present')
    predictions=pd.concat(tables,ignore_index=True)
    result=evaluate_predictions(predictions,protocol)
    out.mkdir(parents=True)
    predictions.to_csv(out/'predictions.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    write_json(out/'metrics.json',result)
    write_json(out/'folds.json',records)
    (out/'README.md').write_text(report_markdown(result))
    data = EXP/'data'; data.mkdir(exist_ok=True)
    cohort.to_csv(data/'features.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    manifest = {'status':'complete CPU-only exploratory study', 'protocol_sha256':digest(protocol_path),
                'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                'input_sha256':protocol['input_sha256'],'dependency_versions':versions,
                'eligible_rows':len(cohort),'excluded':excluded,'feature_columns':features,
                'evaluation_rows_per_horizon':result['horizons']['20']['n'],
                'forecast_dates':result['horizons']['20']['dates'],
                'first_forecast':predictions.end_date.min(),'last_forecast':predictions.end_date.max(),
                'latest_target':predictions.target_date.max(),'elapsed_seconds':round(time.monotonic()-started,3),
                'output_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()},
                'features_sha256':digest(data/'features.csv.gz'),'gpu_started':False,'downloads':False,'rl_training':False}
    write_json(out/'manifest.json',manifest)
    print(json.dumps(result['continuation'],indent=2),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=EXP/'results/evaluation_v1')
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    if args.verify_only:
        versions=verify_protocol(json.loads((EXP/'protocol.json').read_text()))
        print(json.dumps({'protocol_verified':True,'versions':versions,'gpu_started':False},indent=2))
    else:
        run(out=args.out)


if __name__=='__main__':
    main()
