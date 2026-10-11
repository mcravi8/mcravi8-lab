"""CPU-only preparation of fixed framing, styles, controls and frozen inference jobs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

EXP=Path(__file__).resolve().parents[2];REPO=EXP.parent
sys.path.insert(0,str(EXP))
from shared.src.render import render_chart
from shared.src.jev_api import parity_policy
A=EXP/'06a-candles';B=EXP/'06b-chart-patterns'
MAP={'Head and Shoulders':'Inverse Head and Shoulders','Inverse Head and Shoulders':'Head and Shoulders',
     'Double Top':'Double Bottom','Double Bottom':'Double Top','Triangle':'Triangle'}
QUESTIONS={
 'candles':['Which candlestick pattern ends at the right edge of the target chart? Select one option.',
            'Identify the final candlestick formation at the right edge of the target chart. Which listed class fits?'],
 'patterns':['Which completed chart pattern best describes the whole target chart? Select one option.',
             'Classify the main completed geometric formation visible in the target chart using one listed class.']}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def load_synth():
    spec=importlib.util.spec_from_file_location('exp06_day2_synth',B/'src/synth.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def read_bars(root,row):return pd.read_csv(root/row['bars_file'],index_col='Date',parse_dates=True)


def invert_prices(frame):
    """Price inversion preserves time and OHLC order; volume is unchanged."""
    out=frame.copy();center=float(frame.High.max()+frame.Low.min())
    out['Open']=center-frame.Open;out['Close']=center-frame.Close
    out['High']=center-frame.Low;out['Low']=center-frame.High
    return out


def style_subset(index,n=100,seed=606227):
    """Fixed label/split stratification, without consulting model or review outcomes."""
    if len(index)<n:raise ValueError(f'Need {n} charts for the frozen style subset, have {len(index)}')
    groups=[]
    for _,group in index.groupby(['label','split'],sort=True):
        keys=group.file.map(lambda f:hashlib.sha256(f'{seed}:{f}'.encode()).hexdigest())
        groups.append(group.assign(_key=keys).sort_values('_key').drop(columns='_key').to_dict('records'))
    chosen=[]
    while len(chosen)<n:
        for group in groups:
            if group:chosen.append(group.pop(0))
            if len(chosen)==n:break
    return pd.DataFrame(chosen)


def estimate(jobs):
    predictions=sum(j['n_rows'] for j in jobs)
    parity=40*len(jobs)+80
    total=predictions+parity
    return {'jobs':len(jobs),'prediction_calls':predictions,'job_parity_calls':40*len(jobs),
            'global_parity_calls':80,'total_decide_calls':total,
            'gpu_minutes_range':[round(total*.6/60+5,1),round(total*1.8/60+15,1)],
            'assumptions':'Fresh successful run, no retries/resumes: two global 20-chart repeatability gates plus 40 calls per job. Planning throughput 0.6–1.8 GPU seconds per single-image call with concurrency1, batch1, eager/cacheless execution without batch invariance; add 5–15 startup minutes. Historical Exp03 request median 594.704ms is context, not measured throughput for these larger charts. Serial parity, hardware and caching can move runtime outside this range. No timing measured for corrected06 serial configuration.'}


def render_task(task):
    frame, target, size, style, volume_color = task
    render_chart(frame, target, n_bars=len(frame), size=size, style=style, volume_color=volume_color)


def prepare():

    a=pd.read_csv(A/'results/index.csv',keep_default_na=False);real=pd.read_csv(B/'results/real_index.csv',keep_default_na=False)
    fixed_real=pd.read_csv(B/'results/real_fixed100_index.csv',keep_default_na=False);synthetic=pd.read_csv(B/'results/synthetic_index.csv',keep_default_na=False)
    a['data_origin']='market';real['data_origin']='market';fixed_real['data_origin']='market'
    synthetic['data_origin']='synthetic'
    synthetic['id']=synthetic.template_id+'_'+synthetic.noise_number.astype(str)
    a['source_collection']='candles';real['source_collection']='real_variable'
    fixed_real['source_collection']='real_fixed100';synthetic['source_collection']='synthetic_variable'
    a['evidence_role']='benchmark';synthetic['evidence_role']='benchmark'
    real['evidence_role']='showcase_only';fixed_real['evidence_role']='showcase_only'
    synth=load_synth()
    tasks=[]
    def queue(frame,target,size=(1024,640),style='candle',volume_color='neutral'):
        tasks.append((frame,target,size,style,volume_color))
    def flush():
        if tasks:
            with ProcessPoolExecutor(max_workers=4) as pool:
                list(pool.map(render_task,tasks,chunksize=8))
            print(f'Rendered {len(tasks)} additional CPU charts',flush=True)
            tasks.clear()
    def bars_csv(frame,name):
        path=B/'data/day2'/name;path.parent.mkdir(parents=True,exist_ok=True)
        frame.to_csv(path,index_label='Date',float_format='%.12g')
        return str(path.relative_to(B))
    collections=[];assets={};variants=[];paraphrases=[];controls=[]
    def add(name,root,index,kind,role='primary',control='image',expected=None):
        flush()
        index=index.copy().fillna('')
        accuracy_valid=role not in ['time_placebo','showcase']
        index['accuracy_valid']=accuracy_valid
        evidence_role='showcase_only' if role=='showcase' else 'benchmark'
        folder=root/'results/arms';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'{name}.csv';index.to_csv(path,index=False)
        collections.append({'name':name,'root':str(root.relative_to(REPO)),'index':str(path.relative_to(root)),
                            'kind':kind,'role':role,'control':control,'n_rows':len(index),
                            'evidence_role':evidence_role,'accuracy_valid':accuracy_valid,
                            'expected_class_map':expected or {},'data_origin':'+'.join(sorted(index.data_origin.unique()))})
        for file in index.file:assets[str((root/file).relative_to(REPO))]=digest(root/file)
        assets[str(path.relative_to(REPO))]=digest(path)
    add('candles',A,a,'candles')
    add('real_variable',B,real,'patterns',role='showcase')
    add('real_fixed100',B,fixed_real,'patterns',role='showcase')
    add('synthetic_variable',B,synthetic,'patterns')
    fixed_rows=[]
    fixed_dir=B/'charts/fixed100';fixed_dir.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(synthetic.to_dict('records')):
        frame=synth.fixed_frame(read_bars(B,row));path=fixed_dir/Path(row['file']).name
        queue(frame,path)
        fixed_rows.append({**row,'file':str(path.relative_to(B)),'n_bars':100,'framing':'fixed_100',
                           'bars_file':bars_csv(frame,'fixed100/'+path.stem+'.csv'),'source_collection':'synthetic_fixed100'})
        if (i+1)%200==0:print(f'Prepared synthetic fixed100 {i+1}/1200',flush=True)
    add('synthetic_fixed100',B,pd.DataFrame(fixed_rows),'patterns')
    # All secondary subsets are fixed before inference. Synthetic controls use
    # distinct clean parents; full main arms retain all six noise realizations.
    clean=synthetic[synthetic.is_clean.astype(str).str.lower().eq('true')]
    a_subset=style_subset(a[a.split=='test'])
    b_subset=style_subset(clean)
    for name,root,subset,kind in [('candles',A,a_subset,'candles'),('patterns',B,b_subset,'patterns')]:
        add(name+'_paraphrase',root,subset,kind,role='paraphrase')
        paraphrases.append({'dataset':name,'n_rows':len(subset),'ids':subset.id.tolist(),
                            'arms':['zero_shot','definitions'],'paraphrase':2,
                            'primary_wording_reuses_main_predictions':True})
        # Candle style aliases main predictions; OHLC/line are zero-shot only.
        add(name+'_style_candle',root,subset,kind,role='alias')
        for style in ['ohlc','line']:
            directory=root/'charts/styles'/style;directory.mkdir(parents=True,exist_ok=True);rows=[]
            for row in subset.to_dict('records'):
                frame=read_bars(root,row);target=directory/Path(row['file']).name
                queue(frame,target,size=(640,480) if kind=='candles' else (1024,640),style=style,
                      volume_color='direction' if kind=='candles' else 'neutral')
                rows.append({**row,'file':str(target.relative_to(root)),'source_file':row['file'],'style':style})
            add(name+'_style_'+style,root,pd.DataFrame(rows),kind,role='style')
        variants.append({'dataset':name,'n_parents':len(subset),'ids':subset.id.tolist(),
                         'styles':['candle','ohlc','line'],'arms':['zero_shot'],'paraphrase':1,
                         'candle_reuses_primary_predictions':True,
                         'origin_counts':subset.data_origin.value_counts().to_dict(),
                         'pairing':'source_collection + source_file (or file for candle alias)',
                         'cohort':'market test only' if kind=='candles' else 'distinct clean synthetic parents'})
        add(name+'_mirror_time',root,subset,kind,role='time_placebo',control='mirror')
        controls.append({'dataset':name,'transformation':'time_mirror','n_rows':len(subset),
                         'ids':subset.id.tolist(),'accuracy_valid':False,'arms':['zero_shot'],'paraphrase':1})
    # Eligible price flips require a separately frozen 100-parent subset.
    # Bull Flag and None have no justified in-taxonomy inversion label.
    eligible=style_subset(clean[clean.label.isin(MAP)])
    inversion_rows=[];directory=B/'charts/price_inversion/synthetic';directory.mkdir(parents=True,exist_ok=True)
    for row in eligible.to_dict('records'):
        frame=invert_prices(read_bars(B,row));target=directory/Path(row['file']).name
        queue(frame,target)
        inversion_rows.append({**row,'file':str(target.relative_to(B)),'source_file':row['file'],
                               'source_label':row['label'],'label':MAP[row['label']],
                               'transformation':'price_inversion',
                               'bars_file':bars_csv(frame,'price_inversion/synthetic/'+target.stem+'.csv')})
    add('patterns_mirror_price',B,pd.DataFrame(inversion_rows),'patterns',role='price_control',expected=MAP)
    controls.append({'dataset':'patterns','transformation':'price_inversion','n_rows':len(eligible),
                     'ids':eligible.id.tolist(),'accuracy_valid':True,'arms':['zero_shot'],'paraphrase':1})
    # Delete only obsolete generated arm indexes, keeping raw charts untouched.
    for folder in [A/'results/arms',B/'results/arms']:
        active={Path(c['index']).name for c in collections if REPO/c['root']==folder.parent.parent}
        for path in folder.glob('*.csv'):
            if path.name not in active:path.unlink()
    agreement=json.loads((B/'results/real_agreement.json').read_text())
    agreement_rows=[]
    for label,values in agreement['classes'].items():
        is_none=label=='None'
        agreement_rows.append({'label':label,'decision_unit':'absence-screen window' if is_none else 'completed event',
                               'external':values['external_none' if is_none else 'external_events'],
                               'independent':values['zigzag_none' if is_none else 'zigzag_events'],
                               'matched':values['both_none' if is_none else 'matched_events'],
                               'agreement_jaccard':values['positive_jaccard'],
                               'denominator':values['denominator'],
                               'evidence_type':'detector_overlap_not_model_accuracy'})
    agreement_table=B/'results/detector_agreement.csv'
    pd.DataFrame(agreement_rows).to_csv(agreement_table,index=False,float_format='%.12g')
    jobs=[]
    for c in collections:
        if c['role']=='alias':continue
        arms=['zero_shot'] if c['role'] in ['style','time_placebo','price_control'] else ['zero_shot','definitions']
        number=2 if c['role']=='paraphrase' else 1
        question=QUESTIONS[c['kind']][number-1]
        for arm in arms:
            root=REPO/c['root'];context=EXP/'shared/context'/('candles.txt' if c['kind']=='candles' else 'chart_patterns.txt')
            options=root/'options.json';identity=f"{c['name']}_{arm}_p{number}"
            jobs.append({**c,'id':identity,'arm':arm,'paraphrase':number,'question':question,
                'context_file':str(context.relative_to(REPO)) if arm=='definitions' else None,
                'options_file':str(options.relative_to(REPO)),
                'out':f'results/inference/frozen_v7/{identity}.csv'})
    for path in [A/'options.json',B/'options.json',agreement_table,B/'results/real_agreement.json',
                 EXP/'shared/context/candles.txt',EXP/'shared/context/chart_patterns.txt',EXP/'config/model_contract.json',EXP/'config/gpu_runtime.json',EXP/'requirements-scoring.txt',EXP/'requirements-jev-server.txt',EXP/'requirements-jev-torch.txt',EXP/'shared/results/diagnostics/2026-10-10/runtime_lock.json',
                 EXP/'shared/results/diagnostics/2026-10-10-gdn-attn/diagnosis.json',EXP/'shared/results/diagnostics/2026-10-10-gdn-attn/server.log',
                 EXP/'shared/results/diagnostics/2026-10-10-gdn-attn/runtime_lock.json',EXP/'shared/results/diagnostics/2026-10-10-gdn-attn/session_summary.json',
                 EXP/'shared/results/diagnostics/2026-10-10-v4-parity/diagnosis.json',
                 EXP/'shared/results/diagnostics/2026-10-10-v4-parity/candles.parity.json',
                 EXP/'shared/results/diagnostics/2026-10-10-v4-parity/runtime_lock.json',
                 EXP/'shared/results/diagnostics/2026-10-10-v4-parity/session_summary.json',
                 EXP/'shared/results/diagnostics/2026-10-10-v4-parity/frozen_v4_settings.json',
                 EXP/'shared/results/parity_noise_floor.json',
                 EXP/'shared/results/diagnostics/2026-10-10-v5-parity/diagnosis.json',
                 EXP/'shared/results/diagnostics/2026-10-10-v5-parity/candles.parity.json',
                 EXP/'shared/results/diagnostics/2026-10-10-v5-parity/candles.meta.json',
                 EXP/'shared/results/diagnostics/2026-10-10-v5-parity/original_gate.meta.json',
                 EXP/'shared/results/diagnostics/2026-10-10-v5-parity/runtime_lock.json',
                 EXP/'shared/results/diagnostics/2026-10-10-v5-parity/session_summary.json',
                 EXP/'shared/results/diagnostics/2026-10-10-v5-parity/frozen_v5_settings.json',
                 EXP/'shared/results/diagnostics/2026-10-11-v6-parity/diagnosis.json',
                 EXP/'shared/results/diagnostics/2026-10-11-v6-parity/candles.parity.json',
                 EXP/'shared/results/diagnostics/2026-10-11-v6-parity/candles.meta.json',
                 EXP/'shared/results/diagnostics/2026-10-11-v6-parity/runtime_lock.json',
                 EXP/'shared/results/diagnostics/2026-10-11-v6-parity/session_summary.json',
                 EXP/'shared/results/diagnostics/2026-10-11-v6-parity/frozen_v6_settings.json',
                 EXP/'shared/src/score.py',EXP/'shared/src/jev_api.py',EXP/'shared/src/evaluate.py',EXP/'shared/src/render.py',Path(__file__),EXP/'scripts/run_gpu.py',
                 REPO/'01-jev-tsla-charts/scripts/remote_serve.sh']:
        assets[str(path.relative_to(REPO))]=digest(path)
    config={'schema_version':7,'status':'Matteo-approved final near-tie exception after v6 failure; v7 not GPU run','seed':606227,
        'model_contract':'06-jev-patterns/config/model_contract.json','gpu_runtime':'06-jev-patterns/config/gpu_runtime.json','questions':QUESTIONS,
        'parity_gate':parity_policy(),
        'arms':['zero_shot','definitions'],
        'reference_arm':{'status':'removed','enabled':False,'reason':'Trimmed before inference; no reference-image jobs or enable flag.'},
        'primary_evaluation':{'candles':'test only','real':'showcase_only: no aggregate or per-class model accuracy claims',
                              'synthetic':'all eval parents, cluster noise/framing/style versions by template_id'},
        'real_showcase':{'n_parents':150,'framings':['variable','fixed_100'],'accuracy_valid':False,
                         'reason':'Sparse, imbalanced detector-selected examples; not a recognition benchmark.'},
        'reported_results':{'parity_noise_floor':{'details':'06-jev-patterns/shared/results/parity_noise_floor.json',
                             'readme':'06-jev-patterns/shared/results/README.md','evidence_type':'historical candle repeatability; post-hoc v7 policy'},
                             'detector_agreement':{'table':str(agreement_table.relative_to(REPO)),
                             'details':str((B/'results/real_agreement.json').relative_to(REPO)),
                             'evidence_type':'detector overlap; not expert label quality or JEV accuracy'}},
        'paraphrase_subsets':paraphrases,'control_subsets':controls,
        'mirror_policy':{'time_mirror':'Position/time-order placebo only; no assumed class map or label accuracy.',
                         'price_inversion_map':MAP,'excluded_price_classes':['Bull Flag','None'],
                         'exclusion_reason':'Bull Flag inverts to out-of-taxonomy Bear Flag; None can contain Bear Flags that invert into a target class.'},
        'styles':variants,'collections':collections,'jobs':jobs,'assets':assets,
        'mixed_origin_policy':'06a controls are market test rows; 06b controls are synthetic only. Real 06b rows are showcase-only. Never pool these cohorts for accuracy or confidence intervals.',
        'call_budget':{'default':estimate(jobs)},'fresh_call_cap':10000,
        'runtime':{'concurrency':1,'batch_size':1,'timeout':120,'retries':3,'max_model_len':16384,'port':8000},
        'resume':'Completed jobs make no new inference calls. Partial jobs rerun their40-call repeatability gate; retries add requests. Global gates run at each GPU session.',
        'execution_status':'not_run','prior_attempt':{'status':'failed global candle gate','decide_calls':40,'prediction_calls':0,
          'diagnosis':'06-jev-patterns/shared/results/diagnostics/2026-10-10/diagnosis.json'},
        'prior_startup_attempt':{'status':'failed GDN_ATTN batch-invariant server startup','decide_calls':0,'prediction_calls':0,
           'diagnosis':'06-jev-patterns/shared/results/diagnostics/2026-10-10-gdn-attn/diagnosis.json'},
        'prior_v4_attempt':{'status':'failed original0.01 candle gate','decide_calls':40,'prediction_calls':0,
          'max_abs_diff':0.018410737935632693,'argmax_agreement_count':20,
          'diagnosis':'06-jev-patterns/shared/results/diagnostics/2026-10-10-v4-parity/diagnosis.json'},
        'prior_v5_attempt':{'status':'failed maximum0.03 candle gate','decide_calls':40,'prediction_calls':0,
          'max_abs_diff':0.04121730474518304,'argmax_agreement_count':20,
          'diagnosis':'06-jev-patterns/shared/results/diagnostics/2026-10-10-v5-parity/diagnosis.json'},
        'prior_v6_attempt':{'status':'failed argmax agreement only','decide_calls':40,'prediction_calls':0,
          'mean_abs_diff':0.0026098296378638183,'max_abs_diff':0.022243287166081593,'argmax_agreement_count':19,
          'diagnosis':'06-jev-patterns/shared/results/diagnostics/2026-10-11-v6-parity/diagnosis.json'}}
    if config['call_budget']['default']['total_decide_calls']>=config['fresh_call_cap']:
        raise ValueError('Trimmed fresh-run call budget must remain below 10,000')
    write_json(EXP/'config/inference_arms.json',config)
    write_json(EXP/'shared/results/day2_preparation.json',{'counts':{c['name']:c['n_rows'] for c in collections},
        'config_sha256':digest(EXP/'config/inference_arms.json'),'call_budget':config['call_budget'],'gpu_started':False,
        'asset_count':len(assets),'styles':variants})
    print(json.dumps(config['call_budget'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();prepare()
