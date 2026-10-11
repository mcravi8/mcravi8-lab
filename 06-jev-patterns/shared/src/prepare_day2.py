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
from PIL import Image, ImageDraw, ImageFont

EXP=Path(__file__).resolve().parents[2];REPO=EXP.parent
sys.path.insert(0,str(EXP))
from shared.src.render import render_chart
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


def style_subset(index,n=200,seed=606227):
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


def reference_board(rows,target):
    width,height=320,235;cols=2;rows_count=(len(rows)+1)//2
    image=Image.new('RGB',(cols*width,rows_count*height),'white');draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=16)
    for i,(label,path) in enumerate(rows):
        x,y=(i%cols)*width,(i//cols)*height
        draw.text((x+6,y+5),label,fill='black',font=font)
        with Image.open(path) as chart:
            chart=chart.convert('RGB');chart.thumbnail((width,205),Image.Resampling.LANCZOS)
            image.paste(chart,(x+(width-chart.width)//2,y+27))
    target.parent.mkdir(parents=True,exist_ok=True);image.save(target)


def prepare_references(a_index):
    refs=EXP/'shared/references';refs.mkdir(parents=True,exist_ok=True)
    options=json.loads((A/'options.json').read_text());examples=[];provenance=[]
    for label in options:
        rows=a_index[(a_index.label==label)&(a_index.split=='train')&a_index.clear.astype(str).str.lower().eq('true')].sort_values('id')
        if rows.empty:raise ValueError(f'No clear training reference for {label}')
        row=rows.iloc[0];examples.append((label,A/row.file));provenance.append({'label':label,'id':row.id,'split':'train','file':str((A/row.file).relative_to(REPO)),'sha256':digest(A/row.file)})
    reference_board(examples,refs/'candles.png')
    synth=load_synth();protocol={**synth.read_protocol(),'seed':606226,'clean_templates':7}
    specs=synth.template_specs(protocol);primary_seeds={p['seed'] for p in synth.template_specs()}
    if primary_seeds & {p['seed'] for p in specs}:raise ValueError('Synthetic references overlap evaluation parents')
    examples=[];synthetic=[]
    cache=B/'data/reference_charts';cache.mkdir(parents=True,exist_ok=True)
    for i,spec in enumerate(specs):
        frame,_,info=synth.make_template(spec,protocol);path=cache/f'reference_{i}.png'
        render_chart(frame,path,n_bars=len(frame),size=(1024,640),volume_color='neutral')
        examples.append((spec['label'],path));synthetic.append({'label':spec['label'],'seed':spec['seed'],'generator_seed':606226,'sha256':digest(path)})
    reference_board(examples,refs/'patterns.png')
    write_json(refs/'provenance.json',{'candles':provenance,'patterns':synthetic,'synthetic_reference_seed':606226,
        'selection':'Fixed before inference. Candles use first clear training ID per class. Pattern sketches use seven independently seeded non-evaluation clean parents.',
        'training_results':'Candle reference IDs are training-only; their own scored training rows are descriptive, never primary evaluation.',
        'boards':{name:digest(refs/name) for name in ['candles.png','patterns.png']}})
    return {'candles':refs/'candles.png','patterns':refs/'patterns.png'}


def estimate(jobs,include_references=False):
    chosen=[j for j in jobs if include_references or not j.get('optional_reference')]
    predictions=sum(j['n_rows'] for j in chosen);parity=40*len(chosen)+80
    weighted=sum(j['n_rows']*(2 if j.get('optional_reference') else 1) for j in chosen)
    weighted+=sum(40*(2 if j.get('optional_reference') else 1) for j in chosen)+80
    return {'jobs':len(chosen),'prediction_calls':predictions,'job_parity_calls':40*len(chosen),
            'global_parity_calls':80,'total_decide_calls':predictions+parity,
            'gpu_minutes_range':[round(weighted*.25/60+5,1),round(weighted*.65/60+15,1)],
            'assumptions':'Fresh complete run, no retries/resumes: two global20-chart repeatability gates plus40 calls per job. Planning throughput0.25–0.65 effective GPU seconds per single-image call at concurrency4; reference calls weighted2x; add5–15 startup minutes. Historical Exp03 request median594.704ms is context, not measured throughput for these larger charts. No GPU timing measured for06.'}


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
    references=prepare_references(a);synth=load_synth()
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
    collections=[];assets={};variants=[]
    def add(name,root,index,kind,role='primary',control='image',expected=None):
        flush()
        index=index.fillna('')
        folder=root/'results/arms';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'{name}.csv';index.to_csv(path,index=False)
        collections.append({'name':name,'root':str(root.relative_to(REPO)),'index':str(path.relative_to(root)),
                            'kind':kind,'role':role,'control':control,'n_rows':len(index),
                            'expected_class_map':expected or {},'data_origin':'+'.join(sorted(index.data_origin.unique()))})
        for file in index.file:assets[str((root/file).relative_to(REPO))]=digest(root/file)
        assets[str(path.relative_to(REPO))]=digest(path)
    add('candles',A,a,'candles');add('real_variable',B,real,'patterns');add('real_fixed100',B,fixed_real,'patterns')
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
    # Candle style is an alias of the already-scored primary row; no duplicate request.
    if len(real)>=200:
        b_style=style_subset(real)
    else:
        clean=synthetic[synthetic.is_clean.astype(str).str.lower().eq('true')]
        b_style=pd.concat([real,style_subset(clean,200-len(real))],ignore_index=True).fillna('')
    for name,root,index,kind in [('candles',A,style_subset(a),'candles'),('patterns',B,b_style,'patterns')]:
        subset=index;add(name+'_style_candle',root,subset,kind,role='alias')
        for style in ['ohlc','line']:
            directory=root/'charts/styles'/style;directory.mkdir(parents=True,exist_ok=True);rows=[]
            for row in subset.to_dict('records'):
                frame=read_bars(root,row);target=directory/Path(row['file']).name
                queue(frame,target,size=(640,480) if kind=='candles' else (1024,640),style=style,
                      volume_color='direction' if kind=='candles' else 'neutral')
                rows.append({**row,'file':str(target.relative_to(root)),'source_file':row['file'],'style':style})
            add(name+'_style_'+style,root,pd.DataFrame(rows),kind,role='style')
        variants.append({'dataset':name,'n_parents':len(subset),'ids':subset.id.tolist() if 'id' in subset else subset.file.tolist(),
                         'styles':['candle','ohlc','line'],'candle_reuses_primary_predictions':True,
                         'origin_counts':subset.data_origin.value_counts().to_dict(),
                         'pairing':'source_collection + source_file (or file for candle alias); analyze market test and synthetic parents separately'})
    inversion_rows=[]
    for name,root,index,kind in [('candles',A,a,'candles'),('real',B,real,'patterns'),('synthetic',B,synthetic,'patterns')]:
        add(name+'_mirror_time',root,index,kind,role='time_placebo',control='mirror')
        if kind=='patterns':
            rows=[];directory=root/'charts/price_inversion'/name;directory.mkdir(parents=True,exist_ok=True)
            for row in index[index.label.isin(MAP)].to_dict('records'):
                frame=invert_prices(read_bars(root,row));target=directory/Path(row['file']).name
                queue(frame,target)
                rows.append({**row,'file':str(target.relative_to(root)),'source_file':row['file'],'source_label':row['label'],
                             'label':MAP[row['label']],'transformation':'price_inversion',
                             'bars_file':bars_csv(frame,'price_inversion/'+name+'/'+target.stem+'.csv')})
            inversion_rows.extend(rows)
    # Pool requests only, so even rare real controls pass the 20-row gate.
    # Evaluation must separate market and synthetic rows using data_origin.
    add('patterns_mirror_price',B,pd.DataFrame(inversion_rows),'patterns',role='price_control',expected=MAP)
    jobs=[]
    for c in collections:
        if c['role']=='alias':continue
        arms=['zero_shot'] if c['role'] in ['time_placebo','price_control'] else ['zero_shot','definitions','references']
        for arm in arms:
            for number,question in enumerate(QUESTIONS[c['kind']],1):
                root=REPO/c['root'];context=EXP/'shared/context'/('candles.txt' if c['kind']=='candles' else 'chart_patterns.txt')
                options=root/'options.json';identity=f"{c['name']}_{arm}_p{number}"
                jobs.append({**c,'id':identity,'arm':arm,'paraphrase':number,'question':question,
                    'context_file':str(context.relative_to(REPO)) if arm!='zero_shot' else None,
                    'options_file':str(options.relative_to(REPO)),
                    'reference_image':str(references[c['kind']].relative_to(REPO)) if arm=='references' else None,
                    'optional_reference':arm=='references','out':f'results/inference/frozen_v1/{identity}.csv',
                    'accuracy_valid':c['role']!='time_placebo'})
    for path in [A/'options.json',B/'options.json',*references.values(),EXP/'shared/references/provenance.json',
                 EXP/'shared/context/candles.txt',EXP/'shared/context/chart_patterns.txt',EXP/'config/model_contract.json',
                 EXP/'shared/src/score.py',EXP/'shared/src/jev_api.py',EXP/'shared/src/evaluate.py',EXP/'shared/src/render.py',Path(__file__),EXP/'scripts/run_gpu.py',
                 REPO/'01-jev-tsla-charts/scripts/remote_serve.sh']:
        assets[str(path.relative_to(REPO))]=digest(path)
    config={'schema_version':1,'status':'frozen before any06 model inference','seed':606227,
        'model_contract':'06-jev-patterns/config/model_contract.json','questions':QUESTIONS,
        'reference_arm':{'supported':True,'enabled_by_default':False,'enable_flag':'--include-references','images_per_call':2,
                         'reason':'Optional additional compute; multi-image support verified in pinned model card and server source.'},
        'primary_evaluation':{'candles':'test only','real':'test only','synthetic':'all eval parents, cluster all noise/framing/style variants by template_id'},
        'mirror_policy':{'time_mirror':'Position/time-order placebo only; no assumed class map or label accuracy.',
                         'price_inversion_map':MAP,'excluded_price_classes':['Bull Flag','None'],
                         'exclusion_reason':'Bull Flag inverts to out-of-taxonomy Bear Flag; None can contain Bear Flags that invert into a target class.'},
        'styles':variants,'collections':collections,'jobs':jobs,'assets':assets,
        'mixed_origin_policy':'Combined style and price-control request indexes retain data_origin and source_collection. Never pool market and synthetic accuracy or confidence intervals.',
        'call_budget':{'default':estimate(jobs),'with_references':estimate(jobs,True)},
        'runtime':{'concurrency':4,'batch_size':16,'timeout':120,'retries':3,'max_model_len':16384,'port':8000},
        'resume':'Completed jobs make no new inference calls. Partial jobs rerun their40-call repeatability gate; retries add requests. Global gates run at each GPU session.',
        'execution_status':'not_run'}
    write_json(EXP/'config/inference_arms.json',config)
    write_json(EXP/'shared/results/day2_preparation.json',{'counts':{c['name']:c['n_rows'] for c in collections},
        'config_sha256':digest(EXP/'config/inference_arms.json'),'call_budget':config['call_budget'],'gpu_started':False,
        'asset_count':len(assets),'styles':variants})
    print(json.dumps(config['call_budget'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();prepare()
