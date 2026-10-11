"""CPU real-pattern scan, independent event agreement, purged selection and charts."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT.parent
sys.path.insert(0,str(EXP));sys.path.insert(0,str(ROOT/'src'))
import external_detector
import zigzag_detector
from shared.src.data import validate_ohlcv
from shared.src.render import render_chart
RULES_PATH=ROOT/'config/real_rules.json'
RULES=json.loads(RULES_PATH.read_text())
CLASSES=json.loads((ROOT/'options.json').read_text())


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def span_iou(a,b):
    overlap=max(0,min(a['completion'],b['completion'])-max(a['start'],b['start'])+1)
    union=max(a['completion'],b['completion'])-min(a['start'],b['start'])+1
    return overlap/union


def collapse(events):
    """Collapse repeated same-class event proposals, before detector comparison."""
    kept=[]
    for event in sorted(events,key=lambda e:(e['completion'],e['label'],e['start'])):
        if any(e['label']==event['label'] and abs(e['completion']-event['completion'])<=5 and span_iou(e,event)>=.5
               for e in kept[-100:]):continue
        kept.append(event)
    return kept


def match_events(left,right):
    used=set();matches=[]
    for a in left:
        choices=[(abs(a['completion']-b['completion']),-span_iou(a,b),i,b) for i,b in enumerate(right)
                 if i not in used and a['label']==b['label'] and abs(a['completion']-b['completion'])<=RULES['agreement']['completion_distance_bars']
                 and span_iou(a,b)>=RULES['agreement']['minimum_span_iou']]
        if choices:
            _,_,i,b=min(choices,key=lambda x:x[:3]);used.add(i);matches.append((a,b))
    return matches


def split_window(index,start,end):
    if start<0 or end>=len(index) or start>end:return None
    if index[end]<=pd.Timestamp(RULES['splits']['train_end']):return 'train'
    if index[start]>=pd.Timestamp(RULES['splits']['test_start']):return 'test'
    return None


def frame_bounds(start,completion):
    span=completion-start+1;cfg=RULES['framing']
    if not cfg['minimum_pattern_span']<=span<=cfg['maximum_pattern_span']:return None
    padding=max(math.ceil(cfg['padding_fraction']*span),math.ceil((60-span)/2))
    if not .1<=padding/span<=.2:return None
    first,last=start-padding,completion+padding
    if not 60<=last-first+1<=120 or start<last-99:return None
    return first,last,padding


def stable_order(row):
    return hashlib.sha256(f"{RULES['seed']}:{row['id']}".encode()).hexdigest()


def sample_nonoverlapping(candidates):
    """Rare-first class order; deterministic ticker/year cells, all-class overlap."""
    selected=[];occupied=defaultdict(list)
    counts=Counter(r['label'] for r in candidates)
    for label in sorted(CLASSES,key=lambda label:(counts[label],CLASSES.index(label))):
        for split in ['train','test']:
            cells=defaultdict(list)
            for row in candidates:
                if row['label']==label and row['split']==split:cells[(row['ticker'],row['year'])].append(row)
            for values in cells.values():values.sort(key=stable_order)
            keys=sorted(cells);number=0
            while keys and number<RULES['sampling']['target_per_split']:
                remaining=[]
                for key in keys:
                    while cells[key]:
                        row=cells[key].pop(0);start,end=row['coverage_start_bar'],row['end_bar']
                        if any(start<=b and end>=a for a,b in occupied[row['ticker']]):continue
                        selected.append(row);occupied[row['ticker']].append((start,end));number+=1;break
                    if cells[key]:remaining.append(key)
                    if number>=RULES['sampling']['target_per_split']:break
                keys=remaining
    return sorted(selected,key=lambda r:(CLASSES.index(r['label']),r['split'],stable_order(r)))


def scan_ticker(ticker):
    path=EXP/'shared/data'/f'{ticker}.parquet';frame=pd.read_parquet(path);validate_ohlcv(frame)
    raw_a,errors=external_detector.scan(frame,ticker);raw_b=zigzag_detector.scan(frame)
    a,b=collapse(raw_a),collapse(raw_b);matches=match_events(a,b)
    stats={label:{'external_events':sum(e['label']==label for e in a),
                  'zigzag_events':sum(e['label']==label for e in b),
                  'matched_events':sum(x['label']==label for x,y in matches)} for label in CLASSES[:-1]}
    candidates=[];excluded=Counter();n=len(frame);close=frame.Close.to_numpy()
    def row_base(label,start,end,pattern_start,completion,input_start,padding,details):
        first=min(start,end-99);split=split_window(frame.index,min(first,input_start),end)
        if split is None:excluded['split_or_history_boundary']+=1;return
        # Exclude data intervals affected by upstream exceptions.
        if any(input_start<=e['end']<=end for e in errors):excluded['external_error_window']+=1;return
        identity=hashlib.sha256(f'{ticker}:{label}:{frame.index[start].date()}:{frame.index[end].date()}'.encode()).hexdigest()[:20]
        return {'id':identity,'ticker':ticker,'label':label,'split':split,'year':int(frame.index[end].year),
                'window_start':str(frame.index[start].date()),'fixed_window_start':str(frame.index[end-99].date()),
                'end_date':str(frame.index[end].date()),'detector_input_start':str(frame.index[input_start].date()),
                'start_bar':start,'end_bar':end,'coverage_start_bar':first,'pattern_start_bar':pattern_start,
                'completion_bar':completion,'padding_bars':padding,'n_bars':end-start+1,**details}
    for x,y in matches:
        start=min(x['start'],y['start']);complete=max(x['completion'],y['completion'])
        bounds=frame_bounds(start,complete)
        if bounds is None:excluded['span_or_padding']+=1;continue
        first,last,padding=bounds
        if min(first,last-99)<0 or last>=n:excluded['insufficient_context_or_tail']+=1;continue
        if any(e['label']!=x['label'] and e['completion']<=last and abs(e['completion']-complete)<=10 and span_iou(e,{'start':start,'completion':complete})>=.3 for e in a+b):
            excluded['multiple_labels']+=1;continue
        final_margins=[e['direction']*(close[last]-(e['slope']*last+e['intercept'])) for e in [x,y]]
        if min(final_margins)<=0:excluded['completion_not_retained']+=1;continue
        warnings=[]
        if min(x['margin_atr'],y['margin_atr'])<.5:warnings.append('weak_breakout')
        if abs(x['completion']-y['completion'])>2:warnings.append('completion_disagreement')
        if y['asymmetry']>.15:warnings.append('asymmetric_geometry')
        if y['height_atr']<4:warnings.append('small_pattern')
        row=row_base(x['label'],first,last,start,complete,min(x['input_start'],y['input_start'],first,last-99),padding,
                     {'subtle':bool(warnings),'subtle_reasons':';'.join(warnings),'visual_review':'pending_user',
                      'external_completion':str(frame.index[x['completion']].date()),'zigzag_completion':str(frame.index[y['completion']].date()),
                      'event_span_iou':span_iou(x,y),'external_event':x,'zigzag_event':y})
        if row:candidates.append(row)
    none_counts=Counter()
    for end in range(259,n,RULES['none']['stride']):
        start=end-119;input_start=end-259
        if split_window(frame.index,input_start,end) is None:continue
        if any(input_start<=e['end']<=end for e in errors):continue
        has_a=any(e['completion']<=end and e['completion']>=start and e['start']<=end for e in a)
        has_b=any(e['completion']<=end and e['completion']>=start and e['start']<=end for e in b)
        none_counts['eligible_windows']+=1;none_counts['external_none']+=not has_a;none_counts['zigzag_none']+=not has_b
        none_counts['both_none']+=not has_a and not has_b
        if not has_a and not has_b:
            row=row_base('None',end-99,end,end-99,end,input_start,0,{'subtle':True,'subtle_reasons':'absence_is_taxonomy_limited',
                         'visual_review':'pending_user','external_completion':'','zigzag_completion':'','event_span_iou':1.,
                         'external_event':None,'zigzag_event':None})
            if row:candidates.append(row)
    return {'ticker':ticker,'input_sha256':digest(path),'rows':n,'first_session':str(frame.index[0].date()),
            'last_session':str(frame.index[-1].date()),'external_scan_anchors':max(0,n-119),
            'stats':stats,'none_stats':dict(none_counts),'raw_event_counts':{'external':len(raw_a),'zigzag':len(raw_b)},
            'external_errors':errors,'excluded':dict(excluded),'candidates':candidates,
            'external_events':a,'zigzag_events':b}


def review_grids(index,root=ROOT):
    out=root/'results/samples/real';out.mkdir(parents=True,exist_ok=True)
    records=[];font=ImageFont.load_default(size=15)
    for label in CLASSES:
        selected=[]
        for split in ['train','test']:
            part=index[(index.label==label)&(index.split==split)].sort_values('id').head(6)
            selected+=part.to_dict('records')
        w,h=384,272;grid=Image.new('RGB',(3*w,4*h+42),'white');draw=ImageDraw.Draw(grid)
        draw.text((8,8),f'{label}: {len(selected)} fixed review examples; * = rule-based subtle flag',fill='black',font=font)
        for i in range(12):
            x,y=(i%3)*w,42+(i//3)*h
            if i<len(selected):
                row=selected[i]
                with Image.open(root/row['file']) as image:grid.paste(image.resize((384,240),Image.Resampling.LANCZOS),(x,y+30))
                draw.text((x+3,y+3),f"{row['id'][:6]} {row['ticker']} {row['end_date']} {'*' if row['subtle'] else ''}",fill='black',font=font)
                records.append({'label':label,'id':row['id'],'file':row['file'],'ticker':row['ticker'],'end_date':row['end_date'],
                                'split':row['split'],'subtle':row['subtle'],'subtle_reasons':row['subtle_reasons'],
                                'visual_review':'pending_user'})
            else:draw.text((x+20,y+90),'Insufficient eligible examples',fill='gray',font=font)
        grid.save(out/(label.lower().replace(' ','_')+'_grid.png'))
    write_json(out/'review_manifest.json',{'selection':RULES['review'],'reviewer':'User review pending', 'charts':records})


def build(reports,no_render=False):
    candidates=[r for report in reports for r in report['candidates']]
    # A constant None length would itself identify the label. Match the pooled
    # positive length distribution separately within train/test before sampling.
    for split in ['train','test']:
        lengths=sorted(r['n_bars'] for r in candidates if r['label']!='None' and r['split']==split)
        if not lengths:
            raise ValueError(f'No positive lengths available to match None in {split}')
        for row in candidates:
            if row['label']!='None' or row['split']!=split:continue
            width=lengths[int(stable_order(row),16)%len(lengths)]
            frame=pd.read_parquet(EXP/'shared/data'/f"{row['ticker']}.parquet")
            row['n_bars']=width;row['start_bar']=row['end_bar']-width+1
            row['window_start']=str(frame.index[row['start_bar']].date())
            row['coverage_start_bar']=min(row['start_bar'],row['end_bar']-99)
            row['pattern_start_bar']=row['start_bar']
            row['id']=hashlib.sha256(f"{row['ticker']}:None:{row['window_start']}:{row['end_date']}".encode()).hexdigest()[:20]
    selected=sample_nonoverlapping(candidates);rows=[];artifacts=[];frames={}
    (ROOT/'data/real').mkdir(parents=True,exist_ok=True);(ROOT/'charts/real').mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['external_event','zigzag_event']} for r in candidates]).to_csv(ROOT/'data/real/candidates.csv',index=False)
    for number,r in enumerate(selected):
        ticker=r['ticker']
        if ticker not in frames:frames[ticker]=pd.read_parquet(EXP/'shared/data'/f'{ticker}.parquet')
        full=frames[ticker];variable=full.iloc[r['start_bar']:r['end_bar']+1];fixed=full.iloc[r['end_bar']-99:r['end_bar']+1]
        assert len(fixed)==100 and 60<=len(variable)<=120
        assert split_window(full.index,full.index.get_loc(pd.Timestamp(r['detector_input_start'])),r['end_bar'])==r['split']
        row={k:v for k,v in r.items() if k not in ['external_event','zigzag_event']}
        for kind,frame in [('variable',variable),('fixed100',fixed)]:
            png=Path(f'charts/real/{r["id"]}_{kind}.png');bars=Path(f'data/real/{r["id"]}_{kind}.csv')
            frame.to_csv(ROOT/bars,index_label='Date',float_format='%.12g')
            if not no_render:render_chart(frame,ROOT/png,n_bars=len(frame),size=(1024,640),volume_color='neutral')
            row['file' if kind=='variable' else 'fixed_file']=png.as_posix()
            row['bars_file' if kind=='variable' else 'fixed_bars_file']=bars.as_posix()
            artifacts.append({'id':r['id'],'framing':kind,'file':png.as_posix(),'bars_file':bars.as_posix(),
                              'png_sha256':None if no_render else digest(ROOT/png),'bars_sha256':digest(ROOT/bars)})
        row['framing']='variable'
        rows.append(row)
        if (number+1)%100==0:print(f'Rendered {number+1}/{len(selected)} real parents (two framings)',flush=True)
    index=pd.DataFrame(rows)
    index.to_csv(ROOT/'results/real_index.csv',index=False)
    fixed=index.copy();fixed['file']=fixed.fixed_file;fixed['bars_file']=fixed.fixed_bars_file;fixed['window_start']=fixed.fixed_window_start;fixed['start_bar']=fixed.end_bar-99;fixed['n_bars']=100;fixed['framing']='fixed_100'
    fixed.to_csv(ROOT/'results/real_fixed100_index.csv',index=False)
    if not no_render:review_grids(index)
    agreement={}
    for label in CLASSES[:-1]:
        counts={key:sum(r['stats'][label][key] for r in reports) for key in ['external_events','zigzag_events','matched_events']}
        union=counts['external_events']+counts['zigzag_events']-counts['matched_events']
        agreement[label]={**counts,'positive_jaccard':counts['matched_events']/union if union else None,'denominator':'one-to-one union of completed detector events before span, split, exclusivity and sample filters'}
    ns={key:sum(r['none_stats'].get(key,0) for r in reports) for key in ['eligible_windows','external_none','zigzag_none','both_none']}
    union=ns['external_none']+ns['zigzag_none']-ns['both_none']
    agreement['None']={**ns,'positive_jaccard':ns['both_none']/union if union else None,'denominator':'union of negative decisions on prespecified 120-bar absence-screen windows'}
    counts={label:{'eligible_train':sum(r['label']==label and r['split']=='train' for r in candidates),
                  'eligible_test':sum(r['label']==label and r['split']=='test' for r in candidates),
                  'selected_train':sum(r['label']==label and r['split']=='train' for r in selected),
                  'selected_test':sum(r['label']==label and r['split']=='test' for r in selected)} for label in CLASSES}
    for record in counts.values():
        record['selected']=record['selected_train']+record['selected_test'];record['below_target_80']=record['selected']<80
    write_json(ROOT/'results/real_agreement.json',{'interpretation':RULES['agreement']['rate'],'classes':agreement,
               'external_errors':sum(len(r['external_errors']) for r in reports),'excluded':dict(sum((Counter(r['excluded']) for r in reports),Counter()))})
    write_json(ROOT/'results/real_label_counts.json',{'classes':counts,'total':len(index),'subtle_flagged':int(index.subtle.sum()),'human_visual_review':'pending'})
    write_json(ROOT/'results/real_selection_details.json',selected)
    manifest={'rules_sha256':digest(RULES_PATH),'sources_sha256':{name:digest(ROOT/'src'/name) for name in ['detect_patterns.py','external_detector.py','zigzag_detector.py']},
              'renderer_sha256':digest(EXP/'shared/src/render.py'),'data_manifest_sha256':digest(EXP/'shared/results/data_manifest.json'),
              'inputs':[{k:r[k] for k in ['ticker','input_sha256','rows','first_session','last_session','external_scan_anchors']} for r in reports],
              'external':RULES['external'],'n_real_parents':len(index),'n_images':0 if no_render else 2*len(index),
              'real_index_sha256':digest(ROOT/'results/real_index.csv'),'fixed_index_sha256':digest(ROOT/'results/real_fixed100_index.csv'),
              'no_gpu_or_model_calls':True,'artifacts':artifacts}
    write_json(ROOT/'results/real_render_manifest.json',manifest)
    print(json.dumps(counts,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--reuse-scan',action='store_true');p.add_argument('--no-render',action='store_true');args=p.parse_args()
    if not 1<=args.workers<=8:p.error('Use1 through8 CPU workers')
    data_manifest=json.loads((EXP/'shared/results/data_manifest.json').read_text())
    cache=ROOT/'data/real_scan';cache.mkdir(parents=True,exist_ok=True)
    sources={name:digest(ROOT/'src'/name) for name in ['external_detector.py','zigzag_detector.py']}
    stamp={'rules_sha256':digest(RULES_PATH),'source_sha256':sources,'data_manifest_sha256':digest(EXP/'shared/results/data_manifest.json')}
    # Reusing detector events must not render against changed raw snapshots.
    for ticker,info in data_manifest['tickers'].items():
        if digest(EXP/'shared/data'/f'{ticker}.parquet')!=info['sha256']:raise ValueError('Input snapshot hash changed')
    if args.reuse_scan:
        if json.loads((cache/'provenance.json').read_text())!=stamp:raise ValueError('Cached scan provenance changed')
        reports=[json.loads((cache/f'{ticker}.json').read_text()) for ticker in data_manifest['tickers']]
    else:
        external_detector.load_upstream(fetch=True)
        reports=[];started=time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(scan_ticker,t):t for t in data_manifest['tickers']}
            for future in as_completed(futures):
                report=future.result();reports.append(report);write_json(cache/f'{report["ticker"]}.json',report)
                print(f"Scanned {report['ticker']} ({len(reports)}/50), {len(report['candidates'])} eligible, {len(report['external_errors'])} exceptions; elapsed {time.monotonic()-started:.0f}s",flush=True)
        write_json(cache/'provenance.json',stamp)
    reports.sort(key=lambda r:r['ticker']);build(reports,args.no_render)


if __name__=='__main__':main()
