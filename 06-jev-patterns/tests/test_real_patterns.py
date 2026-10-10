"""Independent geometry, causal pivots, completion, embargo and overlap fixtures."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'06b-chart-patterns/src'))
import external_detector as external
import zigzag_detector as zigzag
import detect_patterns as real


def bars(close):
    close=np.asarray(close,float);opening=np.r_[close[0],close[:-1]]
    return pd.DataFrame({'Open':opening,'High':np.maximum(close,opening)+.25,
                         'Low':np.minimum(close,opening)-.25,'Close':close,
                         'Volume':np.linspace(2e6,1e6,len(close))},
                         index=pd.bdate_range('2016-01-04',periods=len(close)))


class RealPatternTests(unittest.TestCase):
    def test_upstream_license_pin_and_required_functions(self):
        module=external.load_upstream()
        self.assertEqual(external.RULES['external']['license'],'GPL-3.0')
        self.assertEqual(len(external.FUNCTIONS),6)
        for name in external.FUNCTIONS.values():self.assertTrue(callable(getattr(module,name)))

    def test_exact_upstream_pivot_cache_including_ties_and_prefix_edges(self):
        rng=np.random.default_rng(50);frame=bars(100+np.cumsum(rng.normal(size=250)))
        frame.iloc[66:70,frame.columns.get_loc('High')]=frame.High.iloc[65]
        source=external.load_upstream();records=external.cached_pivots(frame)
        for end in [119,171,249]:
            start=end-119
            for low in [False,True]:
                actual=source.get_max_min(frame.iloc[start:end+1],pivot_type='low' if low else 'both')
                cached=external.pivot_frame(records,frame,start,end,low)
                pd.testing.assert_frame_equal(actual,cached,check_dtype=False,check_names=False)

    def test_independent_golden_reversal_and_triangle_shapes(self):
        cases=[('Head and Shoulders',[110,100,120,100,110],[1,-1,1,-1,1]),
               ('Double Top',[120,100,121],[1,-1,1]),
               ('Triangle',[120,100,116,104,112],[1,-1,1,-1,1])]
        for label,prices,signs in cases:
            positions=np.linspace(0,40,len(prices)).astype(int)
            points=[(int(i),float(p),s,int(i)+2) for i,p,s in zip(positions,prices,signs)]
            found=zigzag.shape_candidates(points,50,1.)
            self.assertIn(label,{r['label'] for r in found})
            if label!='Triangle':
                inverted=[(i,250-p,-s,q) for i,p,s,q in points]
                inverse='Inverse Head and Shoulders' if label=='Head and Shoulders' else 'Double Bottom'
                self.assertIn(inverse,{r['label'] for r in zigzag.shape_candidates(inverted,50,1.)})
        flat=[(i,100.,s,i+2) for i,s in zip([0,10,20,30,40],[1,-1,1,-1,1])]
        self.assertEqual(zigzag.shape_candidates(flat,50,1.),[])

    def test_independent_flag_requires_positive_pole_and_downward_channel(self):
        close=np.full(151,100.);close[120:146]=np.linspace(100,140,26)
        close[146:151]=[137,137.5,136,136.5,142]
        found=zigzag.flag_candidates(close,150,1.)
        self.assertTrue(any(r['label']=='Bull Flag' and r['intercept']==140 for r in found))
        close[120:]=100
        self.assertEqual(zigzag.flag_candidates(close,150,1.),[])

    def test_causal_confirmations_and_completed_event_prefix_stability(self):
        rng=np.random.default_rng(16);frame=bars(100+np.cumsum(rng.normal(size=320)))
        cut=240
        for multiplier in [1.5,2.5]:
            prefix=zigzag.directional_pivots(frame.Close.to_numpy()[:cut],zigzag.true_range_average(frame.iloc[:cut]),multiplier)
            full=zigzag.directional_pivots(frame.Close.to_numpy(),zigzag.true_range_average(frame),multiplier)
            self.assertEqual(prefix,[p for p in full if p[3]<cut])
            self.assertTrue(all(p[0]<p[3] for p in prefix))
        original=zigzag.scan(frame.iloc[:cut])
        frame.iloc[cut:,:4]*=2
        self.assertEqual(original,[r for r in zigzag.scan(frame) if r['completion']<cut])
        a,errors=external.scan(frame.iloc[:cut],'FIXTURE')
        b,other=external.scan(frame,'FIXTURE')
        self.assertEqual(a,[r for r in b if r['completion']<cut])
        self.assertEqual(errors,[e for e in other if e['end']<cut])

    def test_external_completion_boundaries_use_structure_not_last_close(self):
        index=pd.bdate_range('2010-01-01',periods=70)
        result={'points':{'A':(index[10],110),'B':(index[20],100),
                          'C':(index[30],120),'D':(index[40],102),'E':(index[50],111)}}
        direction,m,b=external.boundaries(result,'Head and Shoulders',index)[0]
        self.assertEqual(direction,-1);self.assertAlmostEqual(m*20+b,100);self.assertAlmostEqual(m*40+b,102)
        direction,m,b=external.boundaries(result,'Double Bottom',index)[0]
        self.assertEqual((direction,m,b),(1,0.,100.))

    def test_event_agreement_is_one_to_one_and_tolerates_small_timing_difference(self):
        def event(label,start,end):return {'label':label,'start':start,'completion':end}
        a=[event('Double Top',20,75),event('Double Top',25,76)]
        b=[event('Double Top',22,77),event('Double Bottom',20,75)]
        matched=real.match_events(a,b)
        self.assertEqual(len(matched),1)
        self.assertGreaterEqual(real.span_iou(*matched[0]),.5)
        self.assertEqual(len(real.collapse(a)),1)
        self.assertFalse(real.match_events([a[0]],[event('Double Top',60,130)]))

    def test_padding_fixed100_and_split_embargo_boundaries(self):
        for span in range(44,87):
            first,last,padding=real.frame_bounds(200,199+span)
            self.assertTrue(60<=last-first+1<=120)
            self.assertTrue(.1<=padding/span<=.2)
            self.assertGreaterEqual(200,last-99)
        self.assertIsNone(real.frame_bounds(200,220));self.assertIsNone(real.frame_bounds(200,300))
        dates=pd.bdate_range('2018-01-01','2020-01-01')
        pos=lambda value:dates.get_loc(pd.Timestamp(value))
        self.assertEqual(real.split_window(dates,pos('2018-12-03'),pos('2018-12-31')),'train')
        self.assertIsNone(real.split_window(dates,pos('2018-12-31'),pos('2019-01-02')))
        self.assertIsNone(real.split_window(dates,pos('2019-03-29'),pos('2019-04-30')))
        self.assertEqual(real.split_window(dates,pos('2019-04-01'),pos('2019-08-30')),'test')

    def test_published_selection_has_no_overlap_or_split_crossing(self):
        index=pd.read_csv(ROOT/'06b-chart-patterns/results/real_index.csv',keep_default_na=False)
        fixed=pd.read_csv(ROOT/'06b-chart-patterns/results/real_fixed100_index.csv',keep_default_na=False)
        self.assertEqual(index.id.tolist(),fixed.id.tolist())
        self.assertTrue(fixed.n_bars.eq(100).all())
        self.assertTrue(index.n_bars.between(60,120).all())
        self.assertTrue((pd.to_datetime(index.detector_input_start)<=pd.to_datetime(index.window_start)).all())
        for split,group in index.groupby('split'):
            if split=='train':
                self.assertTrue(pd.to_datetime(group.end_date).le('2018-12-31').all())
            else:
                self.assertEqual(split,'test')
                for column in ['window_start','fixed_window_start','detector_input_start']:
                    self.assertTrue(pd.to_datetime(group[column]).ge('2019-04-01').all())
        for ticker,group in index.groupby('ticker'):
            ordered=group.sort_values('coverage_start_bar')
            self.assertTrue((ordered.end_bar.to_numpy()[:-1]<ordered.coverage_start_bar.to_numpy()[1:]).all())
        positive=index[index.label!='None']
        span=positive.completion_bar-positive.pattern_start_bar+1
        self.assertTrue((positive.padding_bars/span).between(.1,.2).all())
        self.assertTrue((positive.completion_bar<=positive.end_bar).all())
        self.assertTrue((positive.pattern_start_bar>=positive.end_bar-99).all())
        # None must not advertise itself by one constant variable-frame width.
        self.assertGreater(index[index.label=='None'].n_bars.nunique(),5)
        self.assertEqual(set(index.label),set(real.CLASSES))

    def test_overlap_purge_is_across_all_classes_and_both_framings(self):
        rows=[]
        for i,(label,start,end) in enumerate([('Head and Shoulders',0,99),('Double Top',60,159),('None',160,259),('Double Bottom',260,359)]):
            rows.append({'id':str(i),'ticker':'A','year':2015,'split':'train','label':label,
                         'coverage_start_bar':start,'end_bar':end})
        selected=real.sample_nonoverlapping(rows)
        for i,a in enumerate(selected):
            for b in selected[i+1:]:self.assertTrue(a['end_bar']<b['coverage_start_bar'] or b['end_bar']<a['coverage_start_bar'])
        self.assertEqual(selected,real.sample_nonoverlapping(list(reversed(rows))))


if __name__=='__main__':unittest.main()
