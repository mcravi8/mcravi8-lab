"""Independent real-pattern detector: causal close-only directional changes.

No stock-pattern imports, pivots, candidates, volume rules or source algorithms.
Volatility-normalized reversals and geometric ratios are frozen in real_rules.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RULES=json.loads((Path(__file__).resolve().parents[1]/'config/real_rules.json').read_text())['independent']


def true_range_average(frame, window=20):
    h,l,c=frame.High.to_numpy(),frame.Low.to_numpy(),frame.Close.to_numpy()
    previous=np.r_[c[0],c[:-1]]
    tr=np.maximum(h-l,np.maximum(abs(h-previous),abs(l-previous)))
    result=np.full(len(c),np.nan)
    if len(c)>=window:
        result[window-1:]=(np.r_[0.,np.cumsum(tr)][window:]-np.r_[0.,np.cumsum(tr)][:-window])/window
    return result


def directional_pivots(close, volatility, multiplier):
    """Return (position, price, high=1/low=-1, confirmation_position).

    A running extremum is confirmed only after a close reverses by the current
    causal threshold. No symmetric future neighborhood or unconfirmed last leg.
    """
    points=[];direction=0;extreme=0;initial=0
    for t in range(1,len(close)):
        if not np.isfinite(volatility[t]):
            extreme=t;initial=t;continue
        threshold=max(multiplier*volatility[t],RULES['minimum_reversal_fraction']*close[t])
        if direction==0:
            if abs(close[t]-close[initial])>=threshold:
                direction=1 if close[t]>close[initial] else -1
                points.append((initial,float(close[initial]),-direction,t));extreme=t
        elif direction==1:
            if close[t]>=close[extreme]:extreme=t
            elif close[extreme]-close[t]>=threshold:
                points.append((extreme,float(close[extreme]),1,t));direction=-1;extreme=t
        else:
            if close[t]<=close[extreme]:extreme=t
            elif close[t]-close[extreme]>=threshold:
                points.append((extreme,float(close[extreme]),-1,t));direction=1;extreme=t
    return points


def line(a,b):
    m=(b[1]-a[1])/(b[0]-a[0]);return float(m),float(a[1]-m*a[0])


def shape_candidates(points, t, volatility):
    """Return geometry hypotheses from confirmed adjacent turns, independently."""
    result=[]
    for count in (3,5):
        if len(points)<count:continue
        p=points[-count:]; x=np.array([v[0] for v in p]);y=np.array([v[1] for v in p]);sign=np.array([v[2] for v in p])
        if np.min(np.diff(x))<RULES['minimum_pivot_gap'] or not RULES['minimum_formation_bars']<=t-x[0]<=RULES['maximum_formation_bars'] or t-x[-1]>RULES['maximum_wait_for_break']:continue
        height=float(np.ptp(y))
        if height<RULES['minimum_height_atr']*volatility:continue
        def add(label,direction,m,b,asymmetry=0.):
            result.append({'label':label,'start':int(x[0]),'direction':direction,'slope':m,'intercept':b,
                           'height_atr':height/volatility,'asymmetry':asymmetry,'pivots':[int(v) for v in x]})
        if count==3 and np.array_equal(sign,[1,-1,1]) or count==3 and np.array_equal(sign,[-1,1,-1]):
            discrepancy=abs(y[0]-y[2])/height
            if discrepancy<=RULES['reversal_pair_tolerance_fraction_of_height']:
                add('Double Top' if sign[0]==1 else 'Double Bottom',-int(sign[0]),0.,float(y[1]),discrepancy)
        if count==5:
            for orientation,label in [(1,'Head and Shoulders'),(-1,'Inverse Head and Shoulders')]:
                if not np.array_equal(sign,np.array([1,-1,1,-1,1])*orientation):continue
                z=orientation*y
                if (z[2]-max(z[0],z[4])>=RULES['head_prominence_fraction_of_height']*height
                    and abs(z[0]-z[4])<=RULES['shoulder_tolerance_fraction_of_height']*height
                    and abs(z[1]-z[3])<=RULES['neckline_tilt_fraction_of_height']*height
                    and min(z[0],z[4])>max(z[1],z[3])):
                    add(label,-orientation,*line(p[1],p[3]),abs(z[0]-z[4])/height)
            high=[v for v in p if v[2]==1];low=[v for v in p if v[2]==-1]
            if len(high)>=2 and len(low)>=2:
                # Least squares envelopes over confirmed closing extrema.
                upper=np.polyfit([v[0] for v in high],[v[1] for v in high],1)
                lower=np.polyfit([v[0] for v in low],[v[1] for v in low],1)
                left=max(high[0][0],low[0][0]);right=max(high[-1][0],low[-1][0])
                width0=np.polyval(upper,left)-np.polyval(lower,left)
                width1=np.polyval(upper,right)-np.polyval(lower,right)
                tol=RULES['triangle_slope_tolerance_height_per_bar']*height
                if (upper[0]<=tol and lower[0]>=-tol and upper[0]<lower[0]
                    and width0>0 and 0<width1<=RULES['triangle_max_final_width_fraction']*width0
                    and np.polyval(upper,t)>np.polyval(lower,t)):
                    add('Triangle',1,float(upper[0]),float(upper[1]))
                    add('Triangle',-1,float(lower[0]),float(lower[1]))
    return result


def flag_candidates(close, t, volatility):
    """Pole plus downward close channel, independently fitted without pivots."""
    result=[]
    for length in RULES['flag_consolidation_bars']:
        peak=t-length-1
        first=max(0,peak-RULES['flag_pole_lookback'])
        if peak-first<5:continue
        start=first+int(np.argmin(close[first:peak+1]))
        pole=close[peak]-close[start]
        if pole<RULES['flag_min_pole_atr']*volatility or peak-start<3:continue
        channel=close[peak:t]; positions=np.arange(peak,t)
        slope,intercept=np.polyfit(positions,channel,1)
        residual=channel-(slope*positions+intercept)
        if (slope<0 and channel.max()<=close[peak]+.1*volatility
            and 0<(close[peak]-channel.min())/pole<=RULES['flag_max_retracement']
            and np.ptp(residual)<=RULES['flag_max_channel_width_fraction_of_pole']*pole
            and t-start<=RULES['maximum_formation_bars']):
            result.append({'label':'Bull Flag','start':start,'direction':1,'slope':0.,'intercept':float(close[peak]),
                           'height_atr':float(pole/volatility),'asymmetry':float(np.ptp(residual)/pole),'pivots':[start,peak]})
    return result


def scan(frame):
    close=frame.Close.to_numpy(float);volatility=true_range_average(frame,RULES['atr_window'])
    events=[];seen=set()
    for multiplier in RULES['reversal_atr_multipliers']:
        points=directional_pivots(close,volatility,multiplier);known=[];cursor=0
        for t in range(119,len(close)):
            while cursor<len(points) and points[cursor][3]<=t:
                known.append(points[cursor]);cursor+=1
            if not np.isfinite(volatility[t]):continue
            hypotheses=shape_candidates(known,t,volatility[t])
            if multiplier==RULES['reversal_atr_multipliers'][0]:hypotheses+=flag_candidates(close,t,volatility[t])
            for h in hypotheses:
                direction,m,b=h['direction'],h['slope'],h['intercept']
                margin=direction*(close[t]-(m*t+b))/volatility[t]
                # First close crossing for this boundary, not repeated above-level rows.
                previous=direction*(close[t-1]-(m*(t-1)+b))/volatility[t-1]
                if margin>RULES['breakout_atr_buffer'] and previous<=RULES['breakout_atr_buffer']:
                    key=(h['label'],h['start'],tuple(h['pivots']),direction)
                    if key not in seen:
                        events.append({**h,'completion':t,'detected_at':t,'input_start':max(0,t-119),
                                       'margin_atr':float(margin),'scale':multiplier})
                        seen.add(key)
    return events
