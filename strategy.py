"""Soybean oil weekly reversal, RTH-only to avoid overnight gap churn."""
import numpy as np
FAMILY = "grain-soyoil-weekly-reversal"
PARAMS = {"lookback": 5, "start": 30600, "end": 52200, "stop_ticks": 16}
TICK_VALUE = 6.0

def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s,d in panel.items()}
    if 'ZL' not in panel: return out
    d=panel['ZL']; daily=d.close.groupby(d.session.astype(str)).last()
    side=-np.sign((daily/daily.shift(PARAMS['lookback'])-1).shift(1))
    x=side.reindex(d.session.astype(str)).fillna(0).to_numpy(); sec=d.ct_sec.to_numpy()
    out['ZL']=(x*((sec>=PARAMS['start'])&(sec<PARAMS['end']))).astype(np.int64)
    return out

def entry_risk(panel):
    return {s:np.full(len(d),PARAMS['stop_ticks']*TICK_VALUE) if s=='ZL' else np.zeros(len(d)) for s,d in panel.items()}
