"""Energy crack spread: fade extreme completed daily RB/CL return divergence."""
import numpy as np
import pandas as pd

FAMILY = "energy-crack-spread-reversion"
PARAMS = {"lookback": 20, "entry_z": 1.5, "start": 34200, "end": 53400, "stop_ticks": 20}
TICK_VALUE = {"CL": 10.0, "RB": 4.2}

def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if not {"CL", "RB"}.issubset(panel):
        return out
    c, r = panel["CL"], panel["RB"]
    cd = c.close.groupby(c.session.astype(str)).agg(['first', 'last'])
    rd = r.close.groupby(r.session.astype(str)).agg(['first', 'last'])
    crack = (rd['last'] / rd['first'] - 1).sub(cd['last'] / cd['first'] - 1)
    z = ((crack - crack.rolling(PARAMS['lookback']).mean()) / crack.rolling(PARAMS['lookback']).std()).shift(1)
    side = z.reindex(c.session.astype(str)).fillna(0).to_numpy()
    live = (c.ct_sec.to_numpy() >= PARAMS['start']) & (c.ct_sec.to_numpy() < PARAMS['end'])
    out['CL'] = (np.sign(side) * live).astype(np.int64)
    out['RB'] = (-np.sign(side) * live).astype(np.int64)
    return out

def entry_risk(panel):
    return {s: np.full(len(d), PARAMS['stop_ticks'] * TICK_VALUE[s]) if s in TICK_VALUE else np.zeros(len(d)) for s,d in panel.items()}
