"""Treasury auction-window pre-auction impulse continuation."""
import numpy as np
import pandas as pd
FAMILY = "rates-auction-impulse-follow"
PARAMS = {"observe_start": 42600, "entry": 43200, "exit": 46800, "move_ticks": 2}
def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if "ZN" not in panel: return out
    d = panel["ZN"]; sec=d.ct_sec.to_numpy(); session=d.session.astype(str).to_numpy(); close=d.close.to_numpy(); target=out["ZN"]
    for day in np.unique(session):
        ix=np.flatnonzero(session == day)
        if pd.Timestamp(day).weekday() not in (1,2,3): continue
        obs=ix[(sec[ix] >= PARAMS["observe_start"]) & (sec[ix] < PARAMS["entry"])]
        trade=ix[(sec[ix] >= PARAMS["entry"]) & (sec[ix] < PARAMS["exit"])]
        if len(obs) and len(trade):
            move=close[obs[-1]]-close[obs[0]]
            if abs(move) >= PARAMS["move_ticks"] / 64: target[trade]=1 if move > 0 else -1
    return out
def entry_risk(panel): return {s: np.full(len(d), 300.0) for s,d in panel.items()}
