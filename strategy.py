"""RB Friday late-RTH seasonal carry: long after 10:00 CT, flat 14:50."""
import numpy as np
import pandas as pd

FAMILY = "energy-friday-rth-carry"
PARAMS = {"entry": 36000, "exit": 53400, "weekday": 4, "stop_risk": 300}

def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if "RB" not in panel:
        return out
    d = panel["RB"]
    sec = d.ct_sec.to_numpy()
    friday = pd.to_datetime(d.session.astype(str)).dt.dayofweek.to_numpy() == PARAMS["weekday"]
    trade = friday & (sec >= PARAMS["entry"]) & (sec < PARAMS["exit"])
    out["RB"][trade] = 1
    return out

def entry_risk(panel):
    return {s: np.full(len(d), float(PARAMS["stop_risk"])) for s, d in panel.items()}
