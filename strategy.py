"""6E London-to-New-York session handoff momentum."""
import numpy as np

FAMILY = "fx-london-ny-handoff"
PARAMS = {"observe_start": 21600, "entry": 28800, "exit": 43200, "move": 0.0010, "stop_risk": 300}

def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if "6E" not in panel:
        return out
    d = panel["6E"]
    sec = d.ct_sec.to_numpy(); close = d.close.to_numpy(); ses = d.session.astype(str).to_numpy(); target = out["6E"]
    cuts = np.r_[0, np.flatnonzero(ses[1:] != ses[:-1]) + 1, len(d)]
    for start, end in zip(cuts[:-1], cuts[1:]):
        xsec = sec[start:end]
        obs = np.flatnonzero((xsec >= PARAMS["observe_start"]) & (xsec < PARAMS["entry"])) + start
        trade = np.flatnonzero((xsec >= PARAMS["entry"]) & (xsec < PARAMS["exit"])) + start
        if len(obs) and len(trade):
            move = close[obs[-1]] - close[obs[0]]
            if abs(move) >= PARAMS["move"]:
                target[trade] = 1 if move > 0 else -1
    return out

def entry_risk(panel):
    return {s: np.full(len(d), float(PARAMS["stop_risk"])) for s, d in panel.items()}
