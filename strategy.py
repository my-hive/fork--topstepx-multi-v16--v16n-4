"""Wednesday policy-window impulse continuation in 10-year Treasury futures."""
import numpy as np
import pandas as pd

FAMILY = "rates-wednesday-policy-reaction"
PARAMS = {"observe_start": 42600, "entry": 43800, "exit": 53100, "move_ticks": 4}


def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if "ZN" not in panel:
        return out
    d = panel["ZN"]
    sec, sessions, closes = d.ct_sec.to_numpy(), d.session.to_numpy(), d.close.to_numpy()
    target = np.zeros(len(d), dtype=np.int64)
    i = 0
    while i < len(d):
        j = i + 1
        while j < len(d) and sessions[j] == sessions[i]:
            j += 1
        if pd.Timestamp(sessions[i]).weekday() == 2:
            obs = np.flatnonzero((sec[i:j] >= PARAMS["observe_start"]) & (sec[i:j] < PARAMS["entry"])) + i
            trade = np.flatnonzero((sec[i:j] >= PARAMS["entry"]) & (sec[i:j] < PARAMS["exit"])) + i
            if len(obs) and len(trade):
                change = closes[obs[-1]] - closes[obs[0]]
                if abs(change) >= PARAMS["move_ticks"] / 64:
                    target[trade] = 1 if change > 0 else -1
        i = j
    out["ZN"] = target
    return out


def entry_risk(panel):
    return {s: np.full(len(d), 300.0) for s, d in panel.items()}
