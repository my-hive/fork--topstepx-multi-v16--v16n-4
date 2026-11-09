"""Fade prior completed RTH return after delayed open, only after above-median prior range."""
import numpy as np

FAMILY = "index-prior-day-reversal"
PARAMS = {"rth_start": 30600, "rth_end": 54000, "entry": 34200, "exit": 53400, "range_days": 20}


def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if "NQ" not in panel:
        return out
    d = panel["NQ"]
    sec = d.ct_sec.to_numpy()
    session = d.session.astype(str).to_numpy()
    close = d.close.to_numpy()
    target = np.zeros(len(d), dtype=np.int64)
    history = []
    i = 0
    while i < len(d):
        j = i + 1
        while j < len(d) and session[j] == session[i]:
            j += 1
        rth = np.flatnonzero((sec[i:j] >= PARAMS["rth_start"]) & (sec[i:j] < PARAMS["rth_end"])) + i
        if len(rth):
            op, cl = close[rth[0]], close[rth[-1]]
            rng = close[rth].max() - close[rth].min()
            if history and rng >= np.median(history[-PARAMS["range_days"]:]):
                side = -1 if cl > op else 1
                trade = np.flatnonzero((sec[i:j] >= PARAMS["entry"]) & (sec[i:j] < PARAMS["exit"])) + i
                target[trade] = side
            history.append(rng)
        i = j
    out["NQ"] = target
    return out


def entry_risk(panel):
    return {s: np.full(len(d), 300.0) for s, d in panel.items()}
