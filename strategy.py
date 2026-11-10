"""Prior-RTH-return reversal on equity index futures, RTH-only, one position per symbol."""
import numpy as np

FAMILY = "index-prior-day-reversal"
PARAMS = {"start": 34200, "end": 54000, "stop_ticks": 10}
TICK = {"RTY": 5.0, "NQ": 5.0}

def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    for sym in TICK:
        if sym not in panel:
            continue
        d = panel[sym]
        g = d.close.groupby(d.session.astype(str))
        r = (g.last() / g.first() - 1.0).shift(1)
        side = (-np.sign(r)).reindex(d.session.astype(str)).fillna(0.0).to_numpy()
        sec = d.ct_sec.to_numpy()
        out[sym] = (side * ((sec >= PARAMS["start"]) & (sec < PARAMS["end"]))).astype(np.int64)
    return out

def entry_risk(panel):
    return {s: np.full(len(d), PARAMS["stop_ticks"] * TICK[s]) if s in TICK else np.zeros(len(d)) for s, d in panel.items()}
