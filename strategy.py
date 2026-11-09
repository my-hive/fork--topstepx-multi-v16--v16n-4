"""Flight-to-quality: after sharp ES selloff, hold ZN long for fixed window (event study)."""
import numpy as np
import pandas as pd

FAMILY = "rates-equity-shock-flight-to-quality"
PARAMS = {"ret_min": 30, "shock": 0.004, "hold_min": 90, "stop_ticks": 16}
TICK_VALUE = 1000 / 64
EXEC, TGT = "ZN", "ES"

def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if not {EXEC, TGT}.issubset(panel):
        return out
    e, z = panel[TGT], panel[EXEC]
    w = PARAMS["ret_min"]
    r = (e.close / e.close.shift(w) - 1.0)
    trig = (r < -PARAMS["shock"]).reindex(z.index, method="ffill").fillna(False).to_numpy()
    t = z.index.as_unit('ns').asi8
    hold = PARAMS["hold_min"] * 60 * 1_000_000_000
    pos = np.zeros(len(z), dtype=np.int64)
    entry = None
    for i in range(len(z)):
        if entry is not None and t[i] - entry >= hold:
            entry = None
        if entry is None and trig[min(i, len(trig) - 1)]:
            entry = t[i]
        pos[i] = 0 if entry is None else 1
    out[EXEC] = pos
    return out

def entry_risk(panel):
    return {s: (np.full(len(d), PARAMS["stop_ticks"] * TICK_VALUE) if s == EXEC
                else np.zeros(len(d))) for s, d in panel.items()}
