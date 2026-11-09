"""Rates curve slope reversion: trade ZF when ZN/ZF close-ratio z-score reverts."""
import numpy as np
import pandas as pd

FAMILY = "rates-curve-slope-reversion"
PARAMS = {"lookback": 20, "entry_z": 1.5, "start": 34200, "end": 51000, "stop_ticks": 8}
TICK_VALUE = 15.625


def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if not {"ZF", "ZN"}.issubset(panel):
        return out
    f, n = panel["ZF"], panel["ZN"]
    fc = f.close.groupby(f.session.astype(str)).last()
    nc = n.close.groupby(n.session.astype(str)).last()
    ratio = (nc / fc).shift(1)
    roll = ratio.rolling(PARAMS["lookback"])
    z = (ratio - roll.mean()) / roll.std()
    zbar = z.reindex(f.session.astype(str)).fillna(0.0).to_numpy()
    sec = f.ct_sec.to_numpy()
    live = (sec >= PARAMS["start"]) & (sec < PARAMS["end"])
    sig = np.where(live & (zbar > PARAMS["entry_z"]), 1,
                   np.where(live & (zbar < -PARAMS["entry_z"]), -1, 0))
    out["ZF"] = sig.astype(np.int64)
    return out


def entry_risk(panel):
    return {s: (np.full(len(d), PARAMS["stop_ticks"] * TICK_VALUE) if s == "ZF"
                else np.zeros(len(d))) for s, d in panel.items()}
