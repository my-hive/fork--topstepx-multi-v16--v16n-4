"""Hedged ES/NQ intraday relative-value fade of the index ratio vs prior close."""
import numpy as np
import pandas as pd

FAMILY = "index-es-nq-ratio-reversion"
PARAMS = {"ratio_lookback": 20, "entry_z": 1.0, "exit_z": 0.3, "start": 36000, "end": 54000, "stop_ticks": 4}
TICK = {"ES": 12.5, "NQ": 5.0}

def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if not {"ES", "NQ"}.issubset(panel):
        return out
    es, nq = panel["ES"], panel["NQ"]
    nqc = nq.close.reindex(es.index, method="ffill")
    ratio = (es.close / nqc).to_numpy()
    sess = es.session.astype(str)
    daily = (es.close / nqc).groupby(sess).last()
    base = daily.shift(1).reindex(sess).to_numpy()
    sigma = daily.diff().rolling(PARAMS["ratio_lookback"]).std().shift(1).reindex(sess).to_numpy()
    z = (ratio - base) / sigma
    sec = es.ct_sec.to_numpy()
    live = (sec >= PARAMS["start"]) & (sec < PARAMS["end"])
    pos = np.zeros(len(z), dtype=np.int64)
    state = 0
    for i in range(len(z)):
        if not np.isfinite(z[i]):
            state = 0
        elif state == 0:
            if live[i] and z[i] >= PARAMS["entry_z"]:
                state = -1
            elif live[i] and z[i] <= -PARAMS["entry_z"]:
                state = 1
        elif not live[i] or abs(z[i]) <= PARAMS["exit_z"]:
            state = 0
        pos[i] = state
    s_es = pd.Series(pos, index=es.index)
    out["ES"] = pos
    out["NQ"] = (-s_es).reindex(nq.index, method="ffill").fillna(0).astype(np.int64).to_numpy()
    return out

def entry_risk(panel):
    return {s: np.full(len(d), PARAMS["stop_ticks"] * TICK[s]) if s in TICK else np.zeros(len(d)) for s, d in panel.items()}
