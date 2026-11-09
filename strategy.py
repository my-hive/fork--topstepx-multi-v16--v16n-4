"""ZN ZB combined 30d trend + 20d mean-reversion gate; long/short per contract."""
import numpy as np
import pandas as pd

FAMILY = "rates-trend-mr-gate"
PARAMS = {"trend_n": 30, "mr_n": 20, "stop_ticks": 8, "exit_hhmm": 51000}

TICK = {"ZN": 31.25, "ZB": 31.25}

def _feat(d, p):
    close = d["close"].astype(float)
    ses = d["session"].astype(str)
    daily_close = close.groupby(ses).last()
    # Shift at daily granularity: today's completed close never informs today.
    prior = daily_close.shift(1)
    trend = prior.rolling(p["trend_n"]).mean().diff()
    ret = prior.diff()
    mr = -(ret - ret.rolling(p["mr_n"]).mean())
    trend_b = trend.reindex(ses).fillna(0.0).to_numpy()
    mr_b = mr.reindex(ses).fillna(0.0).to_numpy()
    sec = d["ct_sec"].to_numpy()
    active = (sec >= 30600) & (sec < PARAMS["exit_hhmm"])
    sig = np.where(active & (trend_b > 0) & (mr_b > 0), 1,
          np.where(active & (trend_b < 0) & (mr_b < 0), -1, 0))
    return sig

def signals(panel):
    out = {}
    for s, d in panel.items():
        if s in TICK:
            out[s] = _feat(d, PARAMS).astype(np.int64)
        else:
            out[s] = np.zeros(len(d), dtype=np.int64)
    return out

def entry_risk(panel):
    return {s: np.full(len(d), PARAMS["stop_ticks"] * TICK[s])
            if s in TICK else np.zeros(len(d)) for s, d in panel.items()}
