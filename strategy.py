"""MGC stop-run reversion after prior completed-session extreme breach."""
import numpy as np

FAMILY = "metals-stop-run-prior-day-extremes"
PARAMS = {"start": 30600, "end": 46800, "window": 10, "stop_ticks": 10}
TICK_VALUE = 10.0


def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if "MGC" not in panel:
        return out
    d = panel["MGC"]
    session = d.session
    prior_high = d.high.groupby(session).max().shift(1)
    prior_low = d.low.groupby(session).min().shift(1)
    # Map prior completed session levels; current session never enters its own level.
    levels = np.unique(session.to_numpy())
    highs = d.high.groupby(session).max()
    lows = d.low.groupby(session).min()
    hi_map = dict(zip(levels[1:], highs.reindex(levels[:-1]).to_numpy()))
    lo_map = dict(zip(levels[1:], lows.reindex(levels[:-1]).to_numpy()))
    hi = session.map(hi_map).to_numpy(dtype=float)
    lo = session.map(lo_map).to_numpy(dtype=float)
    close = d.close.to_numpy()
    was_above = (close >= hi)
    was_below = (close <= lo)
    touch_high = np.maximum.accumulate(np.where(was_above, np.arange(len(d)), -PARAMS["window"] - 1))
    touch_low = np.maximum.accumulate(np.where(was_below, np.arange(len(d)), -PARAMS["window"] - 1))
    recent_high = np.arange(len(d)) - touch_high <= PARAMS["window"]
    recent_low = np.arange(len(d)) - touch_low <= PARAMS["window"]
    short = recent_high & (close < hi)
    long = recent_low & (close > lo)
    sec = d.ct_sec.to_numpy()
    live = (sec >= PARAMS["start"]) & (sec < PARAMS["end"])
    out["MGC"] = np.where(live, long.astype(int) - short.astype(int), 0)
    return out


def entry_risk(panel):
    return {s: np.full(len(d), PARAMS["stop_ticks"] * TICK_VALUE) if s == "MGC" else np.zeros(len(d)) for s, d in panel.items()}
