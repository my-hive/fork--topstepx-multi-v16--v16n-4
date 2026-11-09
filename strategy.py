"""Rates turn-of-month flow: long 3-month note on last calendar days, flat before settlement."""
import numpy as np

FAMILY = "rates-ultra-turn-month"
PARAMS = {"month_day": 28, "start_sec": 30000, "exit_sec": 54600, "stop_points": 0.20}
PV = 1000.0


def signals(panel):
    p = PARAMS
    result = {}
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if symbol == "UB":
            sec = df.ct_sec.to_numpy()
            day = df.index.day.to_numpy()
            active = (day >= p["month_day"]) & (sec >= p["start_sec"]) & (sec < p["exit_sec"])
            out[active] = 1
        result[symbol] = out
    return result


def entry_risk(panel):
    p = PARAMS
    return {symbol: np.full(len(df), p["stop_points"] * PV) for symbol, df in panel.items()}
