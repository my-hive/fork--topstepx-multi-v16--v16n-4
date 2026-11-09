"""ZN NR7 compression: vectorized session-grouped logic."""
import numpy as np
import pandas as pd

FAMILY = "rates-nr7-breakout"
PARAMS = {"range_days": 7, "observe_start": 30600, "entry": 32400, "exit": 53400, "stop_risk": 300}

def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if "ZN" not in panel:
        return out
    d = panel["ZN"]
    sec = d.ct_sec.to_numpy()
    high = d.high.to_numpy()
    low = d.low.to_numpy()
    target = out["ZN"]
    
    # Fast session split
    ses = d.session.astype(str).to_numpy()
    change_idx = np.flatnonzero(ses[1:] != ses[:-1]) + 1
    session_starts = np.r_[0, change_idx]
    session_ends = np.r_[change_idx, len(ses)]
    
    # 1. Compute per-session high and low across RTH window
    n_sessions = len(session_starts)
    obs_dir = np.zeros(n_sessions, dtype=np.int8)
    rth_rng = np.zeros(n_sessions, dtype=np.float64)
    has_trade = np.zeros(n_sessions, dtype=bool)

    obs_s = PARAMS["observe_start"]
    obs_e = PARAMS["entry"]
    exit_s = PARAMS["exit"]

    for s_idx in range(n_sessions):
        s_start = session_starts[s_idx]
        s_end = session_ends[s_idx]
        s_sec = sec[s_start:s_end]
        
        # mask within session
        m_rth = (s_sec >= obs_s) & (s_sec < exit_s)
        if np.any(m_rth):
            rth_rng[s_idx] = high[s_start:s_end][m_rth].max() - low[s_start:s_end][m_rth].min()
        
        m_obs = (s_sec >= obs_s) & (s_sec < obs_e)
        m_tr = (s_sec >= obs_e) & (s_sec < exit_s)
        if np.any(m_obs) and np.any(m_tr):
            has_trade[s_idx] = True
            h_obs = high[s_start:s_end][m_obs]
            l_obs = low[s_start:s_end][m_obs]
            up = h_obs.max() - h_obs[0]
            dn = l_obs[0] - l_obs.min()
            obs_dir[s_idx] = 1 if up >= dn else -1

    # 2. NR7 filter: prior session's rth_rng is strictly lowest of past range_days
    K = PARAMS["range_days"]
    for s_idx in range(K, n_sessions):
        if not has_trade[s_idx]:
            continue
        # check if s_idx - 1 was NR K
        prior_window = rth_rng[s_idx - K : s_idx]
        if prior_window[-1] > 0 and prior_window[-1] <= np.min(prior_window):
            d_val = obs_dir[s_idx]
            s_start = session_starts[s_idx]
            s_end = session_ends[s_idx]
            s_sec = sec[s_start:s_end]
            m_tr = (s_sec >= obs_e) & (s_sec < exit_s)
            target[s_start:s_end][m_tr] = d_val

    return out

def entry_risk(panel):
    return {s: np.full(len(d), float(PARAMS["stop_risk"])) for s, d in panel.items()}
