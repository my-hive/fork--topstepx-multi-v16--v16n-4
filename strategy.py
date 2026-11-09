"""Rates curve steepener: ZB-vs-ZF spread, trade the morning slope change on the common timeline."""
import numpy as np, pandas as pd
FAMILY = 'rates-curve-steepener-intraday'
PARAMS = {'obs_start': 30000, 'entry': 34200, 'exit': 54000, 'thr': 0.10}

def signals(panel):
    p = PARAMS
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if 'ZB' in panel and 'ZF' in panel:
        a, b = panel['ZB'], panel['ZF']
        idx = a.index.intersection(b.index)
        zb, zf = a.loc[idx], b.loc[idx]
        same = (zb.session.to_numpy() == zf.session.to_numpy()) & (zb.contract.to_numpy() == zb.contract.shift().bfill().to_numpy())
        sec, ses = zb.ct_sec.to_numpy(), zb.session.to_numpy()
        cb, cf = zb.close.to_numpy(), zf.close.to_numpy()
        pos = np.zeros(len(idx), dtype=np.int64)
        i = 0
        while i < len(idx):
            j = i + 1
            while j < len(idx) and ses[j] == ses[i]: j += 1
            q = np.where((sec[i:j] >= p['obs_start']) & (sec[i:j] < p['entry']))[0]
            if len(q):
                m = (cb[i+q[-1]] - cb[i+q[0]]) - (cf[i+q[-1]] - cf[i+q[0]])
                if abs(m) >= p['thr']:
                    d = 1 if m > 0 else -1
                    for k in range(i+q[-1]+1, j):
                        if sec[k] >= p['exit']: break
                        pos[k] = d
            i = j
        s = pd.Series(pos, index=idx)
        out['ZB'] = s.reindex(a.index, fill_value=0).to_numpy().astype(np.int64)
        out['ZF'] = (-s).reindex(b.index, fill_value=0).to_numpy().astype(np.int64)
    return out

def entry_risk(panel):
    return {s: np.full(len(d), 200.0) for s, d in panel.items()}
