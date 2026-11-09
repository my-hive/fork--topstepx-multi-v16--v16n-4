#!/usr/bin/env python3
"""Record one experiment from sim stdout+report into experiments.jsonl.

usage: python3 record.py N FAMILY HYP SYMBOLS KNOBS OUT_TXT REPORT_JSON [PVALUE]
"""
import json, re, sys

n, family, hyp, symbols, knobs, out_txt, rep_json = sys.argv[1:8]
pval = sys.argv[8] if len(sys.argv) > 8 else None

txt = open(out_txt).read()
dev_score = float(re.search(r"^dev_score:\s*([-\d.]+)", txt, re.M).group(1))
folds = json.loads(re.search(r"^folds:\s*(\{.*\})", txt, re.M).group(1))
era = json.loads(re.search(r"^era blocks.*?:\s*(\{.*\})", txt, re.M).group(1))
elig = json.loads(re.search(r"^eligibility checks:\s*(\{.*\})", txt, re.M).group(1))
r = json.load(open(rep_json))
s = r["stressed"]
fs = sorted(folds.values())
rec = {
    "agent": "v16n-4", "n": int(n), "family": family, "hypothesis": hyp,
    "symbols": symbols.split(","), "source_sha256": r["strategy_sha256"],
    "dev_score": dev_score, "fold_score": (fs[0] + fs[1]) / 2.0,
    "positive_year_fraction": s["eligibility"]["positive_year_fraction"],
    "knobs": int(knobs), "folds": folds, "era_blocks": era,
    "total_pnl": sum(v["pnl"] for v in s["per_market"].values()),
    "entries": sum(v["entries"] for v in s["per_market"].values()),
    "eligibility": elig,
    "null_seeds": 0 if pval is None else 20,
    "provisional_p": None if pval is None else float(pval),
}
with open("experiments.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print(json.dumps(rec))
