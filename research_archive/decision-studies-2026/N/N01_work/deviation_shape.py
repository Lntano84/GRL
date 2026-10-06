"""How does X_c - X_d behave world by world?  Rare large deviations or a systematic shift?"""
from __future__ import annotations

import csv
import gzip
import json
import statistics
from pathlib import Path

WORK = Path(__file__).resolve().parent


def load(csv_name, gz_name, n_cases=40):
    rs = list(csv.DictReader((WORK / csv_name).open(encoding="utf-8-sig")))
    ws = json.loads(gzip.open(WORK / gz_name, "rt", encoding="utf-8").read())
    return rs, ws


for label, csv_name, gz_name in (
        ("FROZEN", "N01_results.csv", "N01_worlds.json.gz"),
        ("PROBE", "N01S_results.csv", "N01S_worlds.json.gz")):
    rs, ws = load(csv_name, gz_name)
    print("=" * 90)
    print(f"  {label}: world-level structure of D = X_c - X_d (confirm batch, 2048 worlds)")
    print("=" * 90)
    all_abs = []
    exact = 0
    total = 0
    per_case_nonzero_worlds = []
    for r, pc in zip(rs, ws["per_case"]):
        d = [u - v for u, v in zip(pc["confirm"]["X_c"], pc["confirm"]["X_d"])]
        nz = [abs(x) for x in d if x != 0]
        exact += len(d) - len(nz)
        total += len(d)
        all_abs.extend(abs(x) for x in d)
        per_case_nonzero_worlds.append(len(nz))
    all_abs.sort()

    def q(p):
        return all_abs[min(len(all_abs) - 1, int(p * len(all_abs)))]

    print(f"  (world, comparison) pairs                     : {total}")
    print(f"  pairs with D == 0 exactly                     : {exact} ({exact/total:.2%})")
    print(f"  pairs with D != 0                             : {total-exact} ({(total-exact)/total:.2%})")
    print(f"  |D| quantiles over all pairs: "
          f"p50 {q(0.50)}, p90 {q(0.90)}, p99 {q(0.99)}, p999 {q(0.999)}, max {all_abs[-1]}")
    print(f"  fraction of pairs with |D| == 0               : {exact/total:.4%}")
    print(f"  per-case count of worlds with D != 0: "
          f"min {min(per_case_nonzero_worlds)}, median {statistics.median(per_case_nonzero_worlds)}, "
          f"max {max(per_case_nonzero_worlds)} (of 2048)")
    # cumulative contribution: what share of the total |D| mass sits in the top 1% of pairs?
    top = all_abs[int(0.99 * len(all_abs)):]
    print(f"  top 1% of pairs carry {sum(top)/sum(all_abs):.2%} of the total |D| mass")
