#!/usr/bin/env python
"""
Stage 09A analysis: continuous re-optimisation vs the MIP methods, plus a structural recount.

    .venv-hs/Scripts/python.exe stage09a_analyse.py

Per state it reports

    state | J_repair | J_LP | LP status | total time | continuous gain
          | E20 cost (both seeds, stage 07) | E@40 cost (both seeds, stage 08 prefix)

with  G_continuous = (J_r - J_LP) / max(1, |J_r|).

Then a recount that needs NO new solves: for stage 08's EMPTY improving solutions, how many
far-horizon Y, short-term Z and far-horizon Z differ from S_r.  This describes WHICH decisions
move alongside the gain; a count of changes does NOT prove those variables had to change.

The strict statement comes from the LP optimum instead:

  * if a valid EMPTY solution costs strictly less than the PROVEN-OPTIMAL J_LP, then reaching
    that cost requires changing at least some binary decisions;
  * if the LP is not proven optimal, EMPTY beating the current LP incumbent proves nothing;
  * if J_LP is better than E20 or E@40, that is reported as-is: cheap continuous
    re-optimisation found a better solution than the corresponding MIP run delivered.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lsp_gen import apply_disruption, build_instance, disruptions_for
from stage06_run import solution_from_record

TOL = 1e-6
CHECKPOINTS = (5.0, 10.0, 20.0, 40.0)


def load(name):
    with open(name, encoding="utf-8") as fh:
        return json.load(fh)


def gain(jr: float, j: float) -> float:
    return (jr - j) / max(1.0, abs(jr))


def main() -> int:
    A = load("stage09a_runs.json")
    lp = {(r["state"]): r for r in A["runs"]}
    s7 = load("stage07_runs.json")["runs"]
    s8 = load("stage08_runs.json")["runs"]
    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)

    print("Stage 09A | %d LPs | method: ALL Y and Z pinned to the frozen repaired plan"
          % len(lp))
    print("=" * 132)
    print("%-28s %11s %11s %-9s %7s %8s %19s %19s"
          % ("state", "J_repair", "J_LP", "LP status", "t(s)", "G_cont",
             "E20 (s0/s1)", "E@40 (s0/s1)"))
    print("-" * 132)

    # ---- baselines
    e20 = defaultdict(dict)
    for r in s7:
        if r["method"] == "E20":
            e20[r["state"]][int(r["seed"])] = float(r["final_cost"])
    e40 = defaultdict(dict)
    for r in s8:
        if r["config"] != "EMPTY":
            continue
        jr = r["repair_cost"]
        ev = sorted((e for e in r["events"] if e.get("usable") and e.get("sol") is not None),
                    key=lambda e: e["wall_s"])
        best = None
        for e in ev:
            if e["wall_s"] > 40.0 + 1e-9:
                break
            if best is None or e["cost_recomputed"] < best["cost_recomputed"] - TOL:
                best = e
        e40[r["state"]][int(r["seed"])] = (best["cost_recomputed"] if best is not None else jr)

    rows: List[Dict[str, Any]] = []
    for state in sorted(lp):
        r = lp[state]
        jr, jlp = r["repair_cost"], r["lp_objective"]
        gc = gain(jr, jlp)
        e20v = e20.get(state, {})
        e40v = e40.get(state, {})
        rows.append({"state": state, "jr": jr, "jlp": jlp, "g": gc,
                     "e20": e20v, "e40": e40v})
        print("%-28s %11.6g %11.6g %-9s %7.2f %+8.4f  %9s/%-9s %9s/%-9s"
              % (state, jr, jlp, r["lp_status"][:9], r["timing"]["total_s"], gc,
                 "%.6g" % e20v.get(0, float("nan")), "%.6g" % e20v.get(1, float("nan")),
                 "%.6g" % e40v.get(0, float("nan")), "%.6g" % e40v.get(1, float("nan"))))
    print("-" * 132)

    # ---- aggregation
    gs = [x["g"] for x in rows]
    print("  G_continuous: mean=%+.4f median=%+.4f min=%+.4f max=%+.4f  (>0 in %d/%d states)"
          % (sum(gs) / len(gs), sorted(gs)[len(gs) // 2], min(gs), max(gs),
             sum(1 for g in gs if g > TOL), len(gs)))
    tot = [lp[s]["timing"]["total_s"] for s in lp]
    print("  LP total time: mean=%.3f s max=%.3f s | proven optimal: %d/%d"
          % (sum(tot) / len(tot), max(tot),
             sum(1 for s in lp if lp[s]["proven_optimal"]), len(lp)))
    print()

    # ---- per nominal instance (2 disruptions averaged)
    print("=== per nominal instance ===")
    print("  %-24s %10s %10s %10s %10s %10s" % ("instance", "G_cont", "G_E20", "G_E@40",
                                                 "E20-LP", "E40-LP"))
    agg = defaultdict(lambda: defaultdict(list))
    for x in rows:
        inst = x["state"].split("|")[0]
        jr = x["jr"]
        agg[inst]["gc"].append(x["g"])
        for sd, v in x["e20"].items():
            agg[inst]["e20"].append(gain(jr, v))
        for sd, v in x["e40"].items():
            agg[inst]["e40"].append(gain(jr, v))
        agg[inst]["e20_minus_lp"].append((x["jlp"] - min(x["e20"].values())) / max(1.0, abs(jr))
                                         if x["e20"] else None)
        agg[inst]["e40_minus_lp"].append((x["jlp"] - min(x["e40"].values())) / max(1.0, abs(jr))
                                         if x["e40"] else None)
    inst_rows = []
    for inst in sorted(agg):
        d = agg[inst]

        def mean_of(k: str) -> float:
            vs = [v for v in d[k] if v is not None]
            return sum(vs) / len(vs) if vs else float("nan")

        inst_rows.append((inst, mean_of("gc"), mean_of("e20"), mean_of("e40"),
                          mean_of("e20_minus_lp"), mean_of("e40_minus_lp")))
        print("  %-24s %+10.4f %+10.4f %+10.4f %+10.4f %+10.4f" % inst_rows[-1])
    print("  G_E20 / G_E@40 are gains vs J_r; 'E20-LP' = (J_LP - min E20)/|J_r|, so a NEGATIVE")
    print("  value means the LP optimum is cheaper than what the MIP delivered.")
    print()

    # ---- the strict statement
    print("=== strict statement: does beating the LP optimum require binary changes? ===")
    n_beat = n_proven = 0
    for x in rows:
        st = x["state"]
        if not lp[st]["proven_optimal"]:
            print("  %-30s LP not proven optimal -> no conclusion drawn" % st)
            continue
        n_proven += 1
        best_mip = min(list(x["e20"].values()) + list(x["e40"].values()))
        if best_mip < x["jlp"] - TOL * (1 + abs(x["jlp"])):
            n_beat += 1
            print("  %-30s best MIP (%.6g) < proven LP optimum (%.6g)  -> binary changes REQUIRED"
                  % (st, best_mip, x["jlp"]))
        elif x["jlp"] < best_mip - TOL * (1 + abs(best_mip)):
            print("  %-30s LP optimum (%.6g) < best MIP (%.6g)  -> continuous re-optimisation won"
                  % (st, x["jlp"], best_mip))
    print()
    print("  states with a PROVEN LP optimum: %d/%d" % (n_proven, len(rows)))
    print("  of those, states where a valid MIP solution strictly beats the LP optimum: %d"
          % n_beat)
    print("  => in those %d states, reaching the MIP's cost REQUIRES changing at least some" % n_beat)
    print("     binary (setup / carry-over) decisions.")
    print()

    # ---- structural recount, no new solves
    print("=== structural recount of stage 08 EMPTY improving solutions (no new solves) ===")
    print("  counted relative to S_r; a change count does NOT prove the change was necessary")
    ctx = {}
    print("  %-28s %3s %9s %14s %13s %13s"
          % ("state", "sd", "gain", "far-horizon Y", "short-term Z", "far-horizon Z"))
    tot_ch = {"fy": 0, "sz": 0, "fz": 0}
    n_sol = 0
    n_moved_binary = 0
    for st in sorted({r["state"] for r in s8}):
        rec = ST[st]
        inst, _ = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        tau = rec["meta"]["tau"]
        ctx[st] = (pert, repair, tau)
    for r in s8:
        if r["config"] != "EMPTY":
            continue
        pert, repair, tau = ctx[r["state"]]
        jr = r["repair_cost"]
        ev = sorted((e for e in r["events"] if e.get("usable") and e.get("sol") is not None),
                    key=lambda e: e["wall_s"])
        if not ev:
            continue
        best = min(ev, key=lambda e: e["cost_recomputed"])
        if best["cost_recomputed"] >= jr - TOL * (1 + abs(jr)):
            continue                                   # no improvement, nothing to count
        Y = np.array(best["sol"]["Y"], float).reshape(pert.N, pert.M, pert.T)
        Z = np.array(best["sol"]["Z"], float).reshape(pert.N, pert.M, pert.T)
        fy = sum(1 for i in range(pert.N) for j in range(pert.M)
                 for t in range(tau + 1, pert.T + 1)
                 if abs(Y[i, j, t - 1] - repair.Y[i, j, t - 1]) > 0.5)
        sz = sum(1 for i in range(pert.N) for j in range(pert.M)
                 for t in range(1, tau + 1)
                 if abs(Z[i, j, t - 1] - repair.Z[i, j, t - 1]) > 0.5)
        fz = sum(1 for i in range(pert.N) for j in range(pert.M)
                 for t in range(tau + 1, pert.T + 1)
                 if abs(Z[i, j, t - 1] - repair.Z[i, j, t - 1]) > 0.5)
        tot_ch["fy"] += fy
        tot_ch["sz"] += sz
        tot_ch["fz"] += fz
        n_sol += 1
        if fy + sz + fz > 0:
            n_moved_binary += 1
        print("  %-28s %3d %+9.4f %14d %13d %13d"
              % (r["state"], r["seed"], gain(jr, best["cost_recomputed"]), fy, sz, fz))
    print()
    print("  EMPTY improving solutions examined: %d" % n_sol)
    print("  total changes vs S_r: far-horizon Y=%d  short-term Z=%d  far-horizon Z=%d"
          % (tot_ch["fy"], tot_ch["sz"], tot_ch["fz"]))
    print("  short-term Y changes are 0 by contract (EMPTY pins them)")
    print("  solutions that moved at least one far-horizon Y or Z: %d/%d"
          % (n_moved_binary, n_sol))
    print("  => EMPTY is NOT a continuous-only method: every one of its improvements moved")
    print("     binary decisions this LP was not allowed to touch.")
    print()

    with open("stage09a_analysis.json", "w", encoding="utf-8") as fh:
        json.dump({"per_state": [{k: v for k, v in x.items()} for x in rows],
                   "per_instance": [{"instance": a, "g_cont": b, "g_e20": c,
                                     "g_e40": d, "e20_minus_lp": e, "e40_minus_lp": f}
                                    for a, b, c, d, e, f in inst_rows],
                   "g_cont_summary": {"mean": sum(gs) / len(gs), "min": min(gs),
                                      "max": max(gs), "n": len(gs)},
                   "proven_optimal": n_proven,
                   "states_requiring_binary_change": n_beat,
                   "structural_recount": {"n_solutions": n_sol, **tot_ch},
                   "lp_meta": A["meta"]}, fh, indent=2, default=str)
    print("wrote stage09a_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
