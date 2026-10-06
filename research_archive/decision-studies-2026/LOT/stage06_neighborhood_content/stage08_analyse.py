#!/usr/bin/env python
"""
Stage 08 analysis: when does continuous search produce usable improvement?

    .venv-hs/Scripts/python.exe stage08_analyse.py

Main indicators (these four, only these four)
---------------------------------------------
  1. FIRST USABLE IMPROVEMENT TIME.  The existing numerical tolerance is used; float noise is
     not an improvement.  A run that never improves within 40 s is reported as "NOT REACHED"
     and is NOT given a first-improvement time of 40 s before averaging.
  2. FIRST TIME REACHING A PRACTICAL IMPROVEMENT,
         (J_r - J(t)) / max(1, |J_r|) >= 2%.
     2% remains an engineering reference only.  Censored runs are reported as "NOT REACHED".
  3. IMPROVEMENT COUNTS AND MEAN IMPROVEMENT AT 5 / 10 / 20 / 40 s.
  4. WHETHER FULL's IMPROVING SOLUTIONS CAN BE TRANSFERRED TO EMPTY: does the short-term Y of
     the FULL solution still satisfy the fixing conditions relative to the SAME frozen S_r, and
     does the full objective subproblem verify?

Episode semantics
-----------------
A checkpoint value is the best VALID solution already found by that time in THAT SAME 40 s run.
These are prefixes of one trajectory, NOT independent runs at 5/10/20/40 s budgets, and they are
labelled that way everywhere.  Verification is offline, so a checkpoint value is "the quality of
the solution found by that time", not a real-time delivery result.

Reporting: per rho group, with the nominal instance as the aggregation unit overall.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_model import MODE_AUDITED, Solution
from lsp_neighborhood import build_release_sets
from lsp_repair3 import binary_change_count
from stage06_run import KAPPA, solution_from_record

CHECKPOINTS = (5.0, 10.0, 20.0, 40.0)
THRESHOLD = 0.02
TOL = 1e-6


def gain(jr: float, j: float) -> float:
    return (jr - j) / max(1.0, abs(jr))


def load():
    with open("stage08_runs.json", encoding="utf-8") as fh:
        return json.load(fh)


def events_by_time(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Usable events only, ordered by time."""
    ev = [e for e in r["events"] if e.get("usable") and e.get("sol") is not None]
    return sorted(ev, key=lambda e: e["wall_s"])


def best_by(ev: List[Dict[str, Any]], t: float, jr: float) -> Optional[Dict[str, Any]]:
    """Best-cost valid event already found by time t (None -> use S_r)."""
    best = None
    for e in ev:
        if e["wall_s"] > t + 1e-9:
            break
        if best is None or e["cost_recomputed"] < best["cost_recomputed"] - TOL:
            best = e
    return best


def first_reaching(ev: List[Dict[str, Any]], jr: float, thr: float) -> Optional[float]:
    best_gain = -float("inf")
    for e in ev:
        g = gain(jr, e["cost_recomputed"])
        best_gain = max(best_gain, g)
        if best_gain >= thr - 1e-12:
            return e["wall_s"]
    return None


def nominal_instances(rows: List[Dict[str, Any]], key) -> Dict[str, float]:
    """Average per (instance, disruption), then per instance."""
    per_state = defaultdict(list)
    for r in rows:
        v = key(r)
        if v is None:
            continue
        per_state[r["state"]].append(v)
    per_inst = defaultdict(list)
    for state, vals in per_state.items():
        per_inst[state.split("|")[0]].append(sum(vals) / len(vals))
    return {k: sum(v) / len(v) for k, v in per_inst.items()}


def main() -> int:
    D = load()
    runs = D["runs"]
    print("runs: %d | budget %.0f s | configs %s | seeds %s"
          % (len(runs), D["meta"]["budget_s"], D["meta"]["configs"], D["meta"]["seeds"]))
    print("scope: %s" % D["meta"]["scope"])
    print("checkpoints: %s s  (PREFIXES of one 40 s run, not independent runs)"
          % D["meta"]["checkpoints_s"])
    print("verification is offline: a checkpoint value is the quality found BY that time")
    print()

    # ---------------------------------------------------------------- event/output overview
    print("=== event overview ===")
    print("  %-6s %-6s %5s %8s %8s %9s %9s"
          % ("config", "rho", "n", "runs_ev", "events", "usable", "improved"))
    for cfg in ("EMPTY", "FULL"):
        for rho in ("0.75", "1.10"):
            sub = [r for r in runs if r["config"] == cfg and abs(r["rho"] - float(rho)) < 1e-9]
            nev = sum(r["n_events"] for r in sub)
            nus = sum(r["n_usable_events"] for r in sub)
            nimp = sum(1 for r in sub if any(e.get("improved_over_repair")
                                             for e in events_by_time(r)))
            print("  %-6s %-6s %5d %8d %8d %9d %9d"
                  % (cfg, rho, len(sub), sum(1 for r in sub if r["n_events"]), nev, nus, nimp))
    print()

    # ---------------------------------------------------------------- 1. first improvement
    print("=== indicator 1: first USABLE improvement time (censored runs stay censored) ===")
    first_rows = []
    for r in runs:
        ev = events_by_time(r)
        jr = r["repair_cost"]
        t = next((e["wall_s"] for e in ev if e["cost_recomputed"] < jr - TOL * (1 + abs(jr))), None)
        first_rows.append({"config": r["config"], "rho": r["rho"], "state": r["state"],
                           "seed": r["seed"], "t": t, "jr": jr})
    for cfg in ("EMPTY", "FULL"):
        for rho in ("0.75", "1.10"):
            sub = [x for x in first_rows if x["config"] == cfg and abs(x["rho"] - float(rho)) < 1e-9]
            got = [x["t"] for x in sub if x["t"] is not None]
            print("  %-6s rho%-5s reached %2d/%-2d  times=%s"
                  % (cfg, rho, len(got), len(sub),
                     "none" if not got else ", ".join("%.2f" % v for v in sorted(got))))
    allgot = [x["t"] for x in first_rows if x["t"] is not None]
    print("  overall reached %d/%d" % (len(allgot), len(first_rows)))
    print("  NOTE: no censored run is assigned a time; means below use reached runs only.")
    print()

    # ---------------------------------------------------------------- 2. first 2%
    print("=== indicator 2: first time reaching a 2%% practical improvement ===")
    sec_rows = []
    for r in runs:
        ev = events_by_time(r)
        t = first_reaching(ev, r["repair_cost"], THRESHOLD)
        sec_rows.append({"config": r["config"], "rho": r["rho"], "state": r["state"],
                         "seed": r["seed"], "t": t})
    for cfg in ("EMPTY", "FULL"):
        for rho in ("0.75", "1.10"):
            sub = [x for x in sec_rows if x["config"] == cfg and abs(x["rho"] - float(rho)) < 1e-9]
            got = [x["t"] for x in sub if x["t"] is not None]
            print("  %-6s rho%-5s reached %2d/%-2d  times=%s"
                  % (cfg, rho, len(got), len(sub),
                     "none" if not got else ", ".join("%.2f" % v for v in sorted(got))))
    print("  overall reached %d/%d" % (sum(1 for x in sec_rows if x["t"] is not None),
                                       len(sec_rows)))
    print()

    # ---------------------------------------------------------------- 3. checkpoints
    print("=== indicator 3: improvement at each checkpoint (prefix trajectory) ===")
    print("  %-6s %-6s %6s %10s %10s %10s %10s"
          % ("config", "rho", "metric", *["%ds" % c for c in CHECKPOINTS]))
    per_cp: Dict[str, Any] = {}
    for cfg in ("EMPTY", "FULL"):
        for rho in ("0.75", "1.10"):
            sub = [r for r in runs if r["config"] == cfg and abs(r["rho"] - float(rho)) < 1e-9]
            for label, fn in (("improved#", lambda e, jr: e is not None
                               and e["cost_recomputed"] < jr - TOL * (1 + abs(jr))),
                              ("mean gain", None)):
                cells = []
                for cp in CHECKPOINTS:
                    if fn is not None:
                        cells.append("%d/%d" % (sum(1 for r in sub
                                                    if fn(best_by(events_by_time(r), cp,
                                                                  r["repair_cost"]),
                                                           r["repair_cost"])), len(sub)))
                    else:
                        gs = []
                        for r in sub:
                            e = best_by(events_by_time(r), cp, r["repair_cost"])
                            j = e["cost_recomputed"] if e is not None else r["repair_cost"]
                            gs.append(gain(r["repair_cost"], j))
                        cells.append("%+.4f" % (sum(gs) / len(gs)))
                print("  %-6s %-6s %6s %10s %10s %10s %10s" % (cfg, rho, label, *cells))
                per_cp[("%s|%s|%s" % (cfg, rho, label))] = cells

            # mean gain over IMPROVED runs only, to separate rate from magnitude
            cells = []
            for cp in CHECKPOINTS:
                gs = []
                for r in sub:
                    e = best_by(events_by_time(r), cp, r["repair_cost"])
                    if e is None:
                        continue
                    g = gain(r["repair_cost"], e["cost_recomputed"])
                    if g > TOL:
                        gs.append(g)
                cells.append("-" if not gs else "%+.4f" % (sum(gs) / len(gs)))
            print("  %-6s %-6s %6s %10s %10s %10s %10s"
                  % (cfg, rho, "gain|impr", *cells))
            per_cp[("%s|%s|gain|impr" % (cfg, rho))] = cells
    print()

    # per-instance (nominal instance aggregation unit) at 20 s and 40 s
    print("  per nominal instance, mean gain at 20 s and 40 s:")
    print("  %-24s %14s %14s %14s %14s" % ("instance", "E@20", "F@20", "E@40", "F@40"))
    inst_tab = {}
    for inst in sorted({r["state"].split("|")[0] for r in runs}):
        row = []
        for cp in (20.0, 40.0):
            for cfg in ("EMPTY", "FULL"):
                sub = [r for r in runs if r["config"] == cfg
                       and r["state"].split("|")[0] == inst]
                gs = []
                for r in sub:
                    e = best_by(events_by_time(r), cp, r["repair_cost"])
                    j = e["cost_recomputed"] if e is not None else r["repair_cost"]
                    gs.append(gain(r["repair_cost"], j))
                row.append(sum(gs) / len(gs))
        inst_tab[inst] = row
        print("  %-24s %+14.4f %+14.4f %+14.4f %+14.4f" % (inst, *row))
    print()

    # ---------------------------------------------------------------- 4. transferability
    print("=== indicator 4: can FULL's improving solutions be transferred to EMPTY? ===")
    print("  EMPTY fixes EVERY short-term Y to S_r, so a FULL solution is admissible for EMPTY")
    print("  only if its short-term Y equals the repaired plan exactly.  The reference is the")
    print("  frozen S_r; it is NOT relaxed to accept a wide-neighbourhood solution.")
    print()
    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)
    ctx = {}
    xfer = {"tested": 0, "ok": 0, "rows": []}
    for r in runs:
        if r["config"] != "FULL":
            continue
        peer = next(x for x in runs if x["state"] == r["state"] and x["config"] == "EMPTY"
                    and int(x["seed"]) == int(r["seed"]))
        ev_f = events_by_time(r)
        ev_e = events_by_time(peer)
        if r["state"] not in ctx:
            rec = ST[r["state"]]
            inst, _ = build_instance(rec["meta"]["scale"], rec["meta"]["rho"],
                                     rec["meta"]["seed"])
            dis = next(d for d in disruptions_for(inst)
                       if d.name == rec["disruption"]["name"])
            pert = apply_disruption(inst, dis)
            repair = solution_from_record(pert, rec["repair"])
            tau = rec["meta"]["tau"]
            ctx[r["state"]] = {"pert": pert, "repair": repair, "tau": tau}
        C = ctx[r["state"]]
        for cp in CHECKPOINTS:
            ef = best_by(ev_f, cp, r["repair_cost"])
            ee = best_by(ev_e, cp, peer["repair_cost"])
            jr = r["repair_cost"]
            gf = gain(jr, ef["cost_recomputed"]) if ef is not None else 0.0
            ge = gain(jr, ee["cost_recomputed"]) if ee is not None else 0.0
            if gf <= max(ge, 0.0) + TOL:
                continue                      # FULL not better than EMPTY at this checkpoint
            xfer["tested"] += 1
            Y = np.array(ef["sol"]["Y"], float).reshape(C["pert"].N, C["pert"].M, C["pert"].T)
            viol = 0
            for i in range(C["pert"].N):
                for j in range(C["pert"].M):
                    for t in range(1, C["tau"] + 1):
                        if abs(float(Y[i, j, t - 1]) - float(C["repair"].Y[i, j, t - 1])) > 0.5:
                            viol += 1
            ok = viol == 0
            xfer["ok"] += int(ok)
            xfer["rows"].append({"state": r["state"], "seed": r["seed"], "cp": cp,
                                 "gain_full": gf, "gain_empty": ge,
                                 "short_term_violations": viol, "transferable": ok})
            print("    %-30s s%d %4.0fs  FULL %+.4f vs EMPTY %+.4f  viol=%d  %s"
                  % (r["state"], r["seed"], cp, gf, ge, viol,
                     "TRANSFERABLE" if ok else "BLOCKED"))
    print()
    print("  FULL better than EMPTY at a checkpoint: %d cases | transferable to EMPTY: %d"
          % (xfer["tested"], xfer["ok"]))
    if xfer["tested"] and xfer["ok"] == 0:
        print("  => TRANSFER BARRIER: FULL's better solutions do NOT satisfy EMPTY's fixing")
        print("     conditions, so they can be neither submitted nor worked around by moving")
        print("     the reference point.")
    print()

    # ---------------------------------------------------------------- early vs late
    print("=== where does EMPTY's improvement appear? ===")
    ev_times = []
    for r in runs:
        if r["config"] != "EMPTY":
            continue
        jr = r["repair_cost"]
        for e in events_by_time(r):
            if e["cost_recomputed"] < jr - TOL * (1 + abs(jr)):
                ev_times.append((r["state"], r["seed"], e["wall_s"], gain(jr, e["cost_recomputed"])))
    if ev_times:
        ts = sorted(t for _s, _d, t, _g in ev_times)
        print("  EMPTY improving events: %d" % len(ts))
        print("  times: %s" % ", ".join("%.2f" % t for t in ts))
        print("  median %.2f s | before 5 s: %d | before 10 s: %d | before 20 s: %d"
              % (ts[len(ts) // 2], sum(1 for t in ts if t <= 5),
                 sum(1 for t in ts if t <= 10), sum(1 for t in ts if t <= 20)))
    else:
        print("  EMPTY produced NO usable improvement in any of the 32 EMPTY runs")
    print()

    # ---------------------------------------------------------------- consistency with stage 07
    print("=== consistency with stage 07 (E20 = EMPTY 20 s) ===")
    try:
        with open("stage07_runs.json", encoding="utf-8") as fh:
            s7 = json.load(fh)["runs"]
    except OSError:
        s7 = []
    if s7:
        s7e = {(r["state"], int(r["seed"])): float(r["final_cost"]) for r in s7
               if r["method"] == "E20"}
        n_same = n_tot = 0
        for r in runs:
            if r["config"] != "EMPTY":
                continue
            k = (r["state"], int(r["seed"]))
            if k not in s7e:
                continue
            e = best_by(events_by_time(r), 20.0, r["repair_cost"])
            j20 = e["cost_recomputed"] if e is not None else r["repair_cost"]
            n_tot += 1
            n_same += int(abs(j20 - s7e[k]) <= 1e-6 * (1 + abs(s7e[k])))
        print("  20 s prefix of this trace equals stage 07's E20 delivered cost: %d/%d"
              % (n_same, n_tot))
        print("  (a difference is solver noise between batches, not necessarily a protocol")
        print("   difference; the two are independent runs)")

    with open("stage08_analysis.json", "w", encoding="utf-8") as fh:
        json.dump({"checkpoints": per_cp, "instances_20_40": inst_tab,
                   "first_improvement": [{k: v for k, v in x.items()} for x in first_rows],
                   "first_2pct": sec_rows,
                   "transfer": {"tested": xfer["tested"], "ok": xfer["ok"],
                                "rows": xfer["rows"]},
                   "empty_event_times": ev_times,
                   "meta": D["meta"]}, fh, indent=2, default=str)
    print("\nwrote stage08_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
