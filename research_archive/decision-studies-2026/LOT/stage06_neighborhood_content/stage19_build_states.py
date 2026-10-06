"""Stage 19 state builder: apply the ORIGINAL repairer to the frozen `S_N+` plans.

32 new fault states = 8 instances x 4 disruptions.  Nothing is redrawn: the disruptions are the
same 2x2 breadth x duration grid used since Stage 15, and the reference plan is the frozen
`S_N+` (never the old `S_N`).

The stability anchor for these states is kappa = 4 relative to the NEW repair solution, which is
what the Stage 19 runner uses.  Cumulative changes relative to the old `S_N` are NOT constrained.

Writes `stage19_states.json`.  Solves NO MIPs.  Also reports, descriptively, the old vs new
repair costs (a lower nominal cost does NOT guarantee an easier repair).
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import Disruption, apply_disruption, build_instance
from lsp_model import MODE_AUDITED, Solution
from lsp_neighborhood import recovery_period
from lsp_repair3 import repair_minbatch_v3

NOMINAL_FILE = "stage06_nominal.json"
PLUS_INDEX = "stage19_nominal_plus.json"
PLAN_DIR = "stage19_nominal_plus"
SRC_STATES = "stage06_states.json"
OLD_STATES = "stage15_states.json"   # carries the previous repair for ALL FOUR cells
OUT_STATES = "stage19_states.json"
TAU = 6
REPAIR_NAME = "repair_minbatch_v3"

CELL = {"D1_m0_1p": ("single", "1p"), "D1_m0_2p": ("single", "2p"),
        "D2_all_1p": ("all", "1p"), "D2_all_2p": ("all", "2p")}


def build_disruptions(inst) -> List[Disruption]:
    all_m = list(range(inst.M))
    return [
        Disruption("D1_m0_1p", {0: [1]}, "machine 0 down for period 1"),
        Disruption("D1_m0_2p", {0: [1, 2]}, "machine 0 down for periods 1-2"),
        Disruption("D2_all_1p", {j: [1] for j in all_m}, "every machine down for period 1"),
        Disruption("D2_all_2p", {j: [1, 2] for j in all_m}, "every machine down for periods 1-2"),
    ]


def load_plus(name: str, inst) -> Tuple[Solution, Dict[str, Any]]:
    path = os.path.join(PLAN_DIR, "%s.json" % name)
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    p = rec["plan"]
    sol = Solution(status="S_N+", objective=float(rec["objective"]),
                   X=np.array(p["X"], float).reshape(inst.N, inst.M, inst.T),
                   Y=np.array(p["Y"], float).reshape(inst.N, inst.M, inst.T),
                   Z=np.array(p["Z"], float).reshape(inst.N, inst.M, inst.T),
                   I=np.array(p["I"], float).reshape(inst.N, inst.T),
                   L=np.array(p["L"], float).reshape(inst.N, inst.T),
                   mode=MODE_AUDITED)
    return sol, rec


def main() -> int:
    with open(SRC_STATES, encoding="utf-8") as fh:
        OLD = json.load(fh)
    with open(OLD_STATES, encoding="utf-8") as fh:
        OLD15 = json.load(fh)
    with open(NOMINAL_FILE, encoding="utf-8") as fh:
        NOM = json.load(fh)
    with open(PLUS_INDEX, encoding="utf-8") as fh:
        PLUS = json.load(fh)

    insts = sorted(PLUS["plans"])
    print("Stage 19 | %d instances x 4 disruptions = %d new fault states"
          % (len(insts), len(insts) * 4))
    print("stability anchor: kappa=4 relative to the NEW repair solution (not to S_N)\n")

    out: Dict[str, Any] = {}
    print("%-19s %-11s %-6s %-3s | %11s %11s %9s | %s"
          % ("instance", "disruption", "breadth", "dur", "J_r(new)", "J_r(old)", "change",
             "feasible"))
    n_feas = 0
    for name in insts:
        rec_old = OLD["%s|D1_m0_2p" % name]
        meta = rec_old["meta"]
        inst, _m = build_instance(meta["scale"], meta["rho"], meta["seed"])
        assert inst.name == name, (inst.name, name)
        plus, plus_rec = load_plus(name, inst)

        chk_plus = check_solution(inst, plus, MODE_AUDITED)
        if not chk_plus.ok:
            raise SystemExit("frozen S_N+ for %s is infeasible: %s"
                             % (name, chk_plus.failed_checks()))

        for dis in build_disruptions(inst):
            pert = apply_disruption(inst, dis)
            rep = repair_minbatch_v3(pert, plus, dis)          # original repairer, new base
            chk = check_solution(pert, rep.solution, MODE_AUDITED)
            if not chk.ok:
                raise SystemExit("repair of S_N+ for %s|%s is INFEASIBLE: %s"
                                 % (name, dis.name, chk.failed_checks()))
            n_feas += 1

            r = {j: recovery_period(dis, j, pert.T) for j in range(pert.M)}
            affected = [j for j in range(pert.M) if dis.delta_for(j) > 0]
            j_new = float(rep.solution.objective)
            j_old = float(OLD15["%s|%s" % (name, dis.name)]["repair"]["objective"])
            bre, dur = CELL[dis.name]

            out["%s|%s" % (name, dis.name)] = {
                "meta": meta,
                "disruption": dis.as_dict(),
                "cell": {"breadth": bre, "duration": dur,
                         "n_affected": len(affected)},
                "recovery_periods": {str(j): r[j] for j in range(pert.M)},
                "nominal_plus_objective": float(plus.objective),
                "old_nominal_objective": float(NOM[name]["objective"]),
                "D_scale": max(1.0, abs(float(NOM[name]["objective"]))),
                "D_scale_note": "fixed scale = old nominal cost J_{N,n}, identical for the "
                                "fault and no-fault arms",
                "repair": {
                    "name": REPAIR_NAME, "objective": j_new,
                    "old_repair_objective": j_old,
                    "feasible": True, "max_violation": float(chk.max_violation),
                    "base": "S_N+ (stage19_nominal_plus)",
                    "log_text": rep.log_text()[:2000],
                    "plan": {k: np.asarray(getattr(rep.solution, k)).tolist()
                             for k in ("X", "Y", "Z", "I", "L")},
                },
            }
            print("%-19s %-11s %-6s %-3s | %11.2f %11.2f %+9.2f | %s"
                  % (name, dis.name, bre, dur, j_new, j_old, j_new - j_old, "yes"))

    # no-fault arm: the reference solution is S_N+ itself, repairer NOT called
    for name in insts:
        meta = OLD["%s|D1_m0_2p" % name]["meta"]
        plus, _pr = load_plus(name, build_instance(meta["scale"], meta["rho"], meta["seed"])[0])
        j_p = float(plus.objective)
        out["%s|NONE" % name] = {
            "meta": meta,
            "disruption": {"name": "NONE", "down": {}, "note": "no disruption",
                           "description": "no disruption"},
            "cell": {"breadth": "none", "duration": "none", "n_affected": 0},
            "recovery_periods": {},
            "nominal_plus_objective": j_p,
            "old_nominal_objective": float(NOM[name]["objective"]),
            "D_scale": max(1.0, abs(float(NOM[name]["objective"]))),
            "D_scale_note": "fixed scale = old nominal cost J_{N,n}",
            "repair": {"name": "none (S_N+ used directly)", "objective": j_p,
                       "old_repair_objective": None, "feasible": True, "max_violation": 0.0,
                       "base": "S_N+", "log_text": "repairer NOT called",
                       "plan": {k: np.asarray(getattr(plus, k)).tolist()
                                for k in ("X", "Y", "Z", "I", "L")}},
        }

    with open(OUT_STATES, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nstates written: %d (%d fault + 8 no-fault) -> %s"
          % (len(out), len(out) - 8, OUT_STATES))
    print("fault repairs feasible: %d/%d" % (n_feas, len(insts) * 4))

    print("\n=== descriptive: old vs new repair cost, per instance ===")
    print("%-19s %-11s %11s %11s %9s" % ("instance", "disruption", "J_r(old)", "J_r(new)",
                                          "change"))
    for name in insts:
        for dis_name in ("D1_m0_1p", "D1_m0_2p", "D2_all_1p", "D2_all_2p"):
            k = "%s|%s" % (name, dis_name)
            a = out[k]["repair"]["old_repair_objective"]
            b = out[k]["repair"]["objective"]
            print("%-19s %-11s %11.2f %11.2f %+9.2f" % (name, dis_name, a, b, b - a))
    print("\nA lower nominal cost does NOT guarantee an easier repair; no such assumption is "
          "made and no state was redrawn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
