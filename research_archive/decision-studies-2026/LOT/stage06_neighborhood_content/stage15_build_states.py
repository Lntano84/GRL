"""Stage 15 state builder + preflight.

Crosses disruption BREADTH (machine 0 only / all machines) with disruption DURATION
(1 period / 2 periods), all outages starting at period 1, on the 8 frozen development
instances (large_rho{0.75,1.10}_s{2..5}).

* The two EXISTING cells (`D1_m0_2p`, `D2_all_1p`) are rebuilt from the SAME frozen nominal plan
  so that all four cells come from one batch; their repair objectives are checked against the
  frozen Stage 06 values.
* The two NEW cells (`D1_m0_1p`, `D2_all_2p`) call the ORIGINAL repairer
  `repair_minbatch_v3` on the SAME frozen nominal plan.  No nominal plan is re-solved and no
  instance is redrawn on the basis of repair cost or expected gain.
* Every repair is re-checked from scratch by `check_solution` and must be feasible.

Writes `stage15_states.json`.  Solves NO MIPs.
"""

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import Disruption, apply_disruption, build_instance
from lsp_model import MODE_AUDITED, Solution
from lsp_neighborhood import recovery_period
from lsp_repair3 import repair_minbatch_v3

SRC_STATES = "stage06_states.json"
NOMINAL_FILE = "stage06_nominal.json"
OUT_STATES = "stage15_states.json"
TAU = 6
REPAIR_NAME = "repair_minbatch_v3"

# cell label -> (breadth, duration);  r_j follows from the down periods
CELL = {
    "D1_m0_1p": ("single", "1p"),
    "D1_m0_2p": ("single", "2p"),
    "D2_all_1p": ("all", "1p"),
    "D2_all_2p": ("all", "2p"),
}


def build_disruptions(inst) -> List[Disruption]:
    """The 2x2 breadth x duration grid; every outage starts at period 1."""
    all_m = list(range(inst.M))
    return [
        Disruption("D1_m0_1p", {0: [1]}, "machine 0 down for period 1"),
        Disruption("D1_m0_2p", {0: [1, 2]}, "machine 0 down for periods 1-2"),
        Disruption("D2_all_1p", {j: [1] for j in all_m}, "every machine down for period 1"),
        Disruption("D2_all_2p", {j: [1, 2] for j in all_m}, "every machine down for periods 1-2"),
    ]


def load_nominal(inst, rec) -> Solution:
    return Solution(status=rec["status"], objective=float(rec["objective"]),
                    X=np.array(rec["solution"]["X"], float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(rec["solution"]["Y"], float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(rec["solution"]["Z"], float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(rec["solution"]["I"], float).reshape(inst.N, inst.T),
                    L=np.array(rec["solution"]["L"], float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def main() -> int:
    with open(SRC_STATES, encoding="utf-8") as fh:
        ALL = json.load(fh)
    with open(NOMINAL_FILE, encoding="utf-8") as fh:
        NOM = json.load(fh)

    states = sorted(s for s in ALL if ALL[s]["meta"]["scale"] == "large")
    insts = sorted({ALL[s]["meta"]["name"] for s in states})
    print("nominal instances: %d | source states: %d" % (len(insts), len(states)))
    print("grid: breadth {single = machine 0 only, all = every machine}"
          "  x  duration {1p, 2p}, all outages start at period 1\n")

    # frozen repair objective per (instance, disruption), for the two existing cells
    frozen = {}
    for s in states:
        r = ALL[s]
        frozen[(r["meta"]["name"], r["disruption"]["name"])] = r["repair"]["objective"]
    frozen_nom = {ALL[s]["meta"]["name"]: ALL[s]["nominal_objective"] for s in states}

    out: Dict[str, Any] = {}
    n_exact = n_frozen = 0
    print("%-19s %-11s %-6s %-3s | %-9s %5s | %-11s %-11s %-9s | %s"
          % ("instance", "disruption", "breadth", "dur", "r_j", "relZ", "J_r(new)", "J_r(frozen)",
             "delta", "feasible"))
    for ins_name in insts:
        meta = ALL["%s|D1_m0_2p" % ins_name]["meta"]
        inst, _meta_built = build_instance(meta["scale"], meta["rho"], meta["seed"])
        assert inst.name == ins_name, (inst.name, ins_name)
        nom = load_nominal(inst, NOM[ins_name])
        d_nom = abs(float(nom.objective) - float(frozen_nom[ins_name]))
        assert d_nom < 1e-9, ("nominal objective mismatch", ins_name, d_nom)

        for dis in build_disruptions(inst):
            pert = apply_disruption(inst, dis)
            rep = repair_minbatch_v3(pert, nom, dis)
            chk = check_solution(pert, rep.solution, MODE_AUDITED)
            if not chk.ok:
                raise SystemExit("repair for %s|%s is INFEASIBLE: %s"
                                 % (ins_name, dis.name, chk.failed_checks()))

            r = {j: recovery_period(dis, j, pert.T) for j in range(pert.M)}
            # RECOVERY releases Z_{i,j,r_j}: one (item, machine, r_j) slot per item for every
            # affected machine whose recovery period falls inside the short-term window.
            affected = [j for j in range(pert.M) if dis.delta_for(j) > 0]
            in_window = [j for j in affected if r[j] <= TAU]
            n_relZ = pert.N * len(in_window)
            bre, dur = CELL[dis.name]

            fz = frozen.get((ins_name, dis.name))
            if fz is None:
                delta_s, flag = "-", "new"
            else:
                n_frozen += 1
                dd = abs(float(rep.solution.objective) - float(fz))
                delta_s = "%.3e" % dd
                flag = "EXACT" if dd < 1e-9 else "DIFF"
                if dd < 1e-9:
                    n_exact += 1

            print("%-19s %-11s %-6s %-3s | %-9s %5d | %-11.2f %-11s %-9s | %s"
                  % (ins_name, dis.name, bre, dur,
                     ",".join(str(r[j]) for j in range(pert.M)), n_relZ,
                     rep.solution.objective, ("%.2f" % fz) if fz else "-", delta_s,
                     "yes (mv=%.1e)" % chk.max_violation))

            out["%s|%s" % (ins_name, dis.name)] = {
                "meta": meta,
                "disruption": dis.as_dict(),
                "cell": {"breadth": bre, "duration": dur, "label": "%s / %s" % (bre, dur),
                         "n_affected": len(affected), "outage": {str(j): dis.down[j]
                                                                 for j in dis.down}},
                "recovery_periods": {str(j): r[j] for j in range(pert.M)},
                "n_rel_Z_recovery": n_relZ,
                "nominal_objective": float(nom.objective),
                "nominal_status": NOM[ins_name]["status"],
                "nominal_gap": NOM[ins_name].get("mip_gap"),
                "nominal_source": NOMINAL_FILE,
                "repair": {
                    "name": REPAIR_NAME, "objective": float(rep.solution.objective),
                    "feasible": True, "max_violation": float(chk.max_violation),
                    "log_text": rep.log_text()[:2000],
                    "plan": {k: np.asarray(getattr(rep.solution, k)).tolist()
                             for k in ("X", "Y", "Z", "I", "L")},
                },
            }

    with open(OUT_STATES, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("\nstates written: %d -> %s" % (len(out), OUT_STATES))
    print("existing cells: %d/%d rebuild the frozen repair objective EXACTLY"
          % (n_exact, n_frozen))
    if n_exact != n_frozen:
        print("  !! not all existing cells reproduced; investigate before running MIPs")
    print("new cells: %d" % (len(out) - n_frozen))
    print("all repairs feasible: %s" % all(v["repair"]["feasible"] for v in out.values()))
    print("\nreleased Z per cell (RECOVERY):")
    for name in sorted(CELL):
        v = [x for k, x in out.items() if x["disruption"]["name"] == name]
        if v:
            print("  %-11s %-6s %-3s  r_j=%s  n_rel_Z=%s"
                  % (name, CELL[name][0], CELL[name][1],
                     sorted({tuple(sorted(x["recovery_periods"].items())) for x in v}),
                     sorted({x["n_rel_Z_recovery"] for x in v})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
