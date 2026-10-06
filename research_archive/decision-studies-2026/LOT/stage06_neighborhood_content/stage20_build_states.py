"""Stage 20 state builder: 16 fault states on the frozen NOM-160 output Sbar_N.

8 nominal instances x 2 disruptions (D1_m0_2p, D2_all_1p) = 16 states.

The two disruptions are the original main faults.  They are reused to control the scale and were
NOT chosen on Stage 19 gains.  This round does not study breadth/duration effects.

For each state: apply the original repairer to Sbar_N to obtain the new reference S_r.  The
stability anchor is kappa = 4 relative to this S_r (post-disruption stability stays anchored on
the new repair solution, per the original protocol).

Writes `stage20_states.json`.  Solves NO MIPs.
"""

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import Disruption, apply_disruption, build_instance
from lsp_model import MODE_AUDITED
from lsp_repair3 import repair_minbatch_v3

BAR_INDEX = "stage20_nominal_bar.json"
BAR_DIR = "stage20_nominal_bar"
OUT_STATES = "stage20_states.json"
TAU = 6
REPAIR_NAME = "repair_minbatch_v3"
DISRUPTIONS = ("D1_m0_2p", "D2_all_1p")


def build_disruptions(inst) -> List[Disruption]:
    return [
        Disruption("D1_m0_2p", {0: [1, 2]}, "machine 0 down for periods 1-2"),
        Disruption("D2_all_1p", {j: [1] for j in range(inst.M)}, "every machine down for period 1"),
    ]


def main() -> int:
    with open(BAR_INDEX, encoding="utf-8") as fh:
        BAR = json.load(fh)
    insts = sorted(BAR["plans"])
    print("Stage 20 | %d instances x %d disruptions = %d states"
          % (len(insts), len(DISRUPTIONS), len(insts) * len(DISRUPTIONS)))
    print("reference: Sbar_N (NOM-160); stability anchor: kappa=%d relative to the NEW repair "
          "solution\n" % 4)

    out: Dict[str, Any] = {}
    print("%-19s %-11s | %11s %11s %9s | %s"
          % ("instance", "disruption", "J_r(new)", "J_r(old)", "change", "feasible"))
    n_feas = 0
    for name in insts:
        entry = BAR["plans"][name]
        with open(entry["file"], encoding="utf-8") as fh:
            rec = json.load(fh)
        meta = rec["meta"]
        inst, _m = build_instance(meta["scale"], meta["rho"], meta["seed"])
        assert inst.name == name, (inst.name, name)
        p = rec["plan"]
        from lsp_model import Solution
        bar = Solution(status="Sbar_N", objective=float(rec["objective"]),
                       X=np.array(p["X"], float).reshape(inst.N, inst.M, inst.T),
                       Y=np.array(p["Y"], float).reshape(inst.N, inst.M, inst.T),
                       Z=np.array(p["Z"], float).reshape(inst.N, inst.M, inst.T),
                       I=np.array(p["I"], float).reshape(inst.N, inst.T),
                       L=np.array(p["L"], float).reshape(inst.N, inst.T),
                       mode=MODE_AUDITED)

        for dis in build_disruptions(inst):
            pert = apply_disruption(inst, dis)
            rep = repair_minbatch_v3(pert, bar, dis)
            chk = check_solution(pert, rep.solution, MODE_AUDITED)
            if not chk.ok:
                raise SystemExit("repair of Sbar_N for %s|%s INFEASIBLE: %s"
                                 % (name, dis.name, chk.failed_checks()))
            n_feas += 1
            j_new = float(rep.solution.objective)
            j_old = float(entry["stage18_objective"])
            out["%s|%s" % (name, dis.name)] = {
                "meta": meta,
                "disruption": dis.as_dict(),
                "cell": {"breadth": "single" if dis.name.startswith("D1") else "all",
                         "duration": "2p" if dis.name.endswith("2p") else "1p"},
                "bar_objective": float(bar.objective),
                "old_nominal_objective": float(rec["old_nominal_objective"]),
                "D_n": max(1.0, abs(float(rec["old_nominal_objective"]))),
                "D_n_note": "fixed scale = old nominal cost J_{N,n}",
                "repair": {"name": REPAIR_NAME, "objective": j_new,
                           "feasible": True, "max_violation": float(chk.max_violation),
                           "base": "Sbar_N (NOM-160)",
                           "log_text": rep.log_text()[:2000],
                           "plan": {k: np.asarray(getattr(rep.solution, k)).tolist()
                                    for k in ("X", "Y", "Z", "I", "L")}},
            }
            print("%-19s %-11s | %11.2f %11.2f %+9.2f | yes"
                  % (name, dis.name, j_new, j_old, j_new - j_old))

    with open(OUT_STATES, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nstates written: %d -> %s ; repairs feasible %d/%d"
          % (len(out), OUT_STATES, n_feas, len(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
