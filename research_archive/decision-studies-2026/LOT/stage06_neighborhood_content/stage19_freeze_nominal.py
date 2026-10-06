"""Stage 19 step 0: freeze the 8 improved nominal plans as `S_N+`.

Source: the delivered plan of each Stage 18 run (RINS-LP-ONE, seed 1, 20 s, kappa = 4 vs S_N).
Each plan is written to its own versioned file, and the originals in `stage06_nominal.json`
are NOT touched.

Frozen BEFORE any new disruption state is generated.  No plan is selected or replaced on the
basis of post-disruption performance.  No multi-seed best-of selection is performed.

Records per plan: source run, seed, objective value, the complete plan, and file SHA-256.
"""

import hashlib
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_gen import build_instance
from lsp_model import MODE_AUDITED
from lsp_checker import check_solution
from lsp_model import Solution

NOMINAL_FILE = "stage06_nominal.json"
RUNS18_FILE = "stage18_runs.json"
PLAN_DIR = "stage19_nominal_plus"
INDEX_FILE = "stage19_nominal_plus.json"

PLAN_KEYS = ("X", "Y", "Z", "I", "L")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    with open(NOMINAL_FILE, encoding="utf-8") as fh:
        NOM = json.load(fh)
    with open(RUNS18_FILE, encoding="utf-8") as fh:
        R18 = json.load(fh)["runs"]

    os.makedirs(PLAN_DIR, exist_ok=True)
    index: Dict[str, Any] = {
        "meta": {
            "name": "S_N+ (improved nominal plans)",
            "generated_by": "RINS-LP-ONE, seed 1, 20 s end-to-end, kappa = 4 relative to S_N",
            "source_stage": 18,
            "source_file": RUNS18_FILE,
            "original_file_untouched": NOMINAL_FILE,
            "frozen_before": "any stage 19 disruption state was generated",
            "selection": "no plan selected or replaced on post-disruption performance; "
                         "no multi-seed best-of",
            "caution": "these plans were produced BY RINS, so later results are a sensitivity "
                       "check on this nominal-plan GENERATION PROTOCOL, not an independent, "
                       "unconditional confirmation of algorithmic generality",
            "generation_cost_s": sum(r["timing"]["total_s"] for r in R18),
            "generation_cost_note": "offline preparation; NOT counted in the stage 19 online "
                                    "20 s budget, but disclosed here",
        },
        "plans": {},
    }

    print("%-19s %12s %12s %9s %7s %s"
          % ("instance", "J_N (old)", "J_N+ (new)", "gain %", "kappa", "file"))
    for r in sorted(R18, key=lambda x: x["instance"]):
        name = r["instance"]
        rec = NOM[name]
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        assert inst.name == name, (inst.name, name)

        plan = {k: np.asarray(r["final_solution"][k], float).tolist() for k in PLAN_KEYS}
        sol = Solution(status="S_N+", objective=float(r["final_cost"]),
                       X=np.array(plan["X"], float).reshape(inst.N, inst.M, inst.T),
                       Y=np.array(plan["Y"], float).reshape(inst.N, inst.M, inst.T),
                       Z=np.array(plan["Z"], float).reshape(inst.N, inst.M, inst.T),
                       I=np.array(plan["I"], float).reshape(inst.N, inst.T),
                       L=np.array(plan["L"], float).reshape(inst.N, inst.T),
                       mode=MODE_AUDITED)
        chk = check_solution(inst, sol, MODE_AUDITED)          # NO DISRUPTION instance
        if not chk.ok:
            raise SystemExit("S_N+ for %s is INFEASIBLE on the undisrupted instance: %s"
                             % (name, chk.failed_checks()))
        if abs(chk.cost_recomputed["total"] - float(r["final_cost"])) > 1e-6 * (
                1 + abs(float(r["final_cost"]))):
            raise SystemExit("S_N+ cost mismatch for %s" % name)

        # short-term Y changes relative to S_N must satisfy kappa
        Yn = np.array(rec["solution"]["Y"], float).reshape(inst.N, inst.M, inst.T)
        sy = sum(1 for i in range(inst.N) for j in range(inst.M) for t in range(1, r["tau"] + 1)
                 if abs(float(sol.Y[i, j, t - 1]) - float(Yn[i, j, t - 1])) > 0.5)
        assert sy <= r["kappa"], (name, sy)

        path = os.path.join(PLAN_DIR, "%s.json" % name)
        payload = {"instance": name, "meta": rec["meta"], "objective": float(r["final_cost"]),
                   "old_nominal_objective": float(rec["objective"]),
                   "source": {"stage": 18, "method": "RINS-LP-ONE", "seed": r["seed"],
                              "budget_s": r["budget_s"], "kappa": r["kappa"], "tau": r["tau"],
                              "final_source": r["final_source"],
                              "short_y_changes": sy,
                              "wall_s": r["timing"]["total_s"]},
                   "plan": plan}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)

        digest = sha256_of(path)
        index["plans"][name] = {
            "file": path, "sha256": digest,
            "objective": float(r["final_cost"]),
            "old_nominal_objective": float(rec["objective"]),
            "gain_vs_old": (float(rec["objective"]) - float(r["final_cost"]))
                           / max(1.0, abs(float(rec["objective"]))),
            "rho": rec["meta"]["rho"], "inst_seed": rec["meta"]["seed"],
            "source": {"stage": 18, "method": "RINS-LP-ONE", "seed": r["seed"],
                       "budget_s": r["budget_s"], "kappa": r["kappa"], "tau": r["tau"],
                       "short_y_changes": sy},
            "feasible_undisrupted": True, "max_violation": float(chk.max_violation),
            "sha256": digest,
        }
        print("%-19s %12.2f %12.2f %+9.2f%% %7d %s"
              % (name, rec["objective"], r["final_cost"],
                 100 * index["plans"][name]["gain_vs_old"], sy, path))

    with open(INDEX_FILE, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    print("\nfroze %d plans -> %s (index: %s)" % (len(index["plans"]), PLAN_DIR, INDEX_FILE))
    print("original %s untouched (sha256 %s)" % (NOMINAL_FILE, sha256_of(NOMINAL_FILE)))
    print("offline generation cost: %.1f s total (%.1f s per instance) -- disclosed, not "
          "counted in the online budget"
          % (index["meta"]["generation_cost_s"],
             index["meta"]["generation_cost_s"] / len(index["plans"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
