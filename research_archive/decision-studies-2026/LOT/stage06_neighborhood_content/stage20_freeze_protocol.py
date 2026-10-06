"""Stage 20 step 0: freeze the nominal generation protocol NOM-160 and its output Sbar_N.

NOM-160 is the ALREADY-COMPLETED three-stage pipeline:

    1. the original Stage 06 nominal solve, 120 s per instance;
    2. one 20 s no-disruption RINS-LP-ONE run (Stage 18);
    3. one 20 s no-disruption RINS-LP-ONE run (Stage 19).

The frozen output is the Stage 19 no-disruption delivered plan of each instance, written to an
independent versioned file as `Sbar_N`.  All historical versions are kept.  No further seeds are
added and no plan is selected on post-disruption performance.

Budget note: 160 s is the SUM OF THE THREE STAGE BUDGETS; the actual wall clock is reported
separately.  The earlier 20 s polishing cost must NOT be presented as the complete nominal
generation cost.

Caution recorded with the protocol: it was settled DURING development, so it is NOT an
independently pre-confirmed best practice.  It is frozen here and will not be tuned further.
"""

import hashlib
import json
import os
import sys
from typing import Any, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import build_instance
from lsp_model import MODE_AUDITED, Solution

NOMINAL_FILE = "stage06_nominal.json"
RUNS19_FILE = "stage19_runs.json"
RUNS18_FILE = "stage18_runs.json"
PLUS_INDEX = "stage19_nominal_plus.json"
PLAN_DIR = "stage20_nominal_bar"
INDEX_FILE = "stage20_nominal_bar.json"
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
    with open(RUNS19_FILE, encoding="utf-8") as fh:
        R19 = json.load(fh)["runs"]
    with open(RUNS18_FILE, encoding="utf-8") as fh:
        R18 = json.load(fh)["runs"]
    with open(PLUS_INDEX, encoding="utf-8") as fh:
        PLUS = json.load(fh)

    nofault = {r["instance"]: r for r in R19 if r["arm"] == "no_fault"}
    stage18 = {r["instance"]: r for r in R18}
    insts = sorted(nofault)
    assert len(insts) == 8, len(insts)

    os.makedirs(PLAN_DIR, exist_ok=True)
    index: Dict[str, Any] = {
        "meta": {
            "protocol": "NOM-160",
            "stages": [
                {"stage": "06", "role": "nominal solve", "budget_s": 120.0},
                {"stage": "18", "role": "one no-disruption RINS-LP-ONE", "budget_s": 20.0},
                {"stage": "19", "role": "one no-disruption RINS-LP-ONE", "budget_s": 20.0},
            ],
            "sum_of_stage_budgets_s": 160.0,
            "frozen_output": "Sbar_N = the stage 19 no-disruption delivered plan",
            "historical_versions_kept": [NOMINAL_FILE, PLUS_INDEX, PLAN_DIR],
            "no_further_seeds": True,
            "no_selection_on_post_disruption_performance": True,
            "budget_note": "160 s is the SUM OF THE STAGE BUDGETS, not measured wall clock; the "
                           "earlier 20 s polishing cost must not be reported as the complete "
                           "nominal generation cost",
            "stability_note": "each polishing stage limits short-term-Y changes RELATIVE TO ITS "
                              "OWN INPUT plan; this does NOT mean the final plan changes only "
                              "four positions relative to the original.  These are pre-freeze "
                              "search limits.  The post-disruption stability constraint stays "
                              "anchored on the new repair solution, per the original protocol.",
            "provenance_caution": "this protocol was settled DURING development and is NOT an "
                                  "independently pre-confirmed best practice; it is frozen and "
                                  "will not be tuned further",
        },
        "plans": {},
    }

    print("%-19s %12s %12s %12s %8s %s"
          % ("instance", "J_N(06)", "J_N+(18)", "Sbar_N(19)", "wall19", "file"))
    for name in insts:
        rec = NOM[name]
        meta = nofault[name]["meta"] if "meta" in nofault[name] else None
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        assert inst.name == name, (inst.name, name)

        r19 = nofault[name]
        plan = {k: np.asarray(r19["final_solution"][k], float).tolist() for k in PLAN_KEYS}
        sol = Solution(status="Sbar_N", objective=float(r19["final_cost"]),
                       X=np.array(plan["X"], float).reshape(inst.N, inst.M, inst.T),
                       Y=np.array(plan["Y"], float).reshape(inst.N, inst.M, inst.T),
                       Z=np.array(plan["Z"], float).reshape(inst.N, inst.M, inst.T),
                       I=np.array(plan["I"], float).reshape(inst.N, inst.T),
                       L=np.array(plan["L"], float).reshape(inst.N, inst.T),
                       mode=MODE_AUDITED)
        chk = check_solution(inst, sol, MODE_AUDITED)
        if not chk.ok:
            raise SystemExit("Sbar_N for %s infeasible on the undisrupted instance: %s"
                             % (name, chk.failed_checks()))

        path = os.path.join(PLAN_DIR, "%s.json" % name)
        payload = {"instance": name, "meta": rec["meta"], "objective": float(r19["final_cost"]),
                   "old_nominal_objective": float(rec["objective"]),
                   "protocol": "NOM-160",
                   "source": {"stage19_state": r19["state"], "arm": "no_fault",
                              "method": "RINS-LP-ONE", "seed": r19["seed"],
                              "budget_s": r19["budget_s"], "kappa": r19["kappa"],
                              "tau": r19["tau"], "final_source": r19["final_source"],
                              "wall_s": r19["timing"]["total_s"],
                              "stage18_objective": float(stage18[name]["final_cost"])},
                   "plan": plan}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)

        digest = sha256_of(path)
        same_as_18 = abs(float(stage18[name]["final_cost"]) - float(r19["final_cost"])) < 1e-9
        index["plans"][name] = {
            "file": path, "sha256": digest,
            "objective": float(r19["final_cost"]),
            "old_nominal_objective": float(rec["objective"]),
            "stage18_objective": float(stage18[name]["final_cost"]),
            "identical_to_stage18": bool(same_as_18),
            "gain_vs_old": (float(rec["objective"]) - float(r19["final_cost"]))
                           / max(1.0, abs(float(rec["objective"]))),
            "rho": rec["meta"]["rho"], "inst_seed": rec["meta"]["seed"],
            "feasible_undisrupted": True, "max_violation": float(chk.max_violation),
            "wall_s_stage19": float(r19["timing"]["total_s"]),
        }
        print("%-19s %12.2f %12.2f %12.2f %8.1f %s"
              % (name, rec["objective"], stage18[name]["final_cost"], r19["final_cost"],
                 r19["timing"]["total_s"], path))

    index["meta"]["wall_clock_s"] = {
        "stage18_total": sum(r["timing"]["total_s"] for r in R18),
        "stage19_no_fault_total": sum(r["timing"]["total_s"] for r in nofault.values()),
        "note": "measured wall clock for the two polishing stages only; the 120 s stage is the "
                "stage 06 budget, whose measured wall clock is recorded in stage06_nominal.json",
    }
    with open(INDEX_FILE, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)

    n_same = sum(1 for v in index["plans"].values() if v["identical_to_stage18"])
    print("\nfroze %d plans -> %s (index: %s)" % (len(index["plans"]), PLAN_DIR, INDEX_FILE))
    print("identical to the stage 18 deliverable: %d/%d" % (n_same, len(index["plans"])))
    print("stage 18 wall %.1f s + stage 19 no-fault wall %.1f s (stage 06 wall is recorded in %s)"
          % (index["meta"]["wall_clock_s"]["stage18_total"],
             index["meta"]["wall_clock_s"]["stage19_no_fault_total"], NOMINAL_FILE))
    print("original %s untouched (sha256 %s)" % (NOMINAL_FILE, sha256_of(NOMINAL_FILE)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
