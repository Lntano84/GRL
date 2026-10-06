"""
Stage 03 data-persistence check.

Reads `stage03_landscape.json` back from disk and, for every entry of the landscape table,
rebuilds the stored plan, re-runs the INDEPENDENT checker and confirms

  * the plan is feasible,
  * the recomputed cost equals the stored cost,
  * the stored short-term flip count is reproducible and within kappa,
  * the stored F(S) equals J(empty) - J(S).

Nothing is re-solved. Run after `python run.py`:

    python stage03_savecheck.py
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from lsp_checker import check_solution
from lsp_model import MODE_AUDITED, Solution, build_model
from lsp_repair import binary_change_count, binary_change_count_full
from stage03_run import KAPPA, TOL, load_states


def rebuild(inst, rec: Dict[str, Any]) -> Solution:
    p = rec["plan"]
    return Solution(status="stored", objective=float(rec["cost"]),
                    X=np.array(p["X"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(p["Y"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(p["Z"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(p["I"], dtype=float).reshape(inst.N, inst.T),
                    L=np.array(p["L"], dtype=float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def parse_key(key: str) -> Tuple[Tuple[int, int, int], ...]:
    if key == "EMPTY":
        return tuple()
    out = []
    for part in key.split("|"):
        nums = part.strip("() ").split(",")
        out.append(tuple(int(x) for x in nums))
    return tuple(out)


def main() -> int:
    with open("stage03_landscape.json", encoding="utf-8") as fh:
        D = json.load(fh)
    states = load_states(D["meta"]["source"])

    failures: List[str] = []
    checked = 0
    worst_viol = 0.0
    print("=" * 100)
    print("%-16s %8s %10s %8s %8s  %s" %
          ("state", "entries", "maxviol", "flipsOK", "costOK", "status"))
    print("-" * 100)
    for name, res in D["results"].items():
        st = states[name]
        J_empty_stored = res["J_empty"]
        n_cost_bad = 0
        n_flip_bad = 0
        n_f_bad = 0
        state_worst = 0.0
        for key, rec in res["table"].items():
            sol = rebuild(st.inst, rec)
            chk = check_solution(st.inst, sol, MODE_AUDITED)
            checked += 1
            state_worst = max(state_worst, chk.max_violation)
            worst_viol = max(worst_viol, chk.max_violation)
            if not chk.ok:
                failures.append("%s/%s: infeasible, max violation %.3g"
                                % (name, key, chk.max_violation))
            if abs(chk.cost_recomputed["total"] - rec["cost"]) > TOL * (1 + abs(rec["cost"])):
                n_cost_bad += 1
                failures.append("%s/%s: stored cost %s != recomputed %s"
                                % (name, key, rec["cost"], chk.cost_recomputed["total"]))
            flip = binary_change_count(st.inst, sol.Y, st.repair.Y, st.tau)
            flip_full = binary_change_count_full(st.inst, sol.Y, st.repair.Y)
            if flip != rec["flips_vs_repair"] or flip_full != rec["flips_vs_repair_full"]:
                n_flip_bad += 1
                failures.append("%s/%s: stored flips %s/%s != recomputed %d/%d"
                                % (name, key, rec["flips_vs_repair"],
                                   rec["flips_vs_repair_full"], flip, flip_full))
            if flip > KAPPA + 1e-9:
                failures.append("%s/%s: flips %d exceed kappa %d" % (name, key, flip, KAPPA))
            expect_F = J_empty_stored - rec["cost"]
            if rec["F"] is None or abs(rec["F"] - expect_F) > TOL * (1 + abs(expect_F)):
                n_f_bad += 1
                failures.append("%s/%s: stored F %s != J(empty)-J(S) = %s"
                                % (name, key, rec["F"], expect_F))
        print("%-16s %8d %10.1e %8s %8s  %s" % (
            name, len(res["table"]), state_worst,
            "ok" if n_flip_bad == 0 else "FAIL",
            "ok" if n_cost_bad == 0 else "FAIL",
            "ok" if not (n_cost_bad or n_flip_bad or n_f_bad) else "FAIL"))

    print("=" * 100)
    print("table entries re-verified from disk: %d" % checked)
    print("largest constraint violation over all rebuilt plans: %.3g" % worst_viol)
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:40]:
            print("   -", f)
        return 1
    print("ALL %d LANDSCAPE PLANS RE-VERIFY FROM DISK; costs, flip counts and F(S) agree"
          % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
