"""
Stage 06 data-persistence check, covering every run.

    .venv-hs/Scripts/python.exe stage06_savecheck.py

Re-reads `stage06_runs.json` and re-verifies each of the 512 runs without re-solving:

  * the SELECTED plan is feasible on the original model, its recomputed cost equals the stored
    cost, it respects kappa, and it respects the configuration's fixing conditions
    (every short-term Y outside the release set equals the repaired plan's value);
  * the release set is consistent with the stored `release` list and its size;
  * MATCHED-RANDOM keeps the (machine, period, repair-value) profile of DEPENDENCY-24;
  * selection rule: source `solver` only when the raw incumbent was usable and cheaper;
  * the final output is never worse than the frozen repaired plan;
  * all runs were warm and every one has confirmed MIP-start adoption;
  * timing bookkeeping and CSV agreement.
"""

from __future__ import annotations

import csv
import json
import sys
from typing import Any, Dict, List

import numpy as np

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_model import MODE_AUDITED, Solution
from lsp_neighborhood import CONFIGS, RANDOM_SEEDS, match_random, _profile
from lsp_repair3 import binary_change_count
from stage06_run import BUDGET, KAPPA, solution_from_record

TOL = 1e-6


def rebuild(inst, plan: Dict[str, Any], objective: float) -> Solution:
    return Solution(status="stored", objective=float(objective),
                    X=np.array(plan["X"], float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(plan["Y"], float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(plan["Z"], float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(plan["I"], float).reshape(inst.N, inst.T),
                    L=np.array(plan["L"], float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def main() -> int:
    with open("stage06_runs.json", encoding="utf-8") as fh:
        D = json.load(fh)
    runs = D["runs"]
    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)
    with open("stage06_results.csv", encoding="utf-8-sig") as fh:
        csv_rows = {(r["state"], r["config"], int(r["seed"])): r for r in csv.DictReader(fh)}

    failures: List[str] = []
    n_checked = 0
    worst = 0.0
    n_cold_like = 0
    n_adopted = 0
    per_state_sets: Dict[str, Dict[str, List]] = {}

    print("=" * 100)
    print("%-30s %-16s %3s %9s %11s %9s  %s" %
          ("state", "config", "sd", "repair", "selected", "maxviol", "ok"))
    print("-" * 100)
    for r in runs:
        state, config, seed = r["state"], r["config"], r["seed"]
        rec = ST.get(state)
        if rec is None:
            failures.append("%s: state missing" % state)
            continue
        inst, _ = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        tau = rec["meta"]["tau"]
        jr = float(r["repair_cost"])
        sel_cost = r["selected_cost"]
        raw = r["raw_objective"]
        ok = True

        # ---- release set bookkeeping
        rel = {tuple(u) for u in r["release"]}
        if len(rel) != r["n_released"]:
            failures.append("%s/%s/s%d: release list length mismatch" % (state, config, seed))
            ok = False
        if config == "FULL":
            if rel != set(tuple(u) for u in _legal(pert, tau)):
                failures.append("%s/FULL/s%d: FULL must release every legal candidate"
                                % (state, seed))
                ok = False
        elif config == "EMPTY":
            if rel:
                failures.append("%s/EMPTY/s%d: EMPTY must release nothing" % (state, seed))
                ok = False
        elif len(rel) != 24:
            failures.append("%s/%s/s%d: expected 24 released, got %d"
                            % (state, config, seed, len(rel)))
            ok = False
        per_state_sets.setdefault(state, {})[config] = sorted(rel)

        # ---- matched-random must keep the dependency profile
        if config.startswith("MATCHED-RANDOM"):
            dep = per_state_sets[state].get("DEPENDENCY-24")
            if dep is not None:
                if _profile(list(rel), repair) != _profile(dep, repair):
                    failures.append("%s/%s/s%d: profile differs from DEPENDENCY-24"
                                    % (state, config, seed))
                    ok = False

        # ---- selected plan
        if sel_cost is None or r.get("selected_solution") is None:
            failures.append("%s/%s/s%d: selected plan missing" % (state, config, seed))
            ok = False
        else:
            sel = rebuild(pert, r["selected_solution"], sel_cost)
            chk = check_solution(pert, sel, MODE_AUDITED)
            n_checked += 1
            worst = max(worst, chk.max_violation)
            if not chk.ok:
                failures.append("%s/%s/s%d: INFEASIBLE %s"
                                % (state, config, seed, chk.failed_checks()))
                ok = False
            if abs(chk.cost_recomputed["total"] - sel_cost) > TOL * (1 + abs(sel_cost)):
                failures.append("%s/%s/s%d: stored %.10g != recomputed %.10g"
                                % (state, config, seed, sel_cost, chk.cost_recomputed["total"]))
                ok = False
            flips = binary_change_count(pert, sel.Y, repair.Y, tau)
            if flips > KAPPA + 1e-9:
                failures.append("%s/%s/s%d: flips %d > kappa" % (state, config, seed, flips))
                ok = False
            # fixings: outside the release set every short-term Y must equal the repair value
            bad = []
            for i in range(pert.N):
                for j in range(pert.M):
                    for t in range(1, tau + 1):
                        if (i, j, t) in rel:
                            continue
                        if abs(float(sel.Y[i, j, t - 1]) - float(repair.Y[i, j, t - 1])) > 0.5:
                            bad.append((i, j, t))
            if bad:
                failures.append("%s/%s/s%d: fixing conditions violated at %s"
                                % (state, config, seed, bad[:4]))
                ok = False

        # ---- raw plan
        if r.get("raw_solver_solution") is not None and raw is not None:
            rs = rebuild(pert, r["raw_solver_solution"], raw)
            chk_raw = check_solution(pert, rs, MODE_AUDITED)
            n_checked += 1
            worst = max(worst, chk_raw.max_violation)
            if bool(chk_raw.ok) != bool(r["raw_valid"]):
                failures.append("%s/%s/s%d: raw validity recorded %s re-checks %s"
                                % (state, config, seed, r["raw_valid"], chk_raw.ok))
                ok = False

        # ---- selection rule
        if r["selected_source"] == "solver":
            if raw is None or not r["raw_valid"]:
                failures.append("%s/%s/s%d: chose solver with invalid raw"
                                % (state, config, seed))
                ok = False
            elif raw > jr + TOL * (1 + abs(jr)):
                failures.append("%s/%s/s%d: chose a solver plan worse than the repair"
                                % (state, config, seed))
                ok = False
        else:
            if abs(sel_cost - jr) > TOL * (1 + abs(jr)):
                failures.append("%s/%s/s%d: chose repair but cost differs" % (state, config, seed))
                ok = False
        if sel_cost is not None and sel_cost > jr + TOL * (1 + abs(jr)):
            failures.append("%s/%s/s%d: FINAL OUTPUT worse than repair" % (state, config, seed))
            ok = False
        if r["select_errors"]:
            failures.append("%s/%s/s%d: selection errors %s"
                            % (state, config, seed, r["select_errors"]))
            ok = False

        # ---- every run is warm in this round
        if not r["start_adopted"]:
            n_cold_like += 1
            failures.append("%s/%s/s%d: no confirmed MIP-start adoption"
                            % (state, config, seed))
            ok = False
        else:
            n_adopted += 1
        rep_obj = r["start_reported_objective"]
        if rep_obj is not None and abs(rep_obj - jr) > TOL * (1 + abs(jr)):
            failures.append("%s/%s/s%d: adopted start cost %.10g != repair %.10g"
                            % (state, config, seed, rep_obj, jr))
            ok = False

        # ---- timing
        t = r["timing"]
        parts = t["feature_rank_s"] + t["transfer_s"] + t["solver_s"]
        if t["total_s"] + 1e-6 < parts:
            failures.append("%s/%s/s%d: total %.4f < parts" % (state, config, seed, t["total_s"]))
            ok = False
        if abs(t["overrun_s"] - max(0.0, t["total_s"] - BUDGET)) > 1e-3:
            failures.append("%s/%s/s%d: overrun inconsistent" % (state, config, seed))
            ok = False

        # ---- CSV
        row = csv_rows.get((state, config, seed))
        if row is None:
            failures.append("%s/%s/s%d: missing from CSV" % (state, config, seed))
            ok = False
        elif sel_cost is not None and \
                abs(float(row["selected_cost"]) - sel_cost) > TOL * (1 + abs(sel_cost)):
            failures.append("%s/%s/s%d: csv selected_cost mismatch" % (state, config, seed))
            ok = False

        if not ok:
            print("%-30s %-16s %3d %9.6g %11.6g %9s  FAIL"
                  % (state, config, seed, jr, sel_cost or float("nan"), "-"))
        elif n_checked % 64 == 0:
            print("%-30s %-16s %3d %9.6g %11.6g %9s  ok"
                  % (state, config, seed, jr, sel_cost or float("nan"), "-"))

    # ---- cross-config sanity: all runs of a state share the same repair and the same start
    print("=" * 100)
    for state, sets in sorted(per_state_sets.items()):
        if "DEPENDENCY-24" in sets and "MATCHED-RANDOM-1" in sets:
            ov = len(set(sets["DEPENDENCY-24"]) & set(sets["MATCHED-RANDOM-1"])) / 24.0
            if ov >= 0.999:
                print("  note: %s MATCHED-RANDOM-1 is almost identical to DEPENDENCY-24 "
                      "(overlap %.2f): limited information in the random control" % (state, ov))

    print("plans re-verified from disk: %d" % n_checked)
    print("largest constraint violation: %.3g" % worst)
    print("runs: %d | confirmed MIP-start adoption: %d | without: %d"
          % (len(runs), n_adopted, n_cold_like))
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:40]:
            print("   -", f)
        return 1
    print("ALL %d RUNS RE-VERIFY FROM DISK; release sets, fixings, selection, adoption and "
          "timing agree" % len(runs))
    return 0


def _legal(inst, tau):
    from lsp_neighborhood import legal_candidates
    return legal_candidates(inst, tau)


if __name__ == "__main__":
    sys.exit(main())
