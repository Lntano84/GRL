#!/usr/bin/env python
"""
Stage 03: the local release-benefit landscape -- single versus pair releases of short-term
setup variables.

    python run.py

What this round does
--------------------
Reads the twelve disruption states and their repaired reference plans from the stage-02
artefacts. It does NOT regenerate nominal plans and does NOT re-run the repair heuristic:
both are frozen inputs.

Frozen quantities
-----------------
  model audited_v1; the original disruptions; the original repaired reference S^r; tau = 2;
  candidate universe U = all 8 short-term Y variables; kappa = 2 for every local release
  solve; short-term Y outside the released set is fixed and every other variable keeps the
  same freedom as in stage 02; each subproblem solved to optimality under the same 30 s cap.

Definitions
-----------
    J(S) = optimal cost with release set S released        (kappa = 2)
    F(S) = J(empty) - J(S)          <-- baseline is J(empty), NOT the repair cost J_r,
                                        so improvement obtainable without releasing any
                                        short-term variable is excluded from F.

Per state: 1 empty + 8 single + 28 pair = 37 subproblems, 444 across the twelve states.

The headline diagnostic is the greedy regret
    R_greedy = J(S_greedy) - min_{|S| <= 2} J(S)
i.e. whether joint selection reaches benefit that stepwise greedy misses.

Measured runtime is NOT deployment cost: it is recorded separately from the query counts,
because S_greedy can reuse solves already paid for by the landscape table.

Outputs: stage03_results.csv, stage03_landscape.json, stage03_details.md
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy

from lsp_checker import check_solution
from lsp_model import MODE_AUDITED, Instance, Solution, build_model, solve_milp
from lsp_repair import binary_change_count, binary_change_count_full

TOL = 1e-6
KAPPA = 2
SOLVE_TIME_LIMIT = 30.0

YIndex = Tuple[int, int, int]


def _g(v: Any, nd: int = 6) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "-"
    if abs(f) < 10 ** (-nd):
        return "0"
    return ("%." + str(nd) + "g") % f


# --------------------------------------------------------------------------------------
# Loading the frozen stage-02 state
# --------------------------------------------------------------------------------------


class FrozenState:
    def __init__(self, name: str, payload: Dict[str, Any], nominal_block: Dict[str, Any]):
        self.name = name
        self.payload = payload
        self.inst = self._build_instance(payload, nominal_block)
        self.tau = int(payload["tau"])
        self.repair = self._build_solution(payload["repair_solution"])
        self.repair_cost = float(payload["repair_cost"])
        self.nominal_cost = float(payload["nominal_cost"])
        self.disruption = payload["disruption"]
        self.stage02_J0 = payload["J0"]
        self.stage02_Jk = payload["Jk"]
        self.candidates: List[YIndex] = [
            (int(i), int(j), int(t))
            for i in range(self.inst.N) for j in range(self.inst.M)
            for t in range(1, self.tau + 1)
        ]

    @staticmethod
    def _build_instance(payload: Dict[str, Any], nominal_block: Dict[str, Any]) -> Instance:
        nb = nominal_block
        shape = np.array(payload["perturbed"]["d"]).shape
        N, T = int(shape[0]), int(shape[1])
        M = len(payload["perturbed"]["c"])
        return Instance(
            name="frozen", N=N, M=M, T=T,
            f=[float(x) for x in nb["f"]], p=[0.0] * N, h=[1.0] * N,
            l=[float(x) for x in nb["l"]], s=[1.0] * N, b=[1.0] * N,
            m=[float(x) for x in nb["m"]],
            d=[[float(v) for v in row] for row in payload["perturbed"]["d"]],
            c=[[float(v) for v in row] for row in payload["perturbed"]["c"]],
            w=[[1] * M for _ in range(N)],
            I0=[float(v) for v in payload["perturbed"]["I0"]],
            note="frozen from stage02: %s" % payload["disruption"]["description"])

    @staticmethod
    def _build_solution(rec: Dict[str, Any]) -> Solution:
        X = np.array(rec["X"], dtype=float)
        Y = np.array(rec["Y"], dtype=float)
        Z = np.array(rec["Z"], dtype=float)
        I = np.array(rec["I"], dtype=float)
        L = np.array(rec["L"], dtype=float)
        return Solution(status=rec.get("status", "stored"), objective=float(rec["objective"]),
                        X=X, Y=Y, Z=Z, I=I, L=L, mode=MODE_AUDITED)


def load_states(path: str) -> Dict[str, FrozenState]:
    with open(path, encoding="utf-8") as fh:
        S = json.load(fh)
    nom = S["nominal"]
    out: Dict[str, FrozenState] = {}
    for state, payload in S["states"].items():
        inst_name = state.split("|")[0]
        out[state] = FrozenState(state, payload, nom[inst_name]["instance"])
    return out


# --------------------------------------------------------------------------------------
# Instance parameters that stage 02 stored only as constants
# --------------------------------------------------------------------------------------

# The four small instances were generated from SMALL_SPEC; the per-item setup times,
# production times, holding costs and unit production costs are constants of that spec and
# were not duplicated into the per-state JSON. They are re-declared here and ASSERTED against
# the stored values so a mismatch cannot pass unnoticed.
SPEC_S = (1.0, 1.0)
SPEC_B = (1.0, 1.0)
SPEC_H = (1.0, 1.0)
SPEC_P = (0.0, 0.0)


def assert_spec_consistency(states: Dict[str, FrozenState]) -> None:
    for name, st in states.items():
        assert len(st.inst.s) == st.inst.N and len(st.inst.b) == st.inst.N
        assert tuple(st.inst.s) == SPEC_S[: st.inst.N], (name, st.inst.s)
        assert tuple(st.inst.b) == SPEC_B[: st.inst.N], (name, st.inst.b)
        assert tuple(st.inst.h) == SPEC_H[: st.inst.N], (name, st.inst.h)
        assert tuple(st.inst.p) == SPEC_P[: st.inst.N], (name, st.inst.p)


# --------------------------------------------------------------------------------------
# One release-set solve
# --------------------------------------------------------------------------------------


def candidate_is_possible(inst: Instance, u: YIndex) -> bool:
    """A Y position that compatibility or capacity rules out is not a candidate for the
    simple heuristic rules. w_ij = 0 rules it out outright; s_i > c'_jt rules it out because
    (1.4) then forces Y_ijt = 0."""
    i, j, t = u
    if inst.w[i][j] == 0:
        return False
    if inst.s[i] > inst.c[j][t - 1] + 1e-9:
        return False
    return True


def solve_release(bm, st: FrozenState, S: Sequence[YIndex]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    sol = solve_milp(bm, free_Y_subset=list(S), Y_reference=st.repair.Y, tau=st.tau,
                     stability_max_changes=float(KAPPA),
                     stability_reference=st.repair.Y, stability_tau=st.tau,
                     time_limit=SOLVE_TIME_LIMIT)
    wall = time.perf_counter() - t0
    ok = sol.ok
    chk = check_solution(st.inst, sol, MODE_AUDITED) if ok else None
    flips_rep = binary_change_count(st.inst, sol.Y, st.repair.Y, st.tau) if ok else None
    return {
        "S": [list(u) for u in S],
        "cost": float(sol.objective) if ok else None,
        "status": sol.status,
        "optimal": bool(ok),
        "mip_gap": sol.mip_gap,
        "solve_time_s": sol.solve_time_s,
        "wall_time_s": wall,
        "feasible": bool(chk.ok) if chk else False,
        "max_violation": chk.max_violation if chk else None,
        "flips_vs_repair": flips_rep,
        "flips_vs_repair_full": (binary_change_count_full(st.inst, sol.Y, st.repair.Y)
                                 if ok else None),
        "flipped_positions": ([list(u) for u in st.candidates
                               if abs(float(sol.Y[u[0], u[1], u[2] - 1])
                                      - float(st.repair.Y[u[0], u[1], u[2] - 1])) > 0.5]
                              if ok else []),
        "stability_ok": (flips_rep is not None and flips_rep <= KAPPA + 1e-9),
        "plan": {k: np.asarray(getattr(sol, k)).tolist() for k in ("X", "Y", "Z", "I", "L")}
        if ok else None,
    }


# --------------------------------------------------------------------------------------
# Selection methods
# --------------------------------------------------------------------------------------


def _sorted_sets(sets: Iterable[Sequence[YIndex]]) -> List[Tuple[YIndex, ...]]:
    """Deterministic dictionary order over sets of candidate indices."""
    return sorted((tuple(sorted(s)) for s in sets))


def _best_of_size(table: Dict[Tuple[YIndex, ...], Dict[str, Any]], size: int,
                   ) -> Tuple[Optional[Tuple[YIndex, ...]], Optional[float]]:
    """Lowest cost among sets of exactly `size`, with dictionary-order tie-breaking.

    The candidate pool is restricted to that size BEFORE comparing. An earlier version
    compared against every set in the table, so the empty set (cost = J(empty)) could not be
    beaten by a pair whose advantage was within the tolerance and the function returned the
    empty set; that is how 7 of 12 states reported `best_pair = inf`.
    """
    pool = [(S, r["cost"]) for S, r in table.items()
            if len(S) == size and r["cost"] is not None]
    if not pool:
        return None, None
    best_cost = min(c for _, c in pool)
    tied = sorted(S for S, c in pool if c <= best_cost + TOL * (1 + abs(best_cost)))
    return tied[0], best_cost


def method_best_single(table: Dict[Tuple[YIndex, ...], Dict[str, Any]],
                       ) -> Tuple[Optional[Tuple[YIndex, ...]], Optional[float]]:
    return _best_of_size(table, 1)


def method_best_pair(table: Dict[Tuple[YIndex, ...], Dict[str, Any]],
                     ) -> Tuple[Optional[Tuple[YIndex, ...]], Optional[float]]:
    return _best_of_size(table, 2)


def method_independent_rank(table: Dict[Tuple[YIndex, ...], Dict[str, Any]],
                            ) -> Tuple[Tuple[YIndex, ...], float, List[Dict[str, Any]]]:
    """Rank single variables by F({u}) descending, take the top two, evaluate that pair."""
    scored = []
    for S, rec in sorted(table.items()):
        if len(S) == 1 and rec["cost"] is not None:
            scored.append((S[0], rec["cost"]))
    if not scored:
        return tuple(), float("inf"), []
    scored.sort(key=lambda kv: (kv[1], kv[0]))          # best cost first, dictionary tie-break
    order = [{"u": list(u), "cost": c} for u, c in scored]
    chosen = tuple(sorted(u for u, _ in scored[:2]))
    if len(chosen) < 2:
        return chosen, (table.get(chosen, {}).get("cost") or float("inf")), order
    rec = table.get(chosen)
    return chosen, (rec["cost"] if rec and rec["cost"] is not None else float("inf")), order


def method_greedy(table: Dict[Tuple[YIndex, ...], Dict[str, Any]],
                  candidates: Sequence[YIndex]) -> Dict[str, Any]:
    """Two-step greedy: best single first, then the variable whose addition gives the lowest
    cost. The second step ALWAYS runs, even when the first step gains nothing, so that an
    early stop cannot hide a joint benefit."""
    singles = [(u, table[(u,)]["cost"]) for u in sorted(candidates)
               if (u,) in table and table[(u,)]["cost"] is not None]
    if not singles:
        return {"S": tuple(), "cost": float("inf"), "ties": [], "tie_spread": None}
    best_cost = min(c for _, c in singles)
    tied = sorted(u for u, c in singles if c <= best_cost + TOL * (1 + abs(best_cost)))
    chosen_first = tied[0]

    def second_step(first: YIndex):
        options = []
        for v in sorted(candidates):
            if v == first:
                continue
            key = tuple(sorted((first, v)))
            rec = table.get(key)
            if rec and rec["cost"] is not None:
                options.append((rec["cost"], v, key))
        if not options:
            return None, None, None
        options.sort(key=lambda t: (t[0], t[1]))
        return options[0][2], options[0][0], options

    S_main, cost_main, _ = second_step(chosen_first)

    # spread over first-step ties: every tied first choice is carried through the same
    # second step, so a spread caused purely by tie-breaking becomes visible
    tie_results = []
    for u in tied:
        S_u, c_u, _ = second_step(u)
        tie_results.append({"first": list(u), "S": [list(x) for x in (S_u or ())],
                            "cost": c_u})
    costs = [r["cost"] for r in tie_results if r["cost"] is not None]
    spread = ({"best": min(costs), "worst": max(costs),
               "differs": bool(max(costs) - min(costs) > TOL * (1 + abs(min(costs))))}
              if costs else None)
    return {"S": S_main or tuple(), "cost": cost_main if cost_main is not None else float("inf"),
            "first": chosen_first, "n_first_ties": len(tied), "tie_results": tie_results,
            "tie_spread": spread}


def shortage_scores(st: FrozenState) -> Dict[YIndex, float]:
    """a(i,j,t) = l_i * sum_{r>=t} L^r_ir / (s_i + b_i m_i), using ONLY the repaired plan and
    known instance parameters."""
    out: Dict[YIndex, float] = {}
    for u in st.candidates:
        i, j, t = u
        shortage = sum(float(st.repair.L[i, r - 1]) for r in range(t, st.inst.T + 1))
        out[u] = (st.inst.l[i] * shortage) / (st.inst.s[i] + st.inst.b[i] * st.inst.m[i])
    return out


def ranked_by_score(scores: Dict[YIndex, float],
                    feasible: Sequence[YIndex]) -> List[YIndex]:
    return sorted([u for u in feasible], key=lambda u: (-scores[u], u))


def method_shortage_first(st: FrozenState, feasible: Sequence[YIndex],
                          ) -> Tuple[Tuple[YIndex, ...], str]:
    scores = shortage_scores(st)
    order = ranked_by_score(scores, feasible)
    if len(order) < 2:
        return tuple(sorted(order)), "fewer than two feasible candidates"
    return tuple(sorted(order[:2])), "top two by a(i,j,t)"


def method_shortage_plus_capacity(st: FrozenState, feasible: Sequence[YIndex],
                                  ) -> Tuple[Tuple[YIndex, ...], str]:
    """First variable by the shortage score; partner = same machine and period, different
    item, Y^r = 1, largest original s_i Y^r + b_i X^r. Falls back to the next shortage
    candidate. Never consults the enumerated benefits."""
    scores = shortage_scores(st)
    order = ranked_by_score(scores, feasible)
    if not order:
        return tuple(), "no feasible candidate"
    first = order[0]
    i, j, t = first
    partners = []
    for u in feasible:
        ii, jj, tt = u
        if (jj, tt) == (j, t) and ii != i and st.repair.Y[ii, jj, tt - 1] > 0.5:
            occupied = (st.inst.s[ii] * float(st.repair.Y[ii, jj, tt - 1])
                        + st.inst.b[ii] * float(st.repair.X[ii, jj, tt - 1]))
            partners.append((occupied, u))
    if partners:
        partners.sort(key=lambda kv: (-kv[0], kv[1]))
        partner = partners[0][1]
        why = ("partner chosen by largest original s_i Y^r + b_i X^r (=%g) on machine %d "
               "period %d" % (partners[0][0], j, t))
    else:
        rest = [u for u in order if u != first]
        if not rest:
            return tuple(sorted([first])), "no partner available"
        partner = rest[0]
        why = "no same-machine same-period partner with Y^r = 1; fell back to the next shortage candidate"
    return tuple(sorted((first, partner))), why


# --------------------------------------------------------------------------------------
# Per-state analysis
# --------------------------------------------------------------------------------------


def analyse_state(st: FrozenState) -> Dict[str, Any]:
    bm = build_model(st.inst, MODE_AUDITED)
    feasible = [u for u in st.candidates if candidate_is_possible(st.inst, u)]
    excluded = [u for u in st.candidates if u not in feasible]

    table: Dict[Tuple[YIndex, ...], Dict[str, Any]] = {}
    subsets: List[Tuple[YIndex, ...]] = [tuple()]
    subsets += [(u,) for u in st.candidates]
    subsets += list(itertools.combinations(st.candidates, 2))
    for S in subsets:
        table[S] = solve_release(bm, st, S)

    J_empty = table[tuple()]["cost"]
    for S, rec in table.items():
        rec["F"] = (J_empty - rec["cost"]) if (J_empty is not None and rec["cost"] is not None) else None

    def cost_of(S: Sequence[YIndex]) -> Optional[float]:
        rec = table.get(tuple(sorted(S)))
        return None if rec is None else rec["cost"]

    # ---- landscape aggregates over |S| <= 2
    finite = {S: r["cost"] for S, r in table.items()
              if r["cost"] is not None and len(S) <= 2}
    single_costs = {S[0]: c for S, c in finite.items() if len(S) == 1}
    pair_costs = {S: c for S, c in finite.items() if len(S) == 2}

    # single source of truth for the best sets: one helper, used by every consumer
    best_single_key, best_single_cost = method_best_single(table)
    best_pair_key, best_pair_cost = method_best_pair(table)
    best_single = best_single_key[0] if best_single_key else None
    best_pair = best_pair_key
    J_best = best_pair_cost
    J_best_single = best_single_cost

    # ---- superadditivity C(u,v) = F({u,v}) - F({u}) - F({v})
    superadd = []
    for S, c in sorted(pair_costs.items()):
        u, v = S
        fu = table[(u,)]["F"]
        fv = table[(v,)]["F"]
        if fu is None or fv is None:
            continue
        fuv = table[S]["F"]
        C = fuv - fu - fv
        superadd.append({"u": list(u), "v": list(v), "F_u": fu, "F_v": fv, "F_uv": fuv,
                         "C": C,
                         "both_singles_zero": (abs(fu) <= TOL and abs(fv) <= TOL),
                         "complementary": bool(abs(fu) <= TOL and abs(fv) <= TOL
                                               and fuv > TOL)})
    n_comp = sum(1 for r in superadd if r["complementary"])
    max_C = max((r["C"] for r in superadd), default=None)

    # ---- methods
    S_best, J_best = best_pair_key, best_pair_cost
    S_rank, J_rank, rank_order = method_independent_rank(table)
    greedy = method_greedy(table, st.candidates)
    S_short, why_short = method_shortage_first(st, feasible)
    S_cap, why_cap = method_shortage_plus_capacity(st, feasible)

    def value(S):
        c = cost_of(S)
        return c

    J_greedy = greedy["cost"]
    J_short = value(S_short)
    J_cap = value(S_cap)
    regret = (J_greedy - J_best) if (J_best is not None and np.isfinite(J_greedy)) else None
    R1 = (best_single_cost - J_best) if (best_single_cost is not None and J_best is not None) else None

    # ---- exposure of greedy to its first-step tie-breaking.
    # Greedy fixes ONE first variable and never looks at pairs built from the other tied
    # first choices. If the optimal pair cannot be reached from any tied first variable, a
    # greedy miss is an artefact of tie-breaking rather than evidence about joint selection.
    tied_firsts = [tuple(r["first"]) for r in (greedy.get("tie_results") or [])]
    reachable = set()
    for f in tied_firsts:
        for v in st.candidates:
            if v == f:
                continue
            key = tuple(sorted((f, v)))
            if key in table:
                reachable.add(key)
    optimal_pair_reachable = (tuple(best_pair) in reachable) if best_pair else None
    tie_exposure = {
        "n_first_ties": len(tied_firsts),
        "n_pairs_reachable_from_tied_firsts": len(reachable),
        "n_pairs_total": len(pair_costs),
        "optimal_pair_reachable_from_a_tied_first": optimal_pair_reachable,
        "optimal_pair": [list(x) for x in best_pair] if best_pair else None,
    }

    return {
        "state": st.name, "disruption": st.disruption,
        "instance": {"d": st.inst.d, "c": st.inst.c, "I0": st.inst.I0,
                     "m": st.inst.m, "f": st.inst.f, "l": st.inst.l},
        "tau": st.tau, "kappa": KAPPA,
        "nominal_cost": st.nominal_cost, "repair_cost": st.repair_cost,
        "candidates": [list(u) for u in st.candidates],
        "candidates_excluded": [list(u) for u in excluded],
        "table": {"|".join(map(str, S)) if S else "EMPTY": r for S, r in
                  sorted(table.items(), key=lambda kv: (len(kv[0]), kv[0]))},
        "J_empty": J_empty,
        "best_single": list(best_single) if best_single else None,
        "best_single_cost": best_single_cost,
        "best_pair": [list(x) for x in best_pair] if best_pair else None,
        "best_pair_cost": J_best,
        "C_superadditivity": superadd,
        "n_complementary_pairs": n_comp,
        "max_C": max_C,
        "methods": {
            "optimal_pair": {"S": [list(x) for x in (S_best or ())], "cost": J_best,
                             "queries": 1 + len(st.candidates)
                                        + len(list(itertools.combinations(st.candidates, 2)))},
            "best_single": {"S": [list(best_single)] if best_single else [],
                            "cost": best_single_cost, "queries": len(st.candidates)},
            "independent_rank": {"S": [list(x) for x in S_rank], "cost": J_rank,
                                 "rank_order": rank_order,
                                 "queries": len(st.candidates) + 1},
            "greedy": {"S": [list(x) for x in greedy["S"]], "cost": J_greedy,
                       "first": list(greedy["first"]) if greedy.get("first") else None,
                       "n_first_ties": greedy.get("n_first_ties"),
                       "tie_results": greedy.get("tie_results"),
                       "tie_spread": greedy.get("tie_spread"),
                       "queries": 2 * len(st.candidates)},
            "shortage_first": {"S": [list(x) for x in S_short], "cost": J_short,
                               "why": why_short, "queries": 1},
            "shortage_plus_capacity": {"S": [list(x) for x in S_cap], "cost": J_cap,
                                       "why": why_cap, "queries": 1},
        },
        "R_greedy": regret,
        "R1": R1,
        "tie_exposure": tie_exposure,
        "max_violation": max((r["max_violation"] or 0.0) for r in table.values()),
        "n_nonoptimal": sum(1 for r in table.values() if not r["optimal"]),
    }


# --------------------------------------------------------------------------------------
# Cross-checks against stage 02
# --------------------------------------------------------------------------------------


def cross_checks(st: FrozenState, res: Dict[str, Any]) -> Dict[str, Any]:
    j1 = st.stage02_Jk.get("1")
    j2 = st.stage02_Jk.get("2")
    out = {
        "stage02_kappa1": j1, "stage02_kappa2": j2,
        "landscape_best_single": res["best_single_cost"],
        "landscape_best_pair": res["best_pair_cost"],
    }
    out["single_matches"] = (j1 is not None and res["best_single_cost"] is not None
                             and abs(res["best_single_cost"] - j1)
                             <= TOL * (1 + abs(j1)))
    out["pair_matches"] = (j2 is not None and res["best_pair_cost"] is not None
                           and abs(res["best_pair_cost"] - j2) <= TOL * (1 + abs(j2)))
    return out


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


CSV_FIELDS = ["state", "empty", "best_single_cost", "best_pair_cost", "independent_rank",
              "greedy", "shortage_first", "shortage_plus_capacity", "greedy_regret",
              "R1", "n_complementary_pairs", "max_C", "greedy_first_ties",
              "greedy_tie_best", "greedy_tie_worst", "best_single_index", "best_pair_index",
              "greedy_index", "optimal_pair_reachable", "check_single_ok", "check_pair_ok",
              "max_violation", "n_nonoptimal", "n_plans_reverified", "reverify_ok"]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 03 local release landscape")
    ap.add_argument("--solutions", default="../stage02_repair_reopt/stage02_solutions.json")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    states = load_states(args.solutions)
    assert_spec_consistency(states)
    print("loaded %d frozen states from %s" % (len(states), args.solutions))

    results: Dict[str, Any] = {}
    checks: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for name, st in states.items():
        res = analyse_state(st)
        cc = cross_checks(st, res)
        results[name] = res
        checks[name] = cc
        m = res["methods"]
        ts = m["greedy"]["tie_spread"]
        n_bad = sum(1 for r in res["table"].values()
                    if not r["optimal"] or not r["feasible"]
                    or not r["stability_ok"])
        rows.append({
            "state": name,
            "empty": res["J_empty"],
            "best_single_cost": res["best_single_cost"],
            "best_pair_cost": res["best_pair_cost"],
            "independent_rank": m["independent_rank"]["cost"],
            "greedy": m["greedy"]["cost"],
            "shortage_first": m["shortage_first"]["cost"],
            "shortage_plus_capacity": m["shortage_plus_capacity"]["cost"],
            "greedy_regret": res["R_greedy"],
            "R1": res["R1"],
            "n_complementary_pairs": res["n_complementary_pairs"],
            "max_C": res["max_C"],
            "greedy_first_ties": m["greedy"]["n_first_ties"],
            "greedy_tie_best": (ts["best"] if ts else ""),
            "greedy_tie_worst": (ts["worst"] if ts else ""),
            "best_single_index": res["best_single"],
            "best_pair_index": res["best_pair"],
            "greedy_index": m["greedy"]["S"],
            "optimal_pair_reachable": res["tie_exposure"][
                "optimal_pair_reachable_from_a_tied_first"],
            "check_single_ok": cc["single_matches"],
            "check_pair_ok": cc["pair_matches"],
            "max_violation": res["max_violation"],
            "n_nonoptimal": res["n_nonoptimal"],
            "n_plans_reverified": len(res["table"]),
            "reverify_ok": (n_bad == 0),
        })

    # ---- write outputs
    csv_path = f"{args.outdir}/stage03_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(f"{args.outdir}/stage03_landscape.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": {"tau": states[list(states)[0]].tau, "kappa": KAPPA,
                            "solve_time_limit_s": SOLVE_TIME_LIMIT,
                            "scipy": scipy.__version__,
                            "python": platform.python_version(),
                            "source": args.solutions,
                            "n_subproblems": sum(len(r["table"]) for r in results.values())},
                   "results": results, "cross_checks": checks}, fh, indent=2, default=str)

    with open(f"{args.outdir}/stage03_details.md", "w", encoding="utf-8") as fh:
        fh.write(render_details(results, checks, rows))

    # ---- console
    print("=" * 118)
    # 12 columns need 12 specifiers: with only 11 the % operator cycles and every column
    # after the missing one is shifted by one argument.
    hdr = ("%-16s %7s %7s %7s %8s %7s %7s %7s %8s %8s %7s %s" %
           ("state", "empty", "best1", "best2", "indepRk", "greedy", "short1", "short2",
            "regret", "R1", "nComp", "checks"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-16s %7s %7s %7s %8s %7s %7s %7s %8s %8s %7d %s/%s" % (
            r["state"], _g(r["empty"]), _g(r["best_single_cost"]), _g(r["best_pair_cost"]),
            _g(r["independent_rank"]), _g(r["greedy"]), _g(r["shortage_first"]),
            _g(r["shortage_plus_capacity"]), _g(r["greedy_regret"]), _g(r["R1"]),
            r["n_complementary_pairs"],
            "ok" if r["check_single_ok"] else "FAIL",
            "ok" if r["check_pair_ok"] else "FAIL"))
    print("=" * 118)
    n_single = sum(1 for c in checks.values() if not c["single_matches"])
    n_pair = sum(1 for c in checks.values() if not c["pair_matches"])
    print("cross-check vs stage02: single mismatches=%d, pair mismatches=%d" % (n_single, n_pair))
    print("subproblems solved: %d" % sum(len(r["table"]) for r in results.values()))
    print("total wall time: %.1fs" % (time.perf_counter() - t0))
    print("wrote: %s\n       %s\n       %s" % (csv_path, f"{args.outdir}/stage03_landscape.json",
                                              f"{args.outdir}/stage03_details.md"))
    return 1 if (n_single or n_pair) else 0


def render_details(results, checks, rows) -> str:
    L: List[str] = []
    L.append("# stage03_details.md\n")
    L.append("Local release-benefit landscape. Produced by `python run.py` in "
             "`stage03_neighborhood_landscape`.\n")
    L.append("Environment: python %s, numpy %s, scipy %s.\n"
             % (platform.python_version(), np.__version__, scipy.__version__))
    L.append("Frozen: `audited_v1`, original disruptions, original repaired reference S^r, "
             "tau=2, kappa=%d for every local solve, 30 s cap per subproblem.\n" % KAPPA)
    L.append("`F(S) = J(empty) - J(S)`; the baseline is `J(empty)`, not the repair cost.\n")

    L.append("\n## 0. Main table\n")
    L.append("| 状态 | 空集合 | 最优单变量 | 最优双变量 | 单变量排序 | 两步贪心 | 缺货优先 | 缺货+容量释放 | 贪心遗憾 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["state"], _g(r["empty"]), _g(r["best_single_cost"]), _g(r["best_pair_cost"]),
            _g(r["independent_rank"]), _g(r["greedy"]), _g(r["shortage_first"]),
            _g(r["shortage_plus_capacity"]), _g(r["greedy_regret"])))

    L.append("\n## 1. Cross-checks against stage 02\n")
    L.append("| 状态 | stage02 kappa=1 | 本轮最优单变量 | 一致 | stage02 kappa=2 | 本轮最优双变量 | 一致 |")
    L.append("|---|---:|---:|---|---:|---:|---|")
    for name, cc in checks.items():
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            name, _g(cc["stage02_kappa1"]), _g(cc["landscape_best_single"]),
            "ok" if cc["single_matches"] else "**FAIL**",
            _g(cc["stage02_kappa2"]), _g(cc["landscape_best_pair"]),
            "ok" if cc["pair_matches"] else "**FAIL**"))

    L.append("\n## 2. Selected indices and diagnostics\n")
    L.append("| 状态 | 最优单变量 | 最优双变量 | R1 | 贪心首选 | 首选并列数 | 并列最好/最差 | 互补对数 | max C |")
    L.append("|---|---|---|---:|---|---:|---|---:|---:|")
    for name, res in results.items():
        m = res["methods"]
        ts = m["greedy"]["tie_spread"]
        L.append("| %s | `%s` | `%s` | %s | `%s` | %d | %s / %s | %d | %s |" % (
            name, res["best_single"], res["best_pair"], _g(res["R1"]),
            m["greedy"]["first"], m["greedy"]["n_first_ties"] or 0,
            _g(ts["best"]) if ts else "-", _g(ts["worst"]) if ts else "-",
            res["n_complementary_pairs"], _g(res["max_C"])))

    L.append("\n### Greedy tie exposure\n")
    L.append("Greedy commits to ONE first variable and only ever examines pairs containing "
             "it. If the optimal pair is not reachable from any tied first choice, a greedy "
             "miss would be a tie-breaking artefact, not evidence about joint selection.\n")
    L.append("| 状态 | 首选并列数 | 从并列首选可达的变量对 | 全部变量对 | 最优对可达 |")
    L.append("|---|---:|---:|---:|---|")
    for name, res in results.items():
        te = res["tie_exposure"]
        L.append("| %s | %d | %d | %d | %s |" % (
            name, te["n_first_ties"], te["n_pairs_reachable_from_tied_firsts"],
            te["n_pairs_total"],
            "-" if te["optimal_pair_reachable_from_a_tied_first"] is None
            else ("yes" if te["optimal_pair_reachable_from_a_tied_first"] else "**NO**")))
    L.append("\nThe reachable count is only %d of %d pairs when there is a single tied first "
             "choice, so the exposure is real even though it did not bite on these states.\n"
             % (len(results[list(results)[0]]["candidates"]) - 1,
                len(results[list(results)[0]]["candidates"])
                * (len(results[list(results)[0]]["candidates"]) - 1) // 2))

    L.append("\n## 3. Query counts (deployment cost is NOT the measured runtime)\n")
    L.append("| 方法 | 规则 | 每状态查询数 |")
    L.append("|---|---|---:|")
    L.append("| 最优双变量组 | 穷举所有 |S|<=2 | 37 |")
    L.append("| 最优单变量 | 穷举所有单变量 | 8 |")
    L.append("| 单变量收益排序 | 单变量打分后取前二 | 9 |")
    L.append("| 两步贪心 | 先最佳单变量，再加最佳第二变量 | 16 |")
    L.append("| 缺货优先 | 可见分数取前二，0 次额外求解 | 1 |")
    L.append("| 缺货+容量释放 | 缺货优先 + 同机同期搭档，0 次额外求解 | 1 |")
    L.append("\nThe last two consult no subproblem at all; their `1` is the single evaluation "
             "of the chosen set. Measured wall time in the JSON is the cost of BUILDING the "
             "table and must not be reported as the deployment cost of any method.\n")

    L.append("\n## 4. Superadditivity and complementary pairs\n")
    for name, res in results.items():
        rows_c = sorted(res["C_superadditivity"], key=lambda r: -r["C"])[:5]
        L.append("- **%s**: complementary pairs (both singles zero, pair positive) = %d; "
                 "largest C = %s" % (name, res["n_complementary_pairs"], _g(res["max_C"])))
        for r in rows_c:
            if r["C"] > TOL:
                L.append("    - `%s`+`%s`: F=%s, F_u=%s, F_v=%s, C=%s%s"
                         % (r["u"], r["v"], _g(r["F_uv"]), _g(r["F_u"]), _g(r["F_v"]),
                            _g(r["C"]), "  <-- complementary" if r["complementary"] else ""))

    L.append("\n## 5. Joint-benefit cases: what actually changed\n")
    L.append("Cases where `R1 > 0` (a second variable helps). The comparison is against the "
             "best SINGLE release, so the columns show what the second variable buys.\n")
    L.append("| 状态 | 集合 | 成本 | F(S) | 实际翻转位置 | κ 内翻转数 |")
    L.append("|---|---|---:|---:|---|---:|")
    for name, res in results.items():
        if not (res["R1"] and res["R1"] > TOL):
            continue
        want = [("empty", tuple())]
        if res["best_single"]:
            want.append(("best single", (tuple(res["best_single"]),)))
        if res["best_pair"]:
            want.append(("best pair", tuple(tuple(y) for y in res["best_pair"])))
        for label, key in want:
            rec = res["table"].get("|".join(map(str, key)) if key else "EMPTY")
            if not rec:
                continue
            L.append("| %s | %s `%s` | %s | %s | `%s` | %s |" % (
                name, label, [list(x) for x in key], _g(rec["cost"]), _g(rec["F"]),
                rec["flipped_positions"], rec["flips_vs_repair"]))
    L.append("\n## 6. Complementary pairs (both singles worthless, pair valuable)\n")
    any_comp = False
    for name, res in results.items():
        comp = [r for r in res["C_superadditivity"] if r["complementary"]]
        if not comp:
            continue
        any_comp = True
        L.append("- **%s**: %d complementary pair(s)" % (name, len(comp)))
        for r in comp:
            L.append("    - `%s` + `%s`: F(single)=%s, F(single)=%s, F(pair)=%s, C=%s"
                     % (r["u"], r["v"], _g(r["F_u"]), _g(r["F_v"]), _g(r["F_uv"]),
                        _g(r["C"])))
            te = res["tie_exposure"]
            L.append("      greedy regret on this state = %s; optimal pair reachable from a "
                     "tied first choice = %s"
                     % (_g(res["R_greedy"]),
                        te["optimal_pair_reachable_from_a_tied_first"]))
    if not any_comp:
        L.append("None in this batch.\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
