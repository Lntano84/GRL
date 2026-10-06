"""
Output selection: the rule that Stage 04 got wrong.

Contract
--------
A method always delivers the better of

    J_out = min( J_repair , J_valid_incumbent )

where "better" is by RECOMPUTED cost, and both candidates must pass independent validation.
Stage 04 only fell back to the repaired plan when the solver returned NO incumbent, so a
worse incumbent was returned as-is. That happened in 114 of 288 timing runs, including 8/8 of
the large FULL@20 runs, and it is what produced the apparently dramatic "restricted methods
beat FULL" result.

Validation performed on BOTH candidates
---------------------------------------
  1. the original model constraints, via the independent checker;
  2. the operational-stability budget against the repaired plan;
  3. the method's own fixing conditions;
  4. the cost recomputed from the plan.

An invalid incumbent is an ERROR, not a quiet timeout: the caller records it. A solver that
reports infeasibility while the validated repaired plan satisfies the same subproblem is also
an error, because a feasible witness exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lsp_checker import check_solution
from lsp_model import MODE_AUDITED, Instance, Solution
from lsp_repair3 import binary_change_count

TOL = 1e-6


@dataclass
class Candidate:
    """One candidate plan plus everything needed to judge it."""
    label: str                      # "solver" or "repair"
    present: bool                   # did the source produce a plan at all?
    cost: Optional[float] = None            # recomputed cost
    reported_cost: Optional[float] = None   # what the source claimed
    feasible: bool = False
    max_violation: Optional[float] = None
    stability_flips: Optional[int] = None
    stability_ok: bool = False
    fixing_ok: bool = False
    solution: Optional[Solution] = None
    problems: List[str] = field(default_factory=list)


@dataclass
class Selection:
    chosen: Optional[Candidate]
    source: str                     # "solver" | "repair" | "none"
    reason: str                     # no_incumbent | incumbent_worse_or_tied | incumbent_better
    errors: List[str] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.chosen is not None and not self.errors


def evaluate_candidate(label: str, inst: Instance, sol: Optional[Solution],
                       repair: Solution, kappa: float, tau: int,
                       fixed_Y: Optional[Dict[Tuple[int, int, int], float]] = None
                       ) -> Candidate:
    """Validate one candidate. `fixed_Y` maps (i,j,t) -> required value for the method's
    fixings, for t <= tau only."""
    if sol is None:
        return Candidate(label=label, present=False,
                         problems=["source produced no plan"])
    cand = Candidate(label=label, present=True, solution=sol,
                     reported_cost=float(sol.objective))
    chk = check_solution(inst, sol, MODE_AUDITED)
    cand.feasible = bool(chk.ok)
    cand.max_violation = chk.max_violation
    cand.cost = float(chk.cost_recomputed["total"])
    if not chk.ok:
        cand.problems.append("violates the original model: %s" % chk.failed_checks())
    if abs(cand.cost - cand.reported_cost) > TOL * (1 + abs(cand.cost)):
        cand.problems.append("reported cost %.10g != recomputed %.10g"
                             % (cand.reported_cost, cand.cost))

    flips = binary_change_count(inst, sol.Y, repair.Y, tau)
    cand.stability_flips = flips
    cand.stability_ok = flips <= kappa + 1e-9
    if not cand.stability_ok:
        cand.problems.append("stability budget exceeded: %d flips > kappa=%g" % (flips, kappa))

    if fixed_Y:
        bad = []
        for (i, j, t), want in fixed_Y.items():
            if t > tau:
                continue
            got = float(sol.Y[i, j, t - 1])
            if abs(got - want) > 0.5:
                bad.append("Y[%d,%d,%d]=%g wanted %g" % (i, j, t, got, want))
        cand.fixing_ok = not bad
        if bad:
            cand.problems.append("fixing conditions violated: %s" % bad[:4])
    else:
        cand.fixing_ok = True
    return cand


def is_usable(cand: Candidate, expected_cost: Optional[float] = None) -> bool:
    """A candidate is USABLE only if EVERY check passed.

    The first version of this module only required model feasibility here, so a plan that
    breached the stability budget or the method's fixing conditions could still be selected.
    The conditions are: present, model-feasible, within kappa, respecting the fixings, and
    (when an expected cost is supplied) cost-consistent with the recomputation.
    """
    if not cand.present or cand.solution is None:
        return False
    if not cand.feasible:
        return False
    if not cand.stability_ok:
        return False
    if not cand.fixing_ok:
        return False
    if expected_cost is not None and cand.cost is not None:
        if abs(cand.cost - expected_cost) > TOL * (1 + abs(expected_cost)):
            return False
    return True


def select_output(inst: Instance, solver_sol: Optional[Solution], repair: Solution,
                  kappa: float, tau: int,
                  fixed_Y: Optional[Dict[Tuple[int, int, int], float]] = None,
                  solver_status: str = "") -> Selection:
    """Apply J_out = min(J_repair, J_valid_incumbent)."""
    rep = evaluate_candidate("repair", inst, repair, repair, kappa, tau, None)
    sol = evaluate_candidate("solver", inst, solver_sol, repair, kappa, tau, fixed_Y)
    errors: List[str] = []

    if not is_usable(rep):
        errors.append("the repaired reference plan itself failed validation: %s"
                      % rep.problems)
    if solver_sol is not None and not is_usable(sol):
        failed = [k for k, ok in (("model", sol.feasible),
                                  ("kappa", sol.stability_ok),
                                  ("fixings", sol.fixing_ok)) if not ok]
        errors.append("INVALID INCUMBENT (failed: %s): %s"
                      % (", ".join(failed) or "cost consistency", sol.problems))
    if solver_sol is None and "infeasible" in (solver_status or "").lower():
        errors.append("solver reported INFEASIBLE while the validated repaired plan is a "
                      "feasible witness for the same subproblem")

    usable = [c for c in (sol, rep) if is_usable(c) and c.cost is not None]
    if not usable:
        return Selection(chosen=None, source="none", reason="no_valid_candidate",
                         errors=errors, candidates=[sol, rep])

    best = min(usable, key=lambda c: c.cost)
    if is_usable(sol) and best is sol:
        reason = "incumbent_better"
    elif not sol.present:
        reason = "no_incumbent"
    else:
        reason = "incumbent_worse_or_tied"
    return Selection(chosen=best, source=best.label, reason=reason,
                     errors=errors, candidates=[sol, rep])
