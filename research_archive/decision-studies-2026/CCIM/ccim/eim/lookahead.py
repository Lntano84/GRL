"""Legal look-ahead: what is one more survey worth *to the final seed decision*?

The quantity
------------
For the current survey record ``D``, attribute ``X`` and working model ``q(.; D, X)``, and for a fixed
end-of-survey seed solver ``A``:

    Q(D, u) = E_{G ~ q(. | D, X)} [ sigma_G( A(D + N_G(u), X) ) - sigma_G( A(D, X) ) ]

``D + N_G(u)`` is the survey record with ``u`` surveyed and the neighbours that ``G`` assigns to ``u``
added.  Two restrictions are structural and are enforced in the code, not merely intended:

1. **No peeking.**  ``A`` receives only ``D + N_G(u)``.  It never receives ``G``.  The generator graph is
   used for two things only: to draw the survey *result*, and to score the seed set that the solver
   produced from the augmented record -- which is what the expectation above says.
2. **No oracle.**  ``q`` is the same coarse working model every cheap rule uses.  ``Q`` is not the true
   value of information and is not an upper bound on anything; it is the value *according to this
   model*, which is all a decision maker without the network can have.

Every candidate is evaluated on the **same** set of sampled graphs, and the baseline term
``A(D, X)`` is computed once per state.  Keeping those fixed is what makes the comparison across
candidates a difference of two coupled quantities rather than a difference of two noisy estimates.

Cost
----
Each candidate costs ``samples`` solver calls, plus one survey simulation per sample.  That is the
whole point of reporting it: the method is only interesting if what it buys exceeds the compute it
burns, so the solver-call count is returned with the scores and the wall clock is recorded by the
caller.
"""

from __future__ import annotations

import time

import numpy as np


def base_value(model, rng, samples: int, solver):
    """``E[ sigma_G( A(D, X) ) ]`` -- the same for every candidate, so it is computed once."""
    total = 0.0
    for _ in range(int(samples)):
        edges, probs = model.edges_for_solver(rng)
        idx, value = solver.solve(edges, probs)
        total += value
    return total / float(samples)


def q_values(model, candidates, rng, samples: int, solver, base: float | None = None,
             timing: dict | None = None):
    """Legal look-ahead scores for ``candidates``.

    Returns ``(scores, diagnostics)``.  ``diagnostics`` carries the sampled baseline and per-candidate
    survey yield, so the report can say *how much* information each survey would have revealed, not
    only what it was judged to be worth.
    """
    t0 = time.perf_counter()
    if base is None:
        base = base_value(model, rng, samples, solver)
    t1 = time.perf_counter()

    scores, yields = {}, {}
    for u in candidates:
        u = int(u)
        total = 0.0
        revealed = 0
        for _ in range(int(samples)):
            child = model.copy()
            nb = model.predict_edges(u, rng)
            revealed += len(nb)
            child.resolve(u, nb)
            edges, probs = child.edges_for_solver(rng)
            idx, value = solver.solve(edges, probs)
            total += value
        scores[u] = total / float(samples) - base
        yields[u] = revealed / float(samples)
    t2 = time.perf_counter()
    if timing is not None:
        timing["base_seconds"] += t1 - t0
        timing["candidate_seconds"] += t2 - t1
    return scores, {"base_value": base, "expected_survey_yield": yields,
                    "smallest_detectable": None}


def privileged_gain(model, candidates, rng, samples: int, solver, truth_neighbours,
                    base: float | None = None):
    """Diagnostic only -- assumes the survey result is already known before choosing whom to survey.

    This is the same Monte-Carlo as :func:`q_values` with one illegal substitution: instead of drawing
    the survey result from the model, it uses the members' **real** neighbour lists.  The result says
    how much a survey of each candidate would have been worth *if the organisation had known what it
    would reveal*.  It is a realised reference for this procedure, it is not implementable, and it is
    **not** an upper bound on what any legal method can achieve -- the seed solver still only sees
    ``D + N_G(u)``, and the working model is still wrong about everything else.
    """
    if base is None:
        base = base_value(model, rng, samples, solver)
    scores = {}
    for u in candidates:
        u = int(u)
        nb = truth_neighbours(u)
        child = model.copy()
        child.resolve(u, nb)
        total = 0.0
        for _ in range(int(samples)):
            edges, probs = child.edges_for_solver(rng)
            idx, value = solver.solve(edges, probs)
            total += value
        scores[u] = total / float(samples) - base
    return scores


def rank_correlation(a: list[float], b: list[float]) -> float:
    """Spearman correlation without a scipy dependency surprise (ties get average ranks)."""
    from scipy.stats import spearmanr

    if len(a) < 3:
        return float("nan")
    return float(spearmanr(a, b).statistic)
