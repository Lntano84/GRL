"""A budget-controlled non-learning combinatorial search, with an optional per-move trace.

What it is
----------
Iterated local search over seed SETS of fixed size ``T``: start from the greedy solution and from
random restarts, then repeatedly apply the best improving 1-swap (replace one seed) or 2-swap
(replace two seeds), and keep the best set seen.  The number of ``sigma`` evaluations is capped
explicitly, and the cap is the only knob -- so the search can be compared to a learned policy at
matched cost.

Why this and not something cleverer
-----------------------------------
The point of the comparison is not to build the best possible solver.  It is to answer one question:
**does a decision procedure that looks ahead over multiple seeds buy anything that a cheap,
non-learning procedure cannot?**  Local search is the standard, well-understood, non-learning answer,
and it consumes the same oracle budget as a training run, so a learned policy that cannot beat it at
equal cost has not demonstrated value.  Nothing here is claimed as novel.

The trace
---------
``trace`` is an optional list.  When given, one row is appended per ``sigma`` evaluation, recording
the phase it belongs to, the set before the move, the swapped-out and swapped-in nodes, the value
before and after, whether the move was accepted, and the cumulative query count.  That is the raw
material for two questions the final result cannot answer: how many queries each phase consumed, and
what the proposal step actually looks like as a learning target.

Recording is **side-effect free with respect to the search itself**: no extra randomness is drawn and
no evaluation is reordered, so a traced run returns exactly the same seed set as an untraced one.
``scripts/audit_search_trace.py`` asserts that against the previously recorded values.
"""

from __future__ import annotations

import random

import networkx as nx

from .baselines import Ledger, Result
from .model import DEFAULT_K, DEFAULT_T


def local_search(graph: nx.Graph, T: int = DEFAULT_T, K: int = DEFAULT_K,
                 budget: int = 20_000, seed: int = 0, restarts: int = 6,
                 two_swap_passes: int = 200, trace: list | None = None,
                 checkpoints: tuple = ()) -> Result:
    """Iterated 1-/2-swap local search with an explicit ``sigma``-evaluation budget."""
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    ledger = Ledger()
    best_so_far = -1
    history: list[tuple[int, int]] = []          # (cumulative queries, best value so far)
    state = {"phase": "init", "restart": 0, "S": [], "u": None, "v": None, "before": None}

    def value(seeds) -> int:
        nonlocal best_so_far
        duplicate = ledger.is_cached(seeds, K)
        v = ledger.sigma(graph, seeds, K)
        if v > best_so_far:
            best_so_far = v
        history.append((ledger.decision_evaluations, best_so_far))
        if trace is not None:
            trace.append({
                "q": ledger.decision_evaluations,
                "phase": state["phase"],
                "restart": state["restart"],
                "S": list(state["S"]) if state["S"] else sorted(int(s) for s in seeds),
                "u": state["u"],
                "v": state["v"],
                "before": state["before"],
                "after": v,
                "delta": None if state["before"] is None else v - state["before"],
                "accepted": False,          # filled in by the caller when a move is taken
                "duplicate_query": duplicate,
                "best_so_far": best_so_far,
            })
        return v

    by_degree = sorted(nodes, key=lambda v: (-graph.degree(v), v))
    starts = [tuple(by_degree[:T])]
    for _ in range(restarts):
        starts.append(tuple(sorted(rng.sample(nodes, T))))

    best_seeds, best_value = None, -1
    for restart_index, start in enumerate(starts):
        if ledger.decision_evaluations >= budget:
            break
        current = list(start)
        state.update({"phase": "init", "restart": restart_index, "S": list(current),
                      "u": None, "v": None, "before": None})
        current_value = value(current)
        improved = True
        while improved and ledger.decision_evaluations < budget:
            improved = False
            order = list(range(len(current)))
            rng.shuffle(order)
            for i in order:
                for v in rng.sample(nodes, len(nodes)):
                    if v in current:
                        continue
                    trial = list(current)
                    trial[i] = v
                    state.update({"phase": "1swap", "S": list(current), "u": current[i],
                                  "v": v, "before": current_value})
                    val = value(trial)
                    if val > current_value:
                        if trace is not None:
                            trace[-1]["accepted"] = True
                        current, current_value, improved = trial, val, True
                        break
                    if ledger.decision_evaluations >= budget:
                        break
                if ledger.decision_evaluations >= budget:
                    break
            if improved:
                continue
            for _ in range(two_swap_passes):
                if ledger.decision_evaluations >= budget:
                    break
                i, j = rng.sample(range(T), 2)
                a, b = rng.sample(nodes, 2)
                if a in current or b in current:
                    continue
                trial = list(current)
                trial[i], trial[j] = a, b
                state.update({"phase": "2swap", "S": list(current), "u": (current[i], current[j]),
                              "v": (a, b), "before": current_value})
                val = value(trial)
                if val > current_value:
                    if trace is not None:
                        trace[-1]["accepted"] = True
                    current, current_value, improved = trial, val, True
                    break
        if current_value > best_value:
            best_seeds, best_value = list(current), current_value

    final = ledger.sigma(graph, best_seeds, K, decision=False)

    curve = []
    for c in checkpoints:
        eligible = [b for q, b in history if q <= c]
        curve.append({"checkpoint": c,
                      "best_within_checkpoint": max(eligible) if eligible else None,
                      "reached": ledger.decision_evaluations >= c,
                      "queries_available": max((q for q, _ in history), default=0)})

    return Result("local_search", sorted(best_seeds), final, graph.number_of_nodes(),
                  ledger.as_dict(),
                  {"budget": budget, "budget_used": ledger.decision_evaluations,
                   "budget_exhausted": ledger.decision_evaluations >= budget, "seed": seed,
                   "checkpoint_curve": curve})
