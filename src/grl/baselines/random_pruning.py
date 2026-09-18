"""Random pruning: the matched control for any shortlist-based policy.

Why this exists
---------------
Gate 2 of this project's plan asks for a quality--cost Pareto curve including a ``random-pruning``
control, "to show the learned component rather than the parallelism/screening is responsible for any
gain".  The reasoning is easy to get wrong, so it is worth stating exactly.

``selective_greedy`` and ``adaptive_selective_greedy`` cut cost by evaluating only a shortlist of
``M`` candidates exactly instead of the whole pool.  That saves cascades, and the saving is real --
but on its own it says nothing about whether the *ranking* was any good.  A cheaper policy can beat a
more expensive one simply because the expensive one wasted its budget on candidates that did not
matter.

The control therefore has to hold the screening budget **fixed** and remove only the ranking:
draw the shortlist uniformly at random, evaluate it exactly with the same oracle, take its best
member.  Then

* if the learned-shortlist arm beats random-pruning at equal ``M``, the ranking carries information;
* if it does not, the arm's advantage over full greedy is screening, not learning.

The control is matched on the thing that costs cascades (``M``), not on the thing being tested (the
score), which is what makes it a control rather than a second treatment.

Cost
----
Exactly ``M`` candidate evaluations per step, plus the base spread, i.e. the same as
``selective_greedy`` at the same ``M``.  It never reads the exposure state, so it is charged no
state acquisition -- unlike the state-conditioned scorers, which must pay for ``delta`` first.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class RandomPruningResult:
    """Selected seeds and the per-step record, in the same shape as the other policies."""

    selected_seeds: list[int]
    steps: list[dict]

    def to_dict(self) -> dict:
        return {"selected_seeds": self.selected_seeds, "steps": self.steps}


def random_pruning_greedy(
    candidate_pool: list[int],
    budget: int,
    exact_oracle,
    shortlist_size: int = 8,
    *,
    random_seed: int = 20260917,
    stopping=None,
    spread_of=None,
):
    """Greedy over a uniformly random shortlist of ``shortlist_size`` at each step.

    Parameters are deliberately identical to :func:`grl.algorithms.sequential_im.selective_greedy`
    except that no score is consulted, so the two are directly comparable at equal cost.

    ``stopping`` and ``spread_of`` are accepted and forwarded so that this arm obeys the same
    declared stopping rule as every other arm; comparing arms under different stopping rules is
    audit item P1-3.1.
    """
    from grl.algorithms.sequential_im import (
        FILL_BUDGET,
        SequentialSelectionResult,
        _available,
        _rule_name,
    )

    stopping = FILL_BUDGET if stopping is None else stopping
    if shortlist_size < 1:
        raise ValueError(f"shortlist_size must be >= 1, got {shortlist_size}")

    rng = random.Random(random_seed)
    selected: list[int] = []
    steps: list[dict] = []
    exact_gains: list[float] = []
    observed_spread: list[float] = []

    for step in range(int(budget)):
        available = _available(candidate_pool, selected)
        if not available:
            break
        size = min(int(shortlist_size), len(available))
        shortlist = rng.sample(available, size)
        exact = exact_oracle.score(selected, shortlist, step=step)
        chosen = max(shortlist, key=lambda v: (exact[v], -v))
        exact_gains.append(float(exact[chosen]))
        if spread_of is not None:
            observed_spread.append(float(spread_of(selected + [chosen])))
        steps.append({
            "step": step + 1,
            "chosen": chosen,
            # no predicted score: this arm has no predictor, and inventing one would hide that
            "predicted_score": None,
            "oracle_score": float(exact[chosen]),
            "verified": len(shortlist),
            "shortlist": sorted(shortlist),
            "stopping_rule": _rule_name(stopping),
            "control": "random_pruning",
        })
        selected.append(chosen)
        if stopping.should_stop(exact_gains, observed_spread):
            steps[-1]["stopped_early"] = True
            break

    return SequentialSelectionResult(selected, steps)
