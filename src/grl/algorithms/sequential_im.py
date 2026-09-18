from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass
class SequentialSelectionResult:
    selected_seeds: list[int]
    steps: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StoppingRule:
    """When a sequential policy may stop before spending the whole budget.

    Why this is an argument and not an implementation detail
    -------------------------------------------------------
    The frozen contract fixes the budget at *at most* ``k``, so returning fewer than ``k`` seeds
    is legal.  But "may stop early" is not itself a rule.  If one policy fills all ``k`` slots ---
    including seeds whose exact marginal gain is negative --- while another stops at the first
    non-positive gain, then a comparison between them measures the stopping rule, not the
    selector, and the cost column is not like-for-like.  Both behaviours are admissible; what is
    not admissible is leaving the choice implicit.  The rule is therefore passed explicitly and
    recorded in every returned step as ``stopping_rule``.

    ``fill_budget``
        Never stop early.  Selects exactly ``min(budget, |pool|)`` seeds even when a marginal gain
        is negative.  Admissible under ``|S| <= k``, and the honest default for a policy whose
        only claim is about ranking.
    ``stop_on_non_positive``
        Stop once the last observed **exact** marginal gain is ``<= 0``.
    ``patience``
        Stop once the observed spread has failed to improve for ``patience`` consecutive steps.
        Uses the realised cascade, so it needs one cascade per step and is available to any policy
        that can measure the spread of its own selection.
    """

    name: str
    stop_on_non_positive: bool = False
    patience: int | None = None

    def __post_init__(self) -> None:
        if self.patience is not None and self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")

    @property
    def needs_exact_marginal(self) -> bool:
        return self.stop_on_non_positive

    @property
    def needs_observed_spread(self) -> bool:
        return self.patience is not None or self.stop_on_non_positive

    def should_stop(self, exact_gains: list[float], observed_spread: list[float]) -> bool:
        """Decide whether to stop after the most recent selection.

        A rule that needs a quantity the caller did not supply raises, rather than silently
        deciding on the wrong evidence.
        """
        if self.patience is not None:
            if len(observed_spread) < self.patience + 1:
                return False
            recent = observed_spread[-(self.patience + 1):]
            if all(later <= earlier for earlier, later in zip(recent, recent[1:])):
                return True
        if self.stop_on_non_positive:
            if not exact_gains:
                raise ValueError(
                    f"stopping rule {self.name!r} needs an exact marginal gain, but none was "
                    f"supplied. A prediction-only policy must declare fill_budget or patience, "
                    f"because its own score is not evidence about the true gain."
                )
            if exact_gains[-1] <= 0.0:
                return True
        return False


#: Spend the whole budget.  Explicitly named so that a comparison can say which rule it used.
FILL_BUDGET = StoppingRule("fill_budget")

#: Stop at the first non-positive observed marginal gain.
STOP_ON_NON_POSITIVE = StoppingRule("stop_on_non_positive", stop_on_non_positive=True)

#: Stop when the realised spread has not improved for two consecutive steps.
PATIENCE_2 = StoppingRule("patience", patience=2)


def _rule_name(stopping: StoppingRule) -> str:
    if stopping.patience is not None:
        return f"patience_{stopping.patience}"
    return stopping.name


def _available(candidate_pool: list[int], selected: list[int]) -> list[int]:
    chosen = set(selected)
    return [v for v in candidate_pool if v not in chosen]


def reference_policy_name(oracle, mc_runs: int) -> str:
    """A name for a greedy arm that states what its reference actually is.

    Confound P1-3.3: an early script labelled its Monte-Carlo greedy arm ``full_oracle``.  At
    ``MC = 25`` it is neither exact nor globally optimal, and calling it an oracle invites the
    reader to treat its result as an upper bound.  The name must carry the budget instead.
    """
    if getattr(oracle, "is_exact", False):
        return "exact_greedy"
    return f"mc_greedy_mc{mc_runs}"


def full_oracle_greedy(
    candidate_pool: list[int],
    budget: int,
    oracle,
    *,
    stopping: StoppingRule = FILL_BUDGET,
    spread_of=None,
) -> SequentialSelectionResult:
    """Greedy on exact marginal gains.

    Named for what it is: greedy on the *exact* objective.  It is not globally optimal, and it is
    not an oracle unless ``oracle`` is one --- a Monte-Carlo estimator at 25 runs is neither exact
    nor optimal, so a caller using one must say so (see :func:`reference_policy_name`).

    ``stopping`` decides whether it may return fewer than ``budget`` seeds.  With
    :data:`STOP_ON_NON_POSITIVE` it stops at the first non-positive exact gain, which is the
    behaviour several earlier scripts assumed without recording it.
    """
    selected: list[int] = []
    steps: list[dict] = []
    exact_gains: list[float] = []
    observed_spread: list[float] = []
    for step in range(int(budget)):
        available = _available(candidate_pool, selected)
        if not available:
            break
        scores = oracle.score(selected, available, step=step)
        chosen = max(available, key=lambda v: (scores[v], -v))
        gain = float(scores[chosen])
        exact_gains.append(gain)
        if spread_of is not None:
            observed_spread.append(float(spread_of(selected + [chosen])))
        steps.append({
            "step": step + 1,
            "chosen": chosen,
            "score": gain,
            "verified": len(available),
            "stopping_rule": _rule_name(stopping),
        })
        selected.append(chosen)
        if stopping.should_stop(exact_gains, observed_spread):
            steps[-1]["stopped_early"] = True
            break
    return SequentialSelectionResult(selected, steps)


def learned_greedy(
    candidate_pool: list[int],
    budget: int,
    learned_oracle,
    *,
    stopping: StoppingRule = FILL_BUDGET,
    spread_of=None,
) -> SequentialSelectionResult:
    """Greedy on predicted marginal gains.

    A prediction-only policy cannot use :data:`STOP_ON_NON_POSITIVE`: its own score is not evidence
    about the true gain, and letting it stop on a *predicted* non-positive value would make it
    incomparable with a policy that stops on an observed one.  :data:`PATIENCE_2` is available
    because it needs only the realised spread, which every policy can measure.
    """
    if stopping.needs_exact_marginal:
        raise ValueError(
            f"stopping rule {_rule_name(stopping)!r} needs an exact marginal gain, but "
            f"learned_greedy never evaluates the objective. Use PATIENCE_2, or pass an exact "
            f"oracle through selective_greedy / adaptive_selective_greedy."
        )
    selected: list[int] = []
    steps: list[dict] = []
    observed_spread: list[float] = []
    for step in range(int(budget)):
        available = _available(candidate_pool, selected)
        if not available:
            break
        scores = learned_oracle.score(selected, available, step=step)
        chosen = max(available, key=lambda v: (scores[v], -v))
        if spread_of is not None:
            observed_spread.append(float(spread_of(selected + [chosen])))
        steps.append({
            "step": step + 1,
            "chosen": chosen,
            "predicted_score": float(scores[chosen]),
            "verified": 0,
            "stopping_rule": _rule_name(stopping),
        })
        selected.append(chosen)
        if stopping.should_stop([], observed_spread):
            steps[-1]["stopped_early"] = True
            break
    return SequentialSelectionResult(selected, steps)


def selective_greedy(
    candidate_pool: list[int],
    budget: int,
    learned_oracle,
    exact_oracle,
    top_m: int = 8,
    *,
    stopping: StoppingRule = FILL_BUDGET,
    spread_of=None,
) -> SequentialSelectionResult:
    """Prediction-guided greedy with fixed Top-M exact refinement.

    It observes an exact marginal for every candidate on its shortlist, so any stopping rule is
    available to it.  ``stop_on_non_positive`` is applied to the gain of the chosen seed, i.e. to
    the same quantity :func:`full_oracle_greedy` uses, so the two arms are comparable under it.
    """
    selected: list[int] = []
    steps: list[dict] = []
    exact_gains: list[float] = []
    observed_spread: list[float] = []
    for step in range(int(budget)):
        available = _available(candidate_pool, selected)
        if not available:
            break
        learned = learned_oracle.score(selected, available, step=step)
        shortlist = sorted(available, key=lambda v: (learned[v], -v), reverse=True)[: min(int(top_m), len(available))]
        exact = exact_oracle.score(selected, shortlist, step=step)
        chosen = max(shortlist, key=lambda v: (exact[v], -v))
        exact_gains.append(float(exact[chosen]))
        if spread_of is not None:
            observed_spread.append(float(spread_of(selected + [chosen])))
        steps.append({
            "step": step + 1,
            "chosen": chosen,
            "predicted_score": float(learned[chosen]),
            "oracle_score": float(exact[chosen]),
            "verified": len(shortlist),
            "shortlist": shortlist,
            "stopping_rule": _rule_name(stopping),
        })
        selected.append(chosen)
        if stopping.should_stop(exact_gains, observed_spread):
            steps[-1]["stopped_early"] = True
            break
    return SequentialSelectionResult(selected, steps)


def adaptive_selective_greedy(
    candidate_pool: list[int],
    budget: int,
    learned_oracle,
    exact_oracle,
    initial_m: int = 8,
    batch_m: int = 8,
    residual_beta: float = 1.0,
    min_rounds: int = 2,
    max_m: int | None = None,
    *,
    stopping: StoppingRule = FILL_BUDGET,
    spread_of=None,
) -> SequentialSelectionResult:
    """Adaptive prediction-guided refinement with an empirical acceptance test.

    Candidates are ranked once by the learned oracle at each greedy step.  Exact evaluation starts
    from a small prefix and expands in batches.  A provisional winner is *accepted* when it is
    stable across ``min_rounds`` expansion rounds and its exact score exceeds an empirical upper
    envelope for the best unverified candidate: the learned outsider score plus the largest observed
    exact-minus-learned residual and ``residual_beta`` residual standard deviations.

    What this is NOT
    ----------------
    The envelope is **not** a probabilistic guarantee.  The residual is a maximum over the
    candidates inspected so far, which says nothing about the ones that were not inspected; a
    candidate with a large but unobserved gain is invisible to it.  There is a concrete
    counterexample with no Monte-Carlo noise at all, using the default ``min_rounds = 2``:

    ===========  ====  ====  ====  =====  =====
    candidate      0     1     2      3      4
    learned       10     9     4     -1     -2
    true          10     9     4      1    100
    ===========  ====  ====  ====  =====  =====

    with ``initial_m = 2``, ``batch_m = 1``, ``max_m = 3``.  Nodes 0, 1 and 2 are verified and all
    three residuals are zero, so the envelope for the best unverified candidate (node 3) collapses to
    its predicted score of -1.  The winner is stable and is accepted at node 0, while node 4 -- never
    inspected -- has true gain 100.  The pool was not exhausted, so this is not the exact branch.

    The reported fields therefore separate three stages instead of merging them into one
    "certified" flag:

    ``empirical_accept``
        the envelope test passed.  A heuristic; it can be wrong as shown above.
    ``statistical_certificate``
        reserved for a decision carrying an explicit coverage or confidence statement.  Nothing in
        this repository sets it today; a caller supplying a genuinely valid bound may.
    ``fallback_full_scan``
        the cap was reached with every candidate in the pool evaluated exactly.  This is the only
        branch whose decision is exact with respect to the pool, and it must be *entered*, not
        merely reachable.

    This is an operational first baseline, not a formal probabilistic guarantee.
    """
    initial_m = max(1, int(initial_m))
    batch_m = max(1, int(batch_m))
    min_rounds = max(1, int(min_rounds))
    selected: list[int] = []
    steps: list[dict] = []
    exact_gains: list[float] = []
    observed_spread: list[float] = []

    for step in range(int(budget)):
        available = _available(candidate_pool, selected)
        if not available:
            break
        learned = learned_oracle.score(selected, available, step=step)
        ranked = sorted(available, key=lambda v: (learned[v], -v), reverse=True)
        cap = len(ranked) if max_m is None else min(len(ranked), max(1, int(max_m)))
        verified: dict[int, float] = {}
        winner_history: list[int] = []
        rounds: list[dict] = []
        target = min(initial_m, cap)
        empirical_accept = False
        statistical_certificate = False
        fallback_full_scan = False
        stop_reason = "max_m"

        while True:
            new_nodes = ranked[len(verified):target]
            if new_nodes:
                exact_batch = exact_oracle.score(selected, new_nodes, step=step)
                verified.update(exact_batch)
            winner = max(verified, key=lambda v: (verified[v], -v))
            winner_history.append(winner)

            residuals = [verified[v] - learned[v] for v in verified]
            residual_max = max(residuals) if residuals else 0.0
            residual_mean = sum(residuals) / len(residuals) if residuals else 0.0
            residual_var = (
                sum((x - residual_mean) ** 2 for x in residuals) / len(residuals)
                if residuals else 0.0
            )
            residual_std = math.sqrt(max(0.0, residual_var))

            outsider = ranked[target] if target < len(ranked) else None
            outsider_upper = None
            if outsider is not None:
                outsider_upper = float(learned[outsider] + residual_max + float(residual_beta) * residual_std)

            stable = (
                len(winner_history) >= min_rounds
                and len(set(winner_history[-min_rounds:])) == 1
            )
            if outsider is None:
                # every candidate in the pool was evaluated exactly
                empirical_accept = True
                fallback_full_scan = True
                stop_reason = "all_candidates"
            elif stable and float(verified[winner]) >= float(outsider_upper):
                empirical_accept = True
                stop_reason = "residual_envelope"

            rounds.append({
                "verified": len(verified),
                "winner": winner,
                "winner_exact": float(verified[winner]),
                "winner_predicted": float(learned[winner]),
                "residual_max": float(residual_max),
                "residual_std": float(residual_std),
                "best_unverified": outsider,
                "best_unverified_predicted": None if outsider is None else float(learned[outsider]),
                "best_unverified_upper": outsider_upper,
                "stable": bool(stable),
                "empirical_accept": bool(empirical_accept),
            })

            if empirical_accept:
                break
            if target >= cap:
                stop_reason = "max_m" if cap < len(ranked) else "all_candidates"
                fallback_full_scan = cap >= len(ranked)
                break
            target = min(cap, target + batch_m)

        chosen = max(verified, key=lambda v: (verified[v], -v))
        exact_gains.append(float(verified[chosen]))
        if spread_of is not None:
            observed_spread.append(float(spread_of(selected + [chosen])))
        steps.append({
            "step": step + 1,
            "chosen": chosen,
            "predicted_score": float(learned[chosen]),
            "oracle_score": float(verified[chosen]),
            "verified": len(verified),
            "empirical_accept": bool(empirical_accept),
            "statistical_certificate": bool(statistical_certificate),
            "fallback_full_scan": bool(fallback_full_scan),
            "stop_reason": stop_reason,
            "stopping_rule": _rule_name(stopping),
            "rounds": rounds,
            "shortlist": ranked[:len(verified)],
        })
        selected.append(chosen)
        if stopping.should_stop(exact_gains, observed_spread):
            steps[-1]["stopped_early"] = True
            break

    return SequentialSelectionResult(selected, steps)
