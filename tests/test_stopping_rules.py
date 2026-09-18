"""Tests for the shared stopping rule (audit item P1-3.1).

The confound: under a budget of *at most* k, one policy filled all k slots --- including seeds whose
exact marginal gain was negative --- while another stopped at the first non-positive gain, and the
two were compared.  Both behaviours are admissible; leaving the choice implicit is not, because the
comparison then measures the stopping rule rather than the selector, and the cost column stops being
like-for-like.

These tests pin: (a) each rule does what its name says, (b) a rule that needs evidence the caller did
not supply raises rather than guessing, and (c) every policy records which rule it used.
"""

from __future__ import annotations

import pytest

from grl.algorithms.sequential_im import (
    FILL_BUDGET,
    PATIENCE_2,
    STOP_ON_NON_POSITIVE,
    StoppingRule,
    adaptive_selective_greedy,
    full_oracle_greedy,
    learned_greedy,
    reference_policy_name,
    selective_greedy,
)


class ScriptedOracle:
    """Exact 'oracle' that returns a scripted gain per selection round."""

    is_exact = True

    def __init__(self, gains: list[float]) -> None:
        self.gains = gains
        self.round = -1

    def score(self, seeds, candidates, step=0):
        del seeds, step
        self.round += 1
        gain = self.gains[min(self.round, len(self.gains) - 1)]
        # give every candidate the same gain so the argmax is deterministic (lowest label wins)
        return {v: gain for v in candidates}


class FixedLearned:
    def __init__(self, scores):
        self.scores = scores

    def score(self, seeds, candidates, step=0):
        del seeds, step
        return {v: self.scores[v] for v in candidates if v in self.scores}


# ---------------------------------------------------------------------------------------------
# the rules themselves
# ---------------------------------------------------------------------------------------------
def test_fill_budget_never_stops_early():
    assert FILL_BUDGET.should_stop([-5.0, -1.0], [1.0, 0.5]) is False


def test_stop_on_non_positive_stops_at_zero_and_below():
    rule = STOP_ON_NON_POSITIVE
    assert rule.should_stop([1.0], []) is False
    assert rule.should_stop([1.0, 0.0], []) is True
    assert rule.should_stop([1.0, -0.25], []) is True


def test_stop_on_non_positive_refuses_to_guess_without_an_exact_gain():
    with pytest.raises(ValueError, match="needs an exact marginal gain"):
        STOP_ON_NON_POSITIVE.should_stop([], [1.0, 2.0])


def test_patience_stops_after_the_configured_number_of_flat_steps():
    assert PATIENCE_2.should_stop([], [5.0, 4.0]) is False
    assert PATIENCE_2.should_stop([], [5.0, 4.0, 4.0]) is True
    assert PATIENCE_2.should_stop([], [5.0, 4.0, 4.5]) is False


def test_patience_requires_a_positive_patience():
    with pytest.raises(ValueError, match="patience must be >= 1"):
        StoppingRule("patience", patience=0)


# ---------------------------------------------------------------------------------------------
# the policies must obey and record the rule
# ---------------------------------------------------------------------------------------------
def test_fill_budget_spends_the_whole_budget_even_on_negative_gains():
    pool = list(range(5))
    result = full_oracle_greedy(pool, 3, ScriptedOracle([-1.0, -2.0, -3.0]))
    assert len(result.selected_seeds) == 3
    assert all(s["stopping_rule"] == "fill_budget" for s in result.steps)
    assert not any(s.get("stopped_early") for s in result.steps)


def test_stop_on_non_positive_returns_fewer_seeds_and_says_so():
    pool = list(range(5))
    result = full_oracle_greedy(
        pool, 4, ScriptedOracle([2.0, -0.5, 9.0, 9.0]), stopping=STOP_ON_NON_POSITIVE
    )
    assert len(result.selected_seeds) == 2, "must stop at the first non-positive gain"
    assert result.steps[-1]["stopped_early"] is True
    assert result.steps[-1]["stopping_rule"] == "stop_on_non_positive"


def test_the_two_rules_disagree_on_the_same_instance():
    """This is the confound, made explicit: same oracle, same budget, different |S| and total gain.

    The two rules differ only in what happens *after* the first non-positive gain, and because a
    marginal gain can be negative, filling the budget can end up worse in total than stopping.
    """
    pool = list(range(5))
    gains = [3.0, -1.0, -5.0, -5.0]  # the budget can be spent on further harm
    filled = full_oracle_greedy(pool, 4, ScriptedOracle(list(gains)))
    stopped = full_oracle_greedy(pool, 4, ScriptedOracle(list(gains)),
                                stopping=STOP_ON_NON_POSITIVE)
    assert len(filled.selected_seeds) == 4
    assert len(stopped.selected_seeds) == 2
    total_filled = sum(s["score"] for s in filled.steps)
    total_stopped = sum(s["score"] for s in stopped.steps)
    assert total_filled < total_stopped, (
        "filling the budget can score worse in total, which is exactly why the rule must be "
        "declared rather than assumed"
    )
    # and the cost differs too, so a cost column that ignores the rule is not like-for-like
    assert filled.steps[0]["verified"] >= stopped.steps[0]["verified"]


def test_learned_greedy_refuses_a_rule_it_cannot_evaluate():
    """A prediction-only policy must not stop on its own predicted sign."""
    with pytest.raises(ValueError, match="never evaluates the objective"):
        learned_greedy(list(range(5)), 3, FixedLearned({v: 1.0 for v in range(5)}),
                       stopping=STOP_ON_NON_POSITIVE)


def test_learned_greedy_accepts_patience_with_a_measured_spread():
    spreads = iter([10.0, 9.0, 9.0, 9.0])
    result = learned_greedy(
        list(range(5)), 4, FixedLearned({v: 1.0 for v in range(5)}),
        stopping=PATIENCE_2, spread_of=lambda seeds: next(spreads),
    )
    assert result.steps[0]["stopping_rule"] == "patience_2"
    assert len(result.selected_seeds) <= 4


def test_selective_greedy_records_the_rule():
    pool = list(range(4))
    result = selective_greedy(
        pool, 2, FixedLearned({v: float(4 - v) for v in pool}), ScriptedOracle([1.0, 1.0]),
    )
    assert all(s["stopping_rule"] == "fill_budget" for s in result.steps)


def test_adaptive_selective_greedy_records_the_rule_and_obeys_it():
    pool = list(range(4))
    learned = FixedLearned({v: float(4 - v) for v in pool})
    exact = ScriptedOracle([1.0, -2.0, 1.0, 1.0])
    result = adaptive_selective_greedy(
        pool, 4, learned, exact,
        initial_m=1, batch_m=1, min_rounds=1, max_m=None, stopping=STOP_ON_NON_POSITIVE,
    )
    assert len(result.selected_seeds) == 2
    assert result.steps[-1]["stopped_early"] is True


# ---------------------------------------------------------------------------------------------
# naming the reference for what it is
# ---------------------------------------------------------------------------------------------
def test_reference_name_carries_the_monte_carlo_budget():
    """P1-3.3: 'full_oracle' at MC = 25 is neither exact nor globally optimal."""

    class NoisyEstimator:
        is_exact = False

    assert reference_policy_name(NoisyEstimator(), 25) == "mc_greedy_mc25"
    assert reference_policy_name(ScriptedOracle([1.0]), 25) == "exact_greedy"
    assert "oracle" not in reference_policy_name(NoisyEstimator(), 25)
