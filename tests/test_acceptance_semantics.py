"""Tests for the acceptance semantics of adaptive_selective_greedy (audit item P1-2).

The point of these tests is to make the three stages distinguishable and to pin the counterexample
that shows the residual envelope is not a guarantee.  A future change that quietly widens the
envelope, or that reports a heuristic decision as a certificate, must fail here.
"""

from __future__ import annotations

from grl.algorithms.sequential_im import adaptive_selective_greedy


class FixedLearned:
    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores

    def score(self, seeds, candidates, step=0):
        del seeds, step
        return {v: self.scores[v] for v in candidates if v in self.scores}


class ScriptedExact:
    """Returns true scores for a limited number of candidate queries, then zeros."""

    def __init__(self, truth: dict[int, float], limit: int) -> None:
        self.truth = truth
        self.limit = limit
        self.calls = 0

    def score(self, seeds, candidates, step=0):
        del seeds, step
        self.calls += 1
        if self.calls <= self.limit:
            return {v: self.truth[v] for v in candidates if v in self.truth}
        return {v: 0.0 for v in candidates}


def test_residual_envelope_accepts_the_wrong_candidate_when_it_certifies():
    """Counterexample with NO Monte-Carlo noise and the default min_rounds = 2.

    learned: {0: 10, 1: 9, 2: 4, 3: -1, 4: -2}
    true:    {0: 10, 1: 9, 2: 4, 3:  1, 4: 100}

    The pool is never exhausted (max_m = 3 < 5), so this is not the exact branch.  After verifying
    nodes 0, 1 and 2 the observed residuals are all zero, the envelope for the best unverified
    candidate collapses to its predicted score (-1), the winner is stable across two rounds, and the
    rule accepts node 0 while node 4 -- never inspected -- has true gain 100.
    """
    learned = {0: 10.0, 1: 9.0, 2: 4.0, 3: -1.0, 4: -2.0}
    truth = {0: 10.0, 1: 9.0, 2: 4.0, 3: 1.0, 4: 100.0}

    result = adaptive_selective_greedy(
        [0, 1, 2, 3, 4], 1, FixedLearned(learned), ScriptedExact(truth, limit=2),
        initial_m=2, batch_m=1, max_m=3,
    )
    step = result.steps[0]
    assert step["chosen"] == 0, "the envelope rule picks the predicted-best verified node"
    assert step["oracle_score"] == 10.0
    assert step["empirical_accept"] is True
    assert step["stop_reason"] == "residual_envelope"
    # the decision was accepted empirically and is wrong: the true best is node 4
    assert truth[4] > step["oracle_score"]
    assert max(truth[v] for v in step["shortlist"]) == step["oracle_score"]
    assert step["fallback_full_scan"] is False, "the full scan was never entered"


def test_empirical_accept_and_statistical_certificate_are_separate_fields():
    learned = {0: 10.0, 1: 9.0, 2: 4.0, 3: -1.0, 4: -2.0}
    truth = {0: 10.0, 1: 9.0, 2: 4.0, 3: 1.0, 4: 100.0}
    result = adaptive_selective_greedy(
        [0, 1, 2, 3, 4], 1, FixedLearned(learned), ScriptedExact(truth, limit=2),
        initial_m=2, batch_m=1, max_m=3,
    )
    step = result.steps[0]
    assert step["empirical_accept"] is True
    assert step["statistical_certificate"] is False, (
        "nothing in this repository establishes a coverage or confidence statement"
    )


def test_full_scan_branch_is_entered_when_the_pool_is_exhausted():
    """With max_m = None every candidate is verified, which is the only exact branch."""
    learned = {0: 1.0, 1: 2.0, 2: 3.0}
    truth = {0: 1.0, 1: 2.0, 2: 3.0}

    class ExactAll:
        def score(self, seeds, candidates, step=0):
            del seeds, step
            return {v: truth[v] for v in candidates}

    # min_rounds deliberately large so the stability test never fires before the cap is hit
    result = adaptive_selective_greedy(
        [0, 1, 2], 1, FixedLearned(learned), ExactAll(),
        initial_m=1, batch_m=1, min_rounds=99, max_m=None,
    )
    step = result.steps[0]
    assert step["fallback_full_scan"] is True
    assert step["verified"] == 3
    assert step["stop_reason"] == "all_candidates"
    assert step["chosen"] == 2


def test_max_m_exhaustion_without_full_scan_is_not_a_fallback():
    """Stopping at max_m with candidates left unevaluated must not be reported as a full scan."""
    learned = {v: float(10 - v) for v in range(10)}
    truth = {v: float(v) for v in range(10)}

    class ExactZeros:
        def score(self, seeds, candidates, step=0):
            del seeds, step
            return {v: 0.0 for v in candidates}

    result = adaptive_selective_greedy(
        list(range(10)), 1, FixedLearned(learned), ExactZeros(),
        initial_m=2, batch_m=2, min_rounds=99, max_m=4,
    )
    step = result.steps[0]
    assert step["stop_reason"] == "max_m"
    assert step["verified"] == 4
    assert step["fallback_full_scan"] is False
    assert step["empirical_accept"] is False


def test_no_field_called_certified_remains():
    """The old name implied a guarantee the method does not provide."""
    learned = {0: 1.0, 1: 2.0}
    result = adaptive_selective_greedy(
        [0, 1], 1, FixedLearned(learned),
        ScriptedExact({0: 1.0, 1: 2.0}, limit=10), initial_m=2, batch_m=1, max_m=2,
    )
    step = result.steps[0]
    assert "certified" not in step
    for rnd in step["rounds"]:
        assert "certified" not in rnd
        assert "empirical_accept" in rnd
