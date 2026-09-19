"""Tests for random-stream independence.

Bug 3 of the validity fixes: trial seeds were ``base + offset`` for ``offset in range(trials)``.
With replicate bases 20260917 and 20260918 and 200 trials, the two evaluation streams overlapped in
199 of 200 windows --- a 99.5% overlap between what were reported as independent repeats.

Sharing windows *between arms* is what makes a comparison paired and is desirable.  Sharing them
between *replicates* means the second replicate is not a replicate, and the reported spread across
repeats understates the real variability.
"""

from __future__ import annotations

import pytest

from grl.oracle import trial_seeds
from grl.oracle.targeted_mc import trial_seeds as direct_trial_seeds

#: The namespaces the runner uses, restated here so a change to one is caught.
SELECT_NS = 0
EVAL_NS = 500_000
STATE_NS = 900_000

REPLICATE_BASES = (20260917, 20260918)
TRIALS = 200


def test_replicates_have_disjoint_evaluation_streams():
    sets = [set(trial_seeds(base + EVAL_NS, TRIALS)) for base in REPLICATE_BASES]
    a, b = sets
    assert not (a & b), (
        f"the two replicates share {len(a & b)} of {TRIALS} windows; they are not independent"
    )


def test_the_old_scheme_would_have_been_caught_here():
    """Guard the guard: show that ``base + offset`` is exactly the failure mode."""
    old = [set(range(base, base + TRIALS)) for base in REPLICATE_BASES]
    assert len(old[0] & old[1]) == TRIALS - 1, (
        "the historical scheme overlapped by all but one window; if this changes, the test above "
        "may be checking the wrong thing"
    )


def test_purposes_do_not_share_windows():
    """A method must not select on the windows it is scored on."""
    base = REPLICATE_BASES[0]
    selection = set(trial_seeds(base + SELECT_NS, TRIALS))
    evaluation = set(trial_seeds(base + EVAL_NS, TRIALS))
    state = set(trial_seeds(base + STATE_NS, TRIALS))
    assert not (selection & evaluation)
    assert not (selection & state)
    assert not (evaluation & state)


def test_streams_are_reproducible():
    assert trial_seeds(12345, 5) == trial_seeds(12345, 5)
    assert trial_seeds(12345, 5) != trial_seeds(12346, 5)


def test_stream_length_matches_the_request():
    assert len(trial_seeds(999, 0)) == 0
    assert len(trial_seeds(999, 17)) == 17
    assert len(set(trial_seeds(999, 5000))) == 5000, "seeds within one stream must be distinct"


def test_negative_length_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        trial_seeds(1, -1)


def test_the_module_export_matches_the_implementation():
    assert trial_seeds is direct_trial_seeds
