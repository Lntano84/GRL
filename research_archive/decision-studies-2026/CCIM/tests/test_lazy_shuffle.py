"""Verification that the lazy scanner reproduces the explicit one exactly.

Three things are checked, in the order the requirements set them:

1. over a small candidate space the full permutation is drawn -- every id exactly once, none missing,
   and the integer numbering decodes to a bijection onto ``selected x outside``;
2. with the same random stream the explicit and lazy scanners emit **identical prefixes**, at every
   prefix length, including the truncated prefixes the search actually uses;
3. the two scanners consume the random stream identically (same call count), which is what lets the
   end-to-end trajectories be compared at all.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccim.lazy_candidates import (ExplicitShuffle, LazyShuffle, candidate_count,  # noqa: E402
                                  decode)


def test_full_permutation_no_duplicates_no_omissions():
    for M in (1, 2, 3, 7, 50, 997):
        rng = random.Random(1234 + M)
        scanner = LazyShuffle(M, rng)
        drawn = []
        while True:
            cid = scanner.next_id()
            if cid is None:
                break
            drawn.append(cid)
        assert len(drawn) == M, f"M={M}: drew {len(drawn)} ids"
        assert sorted(drawn) == list(range(M)), f"M={M}: not a permutation"
        assert len(set(drawn)) == M, f"M={M}: duplicates"


def test_permutations_are_uniformish_and_differ_across_seeds():
    """Not a statistical proof, just a guard against a scanner that ignores the RNG."""
    firsts = set()
    for seed in range(200):
        scanner = LazyShuffle(1000, random.Random(seed))
        firsts.add(scanner.next_id())
    assert len(firsts) > 100, "the first drawn id barely varies with the seed"


def test_id_decoding_is_a_bijection():
    selected = [5, 9, 2, 7]
    outside = [0, 1, 3, 11, 13]
    M = candidate_count(selected, outside)
    assert M == len(selected) * len(outside)
    pairs = {decode(i, selected, outside) for i in range(M)}
    assert len(pairs) == M, "the id mapping is not injective"
    assert pairs == {(u, v) for u in selected for v in outside}, "the id mapping is not surjective"


@pytest.mark.parametrize("M", [1, 2, 5, 64, 1000])
@pytest.mark.parametrize("prefix", [0, 1, 2, 7, 63, 999])
def test_lazy_and_explicit_agree_on_every_prefix_with_the_same_rng(M, prefix):
    prefix = min(prefix, M)
    lazy_rng = random.Random(99)
    expl_rng = random.Random(99)
    lazy, expl = LazyShuffle(M, lazy_rng), ExplicitShuffle(M, expl_rng)
    for step in range(prefix):
        a, b = lazy.next_id(), expl.next_id()
        assert a == b, f"M={M} step {step}: lazy {a} != explicit {b}"
    assert lazy.pos == expl.pos == prefix


def test_both_scanners_consume_the_same_number_of_draws():
    """Same call count is what makes an end-to-end trajectory comparison meaningful."""
    lazy_rng, expl_rng = random.Random(7), random.Random(7)
    lazy, expl = LazyShuffle(500, lazy_rng), ExplicitShuffle(500, expl_rng)
    for _ in range(137):
        lazy.next_id()
        expl.next_id()
    assert lazy_rng.random() == expl_rng.random(), "the RNGs diverged"


def test_lazy_does_not_materialise_the_range():
    """A large M must not cost a large allocation: the sparse map holds only touched positions."""
    scanner = LazyShuffle(10_000_000, random.Random(3))
    for _ in range(25):
        scanner.next_id()
    assert len(scanner.moved) <= 50, f"sparse map grew to {len(scanner.moved)} entries"
    assert scanner.pos == 25


def test_exhaustion_returns_none_repeatedly():
    scanner = LazyShuffle(3, random.Random(1))
    drawn = [scanner.next_id() for _ in range(3)]
    assert None not in drawn, f"exhausted early: {drawn}"
    assert sorted(drawn) == [0, 1, 2]
    assert scanner.next_id() is None and scanner.next_id() is None


def test_single_candidate_space():
    scanner = LazyShuffle(1, random.Random(0))
    assert scanner.next_id() == 0
    assert scanner.next_id() is None
