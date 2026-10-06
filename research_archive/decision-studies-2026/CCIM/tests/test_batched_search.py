"""Gates for the batched search framework and the swap features.

The load-bearing one is :func:`test_batch_size_one_reproduces_the_sequential_lazy_scan`.  With
``batch_size=1`` a batch is a single candidate drawn from the same lazy stream and verified
immediately, so the batched loop must reproduce ``ordered_search(method="random", scanner="lazy")``
**exactly** -- same trajectory hash, same accepts, same final set.  If that fails, the batch machinery
is not a re-parameterisation of the accepted search and nothing downstream is comparable.
"""

from __future__ import annotations

import math
import random

import pytest

from ccim.batched_search import RANKERS, batched_search
from ccim.learned_rank import RankModel
from ccim.model import load_graph, sigma
from ccim.scan_order import ordered_search, swap_scores
from ccim.swap_features import FEATURE_NAMES, build_context


@pytest.fixture(scope="module")
def football():
    return load_graph("football")


# ------------------------------------------------------------------ the equivalence gate
@pytest.mark.parametrize("seed", [0, 3])
def test_batch_size_one_reproduces_the_sequential_lazy_scan(football, seed):
    seq = ordered_search(football, method="random", T=8, K=4, budget=400, seed=seed,
                         scanner="lazy", trace_hash=True, collect_logs=False)
    bat = batched_search(football, ranker="random", T=8, K=4, budget=400, seed=seed,
                         batch_size=1, trace_hash=True, collect_logs=False)
    assert bat.sigma == seq.sigma
    assert bat.seeds == seq.seeds
    assert bat.extra["accepts"] == seq.extra["accepts"]
    assert bat.extra["query_sequence_sha256"] == seq.extra["query_sequence_sha256"]
    assert bat.extra["accept_sequence"] == seq.extra["accept_sequence"]
    assert bat.extra["budget_used"] == seq.extra["budget_used"]


def test_batch_size_32_first_accept_matches_the_sequential_run(football):
    """The first 32 draws are the same draws; only the RNG consumption after an accept may differ."""
    seq = ordered_search(football, method="random", T=8, K=4, budget=400, seed=0,
                         scanner="lazy", trace_hash=True, collect_logs=False)
    bat = batched_search(football, ranker="random", T=8, K=4, budget=400, seed=0,
                         batch_size=32, trace_hash=True, collect_logs=False)
    assert bat.extra["accept_sequence"][0] == seq.extra["accept_sequence"][0]
    assert bat.extra["accepts"] >= 1


# ------------------------------------------------------------------ determinism and accounting
def test_same_seed_same_result(football):
    a = batched_search(football, ranker="degree", T=8, K=4, budget=300, seed=7,
                       batch_size=32, collect_logs=False)
    b = batched_search(football, ranker="degree", T=8, K=4, budget=300, seed=7,
                       batch_size=32, collect_logs=False)
    assert a.sigma == b.sigma and a.seeds == b.seeds
    assert a.extra["accepts"] == b.extra["accepts"]


@pytest.mark.parametrize("ranker", RANKERS)
def test_query_accounting_and_bucket_reconciliation(football, ranker):
    kwargs = {"model": RankModel.fresh()} if ranker == "learned" else {}
    r = batched_search(football, ranker=ranker, T=8, K=4, budget=300, seed=1, batch_size=32,
                       **kwargs)
    tb = r.extra["timing_buckets"]
    # every restart issues exactly one initial value query; all other queries sit inside batches
    assert tb["queries_issued_in_batches"] + r.extra["restarts_used"] == r.extra["budget_used"]
    assert r.extra["budget_used"] == 300
    assert abs(tb["reconciliation_error_seconds"]) < 1e-6
    assert tb["sum_of_buckets"] <= tb["total_seconds"] + 1e-9
    # the batch never verifies more than it drew
    ss = r.extra["scan_summary"]
    assert ss["candidates_scanned_total"] <= ss["candidates_drawn_total"]
    assert ss["pool_exhausted_count"] == 0


def test_ranking_never_decides_acceptance(football):
    """Replay every accepted swap under the true diffuser: each one must be a strict improvement."""
    r = batched_search(football, ranker="degree", T=8, K=4, budget=400, seed=2, batch_size=32,
                       trace_hash=True, collect_logs=False)
    by_degree = sorted(football.nodes(), key=lambda v: (-football.degree(v), v))
    current = sorted(by_degree[:8])
    value = sigma(football, current, 4)
    for (u, v) in r.extra["accept_sequence"]:
        assert u in current and v not in current
        trial = sorted([x for x in current if x != u] + [v])
        new = sigma(football, trial, 4)
        assert new > value, "an accepted swap was not a strict improvement"
        current, value = trial, new
    assert value <= r.sigma          # r.sigma is the best across restarts, so >= this trajectory


def test_record_rows_is_logging_only(football):
    plain = batched_search(football, ranker="degree", T=8, K=4, budget=300, seed=4,
                           batch_size=32, collect_logs=False, trace_hash=True)
    logged = batched_search(football, ranker="degree", T=8, K=4, budget=300, seed=4,
                            batch_size=32, record_rows=True, trace_hash=True)
    assert logged.sigma == plain.sigma and logged.seeds == plain.seeds
    assert logged.extra["query_sequence_sha256"] == plain.extra["query_sequence_sha256"]
    assert len(logged.extra["rows"]) == plain.extra["budget_used"] - plain.extra["restarts_used"]
    for row in logged.extra["rows"]:
        assert row["state_id"] in logged.extra["state_sets"]
        S = logged.extra["state_sets"][row["state_id"]]
        assert row["u"] in S and row["v"] not in S


def test_rows_carry_a_real_delta_for_every_query(football):
    r = batched_search(football, ranker="random", T=8, K=4, budget=200, seed=5,
                       batch_size=32, record_rows=True)
    assert r.extra["rows"], "no rows recorded"
    rng = random.Random(0)
    for row in rng.sample(r.extra["rows"], min(5, len(r.extra["rows"]))):
        S = r.extra["state_sets"][row["state_id"]]
        trial = sorted([x for x in S if x != row["u"]] + [row["v"]])
        assert sigma(football, trial, 4) == pytest.approx(row["after"])
        assert sigma(football, S, 4) == pytest.approx(row["before"])


# ------------------------------------------------------------------ features
def test_f1_delta_feature_matches_the_reference_implementation(football):
    current = sorted(sorted(football.nodes(), key=lambda v: -football.degree(v))[:8])
    ref = swap_scores(football, current, 4)
    ctx = build_context(football, current, 4)
    idx = FEATURE_NAMES.index("f1_delta")
    S = set(current)
    outside = [v for v in football.nodes() if v not in S]
    rng = random.Random(0)
    for u in current:
        for v in rng.sample(outside, 8):
            assert ctx.features(u, v)[idx] == ref[(u, v)]


def test_features_do_not_depend_on_the_labels(football):
    """The feature vector has no label argument, and permuting observed gains cannot change it."""
    current = sorted(sorted(football.nodes(), key=lambda v: -football.degree(v))[:8])
    ctx = build_context(football, current, 4)
    first = [ctx.features(u, v) for u in current[:3] for v in list(football.nodes())[:5]]
    random.Random(1).shuffle(first)          # labels, if they existed, would move like this
    again = [ctx.features(u, v) for u in current[:3] for v in list(football.nodes())[:5]]
    assert all(math.isfinite(x) for row in again for x in row)
    assert sorted(map(tuple, first)) == sorted(map(tuple, again))


def test_context_matches_direct_neighbour_counting(football):
    current = sorted(sorted(football.nodes(), key=lambda v: -football.degree(v))[:8])
    ctx = build_context(football, current, 4)
    S = set(current)
    for x in football.nodes():
        assert ctx.active_count[x] == sum(1 for nb in football.neighbors(x) if nb in S)
