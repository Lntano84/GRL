"""Gates for the dataset builder and the split rules.

The load-bearing one is :func:`test_states_from_two_seeds_do_not_collide`.  Every replayed trajectory
numbers its states from 1, so a caller that merges two trajectories into one ``state_sets`` dict keyed by
raw state id will have the second seed silently overwrite the first -- and every row of the first seed
then gets its features computed against a seed set from the *other* run.  That happened, and it is why
``build_dataset`` now treats ``state_id`` as an opaque key and refuses unknown keys outright.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccim.learned_rank import (build_dataset, check_split_disjoint, drop_shared_states,
                               pairwise_accuracy, surviving_keys)
from ccim.model import load_graph
from ccim.swap_features import FEATURE_NAMES, build_context


@pytest.fixture(scope="module")
def football():
    return load_graph("football")


def _two_seed_records(football):
    """Two trajectories whose state ids overlap completely but whose seed sets differ."""
    nodes = sorted(football.nodes(), key=lambda v: -football.degree(v))
    S_a = sorted(nodes[:8])
    S_b = sorted(nodes[8:16])
    records = {"graph": football, "K": 4,
               "state_sets": {"0:1": S_a, "1:1": S_b},
               "rows": [
                   {"state_id": "0:1", "u": S_a[0], "v": nodes[20], "before": 10.0, "after": 11.0},
                   {"state_id": "1:1", "u": S_b[0], "v": nodes[21], "before": 20.0, "after": 19.0},
               ]}
    return records, S_a, S_b, nodes


def test_states_from_two_seeds_do_not_collide(football):
    records, S_a, S_b, nodes = _two_seed_records(football)
    ds = build_dataset(records)
    assert len(ds["S_of_group"]) == 2, "the two trajectories collapsed into one state"
    assert set(map(frozenset, ds["S_of_group"].values())) == {frozenset(S_a), frozenset(S_b)}

    # each row's features must come from its own seed set, not from the other trajectory's
    by_key = {k: g for g, ks in ds["keys_of_group"].items() for k in ks}
    ctx_a = build_context(football, S_a, 4)
    ctx_b = build_context(football, S_b, 4)
    row_a = next(r for r in records["rows"] if r["state_id"] == "0:1")
    row_b = next(r for r in records["rows"] if r["state_id"] == "1:1")
    idx_a = np.nonzero(ds["group"] == by_key["0:1"])[0][0]
    idx_b = np.nonzero(ds["group"] == by_key["1:1"])[0][0]
    assert list(ds["X"][idx_a]) == ctx_a.features(row_a["u"], row_a["v"])
    assert list(ds["X"][idx_b]) == ctx_b.features(row_b["u"], row_b["v"])
    assert not np.array_equal(ds["X"][idx_a], ds["X"][idx_b])


def test_states_with_the_same_S_are_merged_into_one_group(football):
    """The degree-52 initial set begins every trajectory: one state, not one per trajectory."""
    nodes = sorted(football.nodes(), key=lambda v: -football.degree(v))
    S = sorted(nodes[:8])
    other = sorted(nodes[8:16])
    mk = lambda key, u, v, after: {"state_id": key, "u": u, "v": v, "before": 1.0, "after": after}
    ds = build_dataset({"graph": football, "K": 4,
                        "state_sets": {"0:1": S, "1:1": S, "0:2": other},
                        "rows": [mk("0:1", S[0], nodes[20], 2.0),
                                 mk("1:1", S[1], nodes[21], 3.0),
                                 mk("1:1", S[0], nodes[20], 2.0),   # same (S,u,v) as 0:1 -> collapsed
                                 mk("0:2", other[0], nodes[22], 5.0)]})
    assert len(ds["S_of_group"]) == 2
    assert len(ds["y"]) == 3, "the repeated (S,u,v) observation was not collapsed"
    merged = [ks for ks in ds["keys_of_group"].values() if len(ks) > 1]
    assert merged == [["0:1", "1:1"]]
    assert surviving_keys(ds) == {"0:1", "1:1", "0:2"}


def test_raw_integer_state_ids_that_collide_raise(football):
    """Un-namespaced ids are exactly the old bug, so the builder must not silently accept them."""
    nodes = sorted(football.nodes(), key=lambda v: -football.degree(v))
    records = {"graph": football, "K": 4,
               "state_sets": {1: sorted(nodes[:8])},          # only one entry survives the collision
               "rows": [{"state_id": 1, "u": nodes[0], "v": nodes[20],
                         "before": 10.0, "after": 11.0},
                        {"state_id": 7, "u": nodes[0], "v": nodes[22],
                         "before": 10.0, "after": 12.0}]}     # state 7 was overwritten away
    with pytest.raises(KeyError, match="unknown state"):
        build_dataset(records)


def test_duplicate_swap_rows_are_collapsed(football):
    nodes = sorted(football.nodes(), key=lambda v: -football.degree(v))
    S = sorted(nodes[:8])
    row = {"state_id": "0:1", "u": S[0], "v": nodes[20], "before": 10.0, "after": 11.0}
    ds = build_dataset({"graph": football, "K": 4, "state_sets": {"0:1": S},
                        "rows": [row, dict(row), dict(row)]})
    assert len(ds["y"]) == 1


def test_shared_state_is_detected_and_can_be_removed(football):
    nodes = sorted(football.nodes(), key=lambda v: -football.degree(v))
    shared = sorted(nodes[:8])                       # the degree-52 initial set, in miniature
    a_only, b_only = sorted(nodes[8:16]), sorted(nodes[16:24])
    mk = lambda key, S, v: {"state_id": key, "u": S[0], "v": v, "before": 1.0, "after": 2.0}
    train = build_dataset({"graph": football, "K": 4,
                           "state_sets": {"0:1": shared, "0:2": a_only},
                           "rows": [mk("0:1", shared, nodes[30]), mk("0:2", a_only, nodes[31])]})
    val = build_dataset({"graph": football, "K": 4,
                         "state_sets": {"1:1": shared, "1:2": b_only},
                         "rows": [mk("1:1", shared, nodes[32]), mk("1:2", b_only, nodes[33])]})
    datasets = {"train": train, "val": val}
    assert not check_split_disjoint(datasets)["disjoint"]
    removed = drop_shared_states("train", ["val"], datasets)
    assert removed["val"]["states_dropped"] == 1
    assert check_split_disjoint(datasets)["disjoint"]
    assert len(datasets["val"]["y"]) == 1


def test_pairwise_accuracy_ignores_states_without_contrast():
    # state 1: one positive, two negatives.  state 2: one of each.  state 3: no positive -> skipped.
    y = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    group = np.array([1, 1, 1, 2, 2, 3])
    scores = np.array([0.9, 0.1, 0.2, 5.0, -5.0, 0.0])
    out = pairwise_accuracy(scores, y, group)
    assert out["states_usable"] == 2 and out["pairs"] == 3
    assert out["pairwise_accuracy"] == 1.0
    assert pairwise_accuracy(-scores, y, group)["pairwise_accuracy"] == 0.0
    tied = np.array([0.1, 0.1, 0.2, 5.0, -5.0, 0.0])   # positive ties one negative
    assert pairwise_accuracy(tied, y, group)["pairwise_accuracy"] == pytest.approx(1.5 / 3)
    none_usable = pairwise_accuracy(scores, np.zeros(6), group)
    assert none_usable["states_usable"] == 0 and none_usable["pairwise_accuracy"] is None


def test_feature_names_and_vectors_agree(football):
    nodes = sorted(football.nodes(), key=lambda v: -football.degree(v))
    ctx = build_context(football, sorted(nodes[:8]), 4)
    assert len(ctx.features(nodes[0], nodes[20])) == len(FEATURE_NAMES)


def test_numpy_forward_reproduces_the_torch_ordering(football):
    """The fixed-wall-clock comparison uses the numpy forward; it must be the same ranker, not a variant."""
    import torch
    from ccim.learned_rank import RankModel

    model = RankModel.fresh()
    # give the fresh net non-degenerate weights so ties do not mask an ordering mismatch
    torch.manual_seed(3)
    for p in model.net.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.5)
    model.mean = torch.randn(len(FEATURE_NAMES))
    model.std = torch.rand(len(FEATURE_NAMES)) + 0.5

    nodes = sorted(football.nodes(), key=lambda v: -football.degree(v))
    S = sorted(nodes[:8])
    ctx = build_context(football, S, 4)
    pairs = [(u, v) for u in S[:6] for v in nodes[8:24]]
    with torch.no_grad():
        x = torch.tensor([ctx.features(u, v) for (u, v) in pairs], dtype=torch.float32)
        torch_scores = model.net((x - model.mean) / model.std).squeeze(-1).numpy()
    np_scores = model.numpy_scores([ctx.features(u, v) for (u, v) in pairs])
    assert np.allclose(torch_scores, np_scores, atol=1e-5)

    torch_order = [pairs[i] for i in torch.argsort(
        torch.tensor(torch_scores), descending=True, stable=True).tolist()]
    assert model.numpy_ranker()(ctx, pairs) == torch_order
