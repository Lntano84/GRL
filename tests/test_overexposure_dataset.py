"""Tests for the signed-label dataset builder (audit item P1-3, confounds 6 and 7).

P1-3.6  the split was applied to context *identifiers*, and the audit alleged that identical seed
        sets could therefore straddle the split boundary.  **Measured, this does not happen**: the
        generator gives each context index a distinct seed set and draws candidates without
        replacement, so every context is already a distinct state (checked at 12, 24, 48 and 120
        contexts, zero duplicates at every size).  The state-group split is kept as a guard that
        makes the invariant explicit in code, and the tests below say so rather than claiming a
        repaired number.  Nothing in the withdrawn empirical record changes because of it.

P1-3.7  ``label_std`` was written as ``float("nan")``, so a candidate difference was never compared
        against the standard error of estimating it.  In the saturated regime that comparison is the
        whole question, and a NaN silently disabled it.  This one was real: the paired estimator
        already formed per-trial differences, so the standard error was available for free.

These tests build a tiny graph so they run in about a second, and they assert properties rather
than values.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from grl.training.overexposure_dataset import (
    OverexposureMarginalSample,
    build_overexposure_dataset,
    dataset_statistics,
    load_overexposure_dataset,
    save_overexposure_dataset,
)


def make_graph(n: int = 26, seed: int = 3) -> nx.DiGraph:
    """A small connected-ish digraph with normalised in-weights, as the model requires."""
    rng = __import__("random").Random(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for v in range(n):
        sources = rng.sample([u for u in range(n) if u != v], k=min(3, n - 1))
        for u in sources:
            graph.add_edge(u, v, weight=1.0 / len(sources))
    return graph


def config(**overrides) -> dict:
    cfg = {
        "seed": {"budget": 3},
        "overexposure_dataset": {
            "budget": 3,
            "contexts": 48,
            "candidates_per_context": 4,
            "mc_runs": 6,
            "random_seed": 20260917,
        },
    }
    cfg["overexposure_dataset"].update(overrides)
    return cfg


def by_context(samples: list[OverexposureMarginalSample]) -> dict[str, list[OverexposureMarginalSample]]:
    grouped: dict[str, list[OverexposureMarginalSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.context_id, []).append(sample)
    return grouped


def state_key(samples: list[OverexposureMarginalSample]) -> tuple:
    """The canonical state of one context: which seeds, scored against which candidates."""
    return (
        tuple(sorted(samples[0].seed_set)),
        tuple(sorted(s.candidate for s in samples)),
    )


# ---------------------------------------------------------------------------------------------
# P1-3.7 -- label uncertainty must be recorded
# ---------------------------------------------------------------------------------------------
def test_label_std_is_a_real_number_not_nan():
    splits = build_overexposure_dataset(make_graph(), config())
    for name, samples in splits.items():
        assert samples, f"{name} split is empty"
        for sample in samples:
            assert not math.isnan(sample.label_std), (
                f"{name} still records label_std as NaN, so candidate differences cannot be "
                f"compared against estimation error"
            )
            assert sample.label_std >= 0.0
            assert sample.label_trials > 0


def test_more_monte_carlo_trials_shrink_the_recorded_label_error():
    """The recorded error must behave like a standard error, not like a standard deviation."""
    graph = make_graph()
    few = build_overexposure_dataset(graph, config(mc_runs=6))
    many = build_overexposure_dataset(graph, config(mc_runs=48))
    few_mean = sum(s.label_std for s in few["train"]) / len(few["train"])
    many_mean = sum(s.label_std for s in many["train"]) / len(many["train"])
    assert many_mean < few_mean, (
        "an eight-fold increase in trials must reduce the mean recorded label error; if it does "
        "not, label_std is not a standard error"
    )


def test_stats_report_the_label_error_alongside_the_gain():
    splits = build_overexposure_dataset(make_graph(), config())
    stats = dataset_statistics(splits)
    for name in ("train", "validation", "test"):
        assert "mean_label_std" in stats[name], (
            f"{name} statistics omit the label uncertainty"
        )
        assert stats[name]["mean_label_std"] >= 0.0


# ---------------------------------------------------------------------------------------------
# P1-3.6 -- the split must not leak a state across the boundary
# ---------------------------------------------------------------------------------------------
def test_no_canonical_state_appears_in_more_than_one_split():
    splits = build_overexposure_dataset(make_graph(), config())
    keys = {
        name: {state_key(samples) for samples in by_context(samples_list).values()}
        for name, samples_list in splits.items()
    }
    total = sum(len(v) for v in keys.values())
    assert total > 0, "no states were produced at all"
    for a in ("train", "validation", "test"):
        for b in ("train", "validation", "test"):
            if a >= b:
                continue
            overlap = keys[a] & keys[b]
            assert not overlap, (
                f"{len(overlap)} canonical state(s) appear in both {a} and {b}; the split is "
                f"leaking the label across the boundary"
            )


def test_duplicate_states_do_not_occur_and_the_split_groups_anyway():
    """Guard, not a fix --- and the distinction is recorded deliberately.

    The audit alleged (P1-3.6) that splitting by context id lets identical seed sets straddle the
    boundary.  Measured on this builder, it does not: ``_make_seed_set`` gives each index a
    distinct seed set and the candidate draw is without replacement, so every context is already a
    distinct state.  That was checked at 12, 24, 48 and 120 contexts, with zero duplicated states
    at every size, and the script that built the paper's ranking dataset has the same property
    because its context seed depends on the context index.

    So no held-out number is repaired by this test.  The state-group split is kept as a guard: it
    makes "no state crosses the boundary" an invariant of the code rather than an accident of the
    generator, so a future change to ``_make_seed_set`` cannot silently introduce the leak.
    """
    splits = build_overexposure_dataset(make_graph(), config())

    seen: dict[tuple, set[str]] = {}
    for name, samples in splits.items():
        for group in by_context(samples).values():
            seen.setdefault(state_key(group), set()).add(name)
    assert not {k: v for k, v in seen.items() if len(v) > 1}, "a state crosses the split boundary"

    states = [state_key(g) for samples in splits.values() for g in by_context(samples).values()]
    assert len(states) == len(set(states)), (
        "this data has duplicated states after all, so the guard IS doing work and the docstring "
        "above is wrong; update the audit note rather than deleting this assertion"
    )

    # the fingerprint must collapse argument order but not distinct states
    base = [0, 1]
    candidates = [5, 6]
    same_a = [OverexposureMarginalSample("ctx_a", base, c, 2, 1.0, 0.1, 0.0, 1.0)
               for c in candidates]
    same_b = [OverexposureMarginalSample("ctx_b", list(base), c, 2, 1.0, 0.1, 0.0, 1.0)
              for c in reversed(candidates)]
    other = [OverexposureMarginalSample("ctx_c", [7, 8], c, 2, 1.0, 0.1, 0.0, 1.0)
             for c in candidates]
    assert state_key(same_a) == state_key(same_b), "argument order must not create a new state"
    assert state_key(same_a) != state_key(other)


def test_every_context_lands_in_exactly_one_split():
    splits = build_overexposure_dataset(make_graph(), config())
    seen: dict[str, str] = {}
    for name, samples in splits.items():
        for context_id in {s.context_id for s in samples}:
            assert context_id not in seen, (
                f"context {context_id} appears in both {seen[context_id]} and {name}"
            )
            seen[context_id] = name
    assert seen, "no contexts were produced at all"


def test_all_samples_of_a_context_stay_together():
    splits = build_overexposure_dataset(make_graph(), config())
    for name, samples in splits.items():
        by_context: dict[str, int] = {}
        for sample in samples:
            by_context[sample.context_id] = by_context.get(sample.context_id, 0) + 1
        # every context in a split must carry its full candidate list, not a fragment
        assert all(count == 4 for count in by_context.values()), (
            f"{name} contains a partial context: {sorted(set(by_context.values()))}"
        )


# ---------------------------------------------------------------------------------------------
# signed labels must survive the round trip
# ---------------------------------------------------------------------------------------------
def test_signed_labels_and_exposure_survive_a_save_load_round_trip(tmp_path):
    splits = build_overexposure_dataset(make_graph(), config())
    path = save_overexposure_dataset(splits, tmp_path / "dataset.json")
    reloaded = load_overexposure_dataset(path)
    assert set(reloaded) == set(splits)
    for name in splits:
        assert len(reloaded[name]) == len(splits[name])
        for original, restored in zip(splits[name], reloaded[name]):
            assert restored.marginal_gain == original.marginal_gain
            assert restored.label_std == original.label_std
            assert restored.candidate == original.candidate
            assert restored.seed_set == original.seed_set


def test_a_split_that_does_not_sum_to_one_is_refused():
    with pytest.raises(ValueError, match="must contain three fractions summing to 1"):
        build_overexposure_dataset(make_graph(), config(), split=(0.5, 0.2, 0.2))
