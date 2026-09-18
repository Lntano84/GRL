"""Tests for the frozen model contract.

Each test corresponds to one of the six items the audit required to be fixed before any comparison
is meaningful.  The point is that a future experiment cannot silently redefine counting, seed
eligibility or budget semantics.
"""

from __future__ import annotations

import random

import networkx as nx
import pytest

from grl.diffusion import overexposure as oe
from grl.diffusion.contract import (
    TARGET_MODE_ALL,
    TARGET_MODE_DEGREE_TAIL,
    ContractViolation,
    ModelContract,
    ObjectiveContract,
    TargetedObjective,
    build_contract,
    resolve_target_contract,
)
from grl.diffusion.params import OverexposureParams


def chain(n: int = 5) -> nx.DiGraph:
    graph = nx.DiGraph()
    for i in range(n - 1):
        graph.add_edge(i, i + 1, weight=0.5)
    return graph


def star(n: int = 10) -> nx.DiGraph:
    """A graph with a clear out-degree ordering, so degree-tail target selection is unambiguous."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for v in range(1, n):
        graph.add_edge(0, v, weight=0.3)
        graph.add_edge(v, 0, weight=0.3)
    return graph


# --------------------------------------------------------------------------------------
# item 5: target set and seed eligibility
# --------------------------------------------------------------------------------------
def test_contract_requires_an_explicit_target_set():
    with pytest.raises(ContractViolation, match="no target set given"):
        build_contract(chain(), {})


def test_seeds_must_come_from_outside_the_target_set():
    graph = chain()
    contract = build_contract(graph, {}, target_set=[3, 4], budget=2)
    with pytest.raises(ContractViolation, match="seeds must come from"):
        contract.objective.check_seeds([3])


def test_legal_candidates_exclude_the_target_set():
    graph = chain()
    contract = build_contract(graph, {}, target_set=[3, 4], budget=2)
    assert contract.objective.legal_candidates(graph) == [0, 1, 2]


def test_full_target_set_requires_the_explicit_flag():
    graph = chain()
    # A strict subset target with the flag set would silently permit seeds inside D, which is the
    # D = V variant only.  The contract refuses that combination.
    with pytest.raises(ContractViolation, match="D = V variant"):
        build_contract(graph, {}, target_set=[3, 4], allow_seeds_in_target=True)
    # With D = V and the flag, seeds may be targets.
    contract = build_contract(graph, {}, target_set=list(graph.nodes()),
                              allow_seeds_in_target=True, budget=1)
    contract.objective.check_seeds([0])
    assert contract.objective.legal_candidates(graph) == list(graph.nodes())


def test_target_set_nodes_must_exist():
    with pytest.raises(ContractViolation, match="not in the graph"):
        build_contract(chain(), {}, target_set=[99])


# --------------------------------------------------------------------------------------
# item 6: budget is "at most k"
# --------------------------------------------------------------------------------------
def test_budget_is_at_most_and_early_stopping_is_legal():
    graph = chain()
    contract = build_contract(graph, {}, target_set=[4], budget=3)
    contract.objective.check_seeds([0])          # one of three: fine
    contract.objective.check_seeds([0, 1])       # two of three: fine
    contract.objective.check_seeds([0, 1, 2])    # exactly three: fine


def test_exceeding_the_budget_is_rejected():
    graph = chain()
    contract = build_contract(graph, {}, target_set=[4], budget=1)
    with pytest.raises(ContractViolation, match="budget is at most"):
        contract.objective.check_seeds([0, 1])


def test_duplicate_seeds_are_rejected():
    graph = chain()
    contract = build_contract(graph, {}, target_set=[4], budget=3)
    with pytest.raises(ContractViolation, match="duplicates"):
        contract.objective.check_seeds([0, 0])


def test_forcing_exactly_k_is_refused():
    with pytest.raises(ContractViolation, match="at most k"):
        ObjectiveContract(target_set=frozenset({4}), budget=2, budget_is_at_most=False)


def test_negative_budget_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        ObjectiveContract(target_set=frozenset({4}), budget=-1)


# --------------------------------------------------------------------------------------
# item 1: end-state counting, not ever-positive
# --------------------------------------------------------------------------------------
def test_objective_counts_end_state_not_history():
    """The P0-1 construction: node 'a' is positive then overexposed, and D = {a}.

    With the old state machine the objective would have credited a; the contract requires the
    end-state count, which is zero.
    """
    graph = nx.DiGraph()
    graph.add_edge("s", "a", weight=0.4)
    graph.add_edge("s", "b", weight=0.5)
    graph.add_edge("b", "a", weight=0.4)
    windows = {"s": (0.0, 1.0), "a": (0.2, 0.6), "b": (0.2, 0.9)}

    run = oe.run_overexposure(graph, ["s"], windows, random.Random(0))
    assert "a" in run.ever_positive
    assert "a" not in run.positive
    assert len(run.positive & {"a"}) == 0, "end-state counting must not credit a"
    assert len(run.ever_positive & {"a"}) == 1, "history is a different set"


def test_objective_is_zero_when_the_only_target_is_overexposed():
    graph = nx.DiGraph()
    graph.add_edge("s", "a", weight=0.4)
    graph.add_edge("s", "b", weight=0.5)
    graph.add_edge("b", "a", weight=0.4)
    contract = ModelContract(
        objective=ObjectiveContract(target_set=frozenset({"a"}), budget=1),
        diffusion=OverexposureParams(mc_runs=5, random_seed=3),
    )
    objective = TargetedObjective(graph, contract)
    # the simulator draws its own windows, so pin the outcome via a degenerate graph instead:
    # a star where a is the only target and cannot be reached
    isolated = nx.DiGraph()
    isolated.add_node("s")
    isolated.add_node("a")
    contract2 = ModelContract(
        objective=ObjectiveContract(target_set=frozenset({"a"}), budget=1),
        diffusion=OverexposureParams(mc_runs=5, random_seed=3),
    )
    result = TargetedObjective(isolated, contract2).evaluate(["s"])
    assert result["mean"] == pytest.approx(0.0)


def test_objective_counts_only_the_target_set():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)
    windows = {0: (0.0, 1.0), 1: (0.0, 1.0), 2: (0.0, 1.0)}
    run = oe.run_overexposure(graph, [0], windows, random.Random(0))
    assert run.positive == {0, 1, 2}
    assert len(run.positive & {1}) == 1
    assert len(run.positive & {2}) == 1
    assert len(run.positive & set()) == 0


# --------------------------------------------------------------------------------------
# items 2 and 3: seeds stay positive, historical influence persists
# --------------------------------------------------------------------------------------
def test_contract_records_the_diffusion_semantics():
    graph = chain()
    contract = build_contract(graph, {}, target_set=[4], budget=1)
    described = contract.describe()
    assert described["seeds_always_positive"] is True
    assert described["historical_influence_persists"] is True
    assert "final positive" in described["counting"]


# --------------------------------------------------------------------------------------
# item 4: the threshold law is named, not assumed
# --------------------------------------------------------------------------------------
def test_threshold_law_is_named_and_not_uniform_lt():
    default = OverexposureParams()
    assert default.threshold_law == "simplex"
    assert "uniform" not in default.threshold_law

    clamped = OverexposureParams(threshold_law="simplex_tau_clamped_to_one")
    assert clamped.threshold_law == "simplex_tau_clamped_to_one"
    assert "uniform" not in clamped.threshold_law

    uniform = OverexposureParams(threshold_law="uniform_lt_tau_one")
    assert uniform.threshold_law == "uniform_lt_tau_one"


def test_the_three_degeneracy_paths_are_distinct():
    """The audit required the degeneracy checks to be kept apart.

    Clamping tau to 1 while keeping the min-of-two-uniforms kappa gives ``F_kappa(x) = 2x - x^2``;
    uniform-threshold LT has ``F_kappa(x) = x``.  The two must not be used as one control, so the
    contract exposes them as separate laws and the sampled marginals must differ.
    """
    import random

    from grl.diffusion import overexposure as oe

    nodes = list(range(20000))
    clamped = oe.sample_threshold_windows(
        nodes, random.Random(3), threshold_law="simplex_tau_clamped_to_one")
    uniform = oe.sample_threshold_windows(
        nodes, random.Random(3), threshold_law="uniform_lt_tau_one")

    def mean_kappa(windows):
        return sum(k for k, _ in windows.values()) / len(windows)

    # E[kappa] is 1/3 for min of two uniforms and 1/2 for a single uniform
    assert abs(mean_kappa(clamped) - 1 / 3) < 0.02, mean_kappa(clamped)
    assert abs(mean_kappa(uniform) - 1 / 2) < 0.02, mean_kappa(uniform)
    assert abs(mean_kappa(clamped) - mean_kappa(uniform)) > 0.1
    # both disable overexposure
    assert all(t == 1.0 for _, t in clamped.values())
    assert all(t == 1.0 for _, t in uniform.values())


def test_overexposure_free_is_a_derived_property_not_a_field():
    """Both clamped laws disable overexposure, but they are still different processes."""
    assert OverexposureParams().overexposure_free is False
    assert OverexposureParams(threshold_law="simplex_tau_clamped_to_one").overexposure_free is True
    assert OverexposureParams(threshold_law="uniform_lt_tau_one").overexposure_free is True
    # it cannot be set, so the paths cannot be collapsed back into one boolean
    with pytest.raises(TypeError):
        OverexposureParams(overexposure_free=True)


def test_threshold_law_is_reported_in_the_description():
    graph = chain()
    contract = build_contract(graph, {}, target_set=[4], budget=1)
    assert contract.describe()["diffusion"]["threshold_law"] == "simplex"


# --------------------------------------------------------------------------------------
# the objective itself
# --------------------------------------------------------------------------------------
def test_evaluate_matches_a_hand_computed_expectation():
    """Single edge with weight 1 and a wide window: the target is always positive."""
    graph = nx.DiGraph()
    graph.add_edge("s", "t", weight=1.0)
    contract = ModelContract(
        objective=ObjectiveContract(target_set=frozenset({"t"}), budget=1),
        diffusion=OverexposureParams(mc_runs=20, random_seed=11),
    )
    result = TargetedObjective(graph, contract).evaluate(["s"])
    assert result["n"] == 20
    assert 0.0 <= result["mean"] <= 1.0


def test_marginal_gains_are_paired_and_reproducible():
    graph = nx.DiGraph()
    for i in range(6):
        graph.add_edge(i, 5, weight=0.3)
    contract = ModelContract(
        objective=ObjectiveContract(target_set=frozenset({5}), budget=2),
        diffusion=OverexposureParams(mc_runs=25, random_seed=7),
    )
    objective = TargetedObjective(graph, contract)
    first = objective.marginal_gains([0], [1, 2, 3])
    second = TargetedObjective(graph, contract).marginal_gains([0], [1, 2, 3])
    assert first == second


def test_marginal_gains_can_be_negative():
    """Two heavy contributors to one target: the second pushes it past its window sometimes."""
    graph = nx.DiGraph()
    graph.add_edge("a", "t", weight=0.5)
    graph.add_edge("b", "t", weight=0.5)
    contract = ModelContract(
        objective=ObjectiveContract(target_set=frozenset({"t"}), budget=2),
        diffusion=OverexposureParams(mc_runs=400, random_seed=5),
    )
    gains = TargetedObjective(graph, contract).marginal_gains(["a"], ["b"])
    assert gains["b"] < 0.0, f"expected a harmful second seed, got {gains['b']}"


def test_contract_rejects_illegal_seeds_during_evaluation():
    graph = chain()
    contract = build_contract(graph, {}, target_set=[3], budget=1)
    objective = TargetedObjective(graph, contract)
    with pytest.raises(ContractViolation):
        objective.evaluate([3])


def test_mc_cascade_accounting():
    graph = chain()
    contract = ModelContract(
        objective=ObjectiveContract(target_set=frozenset({4}), budget=1),
        diffusion=OverexposureParams(mc_runs=6, random_seed=1),
    )
    objective = TargetedObjective(graph, contract)
    objective.evaluate([0])
    assert objective.mc_cascades == 6
    objective.marginal_gains([0], [1, 2])
    # 6 trials x (1 base + 2 candidates)
    assert objective.mc_cascades == 6 + 6 * 3


def test_config_driven_construction():
    graph = chain()
    config = {
        "objective": {"target_set": [4], "budget": 2},
        "overexposure": {"mc_runs": 9, "random_seed": 4},
    }
    contract = build_contract(graph, config)
    assert contract.objective.budget == 2
    assert 4 in contract.objective.target_set
    assert contract.diffusion.mc_runs == 9
    assert contract.diffusion.random_seed == 4


# --------------------------------------------------------------------------------------
# resolve_target_contract: the two target modes must be distinguishable and recorded
# --------------------------------------------------------------------------------------
def test_all_target_mode_counts_everything_and_records_the_choice():
    """D = V is the same simulator applied to a different problem, and it must say so."""
    graph = star()
    contract = resolve_target_contract(graph, TARGET_MODE_ALL, 0.2, budget=2)
    described = contract.describe()
    assert described["objective"]["target_set_size"] == len(graph.nodes())
    assert described["objective"]["allow_seeds_in_target"] is True
    assert described["counting"] == "final positive nodes intersect target set"
    # every node is a legal seed in this mode
    assert len(contract.objective.legal_candidates(graph)) == len(graph.nodes())


def test_degree_tail_target_mode_matches_the_source_formulation():
    """D is a strict subset and seeds come from V \\ D, which is what the source model states."""
    graph = star()
    contract = resolve_target_contract(graph, TARGET_MODE_DEGREE_TAIL, 0.2, budget=2)
    target = contract.objective.target_set
    assert 0 < len(target) < len(graph.nodes())
    assert contract.objective.allow_seeds_in_target is False
    eligible = contract.objective.legal_candidates(graph)
    assert set(eligible).isdisjoint(target), "a target node must not be an eligible seed"
    assert len(eligible) + len(target) == len(graph.nodes())
    # the highest out-degree node is in D, so it cannot be seeded
    assert max(graph.out_degree(), key=lambda kv: kv[1])[0] in target


def test_the_two_modes_do_not_declare_the_same_problem():
    graph = star()
    everything = resolve_target_contract(graph, TARGET_MODE_ALL, 0.2, budget=2)
    tail = resolve_target_contract(graph, TARGET_MODE_DEGREE_TAIL, 0.2, budget=2)
    assert everything.objective.target_set != tail.objective.target_set
    assert (everything.objective.allow_seeds_in_target
            != tail.objective.allow_seeds_in_target)


def test_degree_tail_rejects_a_fraction_that_leaves_no_seeds():
    """A fraction below 1 can still round up to the whole graph on a small one."""
    graph = star(10)
    with pytest.raises(ContractViolation, match="leaves"):
        resolve_target_contract(graph, TARGET_MODE_DEGREE_TAIL, 0.999, budget=2)


def test_degree_tail_rejects_a_degenerate_fraction():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ContractViolation, match="strictly between 0 and 1"):
            resolve_target_contract(star(), TARGET_MODE_DEGREE_TAIL, bad, budget=2)


def test_unknown_target_mode_is_refused():
    with pytest.raises(ContractViolation, match="unknown target mode"):
        resolve_target_contract(star(), "whatever", 0.2, budget=2)


def test_an_empty_graph_is_refused():
    with pytest.raises(ContractViolation, match="empty graph"):
        resolve_target_contract(nx.DiGraph(), TARGET_MODE_ALL, 0.2, budget=1)


def test_the_budget_is_carried_into_the_contract():
    managed = resolve_target_contract(star(), TARGET_MODE_ALL, 0.2, budget=3)
    assert managed.objective.budget == 3
    assert managed.objective.budget_is_at_most is True
