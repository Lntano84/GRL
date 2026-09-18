"""The frozen model contract: what is counted, who may be a seed, and how the process runs.

Why this module exists
----------------------
The audit found that earlier runs mixed several things that must be kept apart: the simulator
counted every node including seeds while the source model defines a *target* set ``D`` with seeds
drawn from ``V \\ D``; some methods were forced to pick exactly ``k`` seeds although the problem
allows *at most* ``k``; and the reported objective was the number of nodes that were *ever*
positive rather than the number positive at the end.  Each of those changes the number a method is
credited with, so they have to be fixed before any comparison is meaningful.

This module is the single place where those decisions are written down, so that no experiment has
to make them again and no reader has to infer them.

The six frozen items
--------------------
1. **End-state counting.**  ``F_D(S) = E[ |P_inf(S, omega) INTERSECT D| ]`` where ``P_inf`` is the
   set of nodes **positive at the end of the process**.  A node that turned negative is not counted,
   even though its influence on its neighbours persists.  ``grl.diffusion.overexposure`` exposes
   this as ``OverexposureRun.positive``; ``ever_positive`` is a different set and is what drives
   exposure accumulation.
2. **Seeds stay positive.**  A seed is positive by definition and is never re-evaluated, so it is
   never overexposed.  Enforced in ``run_overexposure``.
3. **Historical influence persists.**  ``A^in_t(v)`` retains neighbours that were positively
   activated and later turned negative, so ``delta`` keeps their weight.  This is why
   ``ever_positive``, not ``positive``, feeds exposure.
4. **Threshold law.**  Named by ``OverexposureParams.threshold_law``.  The default is the source
   model's simplex sampling; it is not uniform-threshold LT and must not be described as such.
5. **Target set and seed eligibility.**  ``D`` is explicit, and seeds come from ``V \\ D``.
   Setting ``D = V`` is a *different* problem and is only permitted with
   ``allow_seeds_in_target=True``, which records that the difference was a deliberate choice.
6. **Budget is at most k.**  Methods may return fewer than ``k`` seeds.  A method that stops early
   is not penalised, and a method that is forced to fill ``k`` slots is not comparable to one that
   is allowed to stop.

Everything a caller needs is reachable from :func:`build_contract` and
:meth:`grl.diffusion.contract.TargetedObjective.evaluate`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

from . import overexposure as oe
from .params import OverexposureParams, resolve_overexposure_params


class ContractViolation(ValueError):
    """Raised when a caller asks for something the frozen contract forbids."""


@dataclass(frozen=True)
class ObjectiveContract:
    """Target set, seed eligibility and budget semantics.

    ``target_set`` is stored as a frozenset of node labels.  ``allow_seeds_in_target`` exists so the
    ``D = V`` variant is available but never silently conflated with the source model's ``S ⊆ V \\ D``
    formulation.
    """

    target_set: frozenset
    allow_seeds_in_target: bool = False
    budget: int = 0
    budget_is_at_most: bool = True

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise ValueError(f"budget must be non-negative, got {self.budget}")
        if not self.budget_is_at_most:
            # The source problem is |S| <= k.  A "pick exactly k" variant changes what a stopping
            # rule means and must not be mixed into the same comparison.
            raise ContractViolation(
                "this contract fixes the budget at 'at most k'; pausing or stopping early is "
                "allowed and must not be penalised.  A variant that forces |S| = k is a different "
                "problem and needs its own contract."
            )

    def check_seeds(self, seeds: Iterable) -> None:
        """Validate a **final** seed set: within budget, unique, and outside the target set.

        Use this on the set a method returns.  It is deliberately not applied to intermediate
        exploration states: evaluating ``Delta(v | S)`` for a candidate means looking one step past
        the current set, which is legitimate even when ``S`` already has the full budget.  Enforcing
        the budget there would make exploration impossible and would silently forbid the
        "at most k" semantics the contract exists to protect.
        """
        seed_list = list(seeds)
        if len(seed_list) > self.budget:
            raise ContractViolation(
                f"seed set has {len(seed_list)} nodes, budget is at most {self.budget}"
            )
        if len(set(seed_list)) != len(seed_list):
            raise ContractViolation("seed set contains duplicates")
        self.check_seed_eligibility(seed_list)

    def check_seed_eligibility(self, seeds: Iterable) -> None:
        """Validate only the eligibility rule ``S ⊆ V \\ D`` (or the recorded D = V variant)."""
        seed_list = list(seeds)
        if len(set(seed_list)) != len(seed_list):
            raise ContractViolation("seed set contains duplicates")
        if not self.allow_seeds_in_target:
            overlap = set(seed_list) & set(self.target_set)
            if overlap:
                raise ContractViolation(
                    f"seeds must come from V \\ D, but {sorted(overlap)[:5]} are in the target set. "
                    f"If D = V is intended, set allow_seeds_in_target=True to record that choice."
                )

    def legal_candidates(self, graph: nx.DiGraph, exclude: Iterable = ()) -> list:
        """Nodes eligible to be seeds: everything outside D (unless overridden) and not already chosen."""
        excluded = set(exclude)
        if self.allow_seeds_in_target:
            return [v for v in graph.nodes() if v not in excluded]
        return [v for v in graph.nodes() if v not in excluded and v not in self.target_set]

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_set_size": len(self.target_set),
            "allow_seeds_in_target": self.allow_seeds_in_target,
            "budget": self.budget,
            "budget_is_at_most": self.budget_is_at_most,
        }


@dataclass(frozen=True)
class ModelContract:
    """Objective half plus diffusion half, i.e. the whole frozen contract."""

    objective: ObjectiveContract
    diffusion: OverexposureParams

    def describe(self) -> dict[str, Any]:
        payload = {"objective": self.objective.as_dict(), "diffusion": self.diffusion.as_dict()}
        payload["counting"] = "final positive nodes intersect target set"
        payload["seeds_always_positive"] = True
        payload["historical_influence_persists"] = True
        return payload


def build_contract(
    graph: nx.DiGraph,
    config: Mapping[str, Any] | None = None,
    *,
    target_set: Sequence | None = None,
    allow_seeds_in_target: bool | None = None,
    budget: int | None = None,
) -> ModelContract:
    """Assemble a validated contract from a config plus explicit overrides.

    Defaults follow the source model: ``D`` is a configured node list, seeds come from ``V \\ D``,
    and the budget is ``seed.budget`` interpreted as "at most".
    """
    config = config or {}
    block = config.get("objective") or {}
    if not isinstance(block, Mapping):
        raise ContractViolation("config['objective'] must be a mapping")

    if target_set is None:
        configured = block.get("target_set")
        if configured is None:
            raise ContractViolation(
                "no target set given.  The source model defines a target set D and draws seeds "
                "from V \\ D, so D must be stated explicitly; pass target_set=... or set "
                "objective.target_set in the config.  Use allow_seeds_in_target=True only for the "
                "deliberate D = V variant, which is a different problem."
            )
        target_set = configured

    nodes = set(graph.nodes())
    target = frozenset(target_set)
    unknown = set(target) - nodes
    if unknown:
        raise ContractViolation(f"target set contains nodes not in the graph: {sorted(unknown)[:5]}")

    if allow_seeds_in_target is None:
        allow_seeds_in_target = bool(block.get("allow_seeds_in_target", False))
    if budget is None:
        budget = int(block.get("budget", config.get("seed", {}).get("budget", 0)))

    if allow_seeds_in_target and target != frozenset(nodes):
        raise ContractViolation(
            "allow_seeds_in_target=True is the D = V variant; with a strict subset target it would "
            "silently permit seeds inside D, which is a different problem.  Pass D = V or leave the "
            "flag False."
        )

    objective = ObjectiveContract(
        target_set=target,
        allow_seeds_in_target=allow_seeds_in_target,
        budget=int(budget),
        budget_is_at_most=True,
    )
    return ModelContract(objective=objective,
                         diffusion=resolve_overexposure_params(config))


#: Target-set modes understood by :func:`resolve_target_contract`.
TARGET_MODE_ALL = "all"
TARGET_MODE_DEGREE_TAIL = "degree-tail"
TARGET_MODES = (TARGET_MODE_ALL, TARGET_MODE_DEGREE_TAIL)


def resolve_target_contract(
    graph: nx.DiGraph, target_mode: str, target_fraction: float, budget: int
) -> ModelContract:
    """State which nodes are counted and where seeds may come from --- explicitly.

    The source model is *targeted*: it defines a target set ``D`` and draws seeds from ``V \\ D``,
    counting only positives inside ``D``.  Earlier sweeps in this repository counted every positive
    node and drew seeds from the whole graph.  That is the same simulator applied to a **different
    problem**, and it is what ``all`` selects --- permitted, recorded, and reported, but never
    silently mixed with the source model's formulation.

    ``all``
        ``D = V`` with ``allow_seeds_in_target=True``.  Every positive node is counted and any node
        may be a seed.
    ``degree-tail``
        ``D`` is the top ``target_fraction`` of nodes by out-degree --- the natural reading of "the
        nodes we are trying to activate" --- and seeds come from ``V \\ D`` as the model requires.

    This lives here rather than in each script so that two experiment scripts cannot drift into
    declaring different problems under the same flag name.
    """
    nodes = list(graph.nodes())
    if not nodes:
        raise ContractViolation("cannot build a contract for an empty graph")
    if target_mode == TARGET_MODE_ALL:
        return build_contract(graph, {}, target_set=nodes, allow_seeds_in_target=True,
                              budget=budget)
    if target_mode == TARGET_MODE_DEGREE_TAIL:
        if not 0.0 < target_fraction < 1.0:
            raise ContractViolation(
                f"target_fraction must lie strictly between 0 and 1, got {target_fraction}"
            )
        degree = dict(graph.out_degree())
        ordered = sorted(nodes, key=lambda v: (-degree[v], v))
        cut = max(1, int(round(target_fraction * len(ordered))))
        if cut >= len(ordered):
            raise ContractViolation(
                f"target_fraction {target_fraction} selects all {len(ordered)} nodes, which leaves "
                f"no eligible seeds; use target_mode={TARGET_MODE_ALL!r} to record that choice"
            )
        return build_contract(graph, {}, target_set=ordered[:cut],
                              allow_seeds_in_target=False, budget=budget)
    raise ContractViolation(
        f"unknown target mode {target_mode!r}; choose from {TARGET_MODES}"
    )


class TargetedObjective:
    """Evaluate the contracted objective by simulation.

    ``F_D(S)`` is the Monte-Carlo mean of ``|positive_at_end ∩ D|``.  ``marginal_gains`` returns the
    paired difference per candidate, sharing windows within a trial so the differences carry no
    window-draw noise.  Both count *end-state* positives, never "ever positive".
    """

    def __init__(self, graph: nx.DiGraph, contract: ModelContract) -> None:
        self.graph = graph
        self.contract = contract
        self._nodes = list(graph.nodes())
        self._target = set(contract.objective.target_set)
        self.mc_cascades = 0

    # -- internals ---------------------------------------------------------------------
    def _windows(self, rng: random.Random) -> dict:
        return oe.sample_threshold_windows(
            self._nodes, rng,
            overexposure_free=self.contract.diffusion.overexposure_free,
            window_lo=self.contract.diffusion.window_lo,
        )

    def _positive_in_target(self, seeds: Sequence, windows: dict, rng: random.Random) -> int:
        # Only eligibility is enforced during evaluation: the exploration state may be one step
        # past the budget, and it is the caller's job to return a set that satisfies check_seeds.
        self.contract.objective.check_seed_eligibility(seeds)
        self.mc_cascades += 1
        run = oe.run_overexposure(
            self.graph, list(seeds), windows, rng,
            activation_mode=self.contract.diffusion.activation_mode,
        )
        return len(run.positive & self._target)

    # -- public API --------------------------------------------------------------------
    def evaluate(self, seeds: Sequence, mc_runs: int | None = None,
                 random_seed: int | None = None) -> dict[str, float]:
        mc = int(mc_runs or self.contract.diffusion.mc_runs)
        seed = int(random_seed if random_seed is not None else self.contract.diffusion.random_seed)
        values = []
        for offset in range(mc):
            rng = random.Random(seed + offset)
            values.append(float(self._positive_in_target(seeds, self._windows(rng), rng)))
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values) if len(values) > 1 else 0.0
        return {"mean": mean, "stderr": (var / len(values)) ** 0.5, "n": float(len(values))}

    def marginal_gains(self, seeds: Sequence, candidates: Sequence,
                       mc_runs: int | None = None, random_seed: int | None = None) -> dict:
        if not candidates:
            return {}
        mc = int(mc_runs or self.contract.diffusion.mc_runs)
        seed = int(random_seed if random_seed is not None else self.contract.diffusion.random_seed)
        totals = {c: 0.0 for c in candidates}
        for offset in range(mc):
            rng = random.Random(seed + offset)
            windows = self._windows(rng)
            base = self._positive_in_target(seeds, windows, rng)
            for candidate in candidates:
                totals[candidate] += (
                    self._positive_in_target([*seeds, candidate], windows, rng) - base
                )
        return {c: total / mc for c, total in totals.items()}
