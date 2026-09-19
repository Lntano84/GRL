"""A Monte-Carlo oracle whose every number is the CONTRACTED objective.

Why this exists
---------------
The Go/No-Go runner used :class:`grl.oracle.OverexposureMonteCarloOracle`, which reports
``run.spread`` --- the number of positively activated nodes **in the whole graph**.  The source model
is *targeted*: it counts ``|positive ∩ D|`` for an explicit target set ``D``.  The two are not close.
On Congress-Twitter with ``D`` = the top 20% by out-degree, ``|D| = 95``, a 20-seed configuration
gave:

    run.spread (whole graph)     145
    |positive ∩ D|                24
    |positive \\ D|               121

So 84% of the reported quantity lay outside the target set, and every conclusion drawn from it ---
including a Gate 1(b) verdict and a set of quality--cost curves --- was about a different objective.
Worse, the gate's tolerance was computed in *target nodes* (``|D| / 1000``) and compared against a
difference in *whole-graph* spread, so the units did not match either.

This class fixes that at the source: ``spread``, ``score``, the statistics and the invariant check
all count the same thing the contract counts.  The invariant ``0 <= value <= |D|`` is asserted, so a
regression to the global count cannot pass unnoticed --- with ``|D| = 95`` and a value of 145 the
assertion fires immediately.

The oracle is explicitly **not exact**: ``is_exact`` is False, so
:func:`grl.algorithms.sequential_im.reference_policy_name` names it after its Monte-Carlo budget
rather than calling it an oracle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import networkx as nx

from grl.diffusion import overexposure as oe
from grl.diffusion.contract import ModelContract


class TargetedObjectiveError(ValueError):
    """Raised when a measurement violates the contracted objective's range."""


@dataclass
class TargetedOracleStats:
    """Cost accounting in cascades, with the state read as a breakdown of the total."""

    mc_cascades: int = 0
    state_cascades: int = 0
    state_reads: int = 0
    candidate_evaluations: int = 0
    spread_queries: int = 0

    @property
    def cascades_for_scoring(self) -> int:
        return self.mc_cascades - self.state_cascades

    def as_dict(self) -> dict[str, int]:
        return {
            "mc_cascades": self.mc_cascades,
            "state_cascades": self.state_cascades,
            "state_reads": self.state_reads,
            "candidate_evaluations": self.candidate_evaluations,
            "spread_queries": self.spread_queries,
            "cascades_for_scoring": self.cascades_for_scoring,
        }

    def reset(self) -> None:
        self.mc_cascades = 0
        self.state_cascades = 0
        self.state_reads = 0
        self.candidate_evaluations = 0
        self.spread_queries = 0


def trial_seeds(base: int, n: int, *, stride: int = 1_000_003) -> list[int]:
    """``n`` well-separated trial seeds derived from ``base``.

    The runner originally used ``base + offset``.  With replicate bases 20260917 and 20260918 the
    two replicates' trial seeds overlapped in 199 of 200 draws --- a 99.5% overlap between what were
    reported as independent repeats.  Multiplying by a stride far larger than ``n`` makes streams
    from different bases disjoint by construction.

    Callers must also namespace the *purpose* (selection, evaluation, state) into the base, so that
    a method never selects on the same windows it is scored on.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    return [base * stride + offset for offset in range(n)]


class TargetedMonteCarloOracle:
    """``|positive_at_end ∩ D|`` by simulation, with a paired marginal-gain estimator.

    Parameters
    ----------
    graph
        The graph.
    contract
        A :class:`~grl.diffusion.contract.ModelContract`.  ``D`` and the seed-eligibility rule come
        from here and nowhere else.
    mc_runs
        Cascades per configuration per call.
    random_seed
        Base for the trial streams.  Use :func:`trial_seeds` semantics via the ``purpose`` argument
        when one oracle instance is shared between selection and evaluation.
    """

    is_exact = False

    def __init__(
        self,
        graph: nx.DiGraph,
        contract: ModelContract,
        mc_runs: int = 200,
        random_seed: int = 20260917,
        *,
        enforce_range: bool = True,
    ) -> None:
        if mc_runs <= 0:
            raise ValueError(f"mc_runs must be positive, got {mc_runs}")
        self.graph = graph
        self.contract = contract
        self.mc_runs = int(mc_runs)
        self.random_seed = int(random_seed)
        self.enforce_range = bool(enforce_range)
        self.stats = TargetedOracleStats()
        self._nodes = list(graph.nodes())
        self._target = set(contract.objective.target_set)
        if not self._target:
            raise TargetedObjectiveError("the contract has an empty target set")
        self.target_size = len(self._target)

    # -- internals ------------------------------------------------------------------
    def _check_level(self, value: float, what: str) -> float:
        """A *level* --- a count of positive nodes inside D --- must lie in ``[0, |D|]``."""
        if self.enforce_range and not (-1e-9 <= value <= self.target_size + 1e-9):
            raise TargetedObjectiveError(
                f"{what} = {value} is outside [0, |D|] = [0, {self.target_size}].  The contracted "
                f"objective counts only nodes inside D, so a level above |D| means something is "
                f"counting the whole graph --- the bug this class exists to prevent."
            )
        return value

    def _check_marginal(self, value: float, what: str) -> float:
        """A *marginal* may be NEGATIVE, so it lies in ``[-|D|, |D|]``.

        Applying the level bound here was a real bug in this class's first version: it rejected
        ``paired marginal = -4.0 is outside [0, |D|]`` and silently removed three arms from the run.
        A negative marginal is not an error, it is the phenomenon --- adding a seed can push a
        target past its upper threshold and *reduce* the number of positive targets, which is the
        non-monotonicity the paper is about.  Forbidding it would have hidden the very effect under
        study.
        """
        if self.enforce_range and not (-self.target_size - 1e-9 <= value <= self.target_size + 1e-9):
            raise TargetedObjectiveError(
                f"{what} = {value} is outside [-|D|, |D|] = [{-self.target_size}, "
                f"{self.target_size}].  A marginal is a difference of two levels, so it may be "
                f"negative but cannot exceed |D| in magnitude."
            )
        return value

    def _windows(self, rng: random.Random) -> dict:
        return oe.sample_threshold_windows(
            self._nodes, rng,
            threshold_law=self.contract.diffusion.threshold_law,
            window_lo=self.contract.diffusion.window_lo,
        )

    def _positive_in_target(self, seeds, windows, rng) -> float:
        # Eligibility only: exploration may sit one step past the budget, and it is the caller's job
        # to return a set that satisfies check_seeds.
        self.contract.objective.check_seed_eligibility(seeds)
        self.stats.mc_cascades += 1
        run = oe.run_overexposure(
            self.graph, list(seeds), windows, rng,
            activation_mode=self.contract.diffusion.activation_mode,
        )
        return self._check_level(float(len(run.positive & self._target)), "|positive ∩ D|")

    # -- public API -----------------------------------------------------------------
    def spread(self, seeds) -> dict[str, float]:
        """Mean ``|positive_at_end ∩ D|`` over ``mc_runs`` trials."""
        values = []
        for trial_seed in trial_seeds(self.random_seed, self.mc_runs):
            rng = random.Random(trial_seed)
            values.append(self._positive_in_target(seeds, self._windows(rng), rng))
        self.stats.spread_queries += 1
        mean = sum(values) / len(values)
        var = (sum((v - mean) ** 2 for v in values) / len(values)) if len(values) > 1 else 0.0
        return {"mean": mean, "stderr": (var / len(values)) ** 0.5, "n": float(len(values))}

    def score(self, seeds: list, candidates: list, step: int = 0) -> dict:
        """Paired marginal gain in TARGET nodes, per candidate.

        All candidates share the window draw within a trial, so the differences carry no
        window-draw noise.  The base configuration is subtracted per trial, so the result is a
        difference of two contracted counts --- not of two whole-graph counts.
        """
        if not candidates:
            return {}
        seed_list = list(seeds)
        totals = {int(v): 0.0 for v in candidates}
        for trial_seed in trial_seeds(self.random_seed + 7919 * int(step), self.mc_runs):
            rng = random.Random(trial_seed)
            windows = self._windows(rng)
            base = self._positive_in_target(seed_list, windows, rng)
            for candidate in candidates:
                delta = (self._positive_in_target([*seed_list, candidate], windows, rng) - base)
                totals[int(candidate)] += self._check_marginal(delta, "paired marginal")
        self.stats.candidate_evaluations += len(candidates)
        return {v: total / self.mc_runs for v, total in totals.items()}

    def score_with_uncertainty(self, seeds: list, candidates: list, step: int = 0) -> dict:
        """As :meth:`score`, plus the paired standard error and trial count."""
        if not candidates:
            return {}
        seed_list = list(seeds)
        per_trial: dict[int, list[float]] = {int(v): [] for v in candidates}
        for trial_seed in trial_seeds(self.random_seed + 7919 * int(step), self.mc_runs):
            rng = random.Random(trial_seed)
            windows = self._windows(rng)
            base = self._positive_in_target(seed_list, windows, rng)
            for candidate in candidates:
                per_trial[int(candidate)].append(
                    self._positive_in_target([*seed_list, candidate], windows, rng) - base)
        self.stats.candidate_evaluations += len(candidates)
        out = {}
        for v, values in per_trial.items():
            n = len(values)
            mean = sum(values) / n
            var = (sum((x - mean) ** 2 for x in values) / n) if n > 1 else 0.0
            out[v] = {"mean": mean, "stderr": (var / n) ** 0.5, "n": float(n)}
        return out

    def state(self, seeds, step: int = 0) -> dict:
        """Mean per-node exposure, keyed by **node**, over ``mc_runs`` trials.

        Returns a dict rather than a position-indexed list, because that is what
        :func:`grl.scoring.exposure_scores_delta` consumes and what
        :func:`grl.scoring.mean_exposure_state` returns.  ``OverexposureMonteCarloOracle.state``
        returns a list; the mismatch is silent until an arm calls ``.get`` on it, which is how this
        was found --- the sequential arms failed with ``'list' object has no attribute 'get'``.

        Costs ``mc_runs`` cascades, charged to ``state_cascades`` *and* ``mc_cascades`` so a caller
        charging the primary unit charges the state.
        """
        seed_list = list(seeds)
        totals = {node: 0.0 for node in self._nodes}
        for trial_seed in trial_seeds(self.random_seed + 104_729 * int(step), self.mc_runs):
            rng = random.Random(trial_seed)
            self.stats.mc_cascades += 1
            self.stats.state_cascades += 1
            run = oe.run_overexposure(
                self.graph, seed_list, self._windows(rng), rng,
                activation_mode=self.contract.diffusion.activation_mode,
            )
            for node in self._nodes:
                totals[node] += run.delta.get(node, 0.0)
        self.stats.state_reads += 1
        return {node: value / self.mc_runs for node, value in totals.items()}
