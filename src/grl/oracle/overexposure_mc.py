"""State-tracking Monte-Carlo oracle for the overexposure diffusion model.

This is the ground-truth reference for every experiment in this project.  It replaces
:class:`grl.oracle.marginal.BatchedMonteCarloMarginalOracle`, which samples a *live-edge graph*
and computes reachability.  That construction is valid for independent cascade and is exactly
what is unavailable here: under the threshold-window process a node's activation depends on the
accumulated exposure ``delta`` landing inside its own window, so there is no per-edge random
structure to sample.  See ``docs/GATE1_REPORT.md`` (Gate 1d) for the measurement.

What this oracle does instead
-----------------------------
It runs the cascade itself.  ``score`` returns the paired Monte-Carlo estimate of

    Delta(v | S) = sigma(S union {v}) - sigma(S)

where every configuration in a call shares the same sampled threshold windows within a trial, so
the difference carries no window-draw noise.  ``spread`` returns ``sigma(S)``.  Cost is counted in
*Monte-Carlo cascades*, which is the quantity the paper's cost axis is about; the existing
``OracleStats`` in :mod:`grl.oracle.marginal` counts candidate evaluations and live-edge samples
but has no field for this, so it is extended here rather than overloaded.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import networkx as nx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grl.diffusion import overexposure as oe  # noqa: E402
from grl.diffusion.params import OverexposureParams  # noqa: E402


@dataclass
class OverexposureOracleStats:
    """Cost accounting for the overexposure oracle.

    ``mc_cascades`` is the primary cost unit: the number of single-cascade simulations spent.
    It is the quantity that must be held equal across methods in any comparison, and it is what
    the quality-vs-cost figures are plotted against.

    ``state_reads`` and ``state_cascades`` exist because of confound P1-3.2.  A state-conditioned
    policy needs the realised exposure vector ``delta`` before it can score anything, and obtaining
    that vector costs cascades.  Earlier scripts reported that policy's cost as *zero*, next to
    policies that were charged for every cascade, which makes the cost column meaningless.

    ``state_cascades`` is a **breakdown of** ``mc_cascades``, not a parallel channel: every cascade
    spent on a state read is also counted in ``mc_cascades``.  That is deliberate.  Any caller that
    charges ``mc_cascades`` therefore charges the state automatically and cannot under-report a
    state-conditioned policy by forgetting a second counter.  ``cascades_for_scoring`` gives the
    complement, so the free-state variant can still be studied --- but only by subtracting the
    state spend on purpose.
    """

    mc_cascades: int = 0
    candidate_evaluations: int = 0
    spread_queries: int = 0
    state_reads: int = 0
    state_cascades: int = 0

    @property
    def cascades_for_scoring(self) -> int:
        """Cascades spent on scoring, i.e. everything except state acquisition."""
        return self.mc_cascades - self.state_cascades

    def as_dict(self) -> dict[str, int]:
        return {
            "mc_cascades": self.mc_cascades,
            "candidate_evaluations": self.candidate_evaluations,
            "spread_queries": self.spread_queries,
            "state_reads": self.state_reads,
            "state_cascades": self.state_cascades,
            "cascades_for_scoring": self.cascades_for_scoring,
        }

    def reset(self) -> None:
        self.mc_cascades = 0
        self.candidate_evaluations = 0
        self.spread_queries = 0
        self.state_reads = 0
        self.state_cascades = 0


class OverexposureMonteCarloOracle:
    """Monte-Carlo marginal-gain oracle under the threshold-window process.

    Parameters
    ----------
    graph
        Directed graph with ``weight`` on every edge.  In-weights are expected to be normalised
        (sum to 1 per node), which is the regime the model requires; this class does not enforce
        it because the caller may deliberately study un-normalised graphs.
    mc_runs
        Cascades per configuration per trial.  Higher values reduce label noise; the audit showed
        the negative-marginal regime needs at least a few hundred for rank comparisons.
    random_seed
        Base seed.  The seed used for a scoring call is ``random_seed`` combined with ``step``, so
        the same ``(seeds, step)`` reproduces exactly while different steps are independent.
    params
        Validated overexposure parameters.  ``params.mc_runs`` is ignored here; pass ``mc_runs``
        explicitly so that one oracle instance can be used at several budgets.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        mc_runs: int = 200,
        random_seed: int = 20260917,
        params: OverexposureParams | None = None,
        *,
        window_lo: float | None = None,
    ) -> None:
        if mc_runs <= 0:
            raise ValueError(f"mc_runs must be positive, got {mc_runs}")
        self.graph = graph
        self.mc_runs = int(mc_runs)
        self.random_seed = int(random_seed)
        self.params = params or OverexposureParams()
        # ``window_lo`` is exposed separately so a single oracle can be re-parameterised by a
        # sweep without rebuilding the whole params object.
        self.window_lo = self.params.window_lo if window_lo is None else float(window_lo)
        if not 0.0 <= self.window_lo < 1.0:
            raise ValueError(f"window_lo must lie in [0, 1), got {self.window_lo}")
        self.stats = OverexposureOracleStats()
        self._nodes: list[int] = list(graph.nodes())

    # ----------------------------------------------------------------------------------
    # internal helpers
    # ----------------------------------------------------------------------------------
    def _call_seed(self, step: int) -> int:
        return self.random_seed + 1_000_003 * int(step)

    def _draw_windows(self, rng: random.Random) -> dict[int, tuple[float, float]]:
        return oe.sample_threshold_windows(
            self._nodes,
            rng,
            threshold_law=self.params.threshold_law,
            window_lo=self.window_lo,
        )

    def _spread_once(
        self, seeds: Iterable[int], windows: dict[int, tuple[float, float]], rng: random.Random
    ) -> int:
        self.stats.mc_cascades += 1
        return oe.run_overexposure(
            self.graph,
            list(seeds),
            windows,
            rng,
            activation_mode=self.params.activation_mode,
        ).spread

    # ----------------------------------------------------------------------------------
    # public API
    # ----------------------------------------------------------------------------------
    def state(self, seeds: Iterable[int], step: int = 0) -> list[float]:
        """Mean realised exposure vector ``delta`` of the cascade started at ``seeds``.

        This is the observation a state-conditioned policy is allowed to see: one pass of the
        process, averaged over enough windows to be a stable feature rather than a single noisy
        draw.  The returned list is indexed by position in ``list(graph.nodes())`` so it can be
        turned into a node-ordered feature vector by the caller.

        Cost is ``mc_runs`` cascades.  Confound P1-3.2 was that this cost was reported as zero; the
        cascades are now tagged in ``stats.state_cascades``, which is a breakdown of
        ``stats.mc_cascades``, so a caller that charges ``mc_cascades`` charges the state too.
        """
        seed_list = list(seeds)
        base_seed = self._call_seed(step)
        totals = [0.0] * len(self._nodes)
        for offset in range(self.mc_runs):
            rng = random.Random(base_seed + offset)
            windows = self._draw_windows(rng)
            self.stats.mc_cascades += 1
            self.stats.state_cascades += 1
            run = oe.run_overexposure(
                self.graph,
                seed_list,
                windows,
                rng,
                activation_mode=self.params.activation_mode,
            )
            for position, node in enumerate(self._nodes):
                totals[position] += run.delta.get(node, 0.0)
        self.stats.state_reads += 1
        return [value / self.mc_runs for value in totals]

    def spread(self, seeds: Iterable[int]) -> dict[str, float]:
        """Monte-Carlo estimate of ``sigma(seeds)``."""
        seed_list = list(seeds)
        values = []
        for offset in range(self.mc_runs):
            rng = random.Random(self.random_seed + offset)
            windows = self._draw_windows(rng)
            values.append(float(self._spread_once(seed_list, windows, rng)))
        self.stats.spread_queries += 1
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / len(values)
            stderr = (var / len(values)) ** 0.5
        else:
            stderr = 0.0
        return {"mean": mean, "stderr": stderr, "n": float(len(values))}

    def score(
        self, seeds: list[int], candidates: list[int], step: int = 0
    ) -> dict[int, float]:
        """Paired Monte-Carlo marginal gain ``Delta(v | seeds)`` for every candidate.

        All candidates in one call share the sampled windows within each trial, so the *differences*
        between them are free of window-draw noise even though each individual estimate is noisy.

        Use :meth:`score_with_uncertainty` when the caller needs to compare a candidate difference
        against its own estimation error; this method returns the point estimates alone, which is
        what the greedy selectors want.
        """
        if not candidates:
            return {}
        return {v: row["mean"] for v, row in self.score_with_uncertainty(
            seeds, candidates, step=step).items()}

    def score_with_uncertainty(
        self, seeds: list[int], candidates: list[int], step: int = 0
    ) -> dict[int, dict[str, float]]:
        """Paired marginal gain per candidate, **with** its standard error and trial count.

        Confound P1-3.7 was that the dataset recorded ``label_std`` as ``NaN``, so a candidate
        difference was never compared against the error of estimating it --- which is exactly the
        comparison that decides whether a rank correlation in the saturated regime means anything.
        The paired per-trial differences are formed before averaging, so ``stderr`` here is the
        standard error of the *difference*, not of two independent spreads.
        """
        if not candidates:
            return {}
        seed_list = list(seeds)
        base_seed = self._call_seed(step)
        per_trial: dict[int, list[float]] = {int(v): [] for v in candidates}

        for offset in range(self.mc_runs):
            rng = random.Random(base_seed + offset)
            windows = self._draw_windows(rng)
            base = float(self._spread_once(seed_list, windows, rng))
            for candidate in candidates:
                per_trial[int(candidate)].append(
                    float(self._spread_once([*seed_list, candidate], windows, rng)) - base
                )

        self.stats.candidate_evaluations += len(candidates)
        out: dict[int, dict[str, float]] = {}
        for v, values in per_trial.items():
            n = len(values)
            mean = sum(values) / n
            if n > 1:
                var = sum((x - mean) ** 2 for x in values) / n
                stderr = (var / n) ** 0.5
            else:
                stderr = 0.0
            out[v] = {"mean": mean, "stderr": stderr, "n": float(n)}
        return out
