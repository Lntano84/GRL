"""Non-learning baselines for the deterministic CCIM setting, with one shared cost ledger.

Every method reports the number of ``sigma`` evaluations it spends.  That is the only unit in which a
learned policy and a combinatorial search can be compared honestly: a ``sigma`` evaluation is the
expensive oracle call (one cascade in the general stochastic model, one deterministic closure here),
so the budget is "how many times may I ask the diffusion model a question".

Reported cost is deliberately broken into two numbers:

* ``decision_evaluations`` -- what the method spends to *choose* a seed set;
* ``scoring_evaluations``  -- the one call needed to report what it achieved.

Only the first is a real cost; the second is bookkeeping that every method pays once, including the
zero-cost degree heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from .model import DEFAULT_K, DEFAULT_T, cascade, sigma


@dataclass
class Ledger:
    """Counts ``sigma`` evaluations so that every method is measured in the same unit.

    The budget is **queries issued**, not distinct queries.  A cache hit is cheaper for the caller but
    it is still a decision the method made about where to spend its budget, and counting only distinct
    answers would silently discount a method that asks the same question over and over -- which is
    exactly what an RL agent does while exploring.  ``distinct`` and ``cache_hits`` are reported
    alongside so the saving is visible rather than hidden.
    """

    decision_evaluations: int = 0
    scoring_evaluations: int = 0
    cache_hits: int = 0
    #: Wall-clock split, so "the diffuser is the cost" can be checked rather than assumed.  ``timer``
    #: is injected by the caller (``time.perf_counter`` in the probe) and left as ``None`` in ordinary
    #: runs, which keeps this class free of timing overhead when nobody is measuring.
    timer: object = None
    cache_seconds: float = 0.0
    diffusion_seconds: float = 0.0
    _cache: dict = field(default_factory=dict)

    def sigma(self, graph: nx.Graph, seeds, K: int, *, decision: bool = True) -> int:
        t = self.timer
        t0 = t() if t is not None else 0.0
        key = (tuple(sorted(int(s) for s in seeds)), int(K))
        if decision:
            self.decision_evaluations += 1
        else:
            self.scoring_evaluations += 1
        hit = key in self._cache
        if t is not None:
            t1 = t()
            self.cache_seconds += t1 - t0
        if hit:
            self.cache_hits += 1
            return self._cache[key]
        value = sigma(graph, seeds, K)
        if t is not None:
            self.diffusion_seconds += t() - t1
        self._cache[key] = value
        return value

    def is_cached(self, seeds, K: int) -> bool:
        """Whether this exact set has already been asked for; used to count duplicate queries."""
        return (tuple(sorted(int(s) for s in seeds)), int(K)) in self._cache

    def timing(self) -> dict:
        return {"cache_and_key_seconds": self.cache_seconds,
                "diffusion_seconds": self.diffusion_seconds,
                "instrumented": self.timer is not None}

    def as_dict(self) -> dict:
        issued = self.decision_evaluations + self.scoring_evaluations
        return {"decision_evaluations": self.decision_evaluations,
                "scoring_evaluations": self.scoring_evaluations,
                "queries_issued": issued,
                "distinct_queries": issued - self.cache_hits,
                "cached_repeats": self.cache_hits,
                "total_evaluations": issued}


@dataclass
class Result:
    """One method's outcome on one graph."""

    method: str
    seeds: list[int]
    sigma: int
    n: int
    cost: dict
    extra: dict = field(default_factory=dict)

    @property
    def normalized(self) -> float:
        return self.sigma / self.n

    def as_dict(self) -> dict:
        return {"method": self.method, "seeds": list(self.seeds), "sigma": self.sigma,
                "normalized": self.normalized, "n": self.n, "cost": self.cost, **self.extra}


def degree_baseline(graph: nx.Graph, T: int = DEFAULT_T, K: int = DEFAULT_K) -> Result:
    """Top-``T`` by degree.  Spends ZERO decision evaluations -- it never queries the cascade."""
    ledger = Ledger()
    seeds = sorted(graph.nodes(), key=lambda v: (-graph.degree(v), v))[:T]
    value = ledger.sigma(graph, seeds, K, decision=False)
    return Result("degree", seeds, value, graph.number_of_nodes(), ledger.as_dict(),
                  {"note": "no cascade query is used to choose; degrees are read directly"})


def greedy_baseline(graph: nx.Graph, T: int = DEFAULT_T, K: int = DEFAULT_K) -> Result:
    """The paper's Greedy: at each step add the node with the largest marginal gain in ``sigma``.

    This is the prominent submodular-IM heuristic, and the paper's point is that it can be
    arbitrarily bad once the objective is non-submodular -- on India it finds no effective solution
    at all.  It is deterministic here, so it reports a single seed set, not a distribution.
    """
    ledger = Ledger()
    chosen: list[int] = []
    current = ledger.sigma(graph, chosen, K)
    gains = []
    for _ in range(T):
        best_v, best_sigma = None, current
        for v in graph.nodes():
            if v in chosen:
                continue
            s = ledger.sigma(graph, [*chosen, v], K)
            if s > best_sigma:
                best_v, best_sigma = v, s
        if best_v is None:
            # no node improves; fill the remaining budget with the highest-degree unused node so the
            # seed set still has size T, which is what the budget constrains
            remaining = [v for v in sorted(graph.nodes(), key=lambda x: (-graph.degree(x), x))
                         if v not in chosen]
            if not remaining:
                break
            best_v, best_sigma = remaining[0], current
        chosen.append(best_v)
        gains.append(best_sigma - current)
        current = best_sigma
    value = ledger.sigma(graph, chosen, K, decision=False)
    return Result("greedy", chosen, value, graph.number_of_nodes(), ledger.as_dict(),
                  {"marginal_gains": gains, "note": "the paper's Greedy baseline"})


def random_plus_baseline(graph: nx.Graph, T: int = DEFAULT_T, K: int = DEFAULT_K,
                         runs: int = 50, seed: int = 0) -> Result:
    """The paper's Random+: sample seeds from the effective candidate set, average over ``runs``.

    Reported for reference because the paper's Table 1 includes it and because it fixes the floor of
    the comparison: a method that cannot beat this has achieved nothing.
    """
    import random

    from .model import effective_candidates

    ledger = Ledger()
    rng = random.Random(seed)
    pool = effective_candidates(graph, K) or list(graph.nodes())
    values, best, best_seeds = [], -1, None
    for _ in range(runs):
        seeds = rng.sample(pool, min(T, len(pool)))
        value = ledger.sigma(graph, seeds, K)
        values.append(value)
        if value > best:
            best, best_seeds = value, seeds
    mean = sum(values) / len(values)
    return Result("random+", sorted(best_seeds), best, graph.number_of_nodes(), ledger.as_dict(),
                  {"mean_sigma": mean, "mean_normalized": mean / graph.number_of_nodes(),
                   "runs": runs, "best_sigma": best})


def cascade_size(graph: nx.Graph, seeds, K: int = DEFAULT_K) -> int:
    """Convenience wrapper so other modules do not have to import ``model`` for one call."""
    return len(cascade(graph, seeds, K))
