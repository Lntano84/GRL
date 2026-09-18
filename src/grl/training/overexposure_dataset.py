"""Marginal-gain dataset for the overexposure diffusion model.

Why this does not reuse :mod:`grl.training.marginal_dataset`
-----------------------------------------------------------
Two differences matter, and both would silently break the science:

1. **Estimator.** The existing builder calls ``grl.diffusion.estimate_marginal_gains``, which is
   the *independent-cascade* estimator and samples live-edge graphs.  That construction does not
   exist under the threshold-window process, so labels must come from the state-tracking
   overexposure oracle instead.

2. **Sign of the label.** The existing builder stores
   ``marginal_gain=max(estimate["mean"], 0.0)``, clipping negatives away.  Negative marginal
   gains are a *defining* property of this model --- seeding a node can push an already-active
   neighbour past its overexposure threshold and destroy the spread that neighbour was producing.
   A clipped dataset would train the predictor to ignore exactly the phenomenon the paper is
   about, and would make the harmful-seed case unlearnable.  Labels here are kept signed, and the
   negative share is reported so the effect is visible in the data rather than assumed.

Exposure state
--------------
Each sample also carries the realised exposure vector ``delta`` of the seed state, because the
state-conditioned predictor has the option of reading it.  ``delta`` is produced by one cascade
under the same threshold windows used for the label, so it is the state a deployed policy would
actually observe.  Storing it per sample is O(n) per sample; that is fine at the scales used here
(n <= ~1.5e4, a few thousand samples) and is recorded as a limitation for larger graphs.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.diffusion.params import OverexposureParams, resolve_overexposure_params
from grl.oracle import OverexposureMonteCarloOracle


@dataclass(frozen=True)
class OverexposureMarginalSample:
    context_id: str
    seed_set: list[int]
    candidate: int
    seed_set_size: int
    marginal_gain: float
    label_std: float
    base_spread: float
    extended_spread: float
    label_trials: int = 0
    exposure: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _partial_seed_sequences(
    graph: nx.DiGraph, budget: int, probability: float, nodes: list[int]
) -> list[list[int]]:
    degree = select_high_degree_nodes(graph, min(budget, len(nodes)))
    discount = select_degree_discount_nodes(graph, min(budget, len(nodes)), probability)
    return (
        [degree[:size] for size in range(len(degree) + 1)]
        + [discount[:size] for size in range(len(discount) + 1)]
    )


def _make_seed_set(
    graph: nx.DiGraph,
    nodes: list[int],
    partials: list[list[int]],
    size: int,
    index: int,
    rng: random.Random,
) -> list[int]:
    """Build a seed set of the requested size from several sources.

    Mixing sources matters: if every state were drawn uniformly at random, the predictor would
    only ever see uninformative states and could not learn state conditioning at all.  The
    prefixes of degree and degree-discount orderings supply states that a real greedy policy
    would visit.
    """
    if size == 0:
        return []
    branch = index % 4
    if branch == 0:
        return rng.sample(nodes, size)
    if branch in (1, 3) and partials:
        source = partials[index % len(partials)]
        if len(source) >= size:
            return list(source[:size])
    if branch == 2:
        ranked = select_high_degree_nodes(graph, len(nodes))
        return list(ranked[:size])
    if len(nodes) >= size:
        return rng.sample(nodes, size)
    return list(nodes[:size])


def _split_context_ids(
    context_ids: list[str], split: tuple[float, float, float]
) -> dict[str, set[str]]:
    if len(split) != 3 or not math.isclose(sum(split), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split must contain three fractions summing to 1")
    n = len(context_ids)
    train_end = int(n * split[0])
    valid_end = train_end + int(split[1] * n)
    return {
        "train": set(context_ids[:train_end]),
        "validation": set(context_ids[train_end:valid_end]),
        "test": set(context_ids[valid_end:]),
    }


def build_overexposure_dataset(
    graph: nx.DiGraph,
    config: dict,
    split: tuple[float, float, float] = (0.7, 0.15, 0.15),
    *,
    oracle_mc: int | None = None,
    params: OverexposureParams | None = None,
) -> dict[str, list[OverexposureMarginalSample]]:
    """Build train/validation/test splits of signed marginal-gain labels.

    Contexts are the unit of splitting, and the split is applied *within* each seed-set size, so
    no test context shares a seed-set size distribution imbalance with training.
    """
    cfg = config.get("overexposure_dataset", {}) or {}
    params = params or resolve_overexposure_params(config)
    nodes = list(graph.nodes())
    n = len(nodes)
    budget = min(int(cfg.get("budget", config.get("seed", {}).get("budget", 3))), n)
    candidates_per_context = max(1, int(cfg.get("candidates_per_context", 20)))
    total_contexts = max(1, int(cfg.get("contexts", 60)))
    mc = int(oracle_mc if oracle_mc is not None else cfg.get("mc_runs", params.mc_runs))
    base_seed = int(cfg.get("random_seed", params.random_seed))
    probability = float(config.get("diffusion", {}).get("probability", 0.01))

    oracle = OverexposureMonteCarloOracle(
        graph, mc_runs=mc, random_seed=base_seed, params=params
    )
    rng = random.Random(base_seed)
    partials = _partial_seed_sequences(graph, budget, probability, nodes)
    max_seed_size = min(max(budget, 1), max(n - candidates_per_context, 0))

    contexts: list[tuple[str, list[OverexposureMarginalSample]]] = []
    for index in range(total_contexts):
        size = index % (max_seed_size + 1)
        seeds = _make_seed_set(graph, nodes, partials, size, index, rng)
        available = [v for v in nodes if v not in set(seeds)]
        if not available:
            continue
        candidates = rng.sample(available, min(candidates_per_context, len(available)))

        # The state a deployed policy observes: one cascade under the same windows.
        state = oracle.state(seeds, step=base_seed + index)
        base = oracle.spread(seeds)
        # ``score_with_uncertainty`` rather than ``score``: confound P1-3.7 was that label_std was
        # recorded as NaN, so a candidate difference was never compared against the error of
        # estimating it.  The paired standard error is available at no extra cost because the
        # per-trial differences are formed before averaging anyway.
        scored = oracle.score_with_uncertainty(seeds, candidates, step=base_seed + index)

        context_id = f"ctx_{index:05d}"
        samples = []
        for candidate in candidates:
            row = scored[candidate]
            gain = row["mean"]
            samples.append(OverexposureMarginalSample(
                context_id=context_id,
                seed_set=list(seeds),
                candidate=int(candidate),
                seed_set_size=len(seeds),
                marginal_gain=float(gain),          # SIGNED -- negatives are kept
                label_std=float(row["stderr"]),     # paired stderr, not NaN
                label_trials=int(row["n"]),
                base_spread=float(base["mean"]),
                extended_spread=float(base["mean"] + gain),
                exposure=list(state),
            ))
        contexts.append((context_id, samples))

    result: dict[str, list[OverexposureMarginalSample]] = {
        "train": [], "validation": [], "test": []
    }
    by_size: dict[int, list[tuple[str, list[OverexposureMarginalSample]]]] = {}
    for context in contexts:
        by_size.setdefault(context[1][0].seed_set_size, []).append(context)

    # Confound P1-3.6 alleged that splitting by context id lets identical seed sets straddle the
    # boundary.  Measured on this builder it does not: ``_make_seed_set`` gives each index a distinct
    # seed set and candidates are drawn without replacement, so every context is already a distinct
    # state (zero duplicates at 12/24/48/120 contexts).  The grouping below is therefore a *guard*,
    # not a repair: it makes "no state crosses the boundary" an invariant of this code instead of a
    # property that happens to hold, so changing the generator cannot silently introduce the leak.
    # No previously reported number changes because of it, and none should be claimed to.
    assigned: dict[str, str] = {}
    for size_contexts in by_size.values():
        rng.shuffle(size_contexts)
        # group context ids by their state fingerprint, so all copies of a state travel together
        groups: dict[tuple, list[str]] = {}
        for context_id, samples in size_contexts:
            fingerprint = (
                tuple(sorted(samples[0].seed_set)),
                tuple(sorted(s.candidate for s in samples)),
            )
            groups.setdefault(fingerprint, []).append(context_id)
        group_keys = sorted(groups)
        rng.shuffle(group_keys)
        split_ids = _split_context_ids([f"group_{i:05d}" for i in range(len(group_keys))], split)
        for index, key in enumerate(group_keys):
            for name, ids in split_ids.items():
                if f"group_{index:05d}" in ids:
                    for context_id in groups[key]:
                        assigned[context_id] = name
                    break

    for context_id, samples in contexts:
        result[assigned[context_id]].extend(samples)
    return result


def dataset_statistics(splits: dict[str, list[OverexposureMarginalSample]]) -> dict[str, Any]:
    """Report the signed-label shape, including the negative share, which must not be hidden.

    ``mean_label_std`` is reported next to ``mean_gain`` on purpose: a candidate difference is only
    interpretable against the error of estimating it, so the two belong in the same summary.
    """
    stats: dict[str, Any] = {}
    for name, samples in splits.items():
        if not samples:
            stats[name] = {"n": 0}
            continue
        gains = [s.marginal_gain for s in samples]
        negatives = sum(1 for g in gains if g < 0)
        stds = [s.label_std for s in samples if not math.isnan(s.label_std)]
        stats[name] = {
            "n": len(samples),
            "contexts": len({s.context_id for s in samples}),
            "mean_gain": sum(gains) / len(gains),
            "min_gain": min(gains),
            "max_gain": max(gains),
            "negative_share": negatives / len(gains),
            "mean_label_std": (sum(stds) / len(stds)) if stds else float("nan"),
            "label_std_is_recorded": len(stds) == len(samples),
        }
    return stats


def save_overexposure_dataset(
    splits: dict[str, list[OverexposureMarginalSample]], path: str | Path
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {name: [s.to_dict() for s in samples] for name, samples in splits.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def load_overexposure_dataset(path: str | Path) -> dict[str, list[OverexposureMarginalSample]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: [OverexposureMarginalSample(**item) for item in items]
        for name, items in payload.items()
    }
