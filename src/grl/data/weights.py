"""In-edge weight normalisation, as an explicit and recorded choice.

Why this module exists
----------------------
Confound P1-3.8.  The model accumulates exposure as

    delta(v, t) = sum over u in A^in_t(v) of omega_uv

and requires the in-edge weights of a node to sum to **at most** 1: ``sum_u omega_uv <= 1``.  That
allowance is what makes ``delta`` an influence *share* rather than an unbounded count, and it is
what lets a node's window ``[theta_kappa_v, theta_tau_v]`` inside ``[0, 1]`` be reached at all.

Earlier scripts scaled every node's in-weights so they sum to **exactly** 1 and said nothing about
it.  That is a different model.  It is defensible --- the source model's own datasets ship weights
that already sum to 1 --- but it is a choice with a large effect: it sets the exposure scale, and
the exposure scale decides whether any node is over-exposed at all.  A reader who assumed the
model's ``<= 1`` allowance was respected would draw different conclusions from the same table.

So the choice is a named strategy here, it is recorded in the graph's metadata, and
:func:`verify_in_weight_allowance` refuses a graph the model forbids.  Any sweep that compares
strategies can then say which one produced a number.

The four strategies
-------------------
``sum_to_one``
    Scale each node's in-weights to sum to exactly 1.  The historical behaviour and the source
    model's own convention.  **This is a choice beyond what the model requires**, and a table
    produced with it must say so.
``clip_to_one``
    Leave weights alone unless a node's total exceeds 1, in which case scale that node's weights
    down to exactly 1.  This respects the model's allowance literally: it never invents weight that
    the data did not provide, so sparse or weak graphs keep a small ``delta`` and may show no
    overexposure --- which is an honest outcome, not a failure.
``uniform_share``
    Give every in-edge of a node the same weight ``1 / in_degree``.  Sums to exactly 1 and is
    independent of whatever magnitudes the file happened to carry.
``as_given``
    Use the file's weights unchanged, and raise if any node violates the allowance.  For datasets
    that already respect the model.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

#: Scale every node's in-weights to sum to exactly 1 (the historical behaviour).
SUM_TO_ONE = "sum_to_one"
#: Only scale down, and only when a node exceeds the model's allowance.
CLIP_TO_ONE = "clip_to_one"
#: Give every in-edge of a node the same share ``1 / in_degree``.
UNIFORM_SHARE = "uniform_share"
#: Use the file's weights unchanged; refuse a graph that violates the allowance.
AS_GIVEN = "as_given"

NORMALISATIONS = (SUM_TO_ONE, CLIP_TO_ONE, UNIFORM_SHARE, AS_GIVEN)

#: The metadata key under which the strategy actually applied is recorded on the graph.
METADATA_KEY = "in_weight_normalisation"


class WeightAllowanceError(ValueError):
    """Raised when a graph violates the model's ``sum_u omega_uv <= 1`` requirement."""


def in_weight_totals(graph: nx.DiGraph) -> dict[Any, float]:
    """Sum of in-edge weights per node."""
    totals: dict[Any, float] = {v: 0.0 for v in graph.nodes()}
    for _, v, data in graph.edges(data=True):
        totals[v] += float(data.get("weight", 0.0))
    return totals


def max_in_weight_total(graph: nx.DiGraph) -> float:
    totals = in_weight_totals(graph)
    return max(totals.values(), default=0.0)


def verify_in_weight_allowance(graph: nx.DiGraph, *, tolerance: float = 1e-9) -> None:
    """Raise if any node's in-weights exceed the model's allowance of 1.

    Called on every path that normalises, so a graph the model forbids cannot reach a measurement
    unnoticed.  ``tolerance`` absorbs floating-point drift from a scaling step.
    """
    totals = in_weight_totals(graph)
    offenders = {v: t for v, t in totals.items() if t > 1.0 + tolerance}
    if offenders:
        worst = max(offenders.values())
        sample = sorted(offenders.items(), key=lambda kv: -kv[1])[:3]
        raise WeightAllowanceError(
            f"{len(offenders)} node(s) receive more than the allowed total in-weight of 1 "
            f"(worst {worst:.6f}; e.g. {sample}).  The model requires sum_u omega_uv <= 1, so this "
            f"graph would make delta exceed 1 and no window could contain it."
        )


def normalise_in_weights(
    graph: nx.DiGraph,
    strategy: str = SUM_TO_ONE,
    *,
    copy: bool = False,
) -> nx.DiGraph:
    """Apply ``strategy`` to ``graph`` and record which one was applied.

    Parameters
    ----------
    strategy
        One of :data:`NORMALISATIONS`.
    copy
        When true, normalise a shallow copy and leave the input untouched.  Use this when the same
        loaded graph is measured under several strategies.

    The strategy is written to ``graph.graph[METADATA_KEY]`` so that any result derived from the
    graph can state how its exposure scale was set.  That is the point of the confound fix: the
    choice must be visible in the output, not inferred from the script that happened to run.
    """
    if strategy not in NORMALISATIONS:
        raise ValueError(f"unknown normalisation {strategy!r}; choose from {NORMALISATIONS}")

    target = graph.copy() if copy else graph
    if strategy == AS_GIVEN:
        verify_in_weight_allowance(target)
        target.graph[METADATA_KEY] = strategy
        return target

    if strategy == UNIFORM_SHARE:
        in_degree: dict[Any, int] = {v: 0 for v in target.nodes()}
        for _, v in target.edges():
            in_degree[v] += 1
        for _, v, data in target.edges(data=True):
            degree = in_degree[v]
            data["weight"] = 1.0 / degree if degree else 0.0
        verify_in_weight_allowance(target)
        target.graph[METADATA_KEY] = strategy
        return target

    totals = in_weight_totals(target)
    for _, v, data in target.edges(data=True):
        raw = float(data.get("weight", 0.0))
        total = totals[v]
        if total <= 0.0:
            data["weight"] = 0.0
        elif strategy == SUM_TO_ONE:
            data["weight"] = raw / total
        else:  # CLIP_TO_ONE
            data["weight"] = raw / total if total > 1.0 else raw

    verify_in_weight_allowance(target)
    target.graph[METADATA_KEY] = strategy
    return target


def describe_normalisation(graph: nx.DiGraph) -> dict[str, Any]:
    """What normalisation was applied, and the exposure scale it produced.

    Reported next to any table built from the graph, so the scale is visible rather than assumed.
    """
    totals = in_weight_totals(graph)
    values = sorted(totals.values())
    n = len(values)
    return {
        "strategy": graph.graph.get(METADATA_KEY, "unknown"),
        "max_in_weight_total": values[-1] if values else 0.0,
        "median_in_weight_total": values[n // 2] if values else 0.0,
        "mean_in_weight_total": (sum(values) / n) if n else 0.0,
        "nodes_at_or_above_half": sum(1 for t in values if t >= 0.5),
        "nodes_reaching_one": sum(1 for t in values if t >= 1.0 - 1e-9),
        "n": n,
    }
