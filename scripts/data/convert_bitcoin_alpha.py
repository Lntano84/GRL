"""Convert the Trust Bitcoin-Alpha signed trust network into a weighted edge list.

Source
------
Trust Bitcoin-Alpha, one of the four datasets used by the source model
(Inf. Sci. 744 (2026) 123375).  Distributed by SNAP as
``soc-sign-bitcoinalpha.csv.gz`` with raw size 24,186 rows in the format

    source, target, rating, timestamp

with ``rating in [-10, 10]`` (inclusive) and ratings concentrated on +1 (13,760 of 24,186 edges).

Why a conversion is needed
--------------------------
The threshold-window model is defined on a directed graph: exposure is
``delta(v) = sum_{u in A^in(v)} omega_uv``, a sum of in-edge weights.  The SNAP file gives signed
integer trust ratings, which are not probabilities.  We map them to ``[0, 1]`` by an affine
transform of the rating range,

    omega = (rating + 10) / 20,

so that ``rating = -10 -> 0.0``, ``rating = 0 -> 0.5``, ``rating = +10 -> 1.0``.  A zero weight
means the edge carries no influence; the edge is kept so the topology is unchanged, matching how
the rest of our pipeline treats weights.

Mean-degree caveat
------------------
The source model's Table 2 reports ``|V| = 3783`` and ``<k> = 6.39`` for this dataset.  Our
directed parse gives ``n = 3783`` (exact match) but ``m = 24186``, hence ``2m/n = 12.79``
directed.  Note that ``m/n = 6.39`` exactly, which is the mean degree one gets by treating the
graph as **undirected**.  The source model therefore appears to handle this dataset as
undirected, whereas we need a directed graph for the exposure sum.  We keep it directed and
report both numbers; the discrepancy is a definitional difference, not a data problem.
"""

from __future__ import annotations

import argparse
import gzip
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def convert(src: Path, dst: Path) -> dict:
    edges: list[tuple[int, int, float]] = []
    raw_ratings: Counter[int] = Counter()

    with gzip.open(src, "rt", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            source, target, rating = int(parts[0]), int(parts[1]), int(parts[2])
            raw_ratings[rating] += 1
            weight = (rating + 10) / 20.0
            edges.append((source, target, weight))

    nodes = {s for s, _, _ in edges} | {t for _, t, _ in edges}

    # Normalise in-weights to sum to 1 per node, which is the regime the model requires.
    #
    # Edge case: a node whose in-edges all carry rating -10 maps to weight 0 and therefore has
    # in-sum 0.  Scaling those edges leaves them at 0, so the node could never be activated by
    # any amount of influence.  That is an artefact of the affine map rather than a property of
    # the data, so such nodes fall back to uniform in-weights.  The alternative -- dropping the
    # node -- would change the graph the source model reports.
    in_sum: dict[int, float] = {v: 0.0 for v in nodes}
    in_degree: dict[int, int] = {v: 0 for v in nodes}
    for _, target, weight in edges:
        in_sum[target] += weight
        in_degree[target] += 1

    degenerate = sum(1 for v in nodes if in_degree[v] > 0 and in_sum[v] <= 0.0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(nodes)} {len(edges)}\n")
        for source, target, weight in edges:
            total = in_sum[target]
            if total > 0:
                normalised = weight / total
            else:
                normalised = 1.0 / in_degree[target] if in_degree[target] else 0.0
            handle.write(f"{source} {target} {normalised:.10f}\n")

    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "duplicate_pairs": len(edges) - len({(s, t) for s, t, _ in edges}),
        "self_loops": sum(1 for s, t, _ in edges if s == t),
        "rating_min": min(raw_ratings),
        "rating_max": max(raw_ratings),
        "rating_plus_one": raw_ratings.get(1, 0),
        "nodes_with_zero_in_weight_sum": degenerate,
        "mean_degree_undirected": len(edges) / len(nodes),
        "mean_degree_directed": 2 * len(edges) / len(nodes),
        "output": str(dst),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path,
                        default=ROOT / "data" / "paper" / "bitcoin-alpha.csv.gz")
    parser.add_argument("--dst", type=Path,
                        default=ROOT / "data" / "paper" / "bitcoin-alpha.txt")
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"missing source file: {args.src}")
    stats = convert(args.src, args.dst)
    for key, value in stats.items():
        print(f"  {key:<26} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
