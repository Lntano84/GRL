"""N02 diagnostic: what does the Q_3 screen actually select?

Q_3 sums raw three-hop region sizes, so a large Q_3 could just mean "all four candidates sit in a
dense region" rather than "the two sides genuinely interact at three hops".  This checks the
distribution, the localisation of the selected quadruples, and a normalised alternative.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "N01_work"))
sys.path.insert(0, str(HERE))

from n01_common import Graph, parse_ca_grqc  # noqa: E402
from n02_common import reach_within  # noqa: E402

SAMPLING_SEED = 2026092802
MAX_ATTEMPTS = 100_000


def main() -> int:
    arcs, undirected, node_ids, _ = parse_ca_grqc()
    graph = Graph(arcs, undirected, node_ids)
    lcc = graph.components()[0]
    n = graph.n
    stage2 = json.loads((HERE / "N02_stage2_selection.json").read_text(encoding="utf-8"))
    selected = {tuple(sorted(c["quadruple"])) for c in stage2["comparisons"]}
    q_of = {tuple(sorted(c["quadruple"])): c["Q3"] for c in stage2["comparisons"]}

    r2 = {v: reach_within(graph, v, 2, include_source=True) for v in range(n)}
    r3 = {v: reach_within(graph, v, 3, include_source=True) for v in range(n)}

    rng = random.Random(SAMPLING_SEED)
    seen = set()
    rows = []          # (Q3, q3_normalised, a,b,c,d)
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        quad = tuple(rng.sample(lcc, 4))
        key = tuple(sorted(quad))
        if key in seen:
            continue
        seen.add(key)
        a, b, c, d = quad
        if (r2[a] | r2[b]) & (r2[c] | r2[d]):
            continue
        ac, bd = len(r3[a] & r3[c]), len(r3[b] & r3[d])
        ad, bc = len(r3[a] & r3[d]), len(r3[b] & r3[c])
        q3 = abs(ac + bd - ad - bc)
        denom = ac + bd + ad + bc
        qn = (q3 / denom) if denom else 0.0
        rows.append((q3, qn, a, b, c, d))

    print("=" * 100)
    print("  Q_3 DIAGNOSTIC")
    print("=" * 100)
    pos = [r for r in rows if r[0] > 0]
    print(f"  quadruples passing the original condition: {len(rows)}")
    print(f"  of those, Q_3 > 0: {len(pos)} ({len(pos)/len(rows):.2%})")
    qs = sorted(r[0] for r in pos)
    print(f"  Q_3 over the positive ones: min {qs[0]}, p25 {qs[len(qs)//4]}, "
          f"median {qs[len(qs)//2]}, p90 {qs[int(0.9*len(qs))]}, max {qs[-1]}")
    sel_q = sorted(q_of.values())
    print(f"  Q_3 of the 64 SELECTED        : min {sel_q[0]}, median {sel_q[len(sel_q)//2]}, "
          f"max {sel_q[-1]}")
    pct = sum(1 for q in qs if q <= sel_q[0]) / len(qs)
    print(f"  the selected minimum sits at the {pct:.2%} percentile of the positive pool")

    # how concentrated are the selected nodes?
    sel_nodes = [v for k in selected for v in k]
    all_nodes = [v for _, _, a, b, c, d in rows for v in (a, b, c, d)]
    from collections import Counter
    sel_cnt = Counter(sel_nodes)
    print(f"\n  distinct nodes appearing in the 64 selected quadruples: {len(sel_cnt)} "
          f"(out of {len(lcc)} in the LCC)")
    print(f"  most-reused nodes: {sel_cnt.most_common(10)}")
    sel_deg = statistics.fmean(graph.degree[v] for v in set(sel_nodes))
    pool_deg = statistics.fmean(graph.degree[v] for v in set(all_nodes))
    print(f"  mean degree of selected nodes {sel_deg:.2f} vs pool nodes {pool_deg:.2f}")

    # normalised screen: does it pick a different set?
    by_qn = sorted(rows, key=lambda t: (-t[1], (t[2], t[3], t[4], t[5])))[:64]
    top64_qn = {tuple(sorted((a, b, c, d))) for _, _, a, b, c, d in by_qn}
    overlap = len(top64_qn & selected)
    print(f"\n  top-64 by raw Q_3 vs top-64 by normalised Q_3: overlap {overlap}/64")
    print(f"  (a low overlap would mean the raw screen is dominated by region size, not by asymmetry)")

    out = {
        "n_passing_condition": len(rows),
        "n_q3_positive": len(pos),
        "q3_positive_fraction": len(pos) / len(rows),
        "q3_of_selected": {"min": sel_q[0], "median": sel_q[len(sel_q)//2], "max": sel_q[-1]},
        "selected_min_percentile": pct,
        "distinct_nodes_in_selection": len(sel_cnt),
        "most_reused_nodes": sel_cnt.most_common(10),
        "mean_degree_selected_nodes": sel_deg,
        "mean_degree_pool_nodes": pool_deg,
        "raw_vs_normalised_top64_overlap": overlap,
    }
    (HERE / "N02_q3_diagnostic.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'N02_q3_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
