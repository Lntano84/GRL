"""Does the model reproduce the paper's Greedy number when Greedy is actually run?

The bounded random/top-degree sweep in the sanity check reached only 0.12--0.22 normalized, far below
Table 1's Greedy 0.3217 (Football) and 0.4571 (Polbooks).  That could mean the model is wrong, or it
could mean an 8-subset drawn at random almost never lands in the one community that cascades.  This
script settles it by running the algorithms the paper actually ran.

It also prints the community structure, because the motivating factor in complex contagion is that a
cluster of seeds inside one dense community is what crosses the threshold.
"""
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccim.model import DEFAULT_K, DEFAULT_T, PAPER_TABLE1, degree_ranking, load_graph, sigma


def greedy(graph, T, K):
    """The paper's Greedy baseline: at each step add the node with the largest marginal gain."""
    chosen, current = [], 0
    evaluations = 0
    for _ in range(T):
        best_v, best_gain, best_sigma = None, -1, current
        for v in graph.nodes():
            if v in chosen:
                continue
            s = sigma(graph, [*chosen, v], K)
            evaluations += 1
            if s - current > best_gain:
                best_v, best_gain, best_sigma = v, s - current, s
        chosen.append(best_v)
        current = best_sigma
    return chosen, current, evaluations


def main() -> int:
    for name in ("football", "polbooks"):
        t0 = time.time()
        g = load_graph(name)
        n = g.number_of_nodes()
        print(f"=== {name} n={n} m={g.number_of_edges()} ===", flush=True)

        # community labels, if the GML carries them
        raw = g.graph.get("node_attrs") or []
        communities = collections.Counter()
        for _, data in g.nodes(data=True):
            for key in ("value", "community", "group", "club"):
                if key in data:
                    communities[data[key]] += 1
                    break
        print(f"  community labels found: {dict(list(communities.items())[:14])}"
              f"{' ...' if len(communities) > 14 else ''}", flush=True)

        # what the paper's Greedy actually reaches
        ch, val, ev = greedy(g, DEFAULT_T, DEFAULT_K)
        print(f"  Greedy seeds {ch}", flush=True)
        print(f"  Greedy sigma {val}  ({val / n:.4f} normalized)   "
              f"paper says {PAPER_TABLE1[name]['Greedy']}   [{ev} sigma-evaluations]"
              f"   ({time.time() - t0:.1f}s)", flush=True)

        # seed densely inside one community, to test whether clustering is what crosses threshold
        best_comm = None
        for label in communities:
            members = [v for _, v in
                       [(u, u) for u, d in g.nodes(data=True)
                        if d.get("value", d.get("community")) == label]]
            if len(members) >= DEFAULT_T:
                members = sorted(members, key=lambda v: (-g.degree(v), v))
                s = sigma(g, members[:DEFAULT_T], DEFAULT_K)
                if best_comm is None or s > best_comm[1]:
                    best_comm = (label, s, members[:DEFAULT_T])
        if best_comm:
            print(f"  best single-community {DEFAULT_T}-seed set: community {best_comm[0]} -> "
                  f"sigma {best_comm[1]} ({best_comm[1] / n:.4f})", flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
