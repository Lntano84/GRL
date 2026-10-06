"""Is a full cascade reachable at K=4 on Football at all?

The paper's RL4CCIM reports normalized influence 1.0 on Football at K=4, T=8, and says the setting is
"trivial" there.  Under the count reading of Eq. (1), Greedy reaches 12 and a 1-swap local search only
16.  Before building anything on the model, this script asks whether the ceiling itself is the problem:
raise the seed count far above the budget and see whether the threshold-4 process can take over the
network at all.

If even a large seed set plateaus far below |V|, then K=4 is above the percolation threshold for this
graph under the count reading, and the paper's 1.0 must come from a different mechanism -- which would
have to be identified before any comparison is meaningful.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccim.model import DEFAULT_T, degree_ranking, load_graph, sigma


def greedy_long(graph, T, K):
    """Greedy with a larger budget, to trace the ceiling rather than a single point."""
    chosen, current = [], 0
    for _ in range(T):
        best_v, best_sigma = None, current
        for v in graph.nodes():
            if v in chosen:
                continue
            s = sigma(graph, [*chosen, v], K)
            if s > best_sigma:
                best_v, best_sigma = v, s
        if best_v is None or best_sigma == current:
            break
        chosen.append(best_v)
        current = best_sigma
    return chosen, current


def main() -> int:
    for name in ("football", "polbooks"):
        g = load_graph(name)
        n = g.number_of_nodes()
        degs = sorted((d for _, d in g.degree()), reverse=True)
        print(f"=== {name} n={n}  degree min={degs[-1]} max={degs[0]} "
              f"avg={sum(degs)/n:.2f} ===", flush=True)
        for K in (2, 3, 4):
            trace = []
            for T in (DEFAULT_T, 12, 16, 20, 30, 40, 60):
                _, val = greedy_long(g, T, K)
                trace.append(f"T={T}:{val}({val/n:.3f})")
            full = greedy_long(g, n, K)
            print(f"  K={K}  ceiling trace  " + "  ".join(trace), flush=True)
            print(f"        K={K}  all {n} nodes as seeds -> sigma {full[1]}"
                  f" ({full[1]/n:.3f})", flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
