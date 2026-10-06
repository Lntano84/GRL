"""Which interpretation of p(k) reproduces the paper's own numbers?

Running the paper's Greedy gives sigma 12 on Football against Table 1's 0.3217 (= 37 nodes), so my
first reading of Eq. (1) is not the paper's.  This script sweeps the plausible variants and reports
which one lands on the published values, before any method is built on top of it.

Variants tested
---------------
* ``count``   : node activates iff it has at least K ACTIVE neighbours                 (first reading)
* ``by_K``    : sigma for K = 1..8 under the count reading, to see what K fits the table
* ``stronger``: a 1-swap local search at the paper's K, to estimate how far above Greedy the true
               optimum sits -- if the optimum at K=4 is far below 37, the count reading is dead
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccim.model import DEFAULT_T, PAPER_TABLE1, degree_ranking, load_graph, sigma


def greedy(graph, T, K):
    chosen, current = [], 0
    for _ in range(T):
        best_v, best_sigma = None, current
        for v in graph.nodes():
            if v in chosen:
                continue
            s = sigma(graph, [*chosen, v], K)
            if s > best_sigma:
                best_v, best_sigma = v, s
        if best_v is None:
            break
        chosen.append(best_v)
        current = best_sigma
    return chosen, current


def local_search(graph, start, K, budget=4000):
    """Best-improvement 1-swap local search, for an upper-ish estimate of what is reachable."""
    best = list(start)
    best_val = sigma(graph, best, K)
    used, improved = 0, True
    while improved and used < budget:
        improved = False
        for i in range(len(best)):
            for v in graph.nodes():
                if v in best:
                    continue
                trial = list(best)
                trial[i] = v
                used += 1
                s = sigma(graph, trial, K)
                if s > best_val:
                    best, best_val, improved = trial, s, True
                if used >= budget:
                    break
            if used >= budget:
                break
    return best, best_val, used


def main() -> int:
    for name in ("football", "polbooks"):
        g = load_graph(name)
        n = g.number_of_nodes()
        target = PAPER_TABLE1[name]["Greedy"] * n
        print(f"=== {name} n={n}  paper Greedy = {PAPER_TABLE1[name]['Greedy']} "
              f"= {target:.1f} nodes ===", flush=True)
        print(f"  {'K':>3}{'greedy sigma':>14}{'normalized':>12}", flush=True)
        for K in range(1, 9):
            _, val = greedy(g, DEFAULT_T, K)
            mark = "   <-- matches paper" if abs(val - target) <= 2 else ""
            print(f"  {K:>3}{val:>14}{val / n:>12.4f}{mark}", flush=True)

        print(f"  local search at the paper's K=4 (1-swap, budget 4000):", flush=True)
        start, _ = greedy(g, DEFAULT_T, 4)
        best, val, used = local_search(g, start, 4)
        print(f"    reachable sigma {val} ({val / n:.4f}) after {used} evaluations", flush=True)
        print(f"    paper's RL4CCIM reports {PAPER_TABLE1[name]['RL4CCIM']} "
              f"= {PAPER_TABLE1[name]['RL4CCIM'] * n:.0f} nodes", flush=True)
        print(f"    degree top-8 sigma at K=4: {sigma(g, degree_ranking(g)[:DEFAULT_T], 4)}", flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
