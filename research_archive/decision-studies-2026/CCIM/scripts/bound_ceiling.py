"""How high can sigma actually go at the paper's literal setting (K=4, T=8)?

The paper reports, for Football at K=4, T=8: Greedy 0.3217 (= 37 nodes), DPIM 0.3913 (= 45),
RL4IM 0.5130 (= 59), RL4CCIM 1.0000 (= 115, the whole network).  Greedy under the count reading of
Eq. (1) reaches 12 and a 1-swap local search 16, so this script estimates the CEILING with a much
stronger search.  If the ceiling is well below 37, then no method -- ours, theirs, or an oracle --
can produce Table 1's numbers in this setting, and the discrepancy is in the model or the
normalization rather than in anybody's algorithm.

The search is a randomised iterated local search with 1- and 2-swaps, run from many starts, so it is
a lower bound on the ceiling that is far tighter than plain Greedy.
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccim.model import DEFAULT_T, PAPER_TABLE1, degree_ranking, load_graph, sigma

EVAL_BUDGET = 400_000


def iterated_local_search(graph, T, K, budget, seed=0):
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    best, best_val, used = None, -1, 0

    def evaluate(seeds):
        nonlocal used
        used += 1
        return sigma(graph, seeds, K)

    starts = [tuple(degree_ranking(graph)[:T]),
              tuple(sorted(rng.sample(nodes, T)))]
    for _ in range(6):
        starts.append(tuple(sorted(rng.sample(degree_ranking(graph)[:40], T))))

    for start in starts:
        cur = list(start)
        cur_val = evaluate(cur)
        improved = True
        while improved and used < budget:
            improved = False
            order = list(range(T))
            rng.shuffle(order)
            for i in order:
                for v in rng.sample(nodes, len(nodes)):
                    if v in cur:
                        continue
                    trial = list(cur)
                    trial[i] = v
                    val = evaluate(trial)
                    if val > cur_val:
                        cur, cur_val, improved = trial, val, True
                        break
                if used >= budget:
                    break
            if improved:
                continue
            # one 2-swap pass before giving up on this start
            for _ in range(200 if used < budget else 0):
                i, j = rng.sample(range(T), 2)
                a, b = rng.sample(nodes, 2)
                if a in cur or b in cur:
                    continue
                trial = list(cur)
                trial[i], trial[j] = a, b
                val = evaluate(trial)
                if val > cur_val:
                    cur, cur_val, improved = trial, val, True
                    break
        if cur_val > best_val:
            best, best_val = list(cur), cur_val
        if used >= budget:
            break
    return best, best_val, used


def main() -> int:
    for name in ("football", "polbooks"):
        g = load_graph(name)
        n = g.number_of_nodes()
        ref = PAPER_TABLE1[name]
        print(f"=== {name} n={n}  T={DEFAULT_T} ===", flush=True)
        print(f"  paper: Greedy={ref['Greedy']} ({ref['Greedy'] * n:.0f})  "
              f"DPIM={ref['DPIM']} ({ref['DPIM'] * n:.0f})  RL4IM={ref['RL4IM']} "
              f"({ref['RL4IM'] * n:.0f})  RL4CCIM={ref['RL4CCIM']} ({ref['RL4CCIM'] * n:.0f})",
              flush=True)
        for K in (2, 3, 4, 5):
            t0 = time.time()
            best, val, used = iterated_local_search(g, DEFAULT_T, K, EVAL_BUDGET, seed=K)
            print(f"  K={K}: best 8-seed sigma found {val} ({val / n:.4f}) after {used} evaluations"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
