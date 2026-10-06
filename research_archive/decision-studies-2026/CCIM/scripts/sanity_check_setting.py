"""Sanity check: does the deterministic K=4, T=8 setting behave the way the paper describes?

Nothing here is a result.  It answers four questions before any method is built:

1. Are the graphs the ones the paper used, and is ``sigma`` deterministic?
2. Is the influence range in the paper's regime -- neither everything nor nothing, so the comparison
   is informative rather than trivial?
3. Is a full-cascade 8-seed set reachable at all, i.e. is the paper's normalized 1.0 on Football
   achievable rather than a typo?  Bounded search only: an exhaustive ``C(40, 8)`` sweep is 77M
   evaluations and was the reason the first version of this script had to be killed.
4. What does the paper's own solution filter ``A`` look like here?
"""
import itertools
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccim.model import (DEFAULT_K, DEFAULT_T, PAPER_TABLE1, degree_ranking, effective_candidates,
                        load_graph, sigma)

EXHAUSTIVE_POOL = 20      # C(20, 8) = 125,970 combinations, a few seconds
RANDOM_DRAWS = 20_000


def main() -> int:
    for name in ("football", "polbooks"):
        t0 = time.time()
        g = load_graph(name)
        n = g.number_of_nodes()
        nodes = list(g.nodes())
        print(f"=== {name}: n={n} m={g.number_of_edges()} ===", flush=True)

        # 1. determinism
        seeds = degree_ranking(g)[:DEFAULT_T]
        values = {sigma(g, seeds, DEFAULT_K) for _ in range(5)}
        print(f"  determinism over 5 repeats   : {values}", flush=True)

        # 2. range
        rng = random.Random(0)
        randoms = sorted(sigma(g, rng.sample(nodes, DEFAULT_T), DEFAULT_K)
                         for _ in range(RANDOM_DRAWS))
        deg = sigma(g, degree_ranking(g)[:DEFAULT_T], DEFAULT_K)
        print(f"  sigma(degree top-{DEFAULT_T})          : {deg}  ({deg / n:.4f} normalized)", flush=True)
        print(f"  random {DEFAULT_T}-seed range ({RANDOM_DRAWS}) : {randoms[0]} .. {randoms[-1]}"
              f"   median {randoms[len(randoms) // 2]}   ({randoms[len(randoms) // 2] / n:.4f})",
              flush=True)

        # 3. bounded search for a full cascade
        best, best_set = 0, None
        for combo in itertools.combinations(degree_ranking(g)[:EXHAUSTIVE_POOL], DEFAULT_T):
            s = sigma(g, combo, DEFAULT_K)
            if s > best:
                best, best_set = s, combo
                if best == n:
                    break
        print(f"  best over C({EXHAUSTIVE_POOL},{DEFAULT_T}) top-degree sweep : {best} "
              f"({best / n:.4f}){'  == |V|' if best == n else ''}", flush=True)
        if best < n:
            for _ in range(RANDOM_DRAWS):
                combo = tuple(rng.sample(nodes, DEFAULT_T))
                s = sigma(g, combo, DEFAULT_K)
                if s > best:
                    best, best_set = s, combo
                    if best == n:
                        break
            print(f"  best after {RANDOM_DRAWS} random draws      : {best} ({best / n:.4f})"
                  f"{'  == |V|' if best == n else ''}", flush=True)
        if best_set is not None and best == n:
            print(f"  a full-cascade set           : {sorted(best_set)}", flush=True)

        # 4. the paper's filter A
        A = effective_candidates(g, DEFAULT_K)
        print(f"  paper's effective candidates | {len(A)} of {n}", flush=True)

        ref = PAPER_TABLE1[name]
        print("  Table 1 reference (normalized): "
              + ", ".join(f"{k}={v}" for k, v in ref.items() if v is not None), flush=True)
        print(f"  ({time.time() - t0:.1f}s)\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
