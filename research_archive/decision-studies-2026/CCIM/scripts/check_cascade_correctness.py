"""Is the cascade implementation correct?  Compare it against a brute-force reference.

Before concluding that the paper's numbers are unreachable, the implementation has to be beyond
suspicion.  The fast worklist ``cascade`` is checked against a deliberately naive
scan-until-nothing-changes reference on many small random graphs, over every seed subset of size 1..3
and several thresholds.
"""
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx

from ccim.model import cascade, load_graph, sigma


def brute_force_cascade(graph, seeds, K):
    """Naive reference: rescan every node until a full pass changes nothing."""
    active = set(int(s) for s in seeds)
    changed = True
    while changed:
        changed = False
        for v in graph.nodes():
            if v in active:
                continue
            active_neighbours = sum(1 for u in graph.neighbors(v) if u in active)
            if active_neighbours >= K:
                active.add(v)
                changed = True
    return active


def main() -> int:
    rng = random.Random(12345)
    mismatches = 0
    checks = 0
    for trial in range(60):
        n = rng.randint(6, 14)
        p = rng.uniform(0.15, 0.6)
        g = nx.gnp_random_graph(n, p, seed=rng.randint(0, 10**6))
        g.remove_edges_from(nx.selfloop_edges(g))
        for K in (1, 2, 3, 4):
            for size in (1, 2, 3):
                if size > n:
                    continue
                for seeds in itertools.combinations(range(n), size):
                    a = cascade(g, seeds, K)
                    b = brute_force_cascade(g, seeds, K)
                    checks += 1
                    if a != b:
                        mismatches += 1
                        if mismatches <= 5:
                            print(f"  MISMATCH n={n} K={K} seeds={seeds}: "
                                  f"worklist={sorted(a)} brute={sorted(b)}", flush=True)
    print(f"random-graph checks: {checks}, mismatches: {mismatches}", flush=True)

    # the two real graphs, checked on the same reference for a sample of seed sets
    real_checks = real_mismatch = 0
    for name in ("football", "polbooks"):
        g = load_graph(name)
        nodes = list(g.nodes())
        r = random.Random(7)
        for _ in range(400):
            size = r.randint(1, 12)
            seeds = r.sample(nodes, size)
            for K in (2, 3, 4, 5):
                real_checks += 1
                if cascade(g, seeds, K) != brute_force_cascade(g, seeds, K):
                    real_mismatch += 1
    print(f"real-graph checks: {real_checks}, mismatches: {real_mismatch}", flush=True)
    ok = mismatches == 0 and real_mismatch == 0
    print(f"CASCADE IMPLEMENTATION CORRECT: {ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
