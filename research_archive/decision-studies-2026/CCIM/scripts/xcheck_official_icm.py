"""Cross-check: run the authors' own IC objective and compare it with this repository's estimator.

A re-implementation that agrees with a *description* of the original proves very little.  This script
executes the published ``icm.py`` (from ``kage08/graph_sample_rl``) after two mechanical patches, and
compares its Monte-Carlo estimate of the influence objective against :class:`ccim.xplore.LiveEdgeObjective`
on identical seed sets.

The two patches, and nothing else
---------------------------------
1. ``from numba import jit`` -> a no-op decorator.  ``numba`` is not installed; ``jit`` is a pure speed
   annotation and removing it changes no arithmetic.
2. ``dtype=np.int`` -> ``dtype=int``.  ``np.int`` was removed in numpy 2.x.

Both are printed and recorded in the artifact.  If the two estimators disagree beyond Monte-Carlo error,
the reproduction's reward is not the published reward and every downstream number is void.

The comparison uses ``make_multilinear_objective_samples`` directly rather than their ``influence()``,
because ``influence()`` bundles greedy selection together with evaluation; here only the *estimator* is
under test, on fixed seed sets.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccim.xplore.icm import LiveEdgeObjective  # noqa: E402

REF = ROOT / "_ref"
OUTPUT = ROOT / "results" / "xcheck_official_icm.json"

PATCHES = [("from numba import jit", "_JIT_STUB = None")]


def load_patched(name: str, source: str, extra: str = ""):
    src = source.replace("from numba import jit", "def jit(*a, **k):\n"
                                                  "    def deco(f):\n"
                                                  "        return f\n"
                                                  "    return deco if not a or callable(a[0]) is False else a[0]\n")
    src = src.replace("dtype=np.int)", "dtype=int)")
    module = types.ModuleType(name)
    module.__dict__["np"] = np
    sys.modules[name] = module
    exec(compile(src, f"<official:{name}>", "exec"), module.__dict__)
    return module


def official_value(icm, graph, seeds, samples, seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    # the published influence() sets the propagation probability on the graph before sampling
    for u, v in graph.edges():
        graph[u][v]["p"] = 0.1
    live = icm.sample_live_icm(graph, samples)
    f_all = icm.make_multilinear_objective_samples(live, list(graph.nodes()), list(graph.nodes()),
                                                   np.ones(len(graph)))
    x = np.zeros(len(graph))
    x[list(seeds)] = 1
    return float(f_all(x))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    print("=" * 100)
    print("  CROSS-CHECK -- the authors' icm.py versus ccim.xplore.icm")
    print("=" * 100)

    icm_src = (REF / "gsrl_icm.py").read_text(encoding="utf-8", errors="ignore")
    patched = []
    if "from numba import jit" in icm_src:
        patched.append("numba.jit -> no-op decorator (numba not installed; speed annotation only)")
    if "dtype=np.int)" in icm_src:
        patched.append("np.int -> int (removed in numpy 2.x)")
    icm = load_patched("official_icm", icm_src)
    print("  patches applied to the published source:")
    for p in patched:
        print(f"    - {p}")
    assert len(patched) == 2, f"expected exactly 2 patches, applied {patched}"

    cases = []
    for name, g in [("karate", nx.karate_club_graph()),
                    ("gnp40", nx.gnp_random_graph(40, 0.2, seed=5))]:
        for seeds in ([0], [0, 1, 2], [3, 7, 11, 19]):
            cases.append((name, g, sorted(seeds)))

    rows = []
    print(f"\n  {'graph':<8}{'seeds':<18}{'authors':>12}{'ours':>12}{'rel.diff':>11}")
    worst = 0.0
    for name, g, seeds in cases:
        theirs = official_value(icm, g, seeds, args.samples, seed=12345)
        mine = LiveEdgeObjective(g, samples=args.samples,
                                 rng=np.random.default_rng(12345)).value(seeds)
        rel = abs(mine - theirs) / max(1e-9, theirs)
        worst = max(worst, rel)
        rows.append({"graph": name, "seeds": seeds, "authors": theirs, "ours": mine,
                     "relative_difference": rel})
        print(f"  {name:<8}{str(seeds):<18}{theirs:>12.4f}{mine:>12.4f}{rel:>10.2%}")

    # a generous but meaningful tolerance: two independent Monte-Carlo estimates of the same expectation
    tol = 0.08
    ok = bool(worst < tol)
    print(f"\n  worst relative difference {worst:.2%}  (tolerance {tol:.0%})")
    print(f"  [{'PASS' if ok else 'FAIL'}] the two estimators agree on the same seed sets")
    if not ok:
        print("  the reproduction's reward is not the published reward; downstream numbers are void")

    artifact = {"script": Path(__file__).name,
                "samples": args.samples, "patches": patched, "cases": rows,
                "worst_relative_difference": worst, "tolerance": tol, "pass": ok,
                "note": ("the published icm.py was executed directly; the only edits are the two "
                         "mechanical patches listed")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  artifact: {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
