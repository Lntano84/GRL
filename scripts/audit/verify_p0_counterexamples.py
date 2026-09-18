"""Verify the two P0 counterexamples before changing any model code.

The review raises two concrete counterexamples against the current implementation.  Both are
small enough to run exactly, with no Monte Carlo involved, so they can be settled rather than
argued about.

P0-1  non-seed nodes must be able to go positive -> negative, and the reported spread must count
      nodes that are positive at the END, not nodes that were ever positive.
      graph: s->a=0.4, s->b=0.5, b->a=0.4
      windows: a=[0.2,0.6], b=[0.2,0.9], seed={s}
      expected: round 1 a,b become positive; round 2 a's exposure reaches 0.8 > 0.6 so a goes
      negative; final positive = {s,b}, negative = {a}, spread should count 2 not 3.

P0-2  the tau = 1 boundary.  With the literal rule "kappa <= delta <= tau", delta = 1 and tau = 1
      is INSIDE the window and should activate.  The current code has a special guard
      ``current < 1.0`` that forces negative instead.
      graph: s->v=1, windows v=[0.2,1.0], seed={s}
      expected under the literal rule: v positive, spread = 2.
      current code reportedly: spread = 1, v negative.

Each check prints what the code does next to what the stated rule requires, so the discrepancy (if
any) is attributed to a specific line rather than to the model as a whole.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grl.diffusion import overexposure as oe  # noqa: E402


def report(title: str, run, expected_positive: set, expected_negative: set,
           expected_spread: int) -> bool:
    got_positive = set(run.positive)
    got_negative = set(run.negative)
    # "final positive" = ever positive minus those that turned negative
    final_positive = got_positive - got_negative
    ok_state = (final_positive == expected_positive and got_negative == expected_negative)
    ok_spread = (run.spread == expected_spread)
    print(f"  {title}")
    print(f"    ever_positive (run.positive)  = {sorted(got_positive)}")
    print(f"    negative      (run.negative)  = {sorted(got_negative)}")
    print(f"    FINAL positive                = {sorted(final_positive)}")
    print(f"    run.spread                    = {run.spread}")
    print(f"    expected final positive       = {sorted(expected_positive)}")
    print(f"    expected negative             = {sorted(expected_negative)}")
    print(f"    expected spread               = {expected_spread}")
    print(f"    -> state {'OK' if ok_state else 'MISMATCH'}, "
          f"spread {'OK' if ok_spread else 'MISMATCH'}")
    return ok_state and ok_spread


def check_p0_1() -> bool:
    print("P0-1  positive -> negative transition of a non-seed node, and end-state counting")
    graph = nx.DiGraph()
    graph.add_edge("s", "a", weight=0.4)
    graph.add_edge("s", "b", weight=0.5)
    graph.add_edge("b", "a", weight=0.4)
    windows = {"s": (0.0, 1.0), "a": (0.2, 0.6), "b": (0.2, 0.9)}
    run = oe.run_overexposure(graph, ["s"], windows, random.Random(0))
    return report("seed {s}", run, expected_positive={"s", "b"},
                  expected_negative={"a"}, expected_spread=2)


def check_p0_2() -> bool:
    print()
    print("P0-2  tau = 1 boundary: delta = 1 with tau = 1 is INSIDE the window")
    graph = nx.DiGraph()
    graph.add_edge("s", "v", weight=1.0)
    windows = {"s": (0.0, 1.0), "v": (0.2, 1.0)}
    run = oe.run_overexposure(graph, ["s"], windows, random.Random(0))
    return report("seed {s}", run, expected_positive={"s", "v"},
                  expected_negative=set(), expected_spread=2)


def main() -> int:
    print("=" * 88)
    print("P0 COUNTEREXAMPLE VERIFICATION")
    print("=" * 88)
    print()
    ok1 = check_p0_1()
    ok2 = check_p0_2()
    print()
    print("=" * 88)
    print(f"  P0-1 reproduced as a defect : {not ok1}")
    print(f"  P0-2 reproduced as a defect : {not ok2}")
    print("=" * 88)
    print()
    print("Also reporting the arithmetic behind P0-1 so the expected state is not taken on trust:")
    print("  round 1: s promotes a (delta_a = 0.4 in [0.2,0.6]) and b (delta_b = 0.5 in [0.2,0.9])")
    print("  round 2: b promotes a, delta_a = 0.4 + 0.4 = 0.8 > tau_a = 0.6  -> a must be negative")
    print("  final positive should be {s, b}; a must not be counted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
