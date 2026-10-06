"""Q01 step 2c: pin the flips, find every instance of the pattern, and scan the real matrices.

Three things:
  1. A direct truth-table evaluation of the branch predicates at limeqo.py:89 / 92 / 114 over the
     hidden value, with every other quantity held at the value the two world example specifies.
  2. A source scan listing every place the strategy reads the ground-truth matrix without a mask.
  3. A scan of the four real matrices: how many cells are (runtime < current best) yet would time out
     early, i.e. how many cells this pattern can wrongly retire.  This counts observable situations,
     it does not run the strategy.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "limeqo_mirror" / "src"
DATASET = HERE / "limeqo_mirror" / "dataset"


def truth_table():
    print("=" * 100)
    print("  (1) BRANCH TRUTH TABLE -- every quantity fixed except the hidden runtime")
    print("=" * 100)
    b = 10.0            # current best observed runtime for the query
    t = 2.0             # the value the candidate was cancelled at (the deadline used)
    tol = min(1.0 * b, 15.0 * 6.0)      # alpha*b vs beta*pred, from the example
    print(f"  min_observed[select] = {b}   cancellation deadline t = {t}   "
          f"timeout_tolerance = {tol}")
    print()
    print(f"  {'hidden runtime':>14} {'t<=x (observable)':>18} {'line89 x>=b':>13} "
          f"{'line92 x>=tol':>14} {'legit branch':>14} {'official outcome':>26}")
    rows = []
    for x in (1.0, 2.0, 3.0, 5.0, 9.99, 10.0, 20.0):
        observable = x >= t                     # the ONLY thing feedback reveals
        l89 = x >= b
        l92 = x >= tol
        if not observable:
            legit = "cell still running"
            official = "not selected yet"
        elif x < b:
            legit = "is an improvement -> measure it"
            official = "marked explored, NEVER measured" if l89 else "measured (mask set)"
        elif x < tol:
            legit = "not an improvement, but under tolerance"
            official = "marked explored" if l89 else "measured"
        else:
            legit = "times out -> record a bound"
            official = "timeout recorded" if l92 else "treated as a measurement"
        print(f"  {x:>14} {str(observable):>18} {str(l89):>13} {str(l92):>14} "
              f"{legit:>14} {official:>26}")
        rows.append({"hidden": x, "observable": observable, "line89": l89, "line92": l92,
                     "legit": legit, "official": official})
    print()
    print("  In the example, t=2 and b=10, so the cell times out and (observably) is still running.")
    print("  Whether it is an improvement is decidable ONLY after the fact; x=5 is an improvement,")
    print("  x=20 is not, and both produce the single legal message 'runtime > 2'.")
    return rows


def source_scan():
    print("\n" + "=" * 100)
    print("  (2) SOURCE SCAN -- every unmasked read of the ground-truth matrix")
    print("=" * 100)
    hits = []
    for fname in ("strategies/limeqo.py", "strategies/limeqo_plus.py", "strategies/greedy.py",
                  "strategies/random.py", "strategies/oracle.py", "strategies/qo_advisor.py"):
        p = SRC / fname
        if not p.exists():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
                    and node.value.attr == "matrix":
                line = p.read_text(encoding="utf-8").splitlines()[node.lineno - 1].strip()
                masked = "mask" in line
                hits.append({"file": fname, "line": node.lineno, "code": line,
                             "masked_in_line": masked})
                print(f"  {fname}:{node.lineno:<4} {'[masked]' if masked else '[UNMASKED]':<11} "
                      f"{line[:78]}")
    unmasked = [h for h in hits if not h["masked_in_line"]]
    print(f"\n  unmasked reads of the ground-truth matrix: {len(unmasked)}")
    for h in unmasked:
        print(f"    {h['file']}:{h['line']}  {h['code'][:74]}")
    return hits


def matrix_scan():
    print("\n" + "=" * 100)
    print("  (3) REAL-DATA SCAN -- how many cells can this pattern wrongly retire?")
    print("=" * 100)
    print("  A cell (q,h) is exposed to the pattern when it would be cancelled at some deadline")
    print("  t < min_observed[q] while its true runtime is still below min_observed[q]; then a")
    print("  timeout marks it explored although it WAS an improvement.  Counted per query over the")
    print("  true matrix, as an upper bound on exposure (a real run only visits some cells).")
    out = {}
    for name in ("ceb", "job", "stack", "dsb"):
        p = DATASET / f"{name}-matrix.csv"
        if not p.exists() or p.stat().st_size == 0:
            continue
        with p.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        vals = [[float(x) for x in r[1:]] for r in rows[1:] if r]
        n, m = len(vals), len(vals[0])
        exposed = 0
        total_cells = n * m
        for i in range(n):
            row = vals[i]
            # min_observed is the best value seen so far; take the default plan as the starting best
            best = row[0]
            for j in range(1, m):
                x = row[j]
                if x < best:
                    # any deadline t with x < t < best would time out while x is an improvement
                    exposed += 1
        print(f"    {name:<6} rows {n:>5} cols {m} cells {total_cells:>8,}  "
              f"cells strictly better than the default plan in their own row: {exposed:>8,} "
              f"({exposed/total_cells:.1%})")
        out[name] = {"rows": n, "cols": m, "cells": total_cells,
                     "cells_below_default": exposed,
                     "share": exposed / total_cells}
    print()
    print("  These are exactly the cells that a deadline shorter than the current best would")
    print("  cancel; the official branch then decides their eligibility from the hidden value.")
    return out


def main() -> int:
    tt = truth_table()
    hits = source_scan()
    scan = matrix_scan()
    (HERE / "q01_branch_audit.json").write_text(json.dumps(
        {"truth_table": tt, "source_scan": hits, "matrix_scan": scan}, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'q01_branch_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
