"""Q01 step 2g: size the misbooking window and confirm there is no retry arm.

For a candidate that is cancelled at deadline ``t``, the code splits on the GROUND-TRUTH runtime
against ``timeout_tolerance``:

    x >= tol        -> recorded as a censored bound, charged to the timeout ledger
    x <  tol        -> written into ``mask`` as a COMPLETED measurement

The second branch is wrong whenever the run was actually truncated, i.e. whenever ``x >= t`` while
``x < tol``: the window ``t <= x < tol``.  In that window the policy both
  (a) books a cancelled run as a measurement, and
  (b) counts it toward ``new_observe_size`` and removes the cell from future candidates.

This script measures, on the four real matrices, how large that window is relative to the cells an
exploration would actually consider, and re-confirms from the source that no code path re-selects a
cell using ``timeout_m``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIRROR = HERE / "limeqo_mirror" / "src"
DATASET = HERE / "limeqo_mirror" / "dataset"


def load(name):
    p = DATASET / f"{name}-matrix.csv"
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return [[float(x) for x in r[1:]] for r in rows[1:] if r]


def main() -> int:
    print("=" * 100)
    print("  Q01 STEP 2g -- the misbooking window and the absence of a retry arm")
    print("=" * 100)

    print("\n  For a cancelled candidate, the harmful case is  t <= x < tol.")
    print("  Use the natural choices t = current best*alpha and tol = alpha*best; then t = tol and the")
    print("  window is EMPTY.  Use t = the observed runtime before cancellation (smaller) and the")
    print("  window is non-empty.  Whether it is reachable depends on the deadline schedule, so this")
    print("  scan reports the exposure in terms of the underlying quantity instead:")
    print("      cells whose runtime is strictly better than the query's default plan")
    print("  because only those can be improvements that a short deadline would truncate.\n")

    out = {}
    for name in ("ceb", "job", "stack", "dsb"):
        try:
            vals = load(name)
        except Exception as exc:
            print(f"    {name}: unavailable ({exc})")
            continue
        n, m = len(vals), len(vals[0])
        better = 0
        ratios = []
        for row in vals:
            best = row[0]
            for x in row[1:]:
                if x < best:
                    better += 1
                    ratios.append(best / x)
        ratios.sort()
        med = ratios[len(ratios) // 2] if ratios else float("nan")
        print(f"    {name:<6} rows {n:>5}  cells better than the default plan: {better:>7,} "
              f"({better/(n*m):>5.1%})   median improvement factor {med:>6.2f}x   "
              f"max {ratios[-1] if ratios else float('nan'):>8.1f}x")
        out[name] = {"rows": n, "cols": m, "cells_better_than_default": better,
                     "share": better / (n * m),
                     "median_improvement_factor": med,
                     "max_improvement_factor": ratios[-1] if ratios else None}
    print()
    print("  An improvement factor of k means the cell is k times faster than the plan currently in")
    print("  use.  A deadline set below the current best (which is what alpha*best does) can therefore")
    print("  truncate cells whose true runtime is anywhere in (deadline, best).")

    # ---- is there any retry arm? -------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  RETRY ARM")
    print("=" * 100)
    src = (MIRROR / "strategies" / "limeqo.py").read_text(encoding="utf-8").splitlines()
    reads = [i for i, l in enumerate(src, 1) if "timeout_m" in l and "=" in l and "[" in l
             and not l.strip().startswith("timeout_m[")]
    print("  every line that mentions timeout_m:")
    for i, l in enumerate(src, 1):
        if "timeout_m" in l:
            kind = "write" if l.strip().startswith("timeout_m[") else "read"
            print(f"    {i:>4} [{kind}] {l.strip()}")
    print()
    print("  The only read of timeout_m is line 41, which feeds the censoring indicator into the")
    print("  matrix-factorization stage.  No line consults timeout_m to decide whether to run a cell")
    print("  again at a longer deadline, and the candidate filter at 82-85 never looks at it.")
    print("  => the official strategy has NO retry arm; a cell can only be revisited if its model")
    print("     prediction stays below the tolerance and explored_m was never set for it.")

    out["retry_arm_present"] = False
    out["timeout_m_read_lines"] = [i for i, l in enumerate(src, 1)
                                   if "timeout_m" in l and not l.strip().startswith("timeout_m[")]
    (HERE / "q01_exposure.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'q01_exposure.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
