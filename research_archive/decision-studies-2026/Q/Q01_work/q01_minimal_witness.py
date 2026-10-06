"""Q01 step 2f: the correct minimal witness -- the DECISION TYPE depends on the hidden runtime.

What the policy legitimately knows when a run is cancelled at deadline ``t``:

    "the runtime is at least t"

What the load-bearing branches at limeqo.py:89 and :92 decide:

    line 89 : explored_m[select, same_hints] = 1     (retire the cell from future candidates)
    line 92 : timeout_m[select, same_hints] = tol    (record a censored bound)
              else -> mask[select, same_hints] = 1   (book the cancelled run as a MEASUREMENT)

Both comparisons use ``dataset.matrix[select, hint]``, the GROUND-TRUTH runtime.  The witness holds
every other input fixed and varies only that hidden value inside the band the legal feedback allows
(``x >= t``), then reports how the branch's outcome changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def branch_outcome(x, best, tol, deadline):
    """Transcribe limeqo.py:87-99 with the branching kept in source order."""
    same_hints = ["<hint>"]
    explored_write = None
    timeout_write = None
    mask_write = None

    # line 88-90
    if x >= best:
        explored_write = "explored_m[select, same_hints] = 1"
    # line 92-95
    if x >= tol:
        timeout_write = "timeout_m[select, same_hints] = tolerance ; timeout += tolerance"
        return {"explored": explored_write, "timeout_m": timeout_write, "mask": mask_write,
                "outcome": "recorded as a censored lower bound"}
    # line 97-100
    mask_write = "mask[select, same_hints] = 1 ; explored_m[select, same_hints] = 1"
    return {"explored": explored_write, "timeout_m": timeout_write, "mask": mask_write,
            "outcome": "booked as a MEASUREMENT of the runtime"}


def main() -> int:
    print("=" * 100)
    print("  Q01 STEP 2f -- MINIMAL WITNESS (decision type vs hidden runtime)")
    print("=" * 100)
    deadline = 2.0      # the run was cancelled here
    best = 10.0         # current best observed runtime for the query
    tol = 10.0          # timeout_tolerance = min(alpha*best, beta*pred)
    print(f"  fixed: deadline t = {deadline}, current best = {best}, timeout_tolerance = {tol}")
    print(f"  legal feedback in every row below: 'runtime >= {deadline}'")
    print()
    print(f"  {'hidden x':>9} {'x>=t':>6} {'x>=best':>8} {'x>=tol':>7} "
          f"{'explored_m written':>20} {'timeout_m written':>20} {'mask written':>10}  outcome")
    rows = []
    for x in (0.5, 1.9, 2.0, 4.0, 9.9, 10.0, 25.0):
        r = branch_outcome(x, best, tol, deadline)
        legal = x >= deadline
        rows.append({"x": x, "legal_visible": bool(legal), **r})
        print(f"  {x:>9} {str(legal):>6} {str(x >= best):>8} {str(x >= tol):>7} "
              f"{str(r['explored'] is not None):>20} {str(r['timeout_m'] is not None):>20} "
              f"{str(r['mask'] is not None):>10}  {r['outcome']}")

    print()
    print("  " + "-" * 96)
    print("  THE DEPENDENCE")
    print("  " + "-" * 96)
    print("  Rows with x >= t are the ONLY rows a deployable policy can be in: it was cancelled at")
    print("  t = 2, so anything at or above 2 is consistent with what it saw.  Among those rows:")
    print()
    legal_rows = [r for r in rows if r["legal_visible"]]
    kinds = {}
    for r in legal_rows:
        kinds.setdefault(r["outcome"], []).append(r["x"])
    for k, xs in kinds.items():
        print(f"    x in {xs}  ->  {k}")
    print()
    print("  The same legal observation ('runtime >= 2') therefore leads to two different")
    print("  bookkeeping decisions:")
    print("    * x in [2, 10)  -> the cancelled run is written into mask, i.e. treated as a")
    print("                       completed MEASUREMENT of the runtime;")
    print("    * x in [10, ..) -> it is written into timeout_m, i.e. treated as a censored BOUND.")
    print()
    print("  Both writes name the same cell in ``same_hints``.  A policy that only knows")
    print("  'runtime >= 2' cannot choose between them, and the split is made by reading the")
    print("  ground-truth runtime at limeqo.py:89 and :92.")

    print()
    print("  " + "-" * 96)
    print("  WHY THIS MATTERS FOR THE EXPLORATION BUDGET")
    print("  " + "-" * 96)
    print("  The two outcomes differ in what the run COST and what it BOUGHT:")
    print("    * as a bound, the run is charged to the timeout ledger (timeout_m[..] = tol) and the")
    print("      cell is not counted toward new_observe_size;")
    print("    * as a measurement, it is counted (cnt += 1) and the cell stops being a candidate.")
    print("  So the hidden value decides both the accounting and the candidate set.")

    print()
    print("  " + "-" * 96)
    print("  SCOPE")
    print("  " + "-" * 96)
    print("  ESTABLISHED: the two load-bearing branches read the ground-truth runtime, and two worlds")
    print("    consistent with the same legal feedback produce different bookkeeping and different")
    print("    candidate sets.")
    print("  ESTABLISHED: the same predicate shape appears in random.py:60, greedy.py:74,")
    print("    qo_advisor.py:68, limeqo_plus.py:114/117/139, and limeqo.py:114.")
    print("  NOT ESTABLISHED: that this reverses a published conclusion, or its effect size on the")
    print("    reported results.  That needs the replay rebuilt with the branch replaced.")

    (HERE / "q01_minimal_witness.json").write_text(json.dumps({
        "fixed_context": {"deadline_t": deadline, "best": best, "timeout_tolerance": tol},
        "legal_feedback": f"runtime >= {deadline}",
        "rows": rows,
        "decision_split_among_legal_rows": {k: v for k, v in kinds.items()},
        "established": [
            "limeqo.py:89 and :92 branch on the ground-truth runtime",
            "two worlds consistent with the same legal feedback produce different bookkeeping "
            "(mask vs timeout_m) and different candidate sets",
        ],
        "not_established": ["that published conclusions reverse",
                            "the effect size on reported results"],
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'q01_minimal_witness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
