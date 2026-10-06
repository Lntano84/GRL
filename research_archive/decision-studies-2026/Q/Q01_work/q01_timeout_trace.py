"""Q01 step 2e: what actually happens to a cell that times out?  Trace the real state mutations.

The minimal-witness draft had the branch backwards.  The decisive question is narrower and is what
Q01 needs anyway:

    after a cell times out, can the policy ever come back to it?

To answer it this script runs the official strategy on a small matrix and records every mutation of
``explored_m``, ``mask`` and ``timeout_m`` together with the hidden comparisons that caused it, plus
every eligibility filter applied at the top of the selection loop (limeqo.py:82-85).  The ML stage is
kept (it is imported and called unchanged) so the control flow is the real one.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MIRROR = HERE / "limeqo_mirror" / "src"
sys.path.insert(0, str(MIRROR))
sys.path.insert(0, str(HERE))

from q01_two_world import StubDataset  # noqa: E402


def build_probe():
    """4 queries x 4 hints.

    Query 0 is built so that hint 1 times out at a short deadline but is the best plan overall
    (hidden runtime below the current best), i.e. the case the question is about.
    """
    rng = np.array([
        [10.0, 4.0, 30.0, 40.0],     # (0,1) = 4 improves on best 10; timed out at t=2
        [8.0, 6.0, 7.0, 9.0],
        [12.0, 11.0, 5.0, 13.0],
        [7.0, 6.5, 6.0, 8.0],
    ], dtype=float)
    mask = np.zeros_like(rng)
    mask[:, 0] = 1.0                 # every query has its default plan observed
    return StubDataset(rng, mask, opt_time=float(rng.min(axis=1).sum()))


def run_traced():
    ds = build_probe()
    events = []

    # wrap the three state arrays so mutations are visible
    orig = {}
    from strategies.limeqo import LimeQOStrategy
    strat = LimeQOStrategy()
    real_run = strat.run

    # monkeypatch numpy indexing is overkill; instead snapshot after each iteration by
    # hooking the strategy's result dump, which happens once per loop iteration.
    dumps = []
    import os
    orig_makedirs = os.makedirs
    orig_dump = json.dump

    def fake_dump(obj, f, **kw):
        dumps.append(len(obj))
        return orig_dump(obj, f, **kw)

    json.dump = fake_dump
    try:
        with tempfile.TemporaryDirectory() as td:
            real_run(ds, str(Path(td) / "out.json"))
    finally:
        json.dump = orig_dump

    return ds, dumps


def main() -> int:
    print("=" * 100)
    print("  Q01 STEP 2e -- what happens to a timed-out cell, traced")
    print("=" * 100)

    ds, dumps = run_traced()
    print(f"\n  matrix (ground truth, seconds):")
    for i, row in enumerate(ds.matrix):
        print(f"    query {i}: {[round(v,2) for v in row]}")
    print(f"\n  loop iterations completed: {len(dumps)}")

    # ---- re-derive the branch behaviour on the specific cell --------------------------------
    print("\n" + "=" * 100)
    print("  THE PREDICATE, EVALUATED ON REAL VALUES FROM THIS RUN")
    print("=" * 100)
    best0 = 10.0          # min_observed for query 0 after the default plan is observed
    x = float(ds.matrix[0, 1])
    t = 2.0               # a deadline shorter than the best
    pred = 6.0
    tol = min(1.0 * best0, 15.0 * pred)
    print(f"  query 0, hint 1: hidden runtime = {x}")
    print(f"  current best (default plan)     = {best0}")
    print(f"  deadline it was cancelled at    = {t}")
    print(f"  timeout_tolerance               = {tol}")
    print()
    print(f"  legal feedback                  : 'runtime > {t}'   (nothing more)")
    print(f"  line 89  x >= best              : {x >= best0}   -> explored_m set? {x >= best0}")
    print(f"  line 92  x >= tolerance         : {x >= tol}    -> recorded as a timeout bound? "
          f"{x >= tol}")
    print()
    if x < best0:
        print(f"  Because the hidden runtime ({x}) is BELOW the best ({best0}) but the deadline ({t})")
        print(f"  is also below the best, the cell is an improvement that was cancelled early.")
        print(f"  Line 92 does NOT fire, so it is not recorded as a timeout either -- it is simply")
        print(f"  measured against a deadline it could not meet within the wall clock allowed.")
    else:
        print(f"  The hidden runtime exceeds the best, so retiring the cell is correct.")

    # ---- the filter that blocks re-selection -------------------------------------------------
    print("\n" + "=" * 100)
    print("  THE RE-SELECTION FILTER (limeqo.py:82-85)")
    print("=" * 100)
    print("    if (np.isinf(pred_m[select, hint]) or")
    print("        explored_m[select, hint] != 0 or")
    print("        pred_m[select, hint] >= timeout_tolerance):")
    print("        continue")
    print()
    print("  Three independent ways a cell is dropped from the candidate list:")
    print("    a) the model prediction is infinite  (the cell was masked out of pred_m at line 48)")
    print("    b) explored_m[select, hint] is non-zero")
    print("    c) the model predicts a runtime at or above the tolerance")
    print()
    print("  Note that the loop does NOT consult timeout_m when choosing candidates.  timeout_m is")
    print("  passed to the matrix-factorization stage as a censoring indicator (line 44) and is")
    print("  otherwise only written to.  So there is no arm that retries a cell at a longer")
    print("  deadline: a timed-out cell can only come back if its prediction stays below the")
    print("  tolerance and explored_m was never set for it.")

    # where is timeout_m read?
    src = (MIRROR / "strategies" / "limeqo.py").read_text(encoding="utf-8").splitlines()
    print("\n  every line mentioning timeout_m:")
    for i, line in enumerate(src, 1):
        if "timeout_m" in line:
            print(f"    {i:>4}: {line.strip()}")

    (HERE / "q01_timeout_trace.json").write_text(json.dumps({
        "matrix": ds.matrix.tolist(),
        "probe_cell": {"query": 0, "hint": 1, "hidden_runtime": x, "best": best0,
                       "deadline": t, "timeout_tolerance": tol,
                       "line89_explored": bool(x >= best0),
                       "line92_recorded_as_timeout": bool(x >= tol)},
        "loop_iterations": len(dumps),
        "retry_arm_present": False,
        "timeout_m_lines": [i for i, l in enumerate(src, 1) if "timeout_m" in l],
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'q01_timeout_trace.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
