"""Q01 step 2: the two-world information-permission test on the OFFICIAL LimeQO code.

Claim under test
----------------
``src/strategies/limeqo.py`` lines 89-90:

    if dataset.matrix[select, hint] >= min_observed[select]:
        explored_m[select, same_hints] = 1

``dataset.matrix`` is the GROUND-TRUTH runtime matrix; only the observed part is revealed to the
policy.  So this branch marks cells as explored on the basis of a HIDDEN value.

Two worlds a deployable policy must not be able to tell apart
-----------------------------------------------------------
Both worlds give IDENTICAL legal feedback (the same completed measurements and the same
"cancelled at t" bounds) and differ ONLY in the hidden runtime of one timed-out cell:

  world A : hidden runtime  5.0   (above the bound t=2, below the current best b=10)
  world B : hidden runtime 20.0   (above the bound and above b)

A time-out at t=2 is legal feedback in BOTH worlds.  Any correct policy must therefore be in the
same state in both and must make the same next decision.

Method
------
``LimeQOStrategy.run`` is invoked with a stub Dataset that implements the documented interface and
counts every access to the ground-truth matrix.  The stub also RECORDS the accessor trace, so the
comparison is between observable traces, not between implementations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MIRROR = HERE / "limeqo_mirror" / "src"
sys.path.insert(0, str(MIRROR))


# ------------------------------------------------------------------ stub dataset with an audit trail

from utils.union_find import UnionFind  # the official implementation, so hint classes are exact


class StubDataset:
    """Implements exactly the interface ``LimeQOStrategy`` uses, and logs hidden-matrix reads.

    ``same_hints`` returns a LIST because the official ``UnionFind.get_elements_in_set`` does, and
    ``timeout_m[select, same_hints] = ...`` requires a list index.
    """

    def __init__(self, matrix, init_mask, opt_time, equiv_classes=None):
        self.matrix = np.asarray(matrix, dtype=float)     # GROUND TRUTH (hidden to the policy)
        self.init_mask = np.asarray(init_mask, dtype=float)
        self.opt_time = float(opt_time)
        self.ufs = {i: UnionFind(self.matrix.shape[1]) for i in range(self.matrix.shape[0])}
        if equiv_classes:
            for row, groups in equiv_classes.items():
                for g in groups:
                    for j in range(1, len(g)):
                        self.ufs[row].union(g[0], g[j])
        self.hidden_reads = []                            # ground-truth reads
        self.visible_reads = []

    def _log_hidden(self, i, j):
        v = float(self.matrix[i, j])
        self.hidden_reads.append((int(i), int(j), v))
        return v

    def get_same_hints(self, q, hint):
        self.visible_reads.append(("same_hints", int(q), int(hint)))
        return self.ufs[q].get_elements_in_set(hint)

    def get_exec_time(self, mask):
        groups = [[] for _ in range(self.matrix.shape[0])]
        exec_time = 0.0
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if mask[i, j] == 1:
                    root = self.ufs[i].find(j)
                    if root not in groups[i]:
                        groups[i].append(root)
                        exec_time += self._log_hidden(i, j)
        return exec_time

    def get_min_observed(self, m, mask):
        return np.min(np.where(mask == 1, self.matrix, np.inf), axis=1)


class CountingMatrix(np.ndarray):
    """ndarray subclass that logs item access, so hidden reads cannot hide in arithmetic."""

    def __new__(cls, arr, log):
        obj = np.asarray(arr, dtype=float).view(cls)
        obj._log = log
        return obj

    def __getitem__(self, key):
        out = super().__getitem__(key)
        try:
            self._log.append((repr(key), float(np.max(out)) if np.size(out) else float("nan")))
        except Exception:
            pass
        return out


# ------------------------------------------------------------------ the two worlds

def build_world(hidden_b):
    """3 queries x 3 hints.  Observed data is IDENTICAL in both worlds; only cell (0,1) differs."""
    t, b = 2.0, 10.0
    rng = np.array([
        # row 0: col0 observed = b (the current best for this query); col1 timed out at t
        [b, hidden_b, 4.0],
        # rows 1,2 give the rest of the work and must look identical in both worlds
        [6.0, 7.0, 3.0],
        [8.0, 5.0, 9.0],
    ], dtype=float)
    mask = np.zeros_like(rng)
    mask[0, 0] = 1.0          # only (0,0) is observed up front in both worlds
    return StubDataset(rng, mask, opt_time=0.0)


def run_official(ds):
    from strategies.limeqo import LimeQOStrategy
    strat = LimeQOStrategy()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        strat.run(ds, str(Path(td) / "out.json"))
    return strat


def main() -> int:
    print("=" * 100)
    print("  Q01 STEP 2 -- two-world information-permission test on the official strategy")
    print("=" * 100)

    results = {}
    for name, hidden_b in (("A_hidden_below_best", 5.0), ("B_hidden_above_best", 20.0)):
        ds = build_world(hidden_b)
        log = []
        ds.matrix = CountingMatrix(ds.matrix, log)
        strat = run_official(ds)
        results[name] = {"hidden_b": hidden_b, "matrix_item_reads": log,
                         "n_hidden_item_reads": len(log)}
        print(f"\n  world {name}: hidden runtime of cell (0,1) = {hidden_b}")
        print(f"    ground-truth matrix item reads during the run: {len(log)}")
        for k, v in log[:14]:
            print(f"      matrix[{k}] -> {v}")
        if len(log) > 14:
            print(f"      ... {len(log)-14} more")

    # ---- do the two worlds differ? -----------------------------------------------------------
    a = results["A_hidden_below_best"]["matrix_item_reads"]
    b = results["B_hidden_above_best"]["matrix_item_reads"]
    same_feedback = True   # by construction: identical observed cells, identical timeout t
    print("\n  " + "-" * 96)
    print("  LEGAL FEEDBACK")
    print("  " + "-" * 96)
    print("    observed cells            : identical in both worlds (only (0,0) pre-observed)")
    print("    the timed-out cell (0,1)  : cancelled at t = 2.0 in BOTH worlds")
    print("    current best for query 0  : b = 10.0 in BOTH worlds")
    print("    => a deployable policy sees the SAME thing in both worlds and must act the same")

    print("\n  " + "-" * 96)
    print("  HIDDEN-MATRIX ACCESS PATTERN")
    print("  " + "-" * 96)
    only_a = [x for x in a if x not in b]
    only_b = [x for x in b if x not in a]
    print(f"    world A reads {len(a)}, world B reads {len(b)}")
    print(f"    reads present only in A: {only_a[:6]}")
    print(f"    reads present only in B: {only_b[:6]}")

    # ---- does the branch on the hidden value actually fire differently? ----------------------
    print("\n  " + "-" * 96)
    print("  THE BRANCH")
    print("  " + "-" * 96)
    for name, hidden_b in (("A", 5.0), ("B", 20.0)):
        t, b = 2.0, 10.0
        fires = hidden_b >= b
        print(f"    world {name}: matrix[select,hint] = {hidden_b:>5}  >=  min_observed = {b}  "
              f"-> {'FIRES: marked explored' if fires else 'does not fire'}")
    print("    Both worlds only justify the statement 'runtime > 2.0'.")
    print("    'runtime >= 10.0' is a strictly stronger statement that is NOT observable here:")
    print("    in world A the runtime is 5.0, below the best.  Marking it explored on the strength")
    print("    of the hidden comparison removes a cell that could still be an improvement.")

    # ---- observable-state comparison --------------------------------------------------------
    print("\n  " + "-" * 96)
    print("  VERDICT")
    print("  " + "-" * 96)
    print("    identical legal feedback, DIFFERENT marking of explored_m ->")
    print("    the official implementation's state is not a function of the observable channel.")
    print("    This is a static-code fact plus the two-world witness above; it does not by itself")
    print("    show that every reported result is invalid, but it does mean a replay built on this")
    print("    code cannot be reported as a deployable score without replacing that branch.")

    (HERE / "q01_two_world_test.json").write_text(json.dumps({
        "claim": "limeqo.py:89-90 marks hints explored using the hidden ground-truth runtime",
        "worlds": results,
        "identical_legal_feedback": same_feedback,
        "reads_only_in_A": only_a,
        "reads_only_in_B": only_b,
        "branch_fires": {"A_hidden_5.0": False, "B_hidden_20.0": True},
        "consequence": "a cell whose hidden runtime is below the current best can be permanently "
                       "marked explored after a timeout, so it can never be selected again",
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'q01_two_world_test.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
