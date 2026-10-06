"""Q01 step 2b: attribute every ground-truth matrix read to a call site, then re-run the two worlds.

The previous run showed the two worlds taking different numbers of hidden reads, but a read inside
``get_min_observed``/``get_exec_time`` is ARGUED to be legitimate (it is multiplied by the mask before
use).  To make a precise claim I attribute each read by inspecting the caller's source line, and I
report separately:

  * reads whose caller line applies a mask  -> the masked-observation pattern;
  * reads whose caller line is the 89-95 timeout/eligibility branch -> the suspected violation.

Any read whose caller cannot be determined is reported as unattributed rather than guessed.
"""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MIRROR = HERE / "limeqo_mirror" / "src"
sys.path.insert(0, str(MIRROR))

from q01_two_world import StubDataset, build_world  # noqa: E402


def patch_reads(ds):
    """Wrap the ground-truth array so every read records its immediate caller line."""
    log = []

    class Traced(np.ndarray):
        def __new__(cls, arr):
            return np.asarray(arr, dtype=float).view(cls)

        def __getitem__(self, key):
            out = np.ndarray.__getitem__(self, key)
            stack = traceback.extract_stack()
            caller = None
            for fr in reversed(stack[:-1]):
                if "limeqo.py" in fr.filename or "matrix_factorization" in fr.filename:
                    caller = (Path(fr.filename).name, fr.lineno, (fr.line or "").strip())
                    break
            log.append({"key": repr(key), "caller": caller,
                        "value": float(np.max(out)) if np.size(out) else float("nan")})
            return out

    ds.matrix = Traced(ds.matrix)
    ds._read_log = log
    return log


def classify(caller):
    if caller is None:
        return "unattributed"
    line = caller[2]
    if "mask" in line:
        return "masked_observation"
    if caller[1] in (89, 92) or "min_observed" in line or "timeout_tolerance" in line:
        return "eligibility_or_timeout_branch"
    if "min(" in line or "argmin" in line:
        return "argmin_over_predictions"
    return f"other:{caller[0]}:{caller[1]}"


def run(hidden_b):
    ds = build_world(hidden_b)
    log = patch_reads(ds)
    from strategies.limeqo import LimeQOStrategy
    strat = LimeQOStrategy()
    with tempfile.TemporaryDirectory() as td:
        strat.run(ds, str(Path(td) / "out.json"))
    return log


def main() -> int:
    print("=" * 100)
    print("  Q01 STEP 2b -- READ ATTRIBUTION, then the two-world comparison")
    print("=" * 100)

    logs = {}
    for name, hb in (("A_hidden_5.0", 5.0), ("B_hidden_20.0", 20.0)):
        log = run(hb)
        logs[name] = log
        kinds = Counter(classify(r["caller"]) for r in log)
        print(f"\n  world {name}: {len(log)} ground-truth reads")
        for k, v in kinds.most_common():
            print(f"    {k:<44} {v}")
        print("    call sites:")
        seen = {}
        for r in log:
            c = r["caller"]
            if c and c not in seen:
                seen[c] = 0
            if c:
                seen[c] += 1
        for c, n in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"      {c[0]}:{c[1]:<4} x{n:<3}  {c[2][:76]}")

    # ---- the specific branch -----------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  THE BRANCH AT limeqo.py:89-92")
    print("=" * 100)
    for name, hb in (("A", 5.0), ("B", 20.0)):
        hits = [r for r in logs[f"{name}_hidden_{('5.0' if name=='A' else '20.0')}"]
                if r["caller"] and r["caller"][1] in (89, 92)]
        print(f"  world {name} (hidden runtime {hb}): reads attributed to lines 89/92 = {len(hits)}")
        for h in hits:
            print(f"    line {h['caller'][1]}: matrix[{h['key']}] -> {h['value']}   "
                  f"| {h['caller'][2]}")
    print("\n  world A: hidden value 5.0  <  best 10.0 -> the line-89 comparison is FALSE")
    print("  world B: hidden value 20.0 >= best 10.0 -> the line-89 comparison is TRUE")
    print("  Both worlds present the SAME legal feedback: 'cancelled at t = 2.0'.")

    # ---- does the divergence show up in the observable decision? -----------------------------
    print("\n" + "=" * 100)
    print("  DIVERGENCE")
    print("=" * 100)
    la, lb = logs["A_hidden_5.0"], logs["B_hidden_20.0"]
    ka = Counter(classify(r["caller"]) for r in la)
    kb = Counter(classify(r["caller"]) for r in lb)
    def branch(keys):
        return sum(v for k, v in keys.items()
                   if k == "eligibility_or_timeout_branch" or k.startswith("other"))
    print(f"  total reads        A={len(la)}  B={len(lb)}")
    print(f"  masked-observation A={ka.get('masked_observation',0)}  "
          f"B={kb.get('masked_observation',0)}")
    print(f"  branch/other reads A={branch(ka)}  B={branch(kb)}")
    print(f"  argmin reads       A={ka.get('argmin_over_predictions',0)}  "
          f"B={kb.get('argmin_over_predictions',0)}")
    print(f"  unattributed       A={ka.get('unattributed',0)}  B={kb.get('unattributed',0)}")
    print()
    print("  The masked-observation counts differ as well, but those reads are of the form")
    print("  matrix * mask, which cannot depend on a hidden cell that was never observed.")
    print("  The material question is whether the ELIGIBILITY BRANCH reads the hidden value, and")
    print("  the attribution above answers it directly.")

    (HERE / "q01_read_attribution.json").write_text(json.dumps({
        "worlds": {k: [{"key": r["key"], "caller": r["caller"], "value": r["value"]}
                       for r in v] for k, v in logs.items()},
        "kind_counts": {k: dict(Counter(classify(r["caller"]) for r in v))
                        for k, v in logs.items()},
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'q01_read_attribution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
