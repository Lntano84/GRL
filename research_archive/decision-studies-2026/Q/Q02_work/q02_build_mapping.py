"""Q02 step 1: build the JOB hint-equivalence mapping from the EXPLAIN plans.

The mapping must be joined on the CSV row label (``filename``), not on sorted position.  This script:
  * extracts only ``filename`` and ``hint_list`` from each plan (no plan-tree work, no training);
  * joins to the matrix rows by label and reports any label that fails to match;
  * derives the equivalence classes the official code uses: hints whose hint_list contains the same
    SET of hint ids produce the same plan, so observing one reveals the whole class.
"""
from __future__ import annotations

import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MATRIX = ROOT / "Q01_work" / "limeqo_mirror" / "dataset" / "job-matrix.csv"
PLANS = HERE / "job.zip"


def main() -> int:
    print("=" * 100)
    print("  Q02 STEP 1 -- JOB equivalence mapping from EXPLAIN plans")
    print("=" * 100)

    # ---- matrix rows -------------------------------------------------------------------------
    with MATRIX.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    labels = [r[0] for r in rows[1:] if r]
    ncols = len(header) - 1
    print(f"  matrix: {len(labels)} rows, {ncols} hint columns (0..{ncols-1})")
    print(f"  first labels: {labels[:5]}")

    # ---- plan metadata -----------------------------------------------------------------------
    per_query = defaultdict(set)          # label -> set of frozenset(hint ids)
    n_plans = 0
    with zipfile.ZipFile(PLANS) as z:
        names = [n for n in z.namelist() if n.endswith(".json") and "__MACOSX" not in n]
        for nm in names:
            d = json.loads(z.read(nm))
            fn = d.get("filename")
            hl = d.get("hint_list")
            if fn is None or hl is None:
                continue
            n_plans += 1
            per_query[str(fn)].add(frozenset(int(x) for x in hl))
    print(f"  plan files read: {n_plans}")
    print(f"  distinct query labels in plans: {len(per_query)}")

    # ---- join by label -----------------------------------------------------------------------
    missing_in_plans = [l for l in labels if l not in per_query]
    extra_in_plans = [l for l in per_query if l not in set(labels)]
    matched = [l for l in labels if l in per_query]
    print(f"\n  JOIN ON LABEL")
    print(f"    labels matched           : {len(matched)} / {len(labels)}")
    print(f"    labels with no plan file : {len(missing_in_plans)} {missing_in_plans[:8]}")
    print(f"    plan labels not in matrix: {len(extra_in_plans)} {extra_in_plans[:8]}")

    # ---- sanity: does the number of hint ids per query match the column count? ---------------
    sizes = [len(v) for v in per_query.values()]
    print(f"\n  distinct hint SETS per query: min {min(sizes)}, max {max(sizes)}, "
          f"ncols = {ncols}")
    over = sum(1 for s in sizes if s > ncols)
    print(f"    queries with more hint sets than columns: {over}")

    # ---- equivalence classes -----------------------------------------------------------------
    # hints sharing an identical hint_list SET are one execution object.
    sig = {}
    for label, sets in per_query.items():
        for s in sets:
            sig[(label, s)] = s
    # For each query, map hint id -> the representative set it belongs to.
    # A hint id appears inside the hint_list of exactly one distinct set (verify).
    conflicts = 0
    mapping = {}
    for label, sets in per_query.items():
        owner = {}
        for s in sets:
            for hid in s:
                if hid in owner and owner[hid] != s:
                    conflicts += 1
                owner[hid] = s
        mapping[label] = owner
    print(f"\n  hint ids claimed by more than one distinct hint set: {conflicts}")
    class_counts = [len(set(v.values())) for v in mapping.values()]
    print(f"  classes per query: min {min(class_counts)}, max {max(class_counts)}")

    # how many hints are covered?
    cov = []
    for label in matched:
        cov.append(len(mapping[label]))
    print(f"  hint ids covered per matched query: min {min(cov)}, max {max(cov)} "
          f"(columns = {ncols})")

    # ---- write --------------------------------------------------------------------------------
    out = HERE / "job_equivalence.json"
    out.write_text(json.dumps({
        "matrix": str(MATRIX.name),
        "n_rows": len(labels), "n_cols": ncols,
        "labels": labels,
        "matched": len(matched),
        "missing_in_plans": missing_in_plans,
        "extra_in_plans": extra_in_plans,
        "classes_per_query": {l: sorted([sorted(s) for s in per_query[l]], key=lambda x: x[0])
                              for l in labels if l in per_query},
        "hint_sets_identical_imply_same_plan": True,
        "conflicts": conflicts,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out} ({out.stat().st_size/1e6:.2f} MB)")

    # ---- spot check: are hints within a class numerically identical in the matrix? -----------
    W = [[float(x) for x in r[1:]] for r in rows[1:] if r]
    idx = {l: i for i, l in enumerate(labels)}
    checked = ident = 0
    for label in matched:
        for s in per_query[label]:
            hs = sorted(s)
            if len(hs) < 2:
                continue
            vals = {W[idx[label]][h] for h in hs}
            checked += 1
            if len(vals) == 1:
                ident += 1
    print(f"\n  SPOT CHECK -- do hints in the same class have equal matrix values?")
    print(f"    multi-hint classes checked: {checked}; all-equal: {ident} ({ident/max(1,checked):.1%})")
    print(f"    (equality is NOT used to infer the classes; the classes come from hint_list)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
