"""Audit the existing search logs for the three things the next decision needs.

Deliverables being checked against:
  A. quality-vs-query curve at 100 / 300 / 1000 / 3000 / 5000 queries, five seeds
  B. query-purpose decomposition (initialisation, swaps, gain signs, duplicates, accepted)
  C. a learnable swap table (run seed, S, u, v, before, after, accepted, cumulative queries)

This reads ONLY the artifacts already on disk.  Nothing is re-run and **no checkpoint is
back-filled from a final result**: a value that was never recorded is reported as MISSING, and the
one piece of extra information that does exist -- how many queries a trajectory consumed before it
stopped improving -- is reported separately and labelled as termination metadata rather than a
checkpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "gate1": ROOT / "results" / "gate1_multistep_value.json",
    "gate2": ROOT / "results" / "gate2_shaping.json",
}
WANTED = (100, 300, 1_000, 3_000, 5_000)


def collect() -> dict:
    """Pull every (graph, seed, budget) -> normalized value that is actually recorded."""
    found: dict[str, dict[int, dict[int, float]]] = {}
    termination: dict[str, list[dict]] = {}

    g1 = json.loads(ARTIFACTS["gate1"].read_text(encoding="utf-8"))
    for name, entry in g1["graphs"].items():
        for row in entry.get("search_ladder", []):
            # gate1's ladder was run with a single search seed (0)
            found.setdefault(name, {}).setdefault(0, {})[row["budget"]] = row["normalized"]
            termination.setdefault(name, []).append(
                {"source": "gate1", "search_seed": 0, "budget": row["budget"],
                 "normalized": row["normalized"], "used": row["used"],
                 "exhausted": row["exhausted"]})

    g2 = json.loads(ARTIFACTS["gate2"].read_text(encoding="utf-8"))
    for name, entry in g2["graphs"].items():
        for row in entry.get("search", []):
            for per in row["per_seed"]:
                seed = per["search_seed"]
                found.setdefault(name, {}).setdefault(seed, {})[row["budget"]] = per["normalized"]
                termination.setdefault(name, []).append(
                    {"source": "gate2", "search_seed": seed, "budget": row["budget"],
                     "normalized": per["normalized"], "used": per["used"],
                     "exhausted": None})
    return {"found": found, "termination": termination}


def main() -> int:
    data = collect()
    found, termination = data["found"], data["termination"]

    print("=" * 100)
    print("  A. QUALITY-VS-QUERY CURVE AVAILABLE FROM EXISTING LOGS")
    print("=" * 100)
    for name in sorted(found):
        print(f"\n  {name}")
        header = "    search_seed |" + "".join(f"{b:>10}" for b in WANTED)
        print(header)
        for seed in sorted(found[name]):
            cells = []
            for b in WANTED:
                v = found[name][seed].get(b)
                cells.append(f"{v:>10.4f}" if v is not None else f"{'MISSING':>10}")
            print(f"    {seed:>11} |" + "".join(cells))
        # also show the budgets that DO exist, so nothing recorded is hidden
        budgets = sorted({b for s in found[name].values() for b in s})
        print(f"    recorded budgets: {budgets}")

    print("\n" + "=" * 100)
    print("  A'. TERMINATION METADATA (NOT a checkpoint -- how many queries were spent before the")
    print("      search stopped improving, recorded at run time)")
    print("=" * 100)
    for name in sorted(termination):
        seen = set()
        for row in termination[name]:
            key = (row["source"], row["budget"])
            if key in seen:
                continue
            seen.add(key)
            print(f"    {name:<10} {row['source']}  budget={row['budget']:>6}  "
                  f"used={row['used']:>6}  exhausted={row['exhausted']}  "
                  f"value={row['normalized']:.4f}")

    print("\n" + "=" * 100)
    print("  B / C. FIELDS THE EXISTING LOGS DO NOT CONTAIN")
    print("=" * 100)
    n_seeds = {name: len(found[name]) for name in found}
    print(f"    search seeds recorded: {n_seeds}  (five were run in gate2, one in gate1)")
    missing_b = [
        "initialisation queries (how many the degree/greedy warm start cost)",
        "swap queries (how many were spent on 1-swap and on 2-swap moves)",
        "counts of positive / zero / negative gain swaps",
        "duplicate-set queries and cache hits inside the search",
        "accepted swaps per trajectory",
    ]
    missing_c = [
        "run seed", "current set S", "swapped-out u", "swapped-in v",
        "value before the swap", "value after the swap", "accepted flag",
        "cumulative query count at that row",
    ]
    print("    B -- per-query-purpose decomposition:")
    for m in missing_b:
        print(f"        MISSING: {m}")
    print("    C -- learnable swap table rows:")
    for m in missing_c:
        print(f"        MISSING: {m}")
    print("\n    Reason: ccim/search.py returns only the final Result"
          " (chosen set, value, ledger counts).")
    print("    It keeps no per-move trace, so none of B or C can be recovered from disk.")

    print("\n" + "=" * 100)
    print("  WHAT THE LOGS *DO* SUPPORT (honest summary)")
    print("=" * 100)
    for name in sorted(found):
        s0 = found[name].get(0, {})
        v1000, v5000 = s0.get(1000), s0.get(5000)
        later = [v for b, v in sorted(s0.items()) if b > 5000]
        print(f"    {name}: seed 0 value {v1000 if v1000 is None else round(v1000, 4)} at 1,000 "
              f"queries, {v5000 if v5000 is None else round(v5000, 4)} at 5,000; "
              f"later budgets {[round(v, 4) for v in later]} -- "
              f"{'no further improvement after 5,000' if later and all(abs(v - v5000) < 1e-9 for v in later) else 'see values'}")
    print("\n    NOTE: the 100 / 300 / 3000 checkpoints were never recorded for any seed, so the")
    print("    curve cannot be reconstructed without re-running the search.  Per instruction this")
    print("    script does not do that; it reports the gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
