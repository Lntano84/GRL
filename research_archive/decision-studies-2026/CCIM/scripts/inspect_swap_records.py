"""What is in the replayed rows: label distribution and how far each frozen feature can rank on its own.

This is a *diagnostic*, not a tuning step.  It reads ``results/swap_records.json`` (or any artifact with
the same shape) and reports:

* the distribution of the true swap gain ``delta`` -- how rare an improving swap is;
* per-feature correlation with ``delta``;
* within-state pairwise ranking accuracy of each single feature, using the same metric that will select
  the model checkpoint.

Nothing here changes a configuration.  It answers a question the pilot report has to answer anyway:
*before* any online run, is there a signal in the frozen features at all, and is the class imbalance so
severe that a plain regression on ``delta`` is expected to collapse to a constant?  Publishing this
before the online comparison is the point -- it stops the online result from being read as a surprise in
either direction.

The imbalance warning is the important one.  The replayed trajectories stop at the first improving
candidate, so every state contributes exactly one positive row and ~34 negative rows.  A model trained
with an unweighted squared loss on that mix is dominated by the zeros.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ccim.learned_rank import build_dataset, pairwise_accuracy  # noqa: E402
from ccim.swap_features import FEATURE_NAMES  # noqa: E402
from measure_ca_grqc import FILE, build_undirected, parse_raw  # noqa: E402

RECORDS = ROOT / "results" / "swap_records.json"
OUTPUT = ROOT / "results" / "swap_records_inspection.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    raw = json.loads(args.records.read_text(encoding="utf-8"))
    K = raw["frozen"]["K"]
    rows, _ = parse_raw(FILE)
    graph = build_undirected(rows)

    print("=" * 112)
    print("  WHAT IS IN THE REPLAYED ROWS")
    print("=" * 112)

    report: dict[str, dict] = {}
    for seed, blob in raw["seeds"].items():
        ds = build_dataset({"graph": graph, "K": K, "state_sets": blob["state_sets"],
                            "rows": blob["rows"]})
        y, group = ds["y"], ds["group"]
        pos = y > 0
        n_states = len(np.unique(group))
        sizes = [int((group == g).sum()) for g in np.unique(group)]

        # per-feature Pearson correlation with delta, plus within-state pairwise accuracy
        corr, pw = {}, {}
        for j, name in enumerate(FEATURE_NAMES):
            col = ds["X"][:, j]
            corr[name] = float(np.corrcoef(col, y)[0, 1]) if col.std() > 0 else None
            pw[name] = pairwise_accuracy(col, y, group)["pairwise_accuracy"]
        best_single = max((v for v in pw.values() if v is not None), default=None)

        report[seed] = {
            "rows": int(len(y)), "states": n_states,
            "positives": int(pos.sum()), "positive_rate": float(pos.mean()),
            "delta_unique": {str(int(v)): int((y == v).sum())
                             for v in np.unique(y)[:12]},
            "delta_max": float(y.max()), "delta_min": float(y.min()),
            "delta_mean_positive": float(y[pos].mean()) if pos.any() else None,
            "candidates_per_state": {"min": int(min(sizes)), "max": int(max(sizes)),
                                     "mean": float(np.mean(sizes))},
            "feature_pearson_with_delta": corr,
            "feature_pairwise_accuracy_alone": pw,
            "best_single_feature_pairwise": best_single,
        }

        print(f"\n  seed {seed}: {len(y):,} rows / {n_states} states"
              f"   positive rows {int(pos.sum())} ({100 * pos.mean():.2f}%)")
        print(f"    delta range      : {y.min():.0f} .. {y.max():.0f}"
              f"   mean of positives {y[pos].mean():.2f}" if pos.any() else "")
        print(f"    rows per state   : min {min(sizes)}  max {max(sizes)}  mean {np.mean(sizes):.1f}")
        print(f"    {'feature':<22} {'pearson(delta)':>15} {'pairwise acc alone':>19}")
        for name in FEATURE_NAMES:
            c = corr[name]
            p = pw[name]
            print(f"    {name:<22} {('%.4f' % c) if c is not None else 'n/a':>15}"
                  f" {('%.4f' % p) if p is not None else 'n/a':>19}")
        print(f"    -> best single frozen feature ranks at {best_single:.4f} pairwise"
              f" (0.5 = no signal)")

    artifact = {"script": Path(__file__).name,
                "scope": "diagnostic only; no configuration is chosen here",
                "records_file": args.records.name,
                "frozen": raw["frozen"], "per_seed": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
