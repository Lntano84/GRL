"""Turn the Go/No-Go artifact into the quality--cost table and the gate verdicts.

Why a separate analyser
-----------------------
The runner's own summary prints while the sweep runs and only covers what exists at that moment.
The deliverable is a table read from the finished artifact, with the cost axis and the paired gate
verdicts computed the same way every time --- so a number in the paper can be regenerated rather
than transcribed, the same rule the regime table follows.

It also computes the one comparison the plan asks for and the runner does not: **which arms match
the reference within tolerance at a fraction of its cost**, in the absolute target-count units
:mod:`grl.evaluation.gate` requires.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grl.evaluation.gate import evaluate_gate, tolerance_from_target_fraction  # noqa: E402

ARM_ORDER = [
    "degree_static", "delta2_static", "delta2_sequential", "delta2_patience",
    "random_pruning", "selective_analytic", "adaptive_selective",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_gain(cell: dict, reference: str, method: str) -> tuple[float, float, int]:
    """Mean, standard error and n of the per-trial paired difference reference - method."""
    ref = cell.get("paired_spread", {}).get(reference)
    other = cell.get("paired_spread", {}).get(method)
    if not ref or not other or len(ref) != len(other):
        return float("nan"), float("nan"), 0
    diffs = [a - b for a, b in zip(ref, other)]
    n = len(diffs)
    mean = statistics.fmean(diffs)
    se = statistics.pstdev(diffs) / math.sqrt(n) if n > 1 else float("inf")
    return mean, se, n


def recompute_gate(cell: dict, reference: str, method: str, arm_entry: dict,
                   ref_entry: dict, tolerance: float, required_saving: float) -> dict:
    """Recompute the gate from the paired trials rather than trusting the stored row.

    The stored rows were produced by whatever version of :mod:`grl.evaluation.gate` was current
    when the cell was measured, so an artifact written before a semantics change carries verdicts
    under the old rules.  Recomputing makes the analysis reflect the criterion as it now stands and
    makes the table reproducible from the artifact alone.
    """
    ref = cell.get("paired_spread", {}).get(reference)
    other = cell.get("paired_spread", {}).get(method)
    if not ref or not other or len(ref) != len(other):
        return {}
    gains = [a - b for a, b in zip(ref, other)]
    verdict = evaluate_gate(
        graph=cell["graph"], budget=cell["budget"], reference=reference, method=method,
        paired_gains=gains,
        reference_spread=statistics.fmean(ref),
        method_spread=statistics.fmean(other),
        reference_cascades=ref_entry["mc_cascades"],
        method_cascades=arm_entry["mc_cascades"],
        abs_tolerance=tolerance,
        tolerance_anchor="target_fraction",
        required_saving=required_saving,
    )
    return verdict.as_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "docs" / "results" / "quality_cost.json")
    parser.add_argument("--tolerance-per-thousand", type=float, default=10.0,
                        help="absolute quality tolerance in target nodes per 1000 target nodes. "
                             "Default 10 rather than the plan's 1: the calibration shows 1 is "
                             "unresolvable at any affordable budget (see "
                             "docs/results/gate_calibration.json). 10 is the strictest tolerance "
                             "this design can actually decide.")
    parser.add_argument("--required-saving", type=float, default=0.30)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs" / "results" / "quality_cost_table.json")
    args = parser.parse_args()

    payload = load(args.input)
    cells = payload.get("cells", [])
    reference = payload.get("design", {}).get("reference_arm", "mc_greedy")
    if not cells:
        print("no cells in the artifact")
        return 1

    print("=" * 108)
    print("QUALITY--COST")
    print("=" * 108)
    print(f"  reference: {reference}   complete: {payload.get('complete')}   "
          f"cells: {len(cells)}")
    print()
    print("  'max loss' is the upper end of the paired 95% CI on (reference - method), in TARGET")
    print("  NODES.  It is the largest loss the data still permits, and it is comparable against")
    print("  any tolerance a reader considers substantive --- unlike a binary verdict, which")
    print("  depends on a tolerance someone chose.")
    print()
    print(f"  {'graph':<18}{'k':>4}{'arm':<22}{'spread':>10}{'cascades':>11}"
          f"{'saving':>9}{'mean':>9}{'max loss':>10}{'tol':>8}{'verdict':>12}")

    rows: list[dict] = []
    for cell in sorted(cells, key=lambda c: (c["graph"], c["budget"], c["random_seed"])):
        ref_arm = cell["arms"].get(reference)
        if not ref_arm:
            continue
        tolerance = tolerance_from_target_fraction(
            cell.get("target_size", 0) or 1, per_thousand=args.tolerance_per_thousand)
        for arm in ARM_ORDER:
            if arm not in cell["arms"]:
                continue
            entry = cell["arms"][arm]
            gate = recompute_gate(cell, reference, arm, entry, ref_arm,
                                  tolerance, args.required_saving)
            mean, se, n = paired_gain(cell, reference, arm)
            saving = gate.get("cascade_saving")
            print(f"  {cell['graph']:<18}{cell['budget']:>4}{arm:<22}"
                  f"{entry['spread']:>10.2f}{entry['mc_cascades']:>11}"
                  f"{(saving*100 if saving is not None else float('nan')):>8.0f}%"
                  f"{mean:>+9.2f}{gate.get('max_plausible_loss', float('nan')):>10.2f}"
                  f"{tolerance:>8.2f}{gate.get('verdict',''):>12}")
            rows.append({
                "graph": cell["graph"], "budget": cell["budget"],
                "seed": cell["random_seed"], "arm": arm,
                "spread": entry["spread"], "cascades": entry["mc_cascades"],
                # the state read is a breakdown of mc_cascades; both are reported
                "state_cascades": entry["state_cascades"],
                "reference_spread": ref_arm["spread"],
                "reference_cascades": ref_arm["mc_cascades"],
                "paired_mean_gain": mean, "paired_se": se, "n_pairs": n,
                "cascade_saving": saving,
                "abs_tolerance": tolerance,
                "max_plausible_loss": gate.get("max_plausible_loss"),
                "power_ratio": gate.get("power_ratio"),
                "verdict": gate.get("verdict"),
            })

    # ---------------- the two aggregates the plan asks for ----------------
    print()
    print("=" * 108)
    print("AGGREGATE 1: cost saving and the largest permitted loss, per arm")
    print("=" * 108)
    print("  The plan's criterion is '<= 1% quality loss AND >= 30% fewer online cascades'.  The")
    print("  cost half is a single number; the quality half is reported as the worst-case loss so")
    print("  that a reader can apply their own tolerance to it.")
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    print()
    print(f"  {'arm':<22}{'cells':>7}{'median saving':>15}{'worst max-loss':>16}"
          f"{'median max-loss':>17}{'PASS':>7}{'UNDEC':>7}")
    for arm in ARM_ORDER:
        group = by_arm.get(arm)
        if not group:
            continue
        savings = [g["cascade_saving"] for g in group if g["cascade_saving"] is not None]
        losses = [g["max_plausible_loss"] for g in group
                  if g["max_plausible_loss"] is not None]
        verdicts = [g["verdict"] for g in group if g["verdict"]]
        print(f"  {arm:<22}{len(group):>7}"
              f"{(statistics.median(savings)*100 if savings else float('nan')):>14.0f}%"
              f"{(max(losses) if losses else float('nan')):>16.2f}"
              f"{(statistics.median(losses) if losses else float('nan')):>17.2f}"
              f"{verdicts.count('PASS'):>7}{verdicts.count('UNDECIDED'):>7}")

    print()
    print("=" * 108)
    print("AGGREGATE 2: is the reference even the quality ceiling?")
    print("=" * 108)
    beaten = []
    for cell in cells:
        ref_arm = cell["arms"].get(reference)
        if not ref_arm:
            continue
        for arm in ARM_ORDER:
            if arm not in cell["arms"]:
                continue
            if cell["arms"][arm]["spread"] > ref_arm["spread"]:
                beaten.append((cell["graph"], cell["budget"], arm,
                               cell["arms"][arm]["spread"] - ref_arm["spread"],
                               cell["arms"][arm]["mc_cascades"], ref_arm["mc_cascades"]))
    if beaten:
        print(f"  {len(beaten)} arm-cells BEAT the reference on spread.  Examples:")
        for graph, budget, arm, delta, cost, ref_cost in beaten[:10]:
            print(f"    {graph:<18} k={budget:<4}{arm:<22} +{delta:6.2f} spread at "
                  f"{cost} cascades against the reference's {ref_cost}")
        print()
        print("  This is expected at a finite Monte-Carlo budget and it is why the reference's")
        print("  name carries --reference-mc: greedy on an estimator is neither exact nor globally")
        print("  optimal, so its spread is a reference point, not an upper bound.  Any claim of the")
        print("  form 'within X of optimal' must not be made against it.")
    else:
        print("  No arm beat the reference on spread in any cell.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "source": str(args.input.name),
        "reference": reference,
        "complete": payload.get("complete"),
        "rows": rows,
        "arms_beating_reference": len(beaten),
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
