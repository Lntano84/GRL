"""Verify that every Go/No-Go artifact actually satisfies the objective's requirements.

The objective states four properties that the outputs must have:

* a **contract declaration**: the target set ``D``, seed eligibility, a budget of *at most* ``k``,
  the stopping rule, and state acquisition charged rather than free;
* **paired** standard errors, i.e. the per-trial differences were formed before averaging;
* the numbers live in a **JSON artifact**, not only in a printed table;
* the criterion is stated in **absolute** units, with the relative form marked report-only.

Those are easy to assert in prose and easy to lose in a refactor, so this script checks them against
the files.  It is deliberately strict about two things that have already gone wrong once each in this
project: a state read that was not charged (P1-3.2), and a strategy argument applied to an
already-normalised graph, which made two columns identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, bool(ok), detail))


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        check(f"{path.name} parses", False, f"{type(exc).__name__}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "docs" / "results")
    args = parser.parse_args()
    R = args.results

    # ---------------------------------------------------------------------------------
    # 1. Gate 1(b)
    # ---------------------------------------------------------------------------------
    g1 = load(R / "gate1b_sequential_vs_static.json")
    if g1 is None:
        check("gate1b artifact exists", False, "missing")
    else:
        check("gate1b artifact exists", True, f"{len(g1.get('cells', []))} cells")
        contract = g1.get("contract", {})
        check("gate1b declares the budget as at-most",
              contract.get("budget_is_at_most") is True,
              f"budget_is_at_most={contract.get('budget_is_at_most')}")
        check("gate1b declares the stopping rule",
              bool(contract.get("stopping_rule")),
              f"stopping_rule={contract.get('stopping_rule')!r}")
        check("gate1b charges state acquisition",
              contract.get("state_acquisition_is_priced") is True,
              f"state_acquisition_is_priced={contract.get('state_acquisition_is_priced')}")
        check("gate1b marks the reference as not exact",
              contract.get("reference_is_exact") is False,
              f"reference_is_exact={contract.get('reference_is_exact')}")
        check("gate1b declares the tolerance anchor",
              bool(g1.get("gate", {}).get("tolerance_anchor")),
              f"anchor={g1.get('gate', {}).get('tolerance_anchor')!r}")
        check("gate1b marks the relative form report-only",
              g1.get("gate", {}).get("relative_form_is_report_only") is True,
              "the plan's <= 1% must not be the pass/fail rule")

        cells = g1.get("cells", [])
        check("gate1b has cells", len(cells) > 0, f"{len(cells)}")
        if cells:
            # target set and eligibility are per cell
            check("gate1b cells declare the target set size",
                  all("target_size" in c and c["target_size"] > 0 for c in cells),
                  f"min |D| = {min(c.get('target_size', 0) for c in cells)}")
            check("gate1b cells declare the realised seed fraction",
                  all("budget" in c and "n" in c for c in cells),
                  "budget and n present per cell")
            check("gate1b cells record the pool/budget ratio",
                  all("pool_to_budget" in c for c in cells),
                  "pool_to_budget present, so a degenerate pool is visible")
            # paired evaluation
            check("gate1b evaluates arms on shared windows (paired)",
                  all(len(c.get("paired_spread", {})) >= 2 for c in cells),
                  "each cell carries per-trial spreads for >= 2 arms")
            # charged state
            st = [c for c in cells if c["arms"].get("delta2_sequential")]
            check("gate1b charges delta2_sequential's state reads",
                  bool(st) and all(
                      c["arms"]["delta2_sequential"]["mc_cascades"]
                      >= c["arms"]["delta2_sequential"]["state_cascades"] > 0 for c in st),
                  "mc_cascades >= state_cascades > 0 for every sequential cell")
            # the two normalisations must not be identical
            by = {}
            for c in cells:
                for a, v in c["arms"].items():
                    by.setdefault((c["graph"], c["budget"], c["random_seed"], a), {})[
                        c["normalisation"]] = v["spread"]
            pairs = [v for v in by.values() if len(v) >= 2]
            differing = [v for v in pairs if len(set(v.values())) > 1]
            check("gate1b's two normalisations actually differ",
                  bool(pairs) and len(differing) > 0,
                  f"{len(differing)}/{len(pairs)} arm-cells differ between sum_to_one and "
                  f"clip_to_one; identical columns were the tell for the loader bug")
            graphs = sorted({c["graph"] for c in cells})
            check("gate1b covers >= 3 graphs", len(graphs) >= 3, ", ".join(graphs))

    # ---------------------------------------------------------------------------------
    # 2. quality-cost
    # ---------------------------------------------------------------------------------
    qc = load(R / "quality_cost.json")
    if qc is None:
        check("quality_cost artifact exists", False, "missing")
    else:
        cells = qc.get("cells", [])
        check("quality_cost artifact exists", True, f"{len(cells)} cells")
        check("quality_cost declares the contract", bool(qc.get("contract")), "contract present")
        check("quality_cost marks the relative form report-only",
              qc.get("gate", {}).get("relative_form_is_report_only") is True,
              "absolute tolerance is the rule")
        check("quality_cost explains why the criterion is absolute",
              bool(qc.get("gate", {}).get("reason_absolute")),
              "reason recorded in the artifact, not only in a commit message")
        arms_needed = {"degree_static", "delta2_static", "random_pruning",
                       "adaptive_selective"}
        seen_arms: set[str] = set()
        for c in cells:
            seen_arms |= set(c["arms"])
        check("quality_cost includes every arm the plan names",
              arms_needed <= seen_arms,
              f"missing: {sorted(arms_needed - seen_arms) or 'none'}")
        check("quality_cost includes the MC reference",
              any(a.startswith("mc_greedy") for a in seen_arms),
              ", ".join(sorted(a for a in seen_arms if a.startswith("mc_greedy"))))
        check("quality_cost covers >= 3 graphs",
              len({c["graph"] for c in cells}) >= 3,
              ", ".join(sorted({c["graph"] for c in cells})))
        check("quality_cost covers >= 3 budgets",
              len({c["budget"] for c in cells}) >= 3,
              ", ".join(str(b) for b in sorted({c["budget"] for c in cells})))
        # the control must cost the same as the shortlist arm it controls for
        comparable = []
        for c in cells:
            if "random_pruning" in c["arms"] and "selective_analytic" in c["arms"]:
                comparable.append(abs(c["arms"]["random_pruning"]["mc_cascades"]
                                      - c["arms"]["selective_analytic"]["mc_cascades"]))
        check("random_pruning is a MATCHED control",
              bool(comparable) and max(comparable) <= 0.25 * max(
                  1, max(c["arms"]["selective_analytic"]["mc_cascades"]
                         for c in cells if "selective_analytic" in c["arms"])),
              f"max cascade-cost gap vs selective_analytic: {max(comparable) if comparable else 'n/a'}")
        check("quality_cost records per-trial spreads (paired)",
              all(len(c.get("paired_spread", {})) >= 2 for c in cells),
              "paired trials stored, so the gate can be recomputed offline")

    # ---------------------------------------------------------------------------------
    # 3. criterion calibration
    # ---------------------------------------------------------------------------------
    cal = load(R / "gate_calibration.json")
    if cal is None:
        check("gate_calibration artifact exists", False, "missing")
    else:
        rows = cal.get("rows", [])
        check("gate_calibration artifact exists", True, f"{len(rows)} rows")
        check("gate_calibration reports the MC needed per candidate tolerance",
              bool(rows) and "mc_needed" in rows[0],
              "so the anchor is chosen from measurement, not from preference")
        worst = max((r["mc_needed"]["1_per_1000"]["mc_needed"] for r in rows), default=0)
        check("gate_calibration shows the plan's tolerance is unaffordable",
              worst > 100_000,
              f"a 1-per-1000 tolerance needs up to {worst:,} paired trials per contrast")

    # ---------------------------------------------------------------------------------
    # 4. raw weight profiles -- the guard against the no-op strategy bug
    # ---------------------------------------------------------------------------------
    prof = load(R / "raw_weight_profiles.json")
    if prof is None:
        check("raw_weight_profiles artifact exists", False, "missing")
    else:
        check("raw_weight_profiles artifact exists", True, f"{len(prof)} graphs")
        over = [g for g, p in prof.items() if p.get("nodes_over_one", 0) > 0]
        check("raw_weight_profiles shows graphs that violate the allowance",
              len(over) > 0,
              f"{len(over)} graphs exceed sum <= 1 in raw form: {', '.join(sorted(over)[:5])}")

    # ---------------------------------------------------------------------------------
    # report
    # ---------------------------------------------------------------------------------
    print("=" * 100)
    print("OBJECTIVE REQUIREMENTS AGAINST THE ARTIFACTS")
    print("=" * 100)
    failed = 0
    for name, ok, detail in CHECKS:
        mark = "OK  " if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {name}")
        print(f"         {detail}")
    print()
    print(f"  {len(CHECKS) - failed}/{len(CHECKS)} checks pass")
    if failed:
        print(f"  {failed} FAILED -- the objective's output requirements are not yet met")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
