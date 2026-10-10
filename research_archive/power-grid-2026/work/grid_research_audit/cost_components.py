"""Read saved completed trajectories only; no policies, fits or new forecasts."""
import argparse
import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, choices=[22, 23], default=22)
args = parser.parse_args()
SOURCE = ROOT / f"outputs/grid{args.stage}"
OUT = ROOT / "outputs/grid_research_audit"
OUT.mkdir(exist_ok=True)
summaries = json.loads((SOURCE / "manifest.json").read_text())
by = {(r["week"], r["method"]): r for r in summaries}
terms = {"losses_cost": "losses_mwh", "redispatch_cost": "redispatch_mwh",
         "storage_cost": "storage_throughput_mwh", "curtailment_delta_cost": "curtailment_delta_mwh"}
tables = {}
for s in summaries:
    if not s["complete"]:
        continue  # An early failure must never look like a cheaper complete week.
    path = SOURCE / "runs" / (s["week"] + "__" + s["method"])
    with gzip.open(path / "steps.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert all(r["ledger"] is not None for r in rows)
    totals = {name: math.fsum(r["ledger"]["marginal_cost"]*r["ledger"][field] for r in rows)
              for name, field in terms.items()}
    recomputed = math.fsum(totals.values())
    assert abs(recomputed-s["cost"]) <= math.fsum(r["ledger_tolerance"] for r in rows)
    tables[(s["week"], s["method"])] = {
        "summary": s, "totals": totals, "recomputed": recomputed,
        "curtailed_energy_mwh": math.fsum(r["ledger"]["curtailed_mwh"] for r in rows),
        "cost_rounding_residual": recomputed-s["cost"],
    }
comparisons = []
for (week, method), a in tables.items():
    if method == "FULL" or (week, "FULL") not in tables:
        continue
    b = tables[(week, "FULL")]
    delta = {name: a["totals"][name]-b["totals"][name] for name in terms}
    raw_delta = a["summary"]["cost"]-b["summary"]["cost"]
    assert abs(math.fsum(delta.values())-raw_delta) <= abs(a["cost_rounding_residual"])+abs(b["cost_rounding_residual"])+1e-6
    comparisons.append({"week": week, "method": method, "cost_difference_vs_full": raw_delta,
        "component_differences": delta,
        "curtailed_energy_difference_mwh": a["curtailed_energy_mwh"]-b["curtailed_energy_mwh"],
        "actual_search_calls": a["summary"]["n1_calls"]})
result = {"stage": args.stage, "completed_runs_available": len(summaries),
    "provisional": not (SOURCE / "audit.json").exists(), "comparisons": comparisons,
    "scope": "Descriptive decomposition of observed complete-week costs, not a causal effect of individual actions, training targets, or controller modules. Curtailment delta in the installed score is distinct from total curtailed energy. Failed weeks excluded only from cost comparisons, retained in original outputs.",
    "new_fits": 0, "new_forecasts": 0, "new_physical_steps": 0}
(OUT / f"GRID{args.stage}_cost_components.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"completed_runs": len(summaries), "comparisons": comparisons}))
