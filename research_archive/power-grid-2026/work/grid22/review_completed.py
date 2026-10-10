"""Lightweight provisional comparisons of completed episodes only, no fitting."""
import gzip
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid22"
rows = json.loads((OUT / "manifest.json").read_text())
by = {(r["week"], r["method"]): r for r in rows}
result = []
for r in rows:
    if r["method"] == "FULL": continue
    full = by.get((r["week"], "FULL"))
    if full is None: continue
    paths = [OUT / "runs" / (r["week"] + "__" + m) for m in ["FULL", r["method"]]]
    step_tables = []
    for path in paths:
        with gzip.open(path / "steps.jsonl.gz", "rt", encoding="utf-8") as f:
            step_tables.append([json.loads(line) for line in f])
    prefix = 0
    for a, b in zip(*step_tables):
        if any(a[k] != b[k] for k in ["before_hash", "action_hash", "after_hash", "raw_cost"]): break
        prefix += 1
    calls = []
    for path in paths:
        calls.append([json.loads(p.read_text()) for p in sorted(path.glob("call[0-9][0-9][0-9].json"))])
    shared = []
    bmap = {c["state_hash"]: c for c in calls[1]}
    for a in calls[0]:
        b = bmap.get(a["state_hash"])
        if b is None: continue
        def chosen_rho(c):
            return next((o["rho"] for o in c["outcomes"] if o["pool_id"] == c["chosen_pool_id"]), None)
        shared.append({"step": a["step"], "full_chosen": a["chosen_pool_id"], "alternative_chosen": b["chosen_pool_id"],
                       "full_predicted_rho": chosen_rho(a), "alternative_predicted_rho": chosen_rho(b),
                       "note": "Same observation hash only; no causal full-week attribution or internal-controller-state equality claim."})
    joint = full["complete"] and r["complete"]
    result.append({"week": r["week"], "method": r["method"], "complete": r["complete"], "steps": r["steps"],
                   "identical_physical_prefix_vs_full": prefix, "shared_observation_searches": shared,
                   "cost_gain_vs_full": (full["cost"]-r["cost"])/abs(full["cost"]) if joint else None,
                   "controller_time_reduction_vs_full": 1-r["controller_s"]/full["controller_s"] if joint else None})
output = {"provisional": True, "completed_runs": len(rows), "planned_runs": 28, "comparisons": result,
          "warning": "Developmental progress only, no verdict before full matrix and audit. No fitted or rollout choices depend on this analysis."}
(OUT / "development_progress_review.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps({"completed": len(rows), "last": result[-1] if result else None}))
