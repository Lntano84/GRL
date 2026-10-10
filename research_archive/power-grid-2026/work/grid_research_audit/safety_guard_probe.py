"""Saved teacher-state probe of the original LJN rho_safe=0.9 guard.

No extra model, tunable constant, forecasts, physical trajectory or fitting.
The output cannot be called a deployment result or a whole-week counterfactual.
"""
import argparse
import json
import statistics
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid_research_audit"
SAFE = .9  # Original LJNAgentTopoNN default; not selected from this probe.
parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, choices=[32, 128], default=128)
args = parser.parse_args()
f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
rows = json.loads((ROOT / "outputs/grid21/per_state.json").read_text())
records = []
for row in rows:
    if row["budget"] != args.budget or row["method"] not in ["GNN_seed0", "GNN_seed1", "OLDNN", "MLP_seed0", "BIAS_seed0"]:
        continue
    number = row["state_id"].split("__")[-1]
    path = ROOT / "outputs/grid19/runs" / row["week"] / (number + ".json")
    c = json.loads(path.read_text())
    assert len(c["pool_ids"]) == len(c["outcomes"])
    outcomes = dict(zip(c["pool_ids"], c["outcomes"]))
    reps, seen = [], set()
    for i in c["pool_ids"]:
        h = outcomes[i]["combined_action_hash"]
        if h not in seen:
            seen.add(h)
            reps.append(i)
    shortlist = row["ordered_shortlist"]
    assert len(shortlist) == min(args.budget, len(reps))
    assert set(shortlist) <= set(reps)
    valid = [i for i in shortlist if outcomes[i]["strict_admissible"]]
    choice = max(valid, key=lambda i: (f32(outcomes[i]["reward"]), -i)) if valid else None
    all_valid = [i for i in reps if outcomes[i]["strict_admissible"]]
    full_choice = max(all_valid, key=lambda i: (f32(outcomes[i]["reward"]), -i)) if all_valid else None
    guard = choice is None or outcomes[choice]["rho"] >= SAFE
    delivered = full_choice if guard else choice
    records.append({"week": row["week"], "method": row["method"], "step": row["step"],
        "rho_safe": SAFE, "guard_expands_to_full": guard,
        "shortlist_choice": choice, "full_choice": full_choice, "guarded_choice": delivered,
        "exact_full_choice_retained": delivered == full_choice,
        "unique_queries": len(reps) if guard else len(shortlist), "full_queries": len(reps),
        "shortlist_rho": None if choice is None else outcomes[choice]["rho"]})
groups = []
for method in sorted({r["method"] for r in records}):
    selected = [r for r in records if r["method"] == method]
    groups.append({"method": method, "states": len(selected),
        "full_expansions": sum(r["guard_expands_to_full"] for r in selected),
        "exact_full_retention": sum(r["exact_full_choice_retained"] for r in selected)/len(selected),
        "query_reduction_sum": 1-sum(r["unique_queries"] for r in selected)/sum(r["full_queries"] for r in selected),
        "query_reduction_mean": statistics.mean(1-r["unique_queries"]/r["full_queries"] for r in selected)})
result = {"passed_engineering": bool(records), "budget": args.budget, "groups": groups, "records": records,
    "scope": "Post-hoc guard at original author's fixed safe threshold, on previously exposed saved teacher states. No causal whole-week effect, cost/survival, exact timing, new labels or models. Full ranking retained only as diagnostic reference.",
    "new_fits": 0, "new_forecasts": 0, "new_physical_steps": 0}
(OUT / f"SAFETY_GUARD_K{args.budget}_teacher_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"groups": groups, "records": len(records)}))
