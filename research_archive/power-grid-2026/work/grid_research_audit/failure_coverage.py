"""Post-hoc saved-feedback coverage diagnosis, not a simulated policy result."""
import gzip
import json
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/grid22"
OUT = ROOT / "outputs/grid_research_audit"
manifest = json.loads((SOURCE / "manifest.json").read_text())
by = {(r["week"], r["method"]): r for r in manifest}
f32 = lambda v: struct.unpack("f", struct.pack("f", float(v)))[0]


def calls(path):
    return [(p, json.loads(p.read_text())) for p in sorted(path.glob("call[0-9][0-9][0-9].json"))]


def base_bytes(call_path):
    path = call_path.with_name(call_path.stem + "_state.npz")
    with zipfile.ZipFile(path) as z:
        return z.read("base_action.npy")


results = []
for s in manifest:
    full = by.get((s["week"], "FULL"))
    if s["complete"] or not s["method"].startswith("GNN") or not full or not full["complete"]:
        continue
    fpath = SOURCE / "runs" / (s["week"] + "__FULL")
    apath = SOURCE / "runs" / (s["week"] + "__" + s["method"])
    full_calls = {r["state_hash"]: (p, r) for p, r in calls(fpath)}
    shared = []
    for p, a in calls(apath):
        if a["state_hash"] not in full_calls:
            continue
        fp, f = full_calls[a["state_hash"]]
        same_base = base_bytes(fp) == base_bytes(p)
        entry = {"step": a["step"], "same_observation": True, "same_base_vector": same_base,
                 "actual_chosen": a["chosen_pool_id"], "full_chosen": f["chosen_pool_id"]}
        if same_base:
            outcomes = {r["pool_id"]: r for r in f["outcomes"]}
            assert set(outcomes) == set(a["order"])
            all_valid = [i for i in a["order"] if outcomes[i]["strict_admissible"]]
            cutoff = a["order"][:128]
            valid = [i for i in cutoff if outcomes[i]["strict_admissible"]]
            # If no valid candidate exists, the existing policy would fully fall
            # back. This diagnostic uses saved full feedback, buying nothing.
            considered = cutoff if valid else f["representatives"]
            valid = valid if valid else all_valid
            best = max(valid, key=lambda i: (f32(outcomes[i]["reward"]), -i)) if valid else None
            chosen = a["chosen_pool_id"]
            entry.update({"full_chosen_rank_1based": None if f["chosen_pool_id"] is None else a["order"].index(f["chosen_pool_id"])+1,
                "saved_feedback_K128_canonical_choice": best,
                "saved_choice_predicted_rho": None if chosen is None else outcomes[chosen]["rho"],
                "hypothetical_choice_predicted_rho": None if best is None else outcomes[best]["rho"],
                "hypothetical_required_queries": len(considered),
                "warning": "Saved full feedback at a shared observation/base only; no actual K128 trajectory, continuous-controller-state equality, survival guarantee or causal failure attribution."})
        shared.append(entry)
    results.append({"week": s["week"], "method": s["method"], "actual_failed_step": s["steps"],
                    "full_complete": full["complete"], "shared_inputs": shared})
report = {"provisional": not (SOURCE / "audit.json").exists(), "results": results,
          "new_forecasts": 0, "new_fits": 0, "scope": "Failure diagnosis only; excludes policy-outcome claims."}
(OUT / "GRID22_failure_coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report))
