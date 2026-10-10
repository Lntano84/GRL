"""Saved-feedback diagnostic only: no counterfactual rollout or new fitting.

Native reward is already saved as the exact promoted float32 value. Alternate
canonical-ID decisions here are hypothetical, not delivered policy scores.
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid22"
DEST = ROOT / "outputs/grid_research_audit"
DEST.mkdir(parents=True, exist_ok=True)
completed = json.loads((OUT / "manifest.json").read_text())
records = []
for episode in completed:
    directory = OUT / "runs" / (episode["week"] + "__" + episode["method"])
    for path in sorted(directory.glob("call[0-9][0-9][0-9].json")):
        call = json.loads(path.read_text())
        valid = [r for r in call["outcomes"] if r["strict_admissible"]]
        if valid:
            reward = max(r["reward"] for r in valid)
            tied = sorted(r["pool_id"] for r in valid if r["reward"] == reward)
            canonical = tied[0]
        else:
            tied, canonical = [], None
        chosen = call["chosen_pool_id"]
        assert chosen is None if not valid else chosen in tied
        with zipfile.ZipFile(path.with_name(path.stem + "_state.npz")) as archive:
            # Byte equality includes dtype/shape as well as all stored values;
            # no domain environment or heavy model needs to run during timing.
            base_bytes = archive.read("base_action.npy")
        import hashlib
        records.append({"week": episode["week"], "method": episode["method"],
                        "step": call["step"], "state_hash": call["state_hash"],
                        "saved_base_array_sha256": hashlib.sha256(base_bytes).hexdigest(),
                        "chosen": chosen, "best_reward_ids": tied,
                        "canonical_choice": canonical, "would_change": chosen != canonical,
                        "queries": len(call["outcomes"])})
summaries = []
for episode in completed:
    rows = [r for r in records if r["week"] == episode["week"] and r["method"] == episode["method"]]
    summaries.append({"week": episode["week"], "method": episode["method"],
                      "calls": len(rows), "best_reward_tie_calls": sum(len(r["best_reward_ids"]) > 1 for r in rows),
                      "canonical_would_change_calls": sum(r["would_change"] for r in rows)})
shared = []
full = {(r["week"], r["state_hash"]): r for r in records if r["method"] == "FULL"}
for r in records:
    reference = full.get((r["week"], r["state_hash"]))
    if r["method"] == "FULL" or reference is None:
        continue
    if reference["chosen"] in r["best_reward_ids"] and r["chosen"] != reference["chosen"]:
        shared.append({"week": r["week"], "method": r["method"], "step": r["step"],
                       "full_chosen": reference["chosen"], "alternative_chosen": r["chosen"],
                       "hypothetical_canonical_choice": r["canonical_choice"],
                       "same_observation_hash": True,
                       "same_saved_base_action": r["saved_base_array_sha256"] == reference["saved_base_array_sha256"],
                       "warning": "Internal optimizer memory not independently saved; no causal full-week claim."})
result = {"provisional": True, "completed_runs": len(completed),
          "scope": "Exact saved native reward ties only. No policy improvement or cost counterfactual measured.",
          "summaries": summaries, "shared_observation_tie_differences": shared, "records": records}
(DEST / "GRID22_saved_reward_ties.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"completed_runs": len(completed), "source_calls": len(records),
                  "tied_calls": sum(len(r["best_reward_ids"]) > 1 for r in records),
                  "hypothetical_canonical_changes": sum(r["would_change"] for r in records),
                  "shared_tie_differences": shared}))
