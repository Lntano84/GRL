"""Saved choices at exactly shared public inputs; no counterfactual execution."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, choices=[22, 23], required=True)
args = parser.parse_args()
SOURCE = ROOT / f"outputs/grid{args.stage}"
DEST = ROOT / "outputs/grid_research_audit"
summaries = json.loads((SOURCE / "manifest.json").read_text())


def calls(summary):
    directory = SOURCE / "runs" / (summary["week"] + "__" + summary["method"])
    found = []
    for path in sorted(directory.glob("call[0-9][0-9][0-9].json")):
        c = json.loads(path.read_text())
        with zipfile.ZipFile(path.with_name(path.stem + "_state.npz")) as z:
            base = hashlib.sha256(z.read("base_action.npy")).hexdigest()
        found.append((c, base))
    return found


full = {}
for s in summaries:
    if s["method"] == "FULL":
        for c, base in calls(s):
            key = (s["week"], c["step"], c["state_hash"], base)
            assert key not in full
            full[key] = c

records = []
for s in summaries:
    if s["method"] == "FULL":
        continue
    shared = []
    for c, base in calls(s):
        f = full.get((s["week"], c["step"], c["state_hash"], base))
        if f is None:
            continue
        chosen, teacher = c["chosen_pool_id"], f["chosen_pool_id"]
        fa = {r["pool_id"]: r for r in f["outcomes"]}
        ca = {r["pool_id"]: r for r in c["outcomes"]}
        assert set(fa) == set(c["order"])
        feedback_agrees = all(
            all(r[k] == fa[i][k] for k in ["rho", "reward", "done", "exceptions", "illegal", "ambiguous", "strict_admissible"])
            for i, r in ca.items()
        )
        shared.append({"step": c["step"], "chosen": chosen, "full_chosen": teacher,
            "choice_differs": chosen != teacher,
            "full_chosen_rank_1based": None if teacher is None else c["order"].index(teacher)+1,
            "full_chosen_queried": teacher in ca,
            "feedback_agrees_on_queried_candidates": feedback_agrees,
            "rho": None if chosen is None else ca[chosen]["rho"],
            "full_rho": None if teacher is None else fa[teacher]["rho"],
            "reward": None if chosen is None else ca[chosen]["reward"],
            "full_reward": None if teacher is None else fa[teacher]["reward"]})
    records.append({"week": s["week"], "method": s["method"], "complete": s["complete"],
        "shared_call_count": len(shared), "shared_inputs": shared})

result = {"stage": args.stage, "provisional": not (SOURCE / "audit.json").exists(),
    "completed_runs_available": len(summaries), "records": records,
    "scope": "Matching step, observation hash and exact saved base action. No hidden optimizer-memory equality, counterfactual cost/survival, causal attribution, model fitting or new forecast.",
    "new_fits": 0, "new_forecasts": 0}
DEST.mkdir(exist_ok=True)
(DEST / f"GRID{args.stage}_shared_choice_coverage.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"stage": args.stage, "completed_runs": len(summaries),
    "first_shared_differences": [{"week": r["week"], "method": r["method"],
        "difference": next((c for c in r["shared_inputs"] if c["choice_differs"]), None)} for r in records]}))
