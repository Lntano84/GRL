"""Freeze first closed-loop screening run; not a revision to GRID15 gate."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid16"
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / "design.json").exists()
weeks = ["2035-01-29_4", "2035-04-16_2", "2035-07-23_10", "2035-10-29_15"]
rules = ["FULL", "LOCAL128", "OLDNN128", "ZONE"]
order = [{"week": week, "rule": rules[(j+i)%len(rules)]} for i, week in enumerate(weeks) for j in range(len(rules))]
design = {"stage": "GRID16_CLOSED_LOOP_CHEAP_N1_SCREEN", "weeks": weeks, "rules": rules, "execution_order": order,
    "seed": 0, "controller": "Original NN20 in connected grid, unsafe QP, ALWAYS_RESTORE cadence6 remain; N1 shortlist only changes",
    "oldnn": "Same existing PPO rho-only model. Rank overlapping 280/909 pool actions by its logits; fill uncovered slots from LOCAL128, truncate128. No new fitting; preprocessing unchanged.",
    "fallback": "Original Greedy admissibility. If shortlist returns None and omits candidates, run ALL; reuse same state/base action/default horizon cache so each candidate gets at most one public forecast per N1 call.",
    "primary": "Completion first, then whole-week costs only jointly complete; total pipeline/controller wall and public forecast counts. Full must exactly reproduce GRID15 physical trajectory.",
    "simple_qualification": "No lost FULL-complete week, no more than one step earlier failure in FULL-failed week, mean cost increase <=1% on jointly complete weeks, >=30% macro N1 forecast reduction, >=10% macro controller-time reduction on jointly complete weeks. Development criterion only, no equivalence/significance inference.",
    "scope": "Distinct whole-policy question after GRID15 priority gate failed. No reinterpretation of that gate, no new trained models or sealed evaluation. A cheap winner becomes a baseline, not GRL credit.",
    "restrictions": ["No hidden future or source exhaustive scores in shortlist ranking", "Keep every version, failed attempt, and resource cost", "No physical reset after observing bad outcome", "No candidate extrapolation outside frozen library"],
    "caps": {"wall_s": 3600, "episode_s": 420, "physical_steps": 34000, "public_forecasts": 220000}}
(OUT / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
files = [ROOT / "work/grid16/rollout.py", ROOT / "work/grid16/freeze.py", ROOT / "work/grid16/inspect_pool.py",
    ROOT / "work/grid15/rules.py", ROOT / "work/grid08/common.py", OUT / "design.json", OUT / "pool_preflight.json"]
sealed = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(OUT / "code_freeze.json").write_text(json.dumps(sealed, indent=2), encoding="utf-8")
print("GRID16_FROZEN_NO_TRAINING")
