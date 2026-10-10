"""Freeze equal-role closed-loop evaluation after ranking engineering checks."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid22"
assert json.loads((OUT / "preflight.json").read_text())["passed"]
assert json.loads((ROOT / "outputs/grid21/audit.json").read_text())["passed"]
q = json.loads((ROOT / "outputs/grid21/comparison.json").read_text())
assert q["preserves_base_screen"] and q["extra_vs_old_screen"]
assert not (OUT / "design.json").exists()
weeks = json.loads((ROOT / "outputs/grid21/design.json").read_text())["development_weeks"]
methods = [
    {"method": "FULL", "rule": "FULL", "budget": None, "checkpoint": None},
    {"method": "OLDNN32", "rule": "OLDNN", "budget": 32, "checkpoint": None},
    {"method": "OLDNN128", "rule": "OLDNN", "budget": 128, "checkpoint": None},
    {"method": "GNN0_32", "rule": "GNN_seed0", "budget": 32, "checkpoint": "outputs/grid21/models/gnn_seed0/final.pt"},
    {"method": "GNN1_32", "rule": "GNN_seed1", "budget": 32, "checkpoint": "outputs/grid21/models/gnn_seed1/final.pt"},
    {"method": "MLP0_32", "rule": "MLP_seed0", "budget": 32, "checkpoint": "outputs/grid21/models/mlp_seed0/final.pt"},
    {"method": "BIAS0_32", "rule": "BIAS_seed0", "budget": 32, "checkpoint": "outputs/grid21/models/bias_seed0/final.pt"},
]
order = []
for i, week in enumerate(weeks):
    rest = methods[1:]; rest = rest[i:]+rest[:i]
    order += [{"week": week, **m} for m in [methods[0]]+rest]
d = {"stage": "GRID22_PRIOR_V1_CLOSED_LOOP_DEVELOPMENT", "weeks": weeks, "methods": methods, "execution_order": order,
     "environment_seed": 0, "new_model_fits": 0, "caps": {"wall_s": 7200, "episode_s": 600, "physical_steps": 60000, "public_forecasts": 1000000},
     "role": "Change only actual disconnected-grid topology search; original PPO other stages, unsafe QP and ALWAYS_RESTORE cadence6 unchanged. Author N1 label is not exhaustive N-1 contingency safety.",
     "fairness": "Every comparator gets same known endpoint mask, cooldown pool, exact full-vector canonical aliases, strict admission including illegal/ambiguous rejection, and same-state full fallback with no duplicate forecasts. FULL is a stronger engineered complete baseline; its physical trajectory must reproduce GRID19 exactly.",
     "primary": "Complete/survival first, then cost only on jointly completed full weeks, total controller walltime including features/prior/forward/forecast, and public query count. A failed shorter episode is never rewarded for lower cumulative cost.",
     "selection": "All four existing development weeks, both GNN fitting seeds; seed0 fixed for MLP/bias comparators before closed-loop results. No checkpoint/horizon/hyperparameter change or best-week choice. Strong original PPO included at both32 and128.",
     "investment_screen": {
         "auxiliary_role": "Against FULL, no lost complete week or earlier failure by>1step, each jointly-complete-week cost regret<=1%, mean controller walltime reduction>=10% and N1 query reduction>=30%. Both GNN seeds separately. Development resource screen, not statistical guarantee.",
         "graph_increment": "After survival parity, average both GNN seeds and require >=0.5% mean joint-complete cost gain versus each of OLDNN32, MLP0_32, BIAS0_32; otherwise report any speed/quality tradeoff without unique-GNN claim. Also report OLDNN128 and every individual week/seed.",
         "interpretation": "Even passing auxiliary and graph screens is developmental, not final independent confirmation or RL benefit. Slight weakness warrants diagnosis, no universal failure. Any major lost survival is a material warning. Do not launch RL from one-step scores alone."},
     "timing": "Single instrumented CPU wallclock; vector/forecast logging included in controller timing, costs also report physical and startup separately. No uninstrumented production speed claim.",
     "scope": "Same old development split, two dates overlap training variants. Not final-test. True final confirmation must isolate date families and retain all failures.",
     "provenance": "V0 and V1 checkpoint/source/metrics sealed, no previous artifact overwritten."}
(OUT / "design.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
files = [ROOT / "work/grid22" / f for f in ["adapter.py", "run.py", "preflight.py", "freeze.py"]]
files += [ROOT / x for x in ["work/grid21/prior.py", "work/grid20/components.py", "work/grid17/mask.py", "work/grid08/common.py"]]
files += [ROOT / m["checkpoint"] for m in methods if m["checkpoint"]]
files += [OUT / "design.json"]
(OUT / "code_freeze.json").write_text(json.dumps({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2), encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("FROZEN.28 full developmental rollouts, no new training or RL.\n", encoding="utf-8")
print(json.dumps({"frozen": True, "runs": len(order), "fits": 0}))
