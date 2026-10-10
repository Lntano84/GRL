"""Seal a small supervised assistant trial before fitting or dev access."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid20"
assert not (OUT / "design.json").exists()
assert not (OUT / "models").exists()
prior = json.loads((ROOT / "outputs/grid19/design.json").read_text())
assert json.loads((OUT / "preflight.json").read_text())["passed"]
d = {
    "stage": "GRID20_SUPERVISED_SHORTLIST_V0", "seeds": [0,1],
    "existing_training_weeks": prior["existing_training_weeks"],
    "training_weeks": prior["existing_training_weeks"]+prior["new_training_weeks"],
    "development_weeks": prior["development_weeks"],
    "epochs": 64, "batch_size": 8, "lr": .001, "weight_decay": .0001, "temperature": .01,
    "ridge_alpha": 1., "budgets": [32,128], "caps": {"wall_s": 3600},
    "teacher": "GRID18/19 actual masked full-search decisions, original public forecasts only. Keep early-failed weeks and no-target states. No balancing by dev outcomes.",
    "models": ["GNN_seed0", "GNN_seed1", "MLP_seed0", "MLP_seed1", "FREQUENCY", "RIDGE", "OLDNN", "LOCAL", "PREFIX", "RANDOM"],
    "architecture": "Original two message/mean aggregation layers, width64, with shared candidate decoder on both local busbars, known masked candidate plan and incoming base. Add four public directed edge channels (rho, signed power, voltage, cooldown). MLP gets exactly the same graph arrays, plans and base action. No original trained GNN weights reused.",
    "pre_fit_addendum": "Candidate-local decoder avoids decoding fixed action identity solely from anonymous pooling. Public per-line edge channels restore information lost by old node aggregation. Pure-topology library confirmed by preflight; exact masked aliases collapse to first library ID for every method, target and hypothetical query budget. Before any fitting; not a post-result variant.",
    "targets": "Soft probability exp(-(rho-best_strict_valid_rho)/.01) on one representative per exact physical-action class; normalize. No strict-valid candidate gives no preference target, retained in dev availability denominator.",
    "training": "All fits final64 epochs only, Adam, seeds0/1, equal minibatch order; no early stopping, dev selection or old fitted GNN weights. Ridge alpha1, train-only scaling, frequency from train targets. No RL.",
    "development_access": "Only after all four neural fits plus ridge/frequency saved and training_finished.json committed. Whole-week metadata split, four dev weeks; they are development, not independent sealed final-test. External pretrained PPO may have seen official distribution; disclosed.",
    "delivery": "Rank only input-eligible canonical classes. Simulate firstK hypothetically; if no strictly improving admissible candidate, full-class cached fallback. Pick best original float32 reward among considered strict-valid candidates. No actual new forecasts or learned-policy physical steps at this stage.",
    "primary": "Macro by development week: raw near-best-rho retention (gap<=.01), raw availability, fallback, delivered rho gap, and expected unique public queries. K32 primary versus OLDNN128 plus every equal32 method. Average neural seeds within week, not independent observations.",
    "investment_screen": {
        "closed_loop_priority": "GNN32 macro near-best retention>=.90, macro fallback<=.10, queries at least65% below full canonical search; retention not more than.05 below OLDNN128. Need no independently proven accuracy claim.",
        "additional_graph_signal": "At equal32 budget GNN mean retention at least.05 above best aggregate non-graph whole-method reference (MLP averaged over seeds, OLDNN, RIDGE, FREQUENCY, LOCAL, PREFIX, RANDOM), positive in at least3/4 dev weeks. Statewise best oracle excluded.",
        "otherwise": "Diagnose poor fit, missing public information or role mismatch; preserve V0 and allow one documented small revision before judging current model. These thresholds prioritize work, not proof of impossibility or field rejection. No generic GNN+RL novelty claim."
    },
    "cost_reporting": "Teacher acquisition, fit time, feature construction and forward/ranking separately. Stored-state ranking and expected calls do not establish online wallclock speed, survival or operational cost; closed-loop needed. CPU one thread, no new installs."
}
assert not set(d["training_weeks"]) & set(d["development_weeks"])
(OUT / "design.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
paths = [ROOT / f"work/grid20/{n}.py" for n in ["run","components","preflight","freeze"]]
paths += [ROOT / "work/grid08/common.py", ROOT / "work/grid15/rules.py", OUT / "design.json"]
(OUT / "code_freeze.json").write_text(json.dumps({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2), encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("FROZEN. Waiting for new teacher collection/audit; no fits yet. Original models and pre-fit prototypes preserved.\n", encoding="utf-8")
print(json.dumps({"training_weeks":len(d["training_weeks"]),"development_weeks":len(d["development_weeks"]),"models_fitted":0}))
