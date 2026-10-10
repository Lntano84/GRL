"""Freeze one diagnosed revision, leave V0 and all prior results intact."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid21"
assert not (OUT / "design.json").exists()
assert json.loads((OUT / "preflight.json").read_text())["passed"]
d = json.loads((ROOT / "outputs/grid20/design.json").read_text())
d["stage"] = "GRID21_PPO_PRIOR_BOUNDED_RESIDUAL_V1"; d["prior_KL_weight"] = .1
d["models"] = [f"{kind}_seed{s}" for kind in ["GNN","MLP","BIAS"] for s in d["seeds"]]+["FREQUENCY","RIDGE","OLDNN","LOCAL","PREFIX","RANDOM","PRIOR_ONLY"]
d["architecture"] = "Keep exact V0 encoder and same legal input decoder; zero-initialize final correction head. Scores are -log(1+rank_public_original_PPO_then_LOCAL)+2*tanh(raw_correction/2). The prior covers every canonical eligible action, no hard outside-support exclusion."
d["training"] = "Same training data, 64 epochs, seeds0/1, minibatches8 and optimizer as V0. Minimize target CE plus0.1 KL(q||public_prior). No development checkpoint/model/constant selection. Bias-only has909 state-independent parameters with same prior/objective/budget, two seeds."
d["pre_fit_addendum"] = "V0 failedK32 versus originalPPO but was competitiveK128; modest training data and existing strong PPO warrant retaining that prior. This revision adds the prior only, not the separately hypothesized object/forecast features. V0 original source/checkpoints/metrics stay sealed."
d["investment_screen"] = {
    "preserve_strong_base": "MacroGNN32 near-best no more than.02 below OLDNN32; fallback no more than.02 higher. Check both seeds separately.",
    "extra_signal": "GNN32 retention at least.03 above OLDNN32 or mean deliveredrho gap at least.001 smaller; also compare matched-prior MLP and bias-only. No3/4 strictly-positive requirement: V0 OLDNN already100% in two dev weeks, so that criterion has a ceiling. Does not retroactively change V0.",
    "otherwise": "Preserve revision and honest diagnostics. CompetitiveK128 remains a separate frozen secondary role. No automatic RL launch or universal failure from this development split."
}
d["reference_information"] = "GNN, MLP, bias-only and PRIOR_ONLY share identical legal original-PPO rank prior. Unmodified FREQUENCY/RIDGE are V0-style extra references, not claimed to be prior-aware calibrated rivals."
(OUT / "design.json").write_text(json.dumps(d,indent=2),encoding="utf-8")
paths = [ROOT / f"work/grid21/{name}.py" for name in ["run","prior","build","preflight","freeze"]]
paths += [ROOT / "work/grid20/components.py",ROOT / "work/grid08/common.py",ROOT / "work/grid15/rules.py",OUT / "design.json"]
(OUT / "code_freeze.json").write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("FROZEN V1; no fits yet. One prior-preserving revision, no new teacher queries or RL; V0 sealed.\n",encoding="utf-8")
print(json.dumps({"stage":d["stage"],"model_fits":0,"planned_fits":7,"new_teacher_queries":0}))
