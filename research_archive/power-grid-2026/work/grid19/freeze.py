"""Freeze metadata-only new teacher split and the planned tiny learner."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid19"; OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / "design.json").exists()
assert json.loads((ROOT / "outputs/grid18/audit.json").read_text())["passed"]
prior = json.loads((ROOT / "outputs/grid10/design.json").read_text())
exposed = set(prior["training_weeks"] + prior["evaluation_weeks"])
available = [p.name for p in (ROOT / "work/grid03/formal_env_initialized/chronics").iterdir() if p.is_dir() and p.name not in exposed]
train = []; development = []; strata = []
for quarter in range(4):
    pool = [n for n in available if (int(n[5:7])-1)//3 == quarter]
    ordered = sorted(pool, key=lambda n: hashlib.sha256(("GRID19_SPLIT_20261010_"+n).encode()).hexdigest())
    assert len(ordered) >= 5
    train += ordered[:4]; development += ordered[4:5]
    strata.append({"quarter": quarter+1, "pool": sorted(pool), "new_training": ordered[:4], "development": ordered[4:5]})
assert len(train) == 16 and len(development) == 4 and not set(train)&set(development)
assert not (set(train)|set(development))&exposed
d = {"stage": "GRID19_NEW_MASKED_N1_TEACHER_DATA", "seed": 0, "new_training_weeks": train,
     "development_weeks": development, "existing_training_weeks": prior["training_weeks"], "weeks": train+development,
     "strata": strata, "split": "Public week names only; hash seed20261010, four new training and one development week per quarter. Exclude eight exposed GRID10 weeks. No outcome-based replacement.",
     "teacher": "Exact GRID18 masked full-search controller, original NN20/QP plus ALWAYS_RESTORE. Save only actual search decisions; no engineered trigger or replay upsampling.",
     "source_code": "GRID18 collect.py copied, only output-directory literal changed. All simulation and selection semantics identical.",
     "learner_protocol": {"role": "Supervised candidate priority only, no RL at this stage", "models": ["Original two-message-layer graph encoder with new candidate head", "Same legal-input MLP", "Training-label frequency", "Dual ridge on same flattened legal inputs", "Frozen original PPO prior", "LOCAL", "PREFIX"],
          "graph_inputs": "Original common.graph_features current busbar graph/current observations; neutral proposal fields. Incoming base action additionally supplied equally to GNN and MLP. No forecast outcome used as an input.",
          "targets": "Original pool-ID soft probabilities, exp(-(rho-rho_best)/0.01) on strictly source-admissible public candidates. States with no valid candidate remain in evaluation, but have no supervised preference target.",
          "fits": "Train using original four + sixteen new training weeks only. Two seeds0,1. GNN/MLP 64 epochs, Adam lr0.001, weight_decay0.0001, batch8, no early stopping or development checkpoint selection. All original trained weights preserved; reuse architecture, not holdout-exposed fitted weights.",
          "ridge": "Dual ridge alpha1, feature standardization fitted on training only; negative scores clipped for ranking/frequency fallback. One fixed fit; no sweep.",
          "ranking_budgets": [32, 128], "initial_primary": "32-query shortlist quality/availability and expected public calls with same full-search fallback; compare frozen old-PPO128 quality plus equal32 references. Ranking-only screen, not whole-policy speed or cost.",
          "interpretation": "Tiny developmental role trial; report all seeds, weeks, baselines, time and checkpoints. Slightly worse/mixed results may motivate one documented diagnostic revision, not automatic field rejection. No sealed final-test access, no GNN+RL novelty claim."},
     "caps": {"physical_steps": 44000, "public_forecasts": 1000000, "wall_s": 7200, "episode_s": 600},
     "training_not_yet_started": True}
(OUT / "design.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
files = [ROOT / "work/grid19/collect.py", ROOT / "work/grid19/freeze.py", ROOT / "work/grid17/mask.py", ROOT / "work/grid08/common.py", OUT / "design.json"]
(OUT / "code_freeze.json").write_text(json.dumps({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2), encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("FROZEN, COLLECTION NOT YET STARTED. No model fits. Sixteen new training/four metadata-selected development weeks.\n", encoding="utf-8")
print(json.dumps({"new_training": train, "development": development, "caps": d["caps"]}))
