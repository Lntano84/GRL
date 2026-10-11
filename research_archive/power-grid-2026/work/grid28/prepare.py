import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT / "outputs/grid28"; OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT / "design.json").exists()
D=json.loads((ROOT / "outputs/grid27/design.json").read_text())
D.update(stage="GRID28_LOCAL_SUBSTATION_COMPLETION",source="GRID27 unchanged schedules plus one semantic substation completion.",
         change="Before experimental stop, query all remaining eligible representatives touching the chosen candidate's original substation; includes physical aliases. No future or FULL winner input. It can still miss a better remote action.",
         gate="Same offline critical-state/rho qualification. Do not modify schedules/groups to rescue outcomes. If PATIENCE64-local passes for GNN0/PPO, run protected baseline profile and March pilots; preservation of closed-loop cost remains unproven until run.")
(OUT / "design.json").write_text(json.dumps(D,indent=2),encoding="utf-8")
paths=[Path(__file__).resolve(),ROOT / "work/grid28/policy.py",ROOT / "work/grid27/policy.py",ROOT / "outputs/grid27/design.json"]
(OUT / "schedule_freeze.json").write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("FROZEN: incumbent substation completion only; offline pending, no new simulation/fits/holdout.\n")
print(json.dumps({"frozen":True,"policies":D["policies"],"rankers":3}))
