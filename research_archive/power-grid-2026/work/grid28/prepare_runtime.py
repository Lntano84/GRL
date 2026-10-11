import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT / "outputs/grid28"
assert not (OUT / "runtime_design.json").exists()
D=json.loads((OUT / "design.json").read_text());result=json.loads((OUT / "offline.json").read_text())
for ranker in ["GNN0_128_GUARD","OLDNN128_GUARD"]:
    s=next(r for r in result["summaries"] if r["ranker"]==ranker and r["policy"]=="PATIENCE64")
    assert not s["march_critical_choice_changed"] and s["max_rho_regret"]<=.01
methods=[{"method":"PROTECTED_GNN0","rule":"GNN_seed0","budget":128,"checkpoint":D["rankers"][0]["checkpoint"],"policy":"PROTECTED"},
         {"method":"LOCAL_GNN0","rule":"GNN_seed0","budget":128,"checkpoint":D["rankers"][0]["checkpoint"],"policy":"PATIENCE64"},
         {"method":"LOCAL_PPO","rule":"OLDNN","budget":128,"checkpoint":None,"policy":"PATIENCE64"}]
order=[]
for wi,week in enumerate(D["weeks"]):
    rotation=methods[wi%3:]+methods[:wi%3]
    for m in rotation:order.append({"repeat":0,"week":week,"phase":"pilot" if wi==0 else "remaining",**m})
D.update(methods=methods,execution_order=order,repeats=1,
         timing="Serial fresh worker per trajectory. Same GRID26 act timer and out-of-act archival. Single development repeat, not a stable-speed confirmation.")
D["caps"].update(wall_s=7200,episode_s=1200,physical_steps=40000,public_forecasts=500000)
(OUT / "runtime_design.json").write_text(json.dumps(D,indent=2),encoding="utf-8")
(OUT / "runtime_prepare_freeze.json").write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [Path(__file__).resolve(),OUT / "runtime_design.json",OUT / "offline.json"]},indent=2),encoding="utf-8")
print(json.dumps({"planned_runs":12,"pilot_runs":3,"offline_gate_passed":True}))
