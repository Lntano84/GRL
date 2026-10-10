"""Zero residual exactly preserves public PPO order; no fit or simulations."""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"),str(ROOT / "work/grid20"),str(ROOT / "work/grid21")]
from common import np, torch, make_env, ResidualControl, write_json
from components import public_inputs, tensors, canonical_groups
from prior import public_prior, ResidualScorer
OUT = ROOT / "outputs/grid21"; OUT.mkdir(parents=True,exist_ok=True)
for stage in ["grid19","grid20"]:
    for name,sha in json.loads((ROOT / f"outputs/{stage}/delivery_manifest.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha,name
env = make_env(); obs0 = env.reset(); ctl = ResidualControl(env,obs0)
library = ctl.base.topo_n1_unsafe.topo_act_list; plans = np.stack([a.set_bus for a in library])
subids = [int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0]) for a in library]; checks = []
try:
    for week in json.loads((ROOT / "outputs/grid18/design.json").read_text())["weeks"]:
        file = sorted((ROOT / f"outputs/grid18/runs/{week}").glob("call[0-9][0-9][0-9].json"))[0]
        c = json.loads(file.read_text())
        with np.load(file.with_name(file.stem+"_state.npz")) as f:
            obs = obs0.copy(); obs.from_vect(f["observation"]); base = env.action_space(); base.from_vect(f["base_action"])
        visible = public_inputs(obs,base,env,plans); reps,aliases = canonical_groups(visible,c["pool_ids"])
        prior,order = public_prior(obs,env,ctl,c["pool_ids"],aliases)
        for kind in ["gnn","mlp","bias"]:
            torch.manual_seed(0)
            model = ResidualScorer(kind,2*env.n_sub,2*env.n_line,len(visible["base"]),env.dim_topo,subids)
            with torch.no_grad(): correction = model(**tensors(visible))[0].numpy()
            assert np.array_equal(correction,np.zeros(len(library)))
            assert sorted(reps,key=lambda i:(-float(prior[i]+correction[i]),i)) == order
        changed = json.loads(json.dumps(c))
        for r in changed["outcomes"]: r["rho"]+=1000; r["strict_admissible"] = not r["strict_admissible"]
        repeated,repeated_order = public_prior(obs,env,ctl,c["pool_ids"],aliases)
        assert changed != c and np.array_equal(prior,repeated) and order == repeated_order
        checks.append({"week":week,"canonical_classes":len(reps),"zero_start_exact":True,"label_perturbation_unchanged":True})
finally: env.close()
write_json(OUT / "preflight.json",{"passed":True,"checks":checks,"fits":0,"forecasts":0,"historical_seals_unchanged":True})
print(json.dumps({"passed":True,"fixtures":len(checks),"fits":0}))
