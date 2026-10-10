"""Training-fixture gradient plumbing; backward only, no optimizer or fits."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"),str(ROOT / "work/grid20")]
from common import np, torch, make_env, ResidualControl, write_json
from components import public_inputs, tensors, CandidateScorer, canonical_groups
OUT = ROOT / "outputs/grid20"
env = make_env(); obs0 = env.reset(); ctl = ResidualControl(env,obs0)
library = ctl.base.topo_n1_unsafe.topo_act_list; plans = np.stack([a.set_bus for a in library])
subids = [int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0]) for a in library]
file = ROOT / "outputs/grid18/runs/2035-01-29_4/call000.json"; c = json.loads(file.read_text())
checks = []
try:
    with np.load(file.with_name(file.stem+"_state.npz")) as f:
        obs = obs0.copy(); obs.from_vect(f["observation"])
        base = env.action_space(); base.from_vect(f["base_action"])
    visible = public_inputs(obs,base,env,plans); reps,_ = canonical_groups(visible,c["pool_ids"])
    labels = dict(zip(c["pool_ids"],c["outcomes"])); valid = [i for i in reps if labels[i]["strict_admissible"]]
    assert valid
    target = torch.zeros(len(library)); best = min(labels[i]["rho"] for i in valid)
    target[valid] = torch.exp(-torch.tensor([labels[i]["rho"]-best for i in valid])/.01); target /= target.sum()
    mask = torch.zeros(len(library),dtype=torch.bool); mask[reps] = True
    for kind in ["gnn","mlp"]:
        torch.manual_seed(0)
        model = CandidateScorer(kind,2*env.n_sub,2*env.n_line,len(visible["base"]),env.dim_topo,subids)
        original = {name:value.detach().clone() for name,value in model.state_dict().items()}
        logits = model(**tensors(visible))[0]
        loss = -(target*torch.log_softmax(logits.masked_fill(~mask,-1e9),dim=0)).sum()
        assert torch.isfinite(loss); loss.backward()
        modules = {"encoder":model.core.project if kind=="gnn" else model.core.flat,"head":model.head,"plans":model.plan}
        if kind == "gnn": modules["edge"] = model.edge
        norms = {}
        for name,module in modules.items():
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
            norm = sum(float(g.abs().sum()) for g in gradients); assert norm>0; norms[name] = norm
        assert all(torch.equal(value,original[name]) for name,value in model.state_dict().items())
        checks.append({"kind":kind,"loss":float(loss.detach()),"gradient_l1":norms,"weights_unchanged":True})
finally:
    env.close()
write_json(OUT / "gradient_check.json", {"passed":True,"checks":checks,"optimizer_steps":0,"model_fits":0,"forecasts":0})
print(json.dumps({"passed":True,"checks":checks,"model_fits":0}))
