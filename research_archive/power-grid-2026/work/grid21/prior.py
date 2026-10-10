"""Public original PPO rank prior and zero-start bounded residual scoring."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"),str(ROOT / "work/grid20")]
from common import np, torch, action_vector, digest
from components import CandidateScorer


def public_prior(obs, env, ctl, ids, aliases):
    """Only current observation, known library metadata and input-only aliases."""
    library = ctl.base.topo_n1_unsafe.topo_act_list
    subs = [int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0]) for a in library]
    live = obs.line_status; rho = obs.rho
    top = sorted(np.flatnonzero(live),key=lambda l:(-float(rho[l]),int(l)))[:3]
    roots = sorted({int(s) for l in top for s in [env.line_or_to_subid[l],env.line_ex_to_subid[l]]})
    distance = np.full(env.n_sub,env.n_sub+1,dtype=int); queue = roots.copy(); neighbor = [[] for _ in range(env.n_sub)]; pressure = np.zeros(env.n_sub)
    for s in roots: distance[s] = 0
    for l in np.flatnonzero(live):
        u,v = int(env.line_or_to_subid[l]),int(env.line_ex_to_subid[l]); neighbor[u].append(v); neighbor[v].append(u)
        pressure[u] = max(pressure[u],float(rho[l])); pressure[v] = max(pressure[v],float(rho[l]))
    for s in queue:
        for nxt in neighbor[s]:
            if distance[nxt]>distance[s]+1: distance[nxt] = distance[s]+1; queue.append(nxt)
    tail = sorted(ids,key=lambda i:(int(distance[subs[i]]),-float(pressure[subs[i]]),i))
    keys = {digest(action_vector(a)):i for i,a in enumerate(library)}
    neural = ctl.base.topo_12_unsafe
    mapped = {i:keys[key] for i,a in enumerate(neural.gym_env.action_space.topo_actions_list) if (key:=digest(action_vector(a))) in keys}
    with torch.no_grad(): ranked = neural.get_top_k(neural.gym_env.observation_space.to_gym(obs),len(neural.gym_env.action_space.topo_actions_list))
    raw = [mapped[int(i)] for i in ranked if int(i) in mapped]+tail
    order = []; seen = set()
    for i in raw:
        rep = aliases.get(i)
        if rep is not None and rep not in seen: seen.add(rep); order.append(rep)
    assert set(order) == set(aliases.values())
    logs = np.full(len(library),-1e9,dtype=np.float32)
    logs[order] = -np.log1p(np.arange(len(order),dtype=np.float32))
    return logs,order


class ResidualScorer(torch.nn.Module):
    def __init__(self,kind,n_nodes,max_edges,base_dim,plan_dim,subids):
        super().__init__(); self.kind = kind
        if kind == "bias": self.bias = torch.nn.Parameter(torch.zeros(len(subids)))
        else:
            self.inner = CandidateScorer(kind,n_nodes,max_edges,base_dim,plan_dim,subids)
            torch.nn.init.zeros_(self.inner.head[-1].weight); torch.nn.init.zeros_(self.inner.head[-1].bias)

    def forward(self,x,edges,edge_attr,glob,base,plans):
        if self.kind == "bias": raw = self.bias.unsqueeze(0).expand(x.shape[0] if x.ndim==3 else 1,-1)
        else: raw = self.inner(x,edges,edge_attr,glob,base,plans)
        return 2.*torch.tanh(raw/2.)
