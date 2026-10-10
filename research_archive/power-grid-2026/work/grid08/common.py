"""GRID08: legal residual control on immutable LJN NN20; no hidden chronics access."""
import os, sys, json, time, hashlib, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path[:0] = [str(ROOT / 'work' / x) for x in ['grid01','grid02','grid03','grid04']]
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import grid2op
from lightsim2grid import LightSimBackend
from ljn_nn import make_agent_topoNN
from ljn_nn.modules.convex_optim import OptimModule
from ljn_nn.modules.rewards import MaxRhoReward
from score_adapter import operational_other_rewards, public_cost_ledger, cost_rounding_tolerance, PREFIX
from trace_agent import action_vector, digest, flags
logging.getLogger().setLevel(logging.ERROR)
OUT = ROOT / 'outputs/grid08'

def json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'Unsupported persisted type: {type(value).__name__}')

def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False,default=json_default); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def make_env():
    return grid2op.make(str(ROOT/'work/grid03/formal_env_initialized'), test=True,
        backend=LightSimBackend(), reward_class=MaxRhoReward,
        other_rewards=operational_other_rewards())

class ResidualControl:
    """Author actions unchanged. A separate optimizer proposes safe continuous restoration.

    Restoring candidates never mutate the original optimizer's memory. Both actions
    are simulated at the same state. A one-step forecast screen is NOT a safety proof.
    """
    def __init__(self, env, obs):
        self.env = env
        self.base = make_agent_topoNN(env, str(ROOT/'work/grid02/ljn_nn'))
        self.base.seed(0); self.base.reset(obs)
        self.safe = OptimModule(env, env.action_space)
        self.safe.reset(obs)
        self.solve_status = []
        original = self.safe._solve_problem
        def solve(prob, solver_type=None):
            ok = original(prob, solver_type)
            self.solve_status.append(str(prob.status))
            # Do not silently treat an iteration-limited solve as qualified.
            return bool(ok and prob.status in ('optimal','optimal_inaccurate'))
        self.safe._solve_problem = solve

    def propose(self, obs, reward=0., cadence=6):
        before = digest(obs.to_vect())
        base = self.base.act(obs, reward, False)
        offered = False
        result = {'eligible':False, 'offered':False, 'reason':'outside_cadence', 'solve_status':[]}
        restore = None
        preds = [None, None]
        # Prototype decisions every 30 minutes; all residual comparators share cadence.
        if (int(obs.current_step)+1) % cadence == 0:
            result['reason'] = 'base_not_empty_or_grid_not_safe'
            # Do not combine safe QP with an unmodelled topology/reconnection action.
            empty = self.env.action_space()
            if np.array_equal(action_vector(base), action_vector(empty)) and \
                    0 < float(obs.rho.max()) < .9 and \
                    float(np.abs(obs.target_dispatch).sum()) > .05:
                result['eligible'] = True
                self.solve_status = []
                # Re-estimate DC correction from today's public state, not stale shadow state.
                self.safe.reset(obs)
                self.safe._update_storage_power_obs(obs)
                self.safe.update_parameters(obs, safe=True)
                c, s, r = self.safe.compute_optimum_safe(obs)
                result['solve_status'] = list(self.solve_status)
                if self.solve_status and self.solve_status[-1] in ('optimal','optimal_inaccurate'):
                    restore = self.safe.to_grid2op(obs,c.copy(),s.copy(),r.copy(),
                        base_action=base.copy(),safe=True)
                    for i,a in enumerate([base,restore]):
                        future,_,done,info = obs.simulate(a, time_step=1)
                        valid = not done and not info.get('is_illegal',False) and \
                            not info.get('is_ambiguous',False) and not info.get('exception',[])
                        rho = float(future.rho.max())
                        valid = bool(valid and np.isfinite(rho) and rho>0)
                        cost = None
                        if valid:
                            cost = public_cost_ledger(future,self.env,
                                float(obs.curtailment_mw.sum(dtype=np.float64)))['recomputed_raw_cost']
                        preds[i] = {'valid':valid,'rho':rho,'cost':cost}
                    offered = bool(preds[0]['valid'] and preds[1]['valid'] and preds[1]['rho']<=.93 and
                        not np.array_equal(action_vector(base),action_vector(restore)))
                    result['reason'] = 'offered' if offered else 'forecast_screen_rejected'
                else:
                    result['reason'] = 'safe_qp_not_qualified'
        assert digest(obs.to_vect()) == before
        assert digest(self.env.get_obs().to_vect()) == before
        result.update(offered=offered, predictions=preds)
        return base, restore if offered else None, result

def graph_features(obs, env, proposal):
    """Dynamic busbar graph; same legal node/global arrays provided to MLP and GNN."""
    n = 2*env.n_sub
    x = np.zeros((n,14),dtype=np.float32)
    def node(sub,bus): return np.asarray(sub)+env.n_sub*(np.maximum(np.asarray(bus),1)-1)
    g = node(env.gen_to_subid,obs.gen_bus); l = node(env.load_to_subid,obs.load_bus)
    for channel,arr in enumerate([obs.gen_p,obs.actual_dispatch,obs.target_dispatch,
            obs.gen_margin_up,obs.gen_margin_down,obs.curtailment_mw]):
        np.add.at(x[:,channel],g,np.asarray(arr,dtype=np.float32)/100.)
    np.add.at(x[:,6],l,obs.load_p/100.)
    if env.n_storage:
        st = node(env.storage_to_subid,obs.storage_bus)
        np.add.at(x[:,7],st,obs.storage_charge/100.)
        np.add.at(x[:,8],st,obs.storage_power/100.)
    u = node(env.line_or_to_subid,obs.line_or_bus)
    v = node(env.line_ex_to_subid,obs.line_ex_bus)
    live = np.asarray(obs.line_status,dtype=bool)
    for idx in [u[live],v[live]]:
        np.maximum.at(x[:,9],idx,obs.rho[live])
        np.add.at(x[:,10],idx,1./10.)
        np.add.at(x[:,11],idx,np.abs(obs.p_or[live])/100.)
    x[:,12] = np.tile(obs.time_before_cooldown_sub/12.,2)
    x[:env.n_sub,13]=1.; x[env.n_sub:,13]=-1.
    edges = np.stack([np.concatenate([u[live],v[live]]),np.concatenate([v[live],u[live]])]).astype(np.int64)
    angle = 2*np.pi*(obs.hour_of_day+obs.minute_of_hour/60.)/24.
    glob = [np.sin(angle),np.cos(angle),float(obs.rho.max()),
        float(np.abs(obs.actual_dispatch).sum())/100.,float(np.abs(obs.target_dispatch).sum())/100.,
        float(obs.load_p.sum())/1000.,float(proposal['offered'])]
    for p in proposal['predictions']:
        glob.extend([p['rho'] if p and p['valid'] else 0.,p['cost']/10000. if p and p['valid'] else 0.])
    x = np.clip(x,-100,100)
    glob = np.asarray(glob,dtype=np.float32)
    assert np.isfinite(x).all() and np.isfinite(glob).all()
    return x,edges,glob

class ActorCritic(torch.nn.Module):
    def __init__(self,kind,n_nodes,hidden=64,max_edges=372):
        super().__init__(); self.kind=kind; self.n_nodes=n_nodes; self.max_edges=max_edges
        if kind=='gnn':
            self.project=torch.nn.Linear(14,hidden)
            self.messages=torch.nn.ModuleList([torch.nn.Linear(hidden,hidden) for _ in range(2)])
            self.updates=torch.nn.ModuleList([torch.nn.Linear(2*hidden,hidden) for _ in range(2)])
            encoded=2*hidden
        else:
            # MLP receives the SAME edge endpoint table, padded, as explicit input.
            self.flat=torch.nn.Sequential(torch.nn.Linear(n_nodes*14+2*max_edges,hidden),torch.nn.Tanh(),
                torch.nn.Linear(hidden,2*hidden),torch.nn.Tanh())
            encoded=2*hidden
        self.trunk=torch.nn.Sequential(torch.nn.Linear(encoded+11,64),torch.nn.Tanh())
        self.actor=torch.nn.Linear(64,2); self.critic=torch.nn.Linear(64,1)
        torch.nn.init.zeros_(self.actor.weight); torch.nn.init.zeros_(self.actor.bias)

    def forward(self,x,edges,glob):
        if x.ndim==2:x=x.unsqueeze(0);glob=glob.unsqueeze(0)
        if self.kind=='gnn':
            h=torch.tanh(self.project(x))
            # Edges supplied individually for a variable-topology batch.
            if isinstance(edges,torch.Tensor): edges=[edges]
            for msg,upd in zip(self.messages,self.updates):
                ag=[]
                for j,e in enumerate(edges):
                    a=torch.zeros_like(h[j]); deg=torch.zeros((self.n_nodes,1),dtype=h.dtype)
                    a.index_add_(0,e[1],msg(h[j][e[0]]))
                    deg.index_add_(0,e[1],torch.ones((e.shape[1],1)))
                    ag.append(a/deg.clamp_min(1.))
                h=torch.tanh(upd(torch.cat([h,torch.stack(ag)],dim=-1)))
            enc=torch.cat([h.mean(1),h.max(1).values],dim=-1)
        else:
            if isinstance(edges,torch.Tensor):edges=[edges]
            edge_flat=[]
            for e in edges:
                assert e.shape[1]<=self.max_edges
                padded=torch.full((2,self.max_edges),-1.,dtype=x.dtype)
                padded[:,:e.shape[1]]=e.float()/self.n_nodes
                edge_flat.append(padded.flatten())
            enc=self.flat(torch.cat([x.flatten(1),torch.stack(edge_flat)],dim=1))
        z=self.trunk(torch.cat([enc,glob],dim=-1))
        logits=self.actor(z)
        logits[:,1]=torch.where(glob[:,6]>0.5,logits[:,1],torch.full_like(logits[:,1],-1.e9))
        return logits,self.critic(z).squeeze(-1)

def tensor_features(f):
    return torch.from_numpy(f[0]),torch.from_numpy(f[1]),torch.from_numpy(f[2])
