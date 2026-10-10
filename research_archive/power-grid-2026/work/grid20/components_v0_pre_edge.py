"""Original message-passing core, with a geometry-aware candidate decoder."""
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "work/grid08"))
from common import np, torch, ActorCritic, graph_features, action_vector


def public_inputs(observation, base_action, env, static_plans):
    # No labels, future observations or candidate forecast scores are arguments.
    x, edges, glob = graph_features(observation, env, {"offered": False, "predictions": [None, None]})
    plans = static_plans.copy()
    intent = (base_action.line_set_status > 0) | base_action.line_change_status
    base_bus = base_action.set_bus
    intent |= (base_bus[env.line_or_pos_topo_vect] > 0) | (base_bus[env.line_ex_pos_topo_vect] > 0)
    disconnected = (~observation.line_status) & (~intent)
    plans[:, env.line_or_pos_topo_vect[disconnected]] = 0
    plans[:, env.line_ex_pos_topo_vect[disconnected]] = 0
    # Same right-hand base bus overrides as candidate + base_action.
    positions = np.flatnonzero(base_bus)
    plans[:, positions] = base_bus[positions]
    base_parts = []
    for name in type(base_action).attr_list_vect:
        values = np.asarray(base_action._get_array_from_attr_name(name), dtype=np.float32)
        values = np.nan_to_num(values, nan=0., posinf=100., neginf=-100.)
        scale = 100. if name in ["_redispatch", "_storage_power"] else 2.
        base_parts.append(np.clip(values, -100., 100.)/scale)
    base = np.concatenate(base_parts).astype(np.float32)
    result = {"x": x, "edges": edges, "glob": glob, "base": base, "plans": plans.astype(np.float32)/2.}
    assert all(np.isfinite(value).all() for value in result.values())
    return result


def fingerprint(features):
    h = hashlib.sha256()
    for name in sorted(features):
        a = np.ascontiguousarray(features[name]); h.update(name.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def flat_state(features, n_nodes, max_edges):
    edge_table = np.full((2, max_edges), -1., dtype=np.float32)
    assert features["edges"].shape[1] <= max_edges
    edge_table[:, :features["edges"].shape[1]] = features["edges"].astype(np.float32)/n_nodes
    return np.concatenate([features["x"].ravel(), edge_table.ravel(), features["glob"], features["base"], features["plans"].ravel()])


def canonical_groups(features, pool_ids):
    """Exact duplicate bus plans are equivalent once pure-topology is verified."""
    first = {}; mapping = {}; representatives = []
    for i in pool_ids:
        key = features["plans"][i].tobytes()
        if key not in first: first[key] = i; representatives.append(i)
        mapping[i] = first[key]
    return representatives, mapping


class CandidateScorer(torch.nn.Module):
    """Reuse original node/global encoding; share a decoder across candidates.

    Fixed action IDs are not decoded from an anonymous global graph pool alone.
    Each candidate receives both busbar embeddings of its public substation and
    its known bus assignment descriptor. MLP receives the same full graph table,
    raw local busbar inputs, descriptor and incoming base-action information.
    """
    def __init__(self, kind, n_nodes, max_edges, base_dim, plan_dim, subids):
        super().__init__(); self.kind = kind; self.n_nodes = n_nodes; self.max_edges = max_edges
        self.core = ActorCritic(kind, n_nodes, hidden=64, max_edges=max_edges)
        # The old binary actor/critic/trunk remain in original saved checkpoints;
        # they have no role in supervised candidate scoring and are frozen here.
        for module in [self.core.actor, self.core.critic, self.core.trunk]:
            for p in module.parameters(): p.requires_grad_(False)
        self.plan = torch.nn.Sequential(torch.nn.Linear(plan_dim, 16), torch.nn.Tanh())
        self.base = torch.nn.Sequential(torch.nn.Linear(base_dim, 32), torch.nn.Tanh())
        self.local = torch.nn.Sequential(torch.nn.Linear(28, 128), torch.nn.Tanh()) if kind == "mlp" else None
        self.head = torch.nn.Sequential(torch.nn.Linear(128+128+16+32+11, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
        self.register_buffer("subids", torch.as_tensor(subids, dtype=torch.long))

    def forward(self, x, edges, glob, base, plans):
        if x.ndim == 2:
            x = x.unsqueeze(0); glob = glob.unsqueeze(0); base = base.unsqueeze(0); plans = plans.unsqueeze(0); edges = [edges]
        if self.kind == "gnn":
            h = torch.tanh(self.core.project(x))
            for msg, upd in zip(self.core.messages, self.core.updates):
                messages = []
                for j, e in enumerate(edges):
                    agg = torch.zeros_like(h[j]); degree = torch.zeros((self.n_nodes, 1), dtype=h.dtype)
                    agg.index_add_(0, e[1], msg(h[j][e[0]])); degree.index_add_(0, e[1], torch.ones((e.shape[1], 1)))
                    messages.append(agg/degree.clamp_min(1.))
                h = torch.tanh(upd(torch.cat([h, torch.stack(messages)], dim=-1)))
            encoded = torch.cat([h.mean(1), h.max(1).values], dim=-1)
            local = torch.cat([h[:, self.subids], h[:, self.subids+self.n_nodes//2]], dim=-1)
        else:
            padded = []
            for e in edges:
                p = torch.full((2, self.max_edges), -1., dtype=x.dtype)
                p[:, :e.shape[1]] = e.float()/self.n_nodes; padded.append(p.ravel())
            encoded = self.core.flat(torch.cat([x.flatten(1), torch.stack(padded)], dim=-1))
            local = self.local(torch.cat([x[:, self.subids], x[:, self.subids+self.n_nodes//2]], dim=-1))
        n = len(self.subids)
        context = torch.cat([encoded, self.base(base), glob], dim=-1).unsqueeze(1).expand(-1, n, -1)
        values = torch.cat([local, self.plan(plans), context], dim=-1)
        return self.head(values).squeeze(-1)


def tensors(feature):
    return {k: torch.as_tensor(v, dtype=torch.long if k == "edges" else torch.float32) for k, v in feature.items()}


def batch(rows):
    return {"x": torch.stack([r["tensor"]["x"] for r in rows]),
            "edges": [r["tensor"]["edges"] for r in rows],
            "glob": torch.stack([r["tensor"]["glob"] for r in rows]),
            "base": torch.stack([r["tensor"]["base"] for r in rows]),
            "plans": torch.stack([r["tensor"]["plans"] for r in rows])}
