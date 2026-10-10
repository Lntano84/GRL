"""Known masked, canonical topology candidates with same-call fallback.

No forecast is read until the selected shortlist is actually dispatched. All
comparators share masks, eligibility, physical-action aliasing and admission.
"""
import hashlib
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid17"), str(ROOT / "work/grid20"), str(ROOT / "work/grid21")]
from common import np, torch, action_vector, digest, flags
from components import public_inputs, tensors, fingerprint, canonical_groups
from prior import ResidualScorer, public_prior
from mask import mask_disconnected


class Search:
    def __init__(self, env, ctl, rule, budget, model_path=None):
        self.env, self.ctl, self.rule, self.budget = env, ctl, rule, budget
        self.module = ctl.base.topo_n1_unsafe
        self.library = self.module.topo_act_list
        self.plans = np.stack([a.set_bus for a in self.library])
        self.subids = [int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0]) for a in self.library]
        self.model = None
        self.checkpoint_sha = None
        if model_path is not None:
            self.checkpoint_sha = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
            f = public_inputs(env.get_obs(), env.action_space(), env, self.plans)
            kind = rule.split("_")[0].lower()
            self.model = ResidualScorer(kind, 2*env.n_sub, 2*env.n_line, len(f["base"]), env.dim_topo, self.subids)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            self.model.eval()

    def prepare(self, obs, base):
        eligible = [i for i, sub in enumerate(self.subids) if obs.time_before_cooldown_sub[sub] == 0]
        actions, aliases, reps, hashes = {}, {}, [], {}
        changed = []
        # Full combined vectors, not just bus plans, certify aliases here.
        for i in eligible:
            _, expected, positions, _ = mask_disconnected(self.library[i], base, obs, self.env)
            candidate = self.library[i].copy()
            if positions:
                bus = candidate.set_bus.copy(); bus[positions] = 0; candidate.set_bus = bus
            combined = candidate + base
            key = digest(action_vector(combined))
            assert key == digest(action_vector(expected))
            rep = hashes.get(key)
            if rep is None:
                rep = i; hashes[key] = i; reps.append(i); actions[i] = candidate
            aliases[i] = rep
            if positions: changed.append(i)
        return eligible, reps, aliases, actions, changed

    def rank(self, obs, base, eligible, reps, aliases):
        times = {"feature_s": 0., "prior_s": 0., "forward_rank_s": 0.}
        if self.rule == "FULL": return reps.copy(), None, None, times
        t = time.perf_counter()
        prior, order = public_prior(obs, self.env, self.ctl, eligible, aliases)
        times["prior_s"] = time.perf_counter()-t
        features = None
        if self.model is not None:
            t = time.perf_counter()
            features = public_inputs(obs, base, self.env, self.plans)
            alt_reps, alt_aliases = canonical_groups(features, eligible)
            assert alt_reps == reps and alt_aliases == aliases
            times["feature_s"] = time.perf_counter()-t
            t = time.perf_counter()
            with torch.no_grad(): residual = self.model(**tensors(features))[0].numpy()
            assert np.isfinite(residual).all() and np.max(np.abs(residual)) <= 2.+1e-6
            values = residual + prior
            order = sorted(reps, key=lambda i: (-float(values[i]), i))
            times["forward_rank_s"] = time.perf_counter()-t
        assert set(order) == set(reps) and len(order) == len(reps)
        return order, prior, features, times

    def choose(self, obs, base, reward, done=False, **kwargs):
        started = time.perf_counter(); before = digest(obs.to_vect())
        eligible, reps, aliases, actions, changed = self.prepare(obs, base)
        order, prior, features, times = self.rank(obs, base, eligible, reps, aliases)
        limit = min(float(obs.rho.max()), float(kwargs.get("rho_threshold", obs.rho.max())))
        first = order if self.rule == "FULL" else order[:self.budget]
        outcomes, cache = [], {}
        forecast_s = 0.

        def dispatch(i, phase):
            nonlocal forecast_s
            assert i not in cache
            combined = actions[i]+base
            t = time.perf_counter(); future, rr, ended, info = obs.simulate(combined)
            elapsed = time.perf_counter()-t; forecast_s += elapsed
            fl = flags(info); rho = float(future.rho.max())
            valid = bool(0 < rho < limit and not ended and not fl["exceptions"] and not fl["illegal"] and not fl["ambiguous"])
            row = {"pool_id": i, "combined_hash": digest(action_vector(combined)), "rho": rho, "reward": float(rr),
                   "done": bool(ended), **fl, "strict_admissible": valid, "forecast_s": elapsed, "phase": phase}
            if valid: assert float(np.float32(rr)) > -100.
            outcomes.append(row); cache[i] = row

        for i in first: dispatch(i, "shortlist")
        fallback = not any(r["strict_admissible"] for r in outcomes) and len(first) < len(reps)
        if fallback:
            for i in reps:
                if i not in cache: dispatch(i, "fallback")
        considered = reps if fallback else first
        valid = [i for i in considered if cache[i]["strict_admissible"]]
        chosen = max(valid, key=lambda i: (float(np.float32(cache[i]["reward"])), -i)) if valid else None
        assert digest(obs.to_vect()) == before
        result = None if chosen is None else actions[chosen]
        log = {"state_hash": before, "rule": self.rule, "budget": self.budget, "eligible": eligible,
               "representatives": reps, "aliases": aliases, "order": order, "shortlist": first, "masked_ids": changed,
               "rho_limit": limit, "fallback": fallback, "chosen_pool_id": chosen,
               "returned_action_hash": None if result is None else digest(action_vector(result)),
               "public_queries": len(outcomes), "outcomes": outcomes, "source_s": time.perf_counter()-started,
               "forecast_s": forecast_s, **times,
               "input_hash": None if features is None else fingerprint(features),
               "prior_hash": None if prior is None else digest(prior)}
        return result, log, features, prior
