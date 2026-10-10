"""Same GRID24 decisions, with per-candidate audit work outside online timing."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid24"), str(ROOT / "work/grid08")]
from common import np, flags, action_vector
from guard_adapter import Search as AuditedSearch


class Search(AuditedSearch):
    def prepare(self, obs, base):
        eligible = [i for i, sub in enumerate(self.subids) if obs.time_before_cooldown_sub[sub] == 0]
        intent = (base.line_set_status > 0) | base.line_change_status
        bus = base.set_bus
        intent |= (bus[self.env.line_or_pos_topo_vect] > 0) | (bus[self.env.line_ex_pos_topo_vect] > 0)
        disconnected = (~obs.line_status) & (~intent)
        positions = np.concatenate([self.env.line_or_pos_topo_vect[disconnected], self.env.line_ex_pos_topo_vect[disconnected]])
        reps, aliases, actions, changed, first = [], {}, {}, [], {}
        for i in eligible:
            action = self.library[i].copy()
            values = action.set_bus.copy()
            changed_positions = positions[values[positions] != 0]
            if len(changed_positions):
                values[changed_positions] = 0
                action.set_bus = values
                changed.append(i)
            key = np.ascontiguousarray(action_vector(action + base)).tobytes()
            if key not in first:
                first[key] = i; reps.append(i); actions[i] = action
            aliases[i] = first[key]
        return eligible, reps, aliases, actions, changed

    def choose(self, obs, base, reward, done=False, **kwargs):
        started = time.perf_counter()
        eligible, reps, aliases, actions, changed = self.prepare(obs, base)
        prepared = time.perf_counter()
        order, prior, features, times = self.rank(obs, base, eligible, reps, aliases)
        limit = min(float(obs.rho.max()), float(kwargs.get("rho_threshold", obs.rho.max())))
        first = order if self.rule == "FULL" else order[:self.budget]
        cache, outcomes = {}, []
        forecast_s = 0.

        def dispatch(i, phase):
            nonlocal forecast_s
            assert i not in cache
            start = time.perf_counter()
            future, rr, ended, info = obs.simulate(actions[i] + base)
            elapsed = time.perf_counter() - start
            forecast_s += elapsed
            f = flags(info); rho = float(future.rho.max())
            valid = bool(0 < rho < limit and not ended and not f["exceptions"] and not f["illegal"] and not f["ambiguous"])
            row = {"pool_id": i, "rho": rho, "reward": float(rr), "done": bool(ended),
                   **f, "strict_admissible": valid, "forecast_s": elapsed, "phase": phase}
            cache[i] = row; outcomes.append(row)

        for i in first: dispatch(i, "shortlist")
        valid = [i for i in first if cache[i]["strict_admissible"]]
        choice = max(valid, key=lambda i: (float(np.float32(cache[i]["reward"])), -i)) if valid else None
        safe = float(self.ctl.base.rho_safe)
        assert safe == .9
        reason = "no_admissible_shortlist" if choice is None else (
            "shortlist_above_author_safe" if cache[choice]["rho"] >= safe else "shortlist_below_author_safe")
        fallback = bool(self.rule != "FULL" and len(first) < len(reps) and (choice is None or cache[choice]["rho"] >= safe))
        if fallback:
            for i in reps:
                if i not in cache: dispatch(i, "fallback")
        considered = reps if fallback else first
        valid = [i for i in considered if cache[i]["strict_admissible"]]
        chosen = max(valid, key=lambda i: (float(np.float32(cache[i]["reward"])), -i)) if valid else None
        elapsed = time.perf_counter() - started
        log = {"rule": self.rule, "budget": self.budget, "eligible": eligible, "representatives": reps,
               "aliases": aliases, "order": order, "shortlist": first, "masked_ids": changed,
               "rho_limit": limit, "guard_rho": safe, "guard_reason": reason, "shortlist_choice": choice,
               "fallback": fallback, "chosen_pool_id": chosen, "outcomes": outcomes,
               "public_queries": len(outcomes), "source_s": elapsed, "prepare_s": prepared-started,
               "forecast_s": forecast_s, **times}
        return None if chosen is None else actions[chosen], log, features, prior
