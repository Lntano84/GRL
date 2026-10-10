"""Closed-loop cheap N1 shortlists with source-level full-search fallback."""
import gzip
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid15")]
from common import np, torch, write_json, make_env, ResidualControl, action_vector, digest, flags, PREFIX, public_cost_ledger, cost_rounding_tolerance
from rules import shortlist_rules

OUT = ROOT / "outputs/grid16"
D = json.loads((OUT / "design.json").read_text())
START = time.perf_counter()
counts = {"physical_steps": 0, "public_forecasts": 0, "model_fits": 0}


def read_reference(week):
    with gzip.open(ROOT / f"outputs/grid15/runs/{week}/steps.jsonl.gz", "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def run(week, rule):
    path = OUT / "runs" / f"{week}__{rule}"; path.mkdir(parents=True, exist_ok=False)
    env = make_env(); env.seed(0); env.set_id(week); obs = env.reset()
    ctl = ResidualControl(env, obs); reference = read_reference(week)
    module = ctl.base.topo_n1_unsafe
    all_actions = module.topo_act_list
    all_hashes = [digest(action_vector(a)) for a in all_actions]
    mapping = {key: i for i, key in enumerate(all_hashes)}
    all_subs = [int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0]) for a in all_actions]
    areas = {int(s): int(area) for area, ss in env._game_rules.legal_action.substations_id_by_area.items() for s in ss}
    neural = ctl.base.topo_12_unsafe
    nn_hashes = [digest(action_vector(a)) for a in neural.gym_env.action_space.topo_actions_list]
    nn_to_n1 = {j: mapping[key] for j, key in enumerate(nn_hashes) if key in mapping}
    original_get = module.get_act; original_pool = module._get_tested_action
    context = {"step": 0, "calls": [], "source_wall_s": 0., "nn_s": 0.}
    source_stream = (path / "source_calls.jsonl").open("w", encoding="utf-8")

    def choose(observation, base_action, reward, done=False, **kwargs):
        t0 = time.perf_counter(); key = digest(observation.to_vect())
        tested = original_pool(observation)
        ids = [mapping[digest(action_vector(a))] for a in tested]
        if rule == "FULL":
            selected_positions = list(range(len(tested)))
        else:
            cheap, _ = shortlist_rules(observation.rho, observation.line_status, env.line_or_to_subid,
                 env.line_ex_to_subid, env.n_sub, areas, [all_subs[i] for i in ids], ids, key)
            if rule == "OLDNN128":
                t = time.perf_counter()
                gym_obs = neural.gym_env.observation_space.to_gym(observation)
                with torch.no_grad():
                    nn_order = neural.get_top_k(gym_obs, top_k=len(nn_hashes))
                context["nn_s"] += time.perf_counter()-t
                pos = {i: j for j, i in enumerate(ids)}
                selected_positions = []
                for j in nn_order:
                    i = nn_to_n1.get(int(j))
                    if i in pos: selected_positions.append(pos[i])
                # Complete uncovered slots using the same legal LOCAL ordering.
                seen = set(selected_positions)
                selected_positions += [j for j in cheap["LOCAL128"] if j not in seen]
                selected_positions = selected_positions[:128]
            else:
                selected_positions = cheap[rule]
        shortlist = [tested[i] for i in selected_positions]
        simulate = observation.simulate; cache = {}; cache_hits = 0; public_s = 0.
        pool_phase = "shortlist"

        def with_cache(action, *args, **kw):
            nonlocal cache_hits, public_s
            # Cache exists for this one get_act call only; author GreedyModule
            # invokes default one-step simulate and reads rho/reward/done/info.
            assert not args and not kw
            action_key = digest(action_vector(action))
            if action_key in cache:
                cache_hits += 1
                saved = cache[action_key]
                return SimpleNamespace(rho=saved[0].copy()), saved[1], saved[2], dict(saved[3])
            t = time.perf_counter(); result = simulate(action); elapsed = time.perf_counter()-t; public_s += elapsed
            future, rr, ended, info = result
            cache[action_key] = (future.rho.copy(), rr, ended, dict(info), elapsed, pool_phase)
            return result

        observation.simulate = with_cache
        module._get_tested_action = lambda ignored: shortlist
        fallback = False
        try:
            result = original_get(observation, base_action, reward, done=done, **kwargs)
            if rule != "FULL" and result is None and len(shortlist) < len(tested):
                fallback = True; pool_phase = "fallback"
                module._get_tested_action = lambda ignored: tested
                result = original_get(observation, base_action, reward, done=done, **kwargs)
        finally:
            module._get_tested_action = original_pool
            observation.simulate = simulate
        assert digest(observation.to_vect()) == key
        assert len(cache) <= len(tested) and (not fallback or len(cache) == len(tested))
        row = {"step": context["step"], "state_hash": key, "rule": rule,
             "pool_ids": ids, "shortlist_positions": selected_positions, "shortlist_size": len(shortlist),
             "full_size": len(tested), "public_queries": len(cache), "cache_hits": cache_hits,
             "fallback": fallback, "chosen_pool_id": None if result is None else mapping[digest(action_vector(result))],
             "source_s": time.perf_counter()-t0, "forecast_s": public_s,
             "labels": [{"combined_action_hash": h, "rho": float(v[0].max()), "reward": float(v[1]),
                 "done": bool(v[2]), **flags(v[3]), "forecast_s": v[4], "phase": v[5]} for h, v in cache.items()]}
        context["source_wall_s"] += row["source_s"]; context["calls"].append(row)
        source_stream.write(json.dumps(row, allow_nan=False)+"\n"); source_stream.flush()
        return result

    module.get_act = choose
    started = time.perf_counter(); reward = 0.; cost = 0.; decision_total = 0.; physical_total = 0.; queries_total = 0
    offered = 0; rows = []; query_committed = False
    arrays = {name: [] for name in ["action", "observation", "rho", "gen_p", "load_p", "actual_dispatch", "curtailment_mw", "storage_power"]}
    write_json(path / "metadata.json", {"rule": rule, "week": week, "cost_per_MW": env.gen_cost_per_MW,
               "dt_hours": float(env.delta_time_seconds)/3600., "cadence": 6,
               "action_library_hashes": all_hashes, "fallback": "No source-admissible action -> full source, cache reuses only same state/base-action/horizon outcomes."})
    try:
        with gzip.open(path / "steps.jsonl.gz", "wt", encoding="utf-8") as stream:
            for step in range(1, int(env.max_episode_duration())+1):
                assert time.perf_counter()-START < D["caps"]["wall_s"]
                assert time.perf_counter()-started < D["caps"]["episode_s"]
                context["step"] = step; before = digest(obs.to_vect()); counter = int(env.nb_highres_called); query_committed = False
                previous_curt = float(obs.curtailment_mw.sum(dtype=np.float64))
                t = time.perf_counter(); base, restore, proposal = ctl.propose(obs, reward, cadence=6)
                action = restore if proposal["offered"] else base
                decision_s = time.perf_counter()-t; decision_total += decision_s
                offered += int(proposal["offered"]); ah = digest(action_vector(action))
                t = time.perf_counter(); obs, reward, done, info = env.step(action); physical_s = time.perf_counter()-t; physical_total += physical_s
                counts["physical_steps"] += 1
                queries = int(env.nb_highres_called)-counter; queries_total += queries
                counts["public_forecasts"] += queries; query_committed = True
                raw = float(info["rewards"][f"{PREFIX}_grid_operational_cost"]); cost += raw; fl = flags(info)
                ledger = public_cost_ledger(obs, env, previous_curt) if (obs.gen_p > 0).any() and not fl["exceptions"] else None
                tolerance = float(cost_rounding_tolerance(obs, env)) if ledger else None
                if ledger: assert abs(ledger["recomputed_raw_cost"]-raw) <= tolerance
                complete = bool(obs.current_step >= env.max_episode_duration() or env.chronics_handler.done())
                row = {"step": step, "before_hash": before, "action_hash": ah, "after_hash": digest(obs.to_vect()),
                     "raw_cost": raw, "ledger": ledger, "ledger_tolerance": tolerance, "simulations": queries,
                     "decision_s": decision_s, "physical_s": physical_s, "done": bool(done), "complete": complete,
                     "restore": bool(proposal["offered"]), **fl}
                if rule == "FULL":
                    ref = reference[step-1]
                    assert (before, ah, row["after_hash"], raw) == (ref["before_hash"], ref["action_hash"], ref["after_hash"], ref["raw_cost"]), (week, step)
                rows.append(row); stream.write(json.dumps(row, allow_nan=False)+"\n"); stream.flush()
                for name in arrays:
                    value = action_vector(action) if name == "action" else obs.to_vect() if name == "observation" else getattr(obs, name)
                    arrays[name].append(np.array(value, copy=True))
                assert counts["physical_steps"] <= D["caps"]["physical_steps"]
                assert counts["public_forecasts"] <= D["caps"]["public_forecasts"]
                if step % 512 == 0 or done:
                    write_json(path / "progress.json", {"step": step, "counts": counts})
                    print(json.dumps({"week": week, "rule": rule, "step": step, "source_calls": len(context["calls"]), "wall_s": time.perf_counter()-started}), flush=True)
                if done: break
        if rule == "FULL": assert len(reference) == step
        np.savez_compressed(path / "vectors.npz", **{key: np.stack(values) for key, values in arrays.items()})
        summary = {"week": week, "rule": rule, "steps": step, "complete": complete, "cost": cost,
             "public_forecasts": queries_total, "n1_forecasts": sum(c["public_queries"] for c in context["calls"]),
             "n1_search_calls": len(context["calls"]), "fallbacks": sum(c["fallback"] for c in context["calls"]),
             "cache_hits": sum(c["cache_hits"] for c in context["calls"]),
             "controller_s": decision_total, "n1_source_s": context["source_wall_s"], "pretrained_nn_s": context["nn_s"],
             "physical_s": physical_total, "wall_s": time.perf_counter()-started,
             "restoration_offers": offered, "illegal": sum(r["illegal"] for r in rows), "ambiguous": sum(r["ambiguous"] for r in rows),
             "exception_steps": sum(bool(r["exceptions"]) for r in rows)}
        write_json(path / "summary.json", summary); return summary
    except Exception:
        if not query_committed: counts["public_forecasts"] += int(env.nb_highres_called)-counter
        np.savez_compressed(path / "partial_vectors.npz", **{key: np.stack(v) for key, v in arrays.items() if v})
        write_json(path / "failure.json", {"traceback": traceback.format_exc(), "counts": counts, "rows": len(rows), "wall_s": time.perf_counter()-started})
        raise
    finally:
        source_stream.close(); env.close()


def main():
    assert not (OUT / "finished.json").exists()
    import hashlib
    for name, sha in json.loads((OUT / "code_freeze.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    summaries = []
    for item in D["execution_order"]:
        summaries.append(run(item["week"], item["rule"])); write_json(OUT / "manifest.json", summaries)
    write_json(OUT / "finished.json", {"passed_engineering": True, "summaries": summaries, "counts": counts, "wall_s": time.perf_counter()-START})


if __name__ == "__main__": main()
