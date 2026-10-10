"""Instrument existing full N1 calls, preserve original physical controller exactly."""
import gzip
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/grid08"))
from common import np, write_json, make_env, ResidualControl, action_vector, digest, flags, PREFIX
from rules import shortlist_rules

OUT = ROOT / "outputs/grid15"
D = json.loads((OUT / "design.json").read_text())
START = time.perf_counter()
counts = {"physical_steps": 0, "public_forecasts": 0, "n1_calls": 0, "training_runs": 0}


def reference_rows(week):
    with gzip.open(ROOT / f"outputs/grid14/runs/{week}__AUTHOR/steps.jsonl.gz", "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def run(week):
    path = OUT / "runs" / week
    path.mkdir(parents=True, exist_ok=False)
    env = make_env(); env.seed(0); env.set_id(week); obs = env.reset()
    ctl = ResidualControl(env, obs)
    ref = reference_rows(week)
    pool = ctl.base.topo_n1_unsafe.topo_act_list
    pool_hash = [digest(action_vector(a)) for a in pool]
    assert len(set(pool_hash)) == len(pool)
    index = {key: i for i, key in enumerate(pool_hash)}
    subids = [int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0]) for a in pool]
    if not (OUT / "action_library.npz").exists():
        np.savez_compressed(OUT / "action_library.npz", actions=np.stack([action_vector(a) for a in pool]), subids=np.array(subids))
        write_json(OUT / "public_static.json", {"action_hashes": pool_hash, "n_sub": env.n_sub,
                   "line_or": env.line_or_to_subid, "line_ex": env.line_ex_to_subid,
                   "areas": {int(s): int(area) for area, ss in env._game_rules.legal_action.substations_id_by_area.items() for s in ss},
                   "observation_layout": [{"name": a, "shape": np.asarray(obs._get_array_from_attr_name(a)).shape,
                     "size": np.asarray(obs._get_array_from_attr_name(a)).size} for a in type(obs).attr_list_vect]})
    static = json.loads((OUT / "public_static.json").read_text())
    assert pool_hash == static["action_hashes"]
    areas = {int(k): v for k, v in static["areas"].items()}
    context = {"step": 0, "calls": [], "module_s": 0., "simulate_s": 0., "bookkeeping_s": 0.}
    original = ctl.base.topo_n1_unsafe.get_act

    def tracked(observation, base_action, reward, done=False, **kwargs):
        t0 = time.perf_counter(); state_key = digest(observation.to_vect())
        candidate_results = []
        simulate = observation.simulate
        forecasting_s = 0.; bookkeeping_s = 0.

        def capture(*args, **kw):
            nonlocal forecasting_s, bookkeeping_s
            t = time.perf_counter(); returned = simulate(*args, **kw)
            duration = time.perf_counter()-t; forecasting_s += duration
            t = time.perf_counter()
            future, score, ended, info = returned
            fl = flags(info)
            candidate_results.append({"rho": float(future.rho.max()), "reward": float(score),
                 "done": bool(ended), "illegal": fl["illegal"], "ambiguous": fl["ambiguous"],
                 "exceptions": fl["exceptions"], "forecast_s": duration,
                 "combined_action_hash": digest(action_vector(args[0]))})
            bookkeeping_s += time.perf_counter()-t
            return returned

        observation.simulate = capture
        try:
            selected = original(observation, base_action, reward, done=done, **kwargs)
        finally:
            observation.simulate = simulate
        module_s = time.perf_counter()-t0
        tested = ctl.base.topo_n1_unsafe.tested_action
        assert len(candidate_results) == len(tested)
        ids = [index[digest(action_vector(a))] for a in tested]
        candidate_subs = [subids[i] for i in ids]
        rules, diagnostic = shortlist_rules(observation.rho, observation.line_status,
             env.line_or_to_subid, env.line_ex_to_subid, env.n_sub, areas, candidate_subs, ids, state_key)
        stored_scores = ctl.base.topo_n1_unsafe.resulting_rewards if tested else []
        rho_limit = min(float(observation.rho.max()), float(kwargs["rho_threshold"])) if "rho_threshold" in kwargs else float(observation.rho.max())
        for i, row in enumerate(candidate_results):
            assert row["combined_action_hash"] == digest(action_vector(tested[i]+base_action))
            row["source_admissible"] = bool(0 < row["rho"] < rho_limit and not row["exceptions"] and not row["done"])
            expected = row["reward"] if row["source_admissible"] else ctl.base.topo_n1_unsafe.null_action_reward
            assert float(np.float32(expected)) == float(stored_scores[i])
            row["source_score"] = float(stored_scores[i])
            row["strict_admissible"] = row["source_admissible"] and not row["illegal"] and not row["ambiguous"]
        choice = None if selected is None else index[digest(action_vector(selected))]
        if tested and max(stored_scores) > ctl.base.topo_n1_unsafe.null_action_reward:
            assert choice == ids[int(np.argmax(stored_scores))]
        else:
            assert choice is None
        call_id = len(context["calls"])
        np.savez_compressed(path / f"call{call_id:03d}_state.npz", observation=observation.to_vect(),
             base_action=action_vector(base_action), rho=observation.rho, live=observation.line_status,
             cooldown_sub=observation.time_before_cooldown_sub)
        data = {"week": week, "action_step": context["step"], "state_hash": state_key,
             "pool_ids": ids, "subids": candidate_subs, "rules": rules, "cheap_features": diagnostic,
             "current_rho": float(observation.rho.max()), "rho_limit": rho_limit,
             "source_choice": choice, "outcomes": candidate_results,
             "module_s_instrumented_before_persistence": module_s, "forecast_s": forecasting_s,
             "forecast_record_bookkeeping_s": bookkeeping_s, "source": "Original full N1, unchanged get_act; public feedback only."}
        write_json(path / f"call{call_id:03d}.json", data)
        context["calls"].append({"call_id": call_id, "step": context["step"], "candidates": len(ids)})
        context["module_s"] += module_s; context["simulate_s"] += forecasting_s
        context["bookkeeping_s"] += bookkeeping_s
        counts["n1_calls"] += 1
        assert digest(observation.to_vect()) == state_key
        return selected

    ctl.base.topo_n1_unsafe.get_act = tracked
    reward = 0.; cost = 0.; total_decision_s = 0.; step_s = 0.
    started = time.perf_counter(); query_count = 0; query_committed = False
    try:
        with gzip.open(path / "steps.jsonl.gz", "wt", encoding="utf-8") as stream:
            for step in range(1, int(env.max_episode_duration())+1):
                assert time.perf_counter()-START < D["caps"]["wall_s"]
                assert time.perf_counter()-started < D["caps"]["episode_s"]
                context["step"] = step; before = digest(obs.to_vect()); counter = int(env.nb_highres_called); query_committed = False
                t = time.perf_counter(); base, restore, proposal = ctl.propose(obs, reward, cadence=6)
                decision_s = time.perf_counter()-t; total_decision_s += decision_s
                act = restore if proposal["offered"] else base
                ah = digest(action_vector(act)); t = time.perf_counter()
                obs, reward, done, info = env.step(act); physical_s = time.perf_counter()-t; step_s += physical_s
                counts["physical_steps"] += 1
                queries = int(env.nb_highres_called)-counter
                counts["public_forecasts"] += queries; query_count += queries; query_committed = True
                raw = float(info["rewards"][f"{PREFIX}_grid_operational_cost"]); cost += raw
                after = digest(obs.to_vect()); r = ref[step-1]
                assert (before, ah, after, raw) == (r["before_hash"], r["action_hash"], r["after_hash"], r["raw_cost"]), (week, step)
                assert counts["physical_steps"] <= D["caps"]["physical_steps"]
                assert counts["public_forecasts"] <= D["caps"]["public_forecasts"]
                row = {"step": step, "before_hash": before, "action_hash": ah, "after_hash": after,
                     "raw_cost": raw, "simulations": queries, "decision_s": decision_s, "physics_s": physical_s,
                     "done": bool(done), "complete": bool(obs.current_step >= env.max_episode_duration() or env.chronics_handler.done()), **flags(info)}
                stream.write(json.dumps(row, allow_nan=False)+"\n"); stream.flush()
                if step % 256 == 0 or done:
                    write_json(path / "progress.json", {"step": step, "counts": counts})
                    print(json.dumps({"week": week, "step": step, "n1_calls": len(context["calls"]), "wall_s": time.perf_counter()-started}), flush=True)
                if done: break
        assert len(ref) == step
        summary = {"week": week, "steps": step, "complete": row["complete"], "cost": cost,
             "physical_control_exact_to_grid14": True, "calls": context["calls"],
             "total_decision_s_instrumented": total_decision_s, "n1_module_s_instrumented": context["module_s"],
             "n1_public_forecast_s": context["simulate_s"], "forecast_record_bookkeeping_s": context["bookkeeping_s"],
             "physical_step_s": step_s, "public_forecasts": query_count, "wall_s": time.perf_counter()-started}
        write_json(path / "summary.json", summary)
        return summary
    except Exception:
        # Include query work performed before a step-level gate failure.
        used = int(env.nb_highres_called)-counter
        if not query_committed:
            counts["public_forecasts"] += used
        write_json(path / "failure.json", {"traceback": traceback.format_exc(), "counts_committed": counts,
                   "current_decision_forecasts": used, "n1_calls": context["calls"], "wall_s": time.perf_counter()-started,
                   "current_decision_forecasts_in_total": True,
                   "note": "Current decision count is included in resource total, not an extra amount to add; no unverified model conclusion."})
        raise
    finally:
        env.close()


def main():
    assert not (OUT / "finished.json").exists()
    for relative, expected in json.loads((OUT / "code_freeze.json").read_text()).items():
        import hashlib
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    summaries = []
    for week in D["weeks"]:
        summaries.append(run(week)); write_json(OUT / "manifest.json", summaries)
    write_json(OUT / "finished.json", {"passed": True, "summaries": summaries, "counts": counts, "wall_s": time.perf_counter()-START})


if __name__ == "__main__":
    main()
