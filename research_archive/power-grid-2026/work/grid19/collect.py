"""Known endpoint repair applied only to the complete disconnected-grid search."""
import gzip
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid17")]
from common import np, make_env, ResidualControl, action_vector, digest, flags, write_json, PREFIX, public_cost_ledger, cost_rounding_tolerance
from mask import mask_disconnected
OUT = ROOT / "outputs/grid19"; D = json.loads((OUT / "design.json").read_text())
START = time.perf_counter(); counts = {"physical_steps": 0, "public_forecasts": 0, "model_fits": 0}


def run(week):
    path = OUT / "runs" / week; path.mkdir(parents=True, exist_ok=False)
    env = make_env(); env.seed(0); env.set_id(week); obs = env.reset(); ctl = ResidualControl(env, obs)
    module = ctl.base.topo_n1_unsafe; original_get = module.get_act; original_pool = module._get_tested_action
    library = module.topo_act_list; pool_mapping = {digest(action_vector(a)): i for i, a in enumerate(library)}
    calls = []; context = {"step": 0}

    def repaired(observation, base_action, reward, done=False, **kwargs):
        before = digest(observation.to_vect()); t0 = time.perf_counter(); tested = original_pool(observation)
        ids = [pool_mapping[digest(action_vector(a))] for a in tested]; replacements = []; changes = []
        for candidate in tested:
            original_combined, expected, changed, intent = mask_disconnected(candidate, base_action, observation, env)
            copied = candidate.copy(); bus = copied.set_bus.copy(); bus[changed] = 0
            if changed: copied.set_bus = bus
            assert np.array_equal(action_vector(copied+base_action), action_vector(expected))
            replacements.append(copied)
            changes.append({"changed_positions": changed, "intentional_base_lines": intent, "original_combined_hash": digest(action_vector(original_combined))})
        simulate = observation.simulate; outcomes = []

        def tracked(action, *args, **kw):
            assert not args and not kw
            t = time.perf_counter(); result = simulate(action); future, rr, ended, info = result
            outcomes.append({"combined_action_hash": digest(action_vector(action)), "rho": float(future.rho.max()),
                             "reward": float(rr), "done": bool(ended), **flags(info), "forecast_s": time.perf_counter()-t})
            return result

        observation.simulate = tracked; module._get_tested_action = lambda ignored: replacements
        try:
            selected = original_get(observation, base_action, reward, done=done, **kwargs)
        finally:
            observation.simulate = simulate; module._get_tested_action = original_pool
        assert len(outcomes) == len(replacements) and digest(observation.to_vect()) == before
        choice_index = None if selected is None else next(i for i, a in enumerate(replacements) if a is selected)
        limit = min(float(observation.rho.max()), float(kwargs.get("rho_threshold", observation.rho.max())))
        for i, (outcome, candidate) in enumerate(zip(outcomes, replacements)):
            assert outcome["combined_action_hash"] == digest(action_vector(candidate+base_action))
            admitted = 0 < outcome["rho"] < limit and not outcome["done"] and not outcome["exceptions"]
            outcome["source_admissible"] = bool(admitted)
            outcome["strict_admissible"] = bool(admitted and not outcome["illegal"] and not outcome["ambiguous"])
            score = outcome["reward"] if admitted else -100.
            assert float(np.float32(score)) == float(module.resulting_rewards[i])
        if choice_index is not None: assert choice_index == int(np.argmax(module.resulting_rewards))
        else: assert not len(replacements) or max(module.resulting_rewards) <= -100.
        number = len(calls)
        np.savez_compressed(path / f"call{number:03d}_state.npz", observation=observation.to_vect(), base_action=action_vector(base_action))
        record = {"step": context["step"], "state_hash": before, "pool_ids": ids, "changes": changes,
                  "chosen_index": choice_index, "chosen_pool_id": None if choice_index is None else ids[choice_index],
                  "rho_limit": limit, "outcomes": outcomes, "source_s": time.perf_counter()-t0,
                  "scope": "All public source candidate outcomes, known endpoint repair. No new predictive model."}
        write_json(path / f"call{number:03d}.json", record); calls.append({"step": context["step"], "candidates": len(ids), "changed_candidates": sum(bool(c["changed_positions"]) for c in changes)})
        return selected

    module.get_act = repaired; rows = []; arrays = {"action": [], "observation": []}
    reward = 0.; cost = 0.; controller = 0.; started = time.perf_counter(); committed = True
    count = int(env.nb_highres_called)
    metadata = {"week": week, "cost_per_MW": env.gen_cost_per_MW, "dt_hours": float(env.delta_time_seconds)/3600., "action_hashes": [digest(action_vector(a)) for a in library]}
    write_json(path / "metadata.json", metadata)
    try:
        with gzip.open(path / "steps.jsonl.gz", "wt", encoding="utf-8") as stream:
            for step in range(1, int(env.max_episode_duration())+1):
                assert time.perf_counter()-START < D["caps"]["wall_s"] and time.perf_counter()-started < D["caps"]["episode_s"]
                context["step"] = step; before = digest(obs.to_vect()); count = int(env.nb_highres_called); committed = False
                previous_curt = float(obs.curtailment_mw.sum(dtype=np.float64)); t = time.perf_counter()
                base, restore, proposal = ctl.propose(obs, reward, cadence=6); action = restore if proposal["offered"] else base
                elapsed = time.perf_counter()-t; controller += elapsed
                obs, reward, done, info = env.step(action); counts["physical_steps"] += 1
                forecasts = int(env.nb_highres_called)-count; counts["public_forecasts"] += forecasts; committed = True
                raw = float(info["rewards"][f"{PREFIX}_grid_operational_cost"]); cost += raw; fl = flags(info)
                ledger = public_cost_ledger(obs, env, previous_curt) if (obs.gen_p > 0).any() and not fl["exceptions"] else None
                tolerance = float(cost_rounding_tolerance(obs, env)) if ledger else None
                if ledger: assert abs(ledger["recomputed_raw_cost"]-raw) <= tolerance
                complete = bool(obs.current_step >= env.max_episode_duration() or env.chronics_handler.done())
                row = {"step": step, "before_hash": before, "action_hash": digest(action_vector(action)), "after_hash": digest(obs.to_vect()),
                       "raw_cost": raw, "simulations": forecasts, "decision_s": elapsed, "done": bool(done), "complete": complete,
                       "restore": bool(proposal["offered"]), "ledger": ledger, "ledger_tolerance": tolerance, **fl}
                rows.append(row); stream.write(json.dumps(row, allow_nan=False)+"\n"); stream.flush()
                arrays["action"].append(action_vector(action)); arrays["observation"].append(obs.to_vect().copy())
                assert counts["physical_steps"] <= D["caps"]["physical_steps"] and counts["public_forecasts"] <= D["caps"]["public_forecasts"]
                if step % 512 == 0 or done: print(json.dumps({"week": week, "step": step, "calls": len(calls), "wall_s": time.perf_counter()-started}), flush=True)
                if done: break
        np.savez_compressed(path / "vectors.npz", **{k: np.stack(v) for k, v in arrays.items()})
        result = {"week": week, "steps": step, "complete": complete, "cost": cost, "controller_s": controller,
                  "public_forecasts": sum(r["simulations"] for r in rows), "n1_forecasts": sum(c["candidates"] for c in calls),
                  "calls": calls, "restoration_offers": sum(r["restore"] for r in rows), "illegal": sum(r["illegal"] for r in rows),
                  "ambiguous": sum(r["ambiguous"] for r in rows), "exception_steps": sum(bool(r["exceptions"]) for r in rows), "wall_s": time.perf_counter()-started}
        write_json(path / "summary.json", result); return result
    except Exception:
        if not committed: counts["public_forecasts"] += int(env.nb_highres_called)-count
        np.savez_compressed(path / "partial_vectors.npz", **{k: np.stack(v) for k, v in arrays.items() if v})
        write_json(path / "failure.json", {"traceback": traceback.format_exc(), "counts": counts, "rows": len(rows)})
        raise
    finally:
        env.close()


def main():
    assert not (OUT / "finished.json").exists()
    for name, sha in json.loads((OUT / "code_freeze.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    results = []
    for week in D["weeks"]:
        results.append(run(week)); write_json(OUT / "manifest.json", results)
    write_json(OUT / "finished.json", {"passed_engineering": True, "summaries": results, "counts": counts, "wall_s": time.perf_counter()-START})


if __name__ == "__main__": main()
