"""Frozen-priority auxiliary methods in complete developmental week rollouts."""
import gzip
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid24")]
from common import np, make_env, ResidualControl, action_vector, digest, flags, write_json, PREFIX, public_cost_ledger, cost_rounding_tolerance
from guard_adapter import Search
OUT = ROOT / "outputs/grid24"
D = json.loads((OUT / "design.json").read_text())
START = time.perf_counter()
counts = {"physical_steps": 0, "public_forecasts": 0, "model_fits": 0}


def read_steps(path):
    with gzip.open(path, "rt", encoding="utf-8") as f: return [json.loads(line) for line in f]


def run(item):
    week, method, rule, budget = (item[k] for k in ["week", "method", "rule", "budget"])
    path = OUT / "runs" / (week + "__" + method)
    path.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    env = make_env(); env.seed(0); env.set_id(week); obs = env.reset()
    ctl = ResidualControl(env, obs)
    checkpoint = ROOT / item["checkpoint"] if item.get("checkpoint") else None
    search = Search(env, ctl, rule, budget, checkpoint)
    calls = []; context = {"step": 0}
    old_rows = read_steps(ROOT / f"outputs/grid19/runs/{week}/steps.jsonl.gz") if rule == "FULL" else None

    def replacement(observation, base_action, reward, done=False, **kwargs):
        number = len(calls)
        before_queries = int(env.nb_highres_called)
        selected, log, features, prior = search.choose(observation, base_action, reward, done=done, **kwargs)
        assert int(env.nb_highres_called)-before_queries == log["public_queries"]
        log["step"] = context["step"]
        np.savez_compressed(path / f"call{number:03d}_state.npz", observation=observation.to_vect(), base_action=action_vector(base_action))
        if features is not None:
            np.savez_compressed(path / f"call{number:03d}_inputs.npz", **features, prior=prior)
        write_json(path / f"call{number:03d}.json", log)
        calls.append(log)
        return selected

    ctl.base.topo_n1_unsafe.get_act = replacement
    metadata = {"week": week, "method": method, "rule": rule, "budget": budget, "checkpoint_sha256": search.checkpoint_sha,
                "cost_per_MW": env.gen_cost_per_MW, "dt_hours": float(env.delta_time_seconds)/3600.,
                "cadence": 6, "strict_all_methods": True, "cache_scope": "One state/base/default one-step forecast only; no repeated action query.",
                "library_hashes": [digest(action_vector(a)) for a in search.library]}
    write_json(path / "metadata.json", metadata)
    rows = []; vectors = {"action": [], "observation": []}
    reward = cost = controller_s = physical_s = 0.
    committed = True; counter = int(env.nb_highres_called)
    try:
        with gzip.open(path / "steps.jsonl.gz", "wt", encoding="utf-8") as f:
            for step in range(1, int(env.max_episode_duration())+1):
                assert time.perf_counter()-START < D["caps"]["wall_s"]
                assert time.perf_counter()-started < D["caps"]["episode_s"]
                context["step"] = step
                before = digest(obs.to_vect()); counter = int(env.nb_highres_called); committed = False
                prev_curt = float(obs.curtailment_mw.sum(dtype=np.float64))
                t = time.perf_counter(); base, restore, proposal = ctl.propose(obs, reward, cadence=6)
                action = restore if proposal["offered"] else base
                decision_s = time.perf_counter()-t; controller_s += decision_s
                t = time.perf_counter(); obs, reward, done, info = env.step(action)
                ps = time.perf_counter()-t; physical_s += ps
                forecasts = int(env.nb_highres_called)-counter
                counts["physical_steps"] += 1; counts["public_forecasts"] += forecasts; committed = True
                raw = float(info["rewards"][f"{PREFIX}_grid_operational_cost"]); cost += raw
                fl = flags(info)
                ledger = public_cost_ledger(obs, env, prev_curt) if (obs.gen_p > 0).any() and not fl["exceptions"] else None
                tolerance = float(cost_rounding_tolerance(obs, env)) if ledger else None
                if ledger: assert abs(ledger["recomputed_raw_cost"]-raw) <= tolerance
                complete = bool(obs.current_step >= env.max_episode_duration() or env.chronics_handler.done())
                row = {"step": step, "before_hash": before, "action_hash": digest(action_vector(action)), "after_hash": digest(obs.to_vect()),
                       "raw_cost": raw, "simulations": forecasts, "decision_s": decision_s, "physical_s": ps,
                       "done": bool(done), "complete": complete, "restore": bool(proposal["offered"]),
                       "ledger": ledger, "ledger_tolerance": tolerance, **fl}
                if old_rows is not None:
                    for key in ["step", "before_hash", "action_hash", "after_hash", "raw_cost", "done", "complete"]:
                        assert row[key] == old_rows[step-1][key], (week, step, key)
                rows.append(row); f.write(json.dumps(row, allow_nan=False)+"\n"); f.flush()
                vectors["action"].append(action_vector(action).copy()); vectors["observation"].append(obs.to_vect().copy())
                assert counts["physical_steps"] <= D["caps"]["physical_steps"] and counts["public_forecasts"] <= D["caps"]["public_forecasts"]
                if step % 512 == 0 or done:
                    write_json(path / "progress.json", {"step": step, "calls": len(calls), "counts": counts})
                    print(json.dumps({"week": week, "method": method, "step": step, "calls": len(calls), "wall_s": time.perf_counter()-started}), flush=True)
                if done: break
        if old_rows is not None: assert len(old_rows) == len(rows)
        np.savez_compressed(path / "vectors.npz", **{k: np.stack(v) for k, v in vectors.items()})
        summary = {"week": week, "method": method, "rule": rule, "budget": budget, "steps": step, "complete": complete, "cost": cost,
                   "controller_s": controller_s, "physical_s": physical_s, "public_forecasts": sum(r["simulations"] for r in rows),
                   "n1_forecasts": sum(c["public_queries"] for c in calls), "n1_calls": len(calls), "fallbacks": sum(c["fallback"] for c in calls),
                   "feature_s": sum(c["feature_s"] for c in calls), "prior_s": sum(c["prior_s"] for c in calls),
                   "model_forward_rank_s": sum(c["forward_rank_s"] for c in calls), "n1_source_s": sum(c["source_s"] for c in calls),
                   "restoration_offers": sum(r["restore"] for r in rows), "illegal": sum(r["illegal"] for r in rows),
                   "ambiguous": sum(r["ambiguous"] for r in rows), "exception_steps": sum(bool(r["exceptions"]) for r in rows), "wall_s": time.perf_counter()-started}
        write_json(path / "summary.json", summary)
        return summary
    except Exception:
        if not committed: counts["public_forecasts"] += int(env.nb_highres_called)-counter
        np.savez_compressed(path / "partial_vectors.npz", **{k: np.stack(v) for k, v in vectors.items() if v})
        write_json(path / "failure.json", {"traceback": traceback.format_exc(), "rows": len(rows), "counts": counts})
        raise
    finally:
        env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pilot", "remaining", "all"], default="pilot")
    phase = parser.parse_args().phase
    selected = [x for x in D["execution_order"] if phase == "all" or x["phase"] == phase]
    assert json.loads((ROOT / "outputs/grid22/audit.json").read_text())["passed"]
    assert json.loads((OUT / "preflight.json").read_text())["passed"]
    for name, sha in json.loads((OUT / "code_freeze.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    results = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() else []
    completed = {(s["week"], s["method"]) for s in results}
    for item in selected:
        if (item["week"], item["method"]) in completed:
            print(json.dumps({"preserved_completed": item["week"] + "__" + item["method"]}), flush=True)
            continue
        results.append(run(item)); write_json(OUT / "manifest.json", results)
    aggregate = {"physical_steps": sum(s["steps"] for s in results),
                 "public_forecasts": sum(s["public_forecasts"] for s in results), "model_fits": 0}
    write_json(OUT / ("finished.json" if len(results) == len(D["execution_order"]) else phase + "_finished.json"),
        {"passed_engineering": True, "summaries": results, "counts": aggregate, "phase": phase,
         "wall_s_this_process": time.perf_counter()-START, "wall_s_sum_runs": sum(s["wall_s"] for s in results)})
    print(json.dumps({"complete": True, "runs": len(results), "counts": counts}))


if __name__ == "__main__": main()
