"""Serial controlled end-to-end timing. Recording/validation happen after act."""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"
OUT = ROOT / "outputs/grid26"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def persist(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def worker(item, design):
    entered = time.perf_counter()
    sys.path[:0] = [str(ROOT / "work/grid26"), str(ROOT / "work/grid08")]
    from common import np, make_env, ResidualControl, action_vector, digest, flags, write_json, PREFIX, public_cost_ledger, cost_rounding_tolerance
    from engine import Search
    week, method = item["week"], item["method"]
    path = OUT / "runs" / f"r{item['repeat']}__{week}__{method}"
    path.mkdir(parents=True, exist_ok=False)
    historical = ROOT / "outputs/grid23/runs" / (week + "__FULL")
    with gzip.open(historical / "steps.jsonl.gz", "rt", encoding="utf-8") as f:
        reference = [json.loads(line) for line in f]
    t = time.perf_counter()
    env = make_env(); env.seed(0); env.set_id(week); obs = env.reset()
    ctl = ResidualControl(env, obs)
    checkpoint = ROOT / item["checkpoint"] if item["checkpoint"] else None
    search = Search(env, ctl, item["rule"], item["budget"], checkpoint)
    setup_s = time.perf_counter()-t
    pending = []; context = {"step": 0}
    calls = []; rows = []; vectors = {"action": [], "observation": []}
    started_loop = time.perf_counter()
    controller_s = controller_cpu_s = physical_s = cost = bookkeeping_s = 0.
    reward = 0.; complete = False
    metadata = {**item, "library_hashes": [digest(action_vector(a)) for a in search.library],
        "checkpoint_sha256": search.checkpoint_sha, "cost_per_MW": env.gen_cost_per_MW,
        "dt_hours": float(env.delta_time_seconds)/3600., "cadence": 6, "process_id": os.getpid(),
        "torch_threads": __import__("torch").get_num_threads(), "timing": design["timing"],
        "clock": "perf_counter wall; process_time CPU diagnostic"}
    write_json(path / "metadata.json", metadata)

    def replacement(observation, base_action, previous_reward, done=False, **kwargs):
        # Two small vector copies retain the callback's input before LJN mutates
        # its base action. Hashing/compression and all expensive checks are later.
        state = observation.to_vect().copy(); base = action_vector(base_action).copy()
        selected, log, features, prior = search.choose(observation, base_action, previous_reward, done=done, **kwargs)
        pending.append((state, base, selected, log, features, prior))
        return selected

    ctl.base.topo_n1_unsafe.get_act = replacement
    initial = int(env.nb_highres_called)
    active_simulation = None
    obs_class = type(obs)
    owned_simulate = "simulate" in obs_class.__dict__
    original_simulate = obs_class.simulate
    def measured_simulate(observation, *args, **kwargs):
        if active_simulation is None:
            return original_simulate(observation, *args, **kwargs)
        t0 = time.perf_counter()
        try: return original_simulate(observation, *args, **kwargs)
        finally:
            active_simulation["count"] += 1
            active_simulation["seconds"] += time.perf_counter()-t0
    obs_class.simulate = measured_simulate
    try:
        for step in range(1, int(env.max_episode_duration())+1):
            assert time.perf_counter()-entered < design["caps"]["episode_s"]
            context["step"] = step; pending.clear()
            before = digest(obs.to_vect()); counter = int(env.nb_highres_called)
            prev_curt = float(obs.curtailment_mw.sum(dtype=np.float64))
            simulation = {"count": 0, "seconds": 0.}
            active_simulation = simulation
            cpu = time.process_time(); start = time.perf_counter()
            try:
                base, restore, proposal = ctl.propose(obs, reward, cadence=6)
                action = restore if proposal["offered"] else base
            finally:
                decision_s = time.perf_counter()-start
                decision_cpu_s = time.process_time()-cpu
                active_simulation = None
            controller_s += decision_s; controller_cpu_s += decision_cpu_s
            forecasts = int(env.nb_highres_called)-counter
            assert forecasts == simulation["count"], (step, forecasts, simulation["count"])
            t = time.perf_counter(); obs, reward, done, info = env.step(action)
            ps = time.perf_counter()-t; physical_s += ps
            t = time.perf_counter()
            raw = float(info["rewards"][f"{PREFIX}_grid_operational_cost"]); cost += raw
            f = flags(info)
            ledger = public_cost_ledger(obs, env, prev_curt) if (obs.gen_p > 0).any() and not f["exceptions"] else None
            tolerance = float(cost_rounding_tolerance(obs, env)) if ledger else None
            if ledger: assert abs(ledger["recomputed_raw_cost"]-raw) <= tolerance
            complete = bool(obs.current_step >= env.max_episode_duration() or env.chronics_handler.done())
            row = {"step": step, "before_hash": before, "action_hash": digest(action_vector(action)),
                "after_hash": digest(obs.to_vect()), "raw_cost": raw, "simulations": forecasts,
                "decision_s": decision_s, "decision_cpu_s": decision_cpu_s, "physical_s": ps,
                "all_forecast_s": simulation["seconds"], "all_forecast_calls": simulation["count"],
                "done": bool(done), "complete": complete, "restore": bool(proposal["offered"]),
                "ledger": ledger, "ledger_tolerance": tolerance, **f}
            for key in ["step", "before_hash", "action_hash", "after_hash", "raw_cost", "done", "complete"]:
                assert row[key] == reference[step-1][key], (method, week, step, key)
            for state, base_vector, selected, log, features, prior in pending:
                number = len(calls)
                log["step"] = step; log["state_hash"] = digest(state)
                log["input_hash"] = None if features is None else __import__("components").fingerprint(features)
                log["prior_hash"] = None if prior is None else digest(prior)
                log["returned_action_hash"] = None if selected is None else digest(action_vector(selected))
                base_action = env.action_space(); base_action.from_vect(base_vector)
                previous_obs = obs.copy(); previous_obs.from_vect(state)
                _, _, _, actions, _ = search.prepare(previous_obs, base_action)
                for r in log["outcomes"]: r["combined_hash"] = digest(action_vector(actions[r["pool_id"]] + base_action))
                assert log["source_s"] <= decision_s
                np.savez_compressed(path / f"call{number:03d}_state.npz", observation=state, base_action=base_vector)
                if features is not None: np.savez_compressed(path / f"call{number:03d}_inputs.npz", **features, prior=prior)
                write_json(path / f"call{number:03d}.json", log)
                calls.append(log)
            rows.append(row)
            vectors["action"].append(action_vector(action).copy()); vectors["observation"].append(obs.to_vect().copy())
            bookkeeping_s += time.perf_counter()-t
            if step % 512 == 0 or done:
                write_json(path / "progress.json", {"step": step, "n1_calls": len(calls), "controller_s": controller_s,
                    "public_forecasts": int(env.nb_highres_called)-initial})
                print(json.dumps({"repeat": item["repeat"], "week": week, "method": method, "step": step,
                    "controller_s": controller_s, "wall_s": time.perf_counter()-entered}), flush=True)
            if done: break
        assert len(rows) == len(reference)
        loop_s = time.perf_counter()-started_loop
        t = time.perf_counter()
        with gzip.open(path / "steps.jsonl.gz", "wt", encoding="utf-8") as f:
            for row in rows: f.write(json.dumps(row, allow_nan=False) + "\n")
        np.savez_compressed(path / "vectors.npz", **{k: np.stack(v) for k, v in vectors.items()})
        final_save_s = time.perf_counter()-t
        summary = {**item, "steps": len(rows), "complete": complete, "cost": cost,
            "controller_s": controller_s, "controller_cpu_s": controller_cpu_s,
            "physical_s": physical_s, "setup_s": setup_s, "preloop_import_s": started_loop-entered-setup_s,
            "loop_wall_s": loop_s, "bookkeeping_s": bookkeeping_s, "final_save_s": final_save_s,
            "public_forecasts": sum(r["simulations"] for r in rows), "n1_calls": len(calls),
            "n1_forecasts": sum(c["public_queries"] for c in calls), "fallbacks": sum(c["fallback"] for c in calls),
            "all_forecast_s": sum(r["all_forecast_s"] for r in rows),
            "n1_source_s": sum(c["source_s"] for c in calls), "n1_forecast_s": sum(c["forecast_s"] for c in calls),
            "n1_prepare_s": sum(c["prepare_s"] for c in calls), "prior_s": sum(c["prior_s"] for c in calls),
            "feature_s": sum(c["feature_s"] for c in calls), "forward_rank_s": sum(c["forward_rank_s"] for c in calls),
            "illegal": sum(r["illegal"] for r in rows), "ambiguous": sum(r["ambiguous"] for r in rows),
            "exception_steps": sum(bool(r["exceptions"]) for r in rows), "exact_full_trajectory": True,
            "wall_s": time.perf_counter()-entered, "new_model_fits": 0}
        write_json(path / "summary.json", summary)
        return summary
    except Exception:
        write_json(path / "failure.json", {"traceback": traceback.format_exc(), "rows": len(rows),
            "controller_s": controller_s, "public_forecasts": int(env.nb_highres_called)-initial})
        if rows:
            with gzip.open(path / "partial_steps.jsonl.gz", "wt", encoding="utf-8") as f:
                for r in rows: f.write(json.dumps(r, allow_nan=False)+"\n")
            np.savez_compressed(path / "partial_vectors.npz", **{k: np.stack(v) for k, v in vectors.items()})
        raise
    finally:
        if owned_simulate: obs_class.simulate = original_simulate
        else: delattr(obs_class, "simulate")
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--phase", choices=["pilot", "remaining", "all"], default="pilot")
    args = parser.parse_args()
    design = load(OUT / "design.json")
    assert load(OUT / "preflight.json")["passed"]
    for file, expected in load(OUT / "code_freeze.json").items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == expected, file
    if args.worker is not None:
        worker(design["execution_order"][args.worker], design)
        return
    if args.phase == "remaining": assert load(OUT / "pilot_audit.json")["passed"]
    started = time.perf_counter()
    summaries = load(OUT / "manifest.json") if (OUT / "manifest.json").exists() else []
    existing = {(s["repeat"], s["week"], s["method"]) for s in summaries}
    (OUT / "RUN_STATE.md").write_text(f"RUNNING {args.phase}: one serial worker. Frozen timing design, unchanged weights.\n", encoding="utf-8")
    for index, item in enumerate(design["execution_order"]):
        if args.phase != "all" and item["phase"] != args.phase: continue
        if (item["repeat"], item["week"], item["method"]) in existing: continue
        assert time.perf_counter()-started < design["caps"]["wall_s"]
        with (OUT / f"worker_{index:02d}.log").open("w", encoding="utf-8") as f:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(index)],
                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        assert process.returncode == 0, (index, item, process.returncode)
        path = OUT / "runs" / f"r{item['repeat']}__{item['week']}__{item['method']}" / "summary.json"
        summary = load(path); summaries.append(summary); persist(OUT / "manifest.json", summaries)
        counts = {"physical_steps": sum(s["steps"] for s in summaries), "public_forecasts": sum(s["public_forecasts"] for s in summaries), "new_model_fits": 0}
        assert counts["physical_steps"] <= design["caps"]["physical_steps"] and counts["public_forecasts"] <= design["caps"]["public_forecasts"]
        print(json.dumps({"completed_runs": len(summaries), "planned": len(design["execution_order"]),
            "week": item["week"], "method": item["method"], "repeat": item["repeat"],
            "controller_s": summary["controller_s"], "n1_source_s": summary["n1_source_s"], "complete": summary["complete"]}), flush=True)
    filename = "finished.json" if len(summaries) == len(design["execution_order"]) else args.phase + "_finished.json"
    persist(OUT / filename, {"passed_engineering": True, "summaries": summaries, "counts": counts,
        "serial_workers": 1, "wall_s_this_process": time.perf_counter()-started})
    print(json.dumps({"phase_finished": args.phase, "runs": len(summaries), "counts": counts}), flush=True)


if __name__ == "__main__": main()
