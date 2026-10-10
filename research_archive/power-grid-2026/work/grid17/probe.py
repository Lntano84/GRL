"""Replay frozen physical prefixes, then compare paired public forecasts only."""
import gzip
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid17")]
from common import np, make_env, ResidualControl, action_vector, digest, flags, write_json
from mask import mask_disconnected
OUT = ROOT / "outputs/grid17"
D = json.loads((OUT / "design.json").read_text())
START = time.perf_counter(); counts = {"physical_steps": 0, "public_forecasts": 0, "model_fits": 0}


def check_cap():
    assert time.perf_counter()-START < D["caps"]["wall_s"]
    assert counts["physical_steps"] <= D["caps"]["physical_steps"]
    assert counts["public_forecasts"] <= D["caps"]["public_forecasts"]


def run(target):
    env = make_env(); env.seed(0); env.set_id(target["week"]); obs = env.reset()
    ctl = ResidualControl(env, obs)
    library = ctl.base.topo_n1_unsafe.topo_act_list
    ref = ROOT / f"outputs/grid16/runs/{target['week']}__FULL"
    with gzip.open(ref / "steps.jsonl.gz", "rt", encoding="utf-8") as f: rows = [json.loads(line) for line in f]
    with np.load(ref / "vectors.npz") as f: actions = f["action"].copy()
    path = OUT / target["week"]; path.mkdir(exist_ok=False)
    samples = []
    try:
        for i in range(target["step"]-1):
            before = digest(obs.to_vect()); action = env.action_space(); action.from_vect(actions[i])
            assert before == rows[i]["before_hash"]
            obs, _, done, _ = env.step(action); counts["physical_steps"] += 1
            assert not done and digest(obs.to_vect()) == rows[i]["after_hash"]
            check_cap()
        assert digest(obs.to_vect()) == target["state_hash"]
        call_path = ROOT / target["call_file"]; call = json.loads(call_path.read_text())
        with np.load(call_path.with_name(call_path.stem + "_state.npz")) as f:
            base = env.action_space(); base.from_vect(f["base_action"])
            assert np.array_equal(obs.to_vect(), f["observation"])
        for pool_id in target["candidate_ids"]:
            candidate = library[pool_id]; original, masked, positions, intent = mask_disconnected(candidate, base, obs, env)
            previous = call["outcomes"][call["pool_ids"].index(pool_id)]
            assert digest(action_vector(original)) == previous["combined_action_hash"]
            pair = {"pool_id": pool_id, "changed_positions": positions, "intentional_base_lines": intent}
            vecs = {"before_observation": obs.to_vect().copy(), "base_action": action_vector(base)}
            for kind, action in [("original", original), ("masked", masked)]:
                counter = int(env.nb_highres_called); before = digest(obs.to_vect()); t = time.perf_counter()
                future, reward, done, info = obs.simulate(action)
                used = int(env.nb_highres_called)-counter; counts["public_forecasts"] += used
                assert used == 1 and digest(obs.to_vect()) == before
                result = {"rho": float(future.rho.max()), "reward": float(reward), "done": bool(done),
                          **flags(info), "wall_s": time.perf_counter()-t, "action_hash": digest(action_vector(action))}
                result["forecast_layout"] = [{"name": name, "size": int(np.asarray(future._get_array_from_attr_name(name)).size)} for name in type(future).attr_list_vect]
                assert sum(f["size"] for f in result["forecast_layout"]) == len(future.to_vect())
                result["valid_improving"] = bool(0 < result["rho"] < call["rho_limit"] and not done and not result["exceptions"] and not result["illegal"] and not result["ambiguous"])
                if kind == "original":
                    for key in ["illegal", "ambiguous", "done"]: assert result[key] == previous[key], (key, result, previous)
                    assert abs(result["rho"]-previous["rho"]) <= 1e-5
                    assert abs(result["reward"]-previous["reward"]) <= 1e-6
                pair[kind] = result; vecs[kind+"_action"] = action_vector(action); vecs[kind+"_observation"] = future.to_vect().copy()
                check_cap()
            np.savez_compressed(path / f"pool{pool_id:04d}.npz", **vecs)
            samples.append(pair); write_json(path / "pairs.json", samples)
        write_json(path / "summary.json", {"target": target, "pairs": samples, "prefix_exact": True})
        return {"target": target, "pairs": samples}
    except Exception:
        write_json(path / "failure.json", {"traceback": traceback.format_exc(), "counts": counts, "pairs": samples})
        raise
    finally:
        env.close()


def main():
    assert not (OUT / "finished.json").exists()
    for name, sha in json.loads((OUT / "code_freeze.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    results = []
    for target in D["targets"]:
        results.append(run(target)); write_json(OUT / "manifest.json", results)
    write_json(OUT / "finished.json", {"passed_engineering": True, "results": results, "counts": counts, "wall_s": time.perf_counter()-START})
    print(json.dumps({"pairs": sum(len(r["pairs"]) for r in results), "counts": counts, "wall_s": time.perf_counter()-START}))


if __name__ == "__main__": main()
