"""Fixed tiny supervised trial; no policy rollout or sealed final-test access."""
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid15"), str(ROOT / "work/grid20")]
from common import np, torch, make_env, ResidualControl, digest, action_vector, write_json
from components import public_inputs, fingerprint, tensors, CandidateScorer, flat_state, canonical_groups, batch
from rules import shortlist_rules
OUT = ROOT / "outputs/grid20"; D = json.loads((OUT / "design.json").read_text())
START = time.perf_counter()
env = make_env(); env.seed(0); initial = env.reset(); ctl = ResidualControl(env, initial)
library = ctl.base.topo_n1_unsafe.topo_act_list
static_plans = np.stack([a.set_bus for a in library]); n_actions = len(library)
subids = [int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0]) for a in library]
areas = {int(s): int(area) for area, subs in env._game_rules.legal_action.substations_id_by_area.items() for s in subs}
neural = ctl.base.topo_12_unsafe
pool_keys = {digest(action_vector(a)): i for i, a in enumerate(library)}
nn_to_pool = {j: pool_keys[key] for j, a in enumerate(neural.gym_env.action_space.topo_actions_list) if (key := digest(action_vector(a))) in pool_keys}


def check_cap():
    assert time.perf_counter()-START < D["caps"]["wall_s"]


def source_folder(week):
    stage = "grid18" if week in D["existing_training_weeks"] else "grid19"
    return ROOT / f"outputs/{stage}/runs/{week}"


def load(weeks, split):
    whitelist = D["training_weeks"] if split == "train" else D["development_weeks"]
    assert list(weeks) == whitelist
    result = []
    for week in weeks:
        for file in sorted(source_folder(week).glob("call[0-9][0-9][0-9].json")):
            c = json.loads(file.read_text())
            with np.load(file.with_name(file.stem+"_state.npz")) as f:
                before = initial.copy(); before.from_vect(f["observation"])
                base = env.action_space(); base.from_vect(f["base_action"])
            assert digest(before.to_vect()) == c["state_hash"]
            begun = time.perf_counter()
            features = public_inputs(before, base, env, static_plans)
            feature_s = time.perf_counter()-begun
            ids = [i for i, sub in enumerate(subids) if before.time_before_cooldown_sub[sub] == 0]
            assert ids == c["pool_ids"]
            reps, aliases = canonical_groups(features, ids)
            scores = dict(zip(ids, c["outcomes"]))
            for i, rep in aliases.items():
                assert scores[i]["combined_action_hash"] == scores[rep]["combined_action_hash"]
                for k in ["rho", "reward"]: assert abs(scores[i][k]-scores[rep][k]) < 1e-6
                for k in ["done", "illegal", "ambiguous", "strict_admissible"]: assert scores[i][k] == scores[rep][k]
            valid = [i for i in reps if scores[i]["strict_admissible"]]
            target = np.zeros(n_actions, dtype=np.float32)
            if valid:
                best = min(scores[i]["rho"] for i in valid)
                target[valid] = np.exp(-np.asarray([scores[i]["rho"]-best for i in valid])/D["temperature"])
                target /= target.sum(dtype=np.float64)
            mask = np.zeros(n_actions, dtype=bool); mask[reps] = True
            identifier = f"{week}__{file.stem}"
            folder = OUT / "inputs" / split; folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"{identifier}.npz", **features, target=target, mask=mask, representatives=np.asarray(reps))
            write_json(folder / f"{identifier}.json", {"week": week, "step": c["step"], "source": file.relative_to(ROOT).as_posix(),
                       "input_hash": fingerprint(features), "state_hash": c["state_hash"], "canonical_count": len(reps), "source_count": len(ids),
                       "source_sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "feature_build_s": feature_s})
            row = {"id": identifier, "week": week, "step": c["step"], "features": features, "tensor": tensors(features),
                   "target": torch.as_tensor(target), "mask": torch.as_tensor(mask), "reps": reps, "aliases": aliases,
                   "ids": ids, "scores": scores, "valid": valid, "obs": before, "base_action": base, "input_hash": fingerprint(features), "feature_build_s": feature_s}
            result.append(row)
            check_cap()
    return result


def new_model(kind, seed, base_dim):
    torch.manual_seed(seed)
    return CandidateScorer(kind, env.n_sub*2, env.n_line*2, base_dim, env.dim_topo, subids)


def train_model(kind, seed, records):
    eligible = [r for r in records if r["valid"]]
    assert eligible
    path = OUT / "models" / f"{kind}_seed{seed}"; path.mkdir(parents=True, exist_ok=False)
    model = new_model(kind, seed, len(records[0]["features"]["base"]))
    torch.save(model.state_dict(), path / "initial.pt")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=D["lr"], weight_decay=D["weight_decay"])
    random = np.random.default_rng(seed+20261010); curve = []; begun = time.perf_counter()
    for epoch in range(D["epochs"]):
        order = random.permutation(len(eligible)); total_loss = 0.; seen = 0
        for offset in range(0, len(order), D["batch_size"]):
            selected = [eligible[i] for i in order[offset:offset+D["batch_size"]]]
            logits = model(**batch(selected))
            masks = torch.stack([r["mask"] for r in selected]); target = torch.stack([r["target"] for r in selected])
            logs = torch.nn.functional.log_softmax(logits.masked_fill(~masks, -1e9), dim=1)
            loss = -(target*logs).sum(1).mean(); assert torch.isfinite(loss)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += float(loss.detach())*len(selected); seen += len(selected); check_cap()
        curve.append({"epoch": epoch+1, "training_cross_entropy": total_loss/seen, "wall_s": time.perf_counter()-begun})
        write_json(path / "curve.json", curve)
        if (epoch+1) % 16 == 0:
            print(json.dumps({"model": kind, "seed": seed, **curve[-1]}), flush=True)
    torch.save(model.state_dict(), path / "final.pt")
    stats = {"kind": kind, "seed": seed, "rows": len(records), "preference_rows": len(eligible), "epochs": D["epochs"],
             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "training_s": time.perf_counter()-begun,
             "final_training_cross_entropy": curve[-1]["training_cross_entropy"]}
    write_json(path / "summary.json", stats); model.eval(); return model, stats


def fit_ridge(records):
    usable = [r for r in records if r["valid"]]; t = time.perf_counter()
    matrix = np.stack([flat_state(r["features"], env.n_sub*2, env.n_line*2) for r in usable])
    mean = matrix.mean(0, dtype=np.float64); scale = matrix.std(0, dtype=np.float64); selected = np.flatnonzero(scale > 1e-8)
    assert len(selected)
    z = (matrix[:, selected]-mean[selected])/scale[selected]; z /= math.sqrt(len(selected))
    y = np.stack([r["target"].numpy() for r in usable]).astype(np.float64)
    prior = y.mean(0); weights = np.linalg.solve(z@z.T+np.eye(len(z))*D["ridge_alpha"], y-prior)
    del matrix
    np.savez_compressed(OUT / "ridge.npz", mean=mean[selected], scale=scale[selected], selected=selected, z=z, weights=weights, prior=prior)
    np.save(OUT / "frequency.npy", prior)
    return {"mean": mean[selected], "scale": scale[selected], "selected": selected, "z": z, "weights": weights, "prior": prior}, time.perf_counter()-t


def unique_order(order, aliases):
    result = []; used = set()
    for i in order:
        rep = aliases.get(int(i))
        if rep is not None and rep not in used: used.add(rep); result.append(rep)
    return result


def orders_for(row, models, ridge):
    result = {}; timings = {}
    reps = row["reps"]
    def sort(values): return sorted(reps, key=lambda i: (-float(values[i]), i))
    for name, model in models.items():
        t = time.perf_counter()
        with torch.no_grad(): values = model(**row["tensor"])[0].numpy().copy()
        result[name] = sort(values); timings[name] = time.perf_counter()-t
    t = time.perf_counter(); result["FREQUENCY"] = sort(ridge["prior"]); timings["FREQUENCY"] = time.perf_counter()-t
    t = time.perf_counter(); flat = flat_state(row["features"], env.n_sub*2, env.n_line*2)
    z = (flat[ridge["selected"]]-ridge["mean"])/ridge["scale"]/math.sqrt(len(ridge["selected"]))
    values = np.maximum(0., (z@ridge["z"].T)@ridge["weights"]+ridge["prior"])
    if not np.any(values[reps] > 0): values = ridge["prior"]
    result["RIDGE"] = sort(values); timings["RIDGE"] = time.perf_counter()-t
    t = time.perf_counter()
    cheap, _ = shortlist_rules(row["obs"].rho, row["obs"].line_status, env.line_or_to_subid, env.line_ex_to_subid,
                              env.n_sub, areas, [subids[i] for i in row["ids"]], row["ids"], digest(row["obs"].to_vect()))
    # LOCAL128 does not necessarily cover128 distinct masked actions. Rebuild
    # complete local order by the same public distance/incident-loading recipe.
    rho = row["obs"].rho; live = row["obs"].line_status
    top = sorted(np.flatnonzero(live), key=lambda i: (-float(rho[i]), int(i)))[:3]
    sources = {int(env.line_or_to_subid[i]) for i in top}|{int(env.line_ex_to_subid[i]) for i in top}
    distance = np.full(env.n_sub, env.n_sub+1, dtype=int); queue = sorted(sources)
    for sub in sources: distance[sub] = 0
    neighbors = [[] for _ in range(env.n_sub)]; incident = np.zeros(env.n_sub)
    for i in np.flatnonzero(live):
        u, v = int(env.line_or_to_subid[i]), int(env.line_ex_to_subid[i]); neighbors[u].append(v); neighbors[v].append(u)
        incident[u] = max(incident[u], float(rho[i])); incident[v] = max(incident[v], float(rho[i]))
    for sub in queue:
        for nxt in sorted(neighbors[sub]):
            if distance[nxt] > distance[sub]+1: distance[nxt] = distance[sub]+1; queue.append(nxt)
    local_raw = sorted(row["ids"], key=lambda i: (int(distance[subids[i]]), -float(incident[subids[i]]), i))
    # Guard against changing the already frozen cheap rule semantics.
    assert local_raw[:min(128,len(local_raw))] == [row["ids"][j] for j in cheap["LOCAL128"]]
    result["LOCAL"] = unique_order(local_raw, row["aliases"]); timings["LOCAL"] = time.perf_counter()-t
    t = time.perf_counter()
    with torch.no_grad(): nn_order = neural.get_top_k(neural.gym_env.observation_space.to_gym(row["obs"]), len(neural.gym_env.action_space.topo_actions_list))
    candidates = [nn_to_pool[int(j)] for j in nn_order if int(j) in nn_to_pool]
    result["OLDNN"] = unique_order(candidates+local_raw, row["aliases"]); timings["OLDNN"] = time.perf_counter()-t
    t = time.perf_counter(); result["PREFIX"] = reps.copy(); timings["PREFIX"] = time.perf_counter()-t
    t = time.perf_counter()
    result["RANDOM"] = sorted(reps, key=lambda i: hashlib.sha256(f"GRID20_RANDOM_20261010:{row['input_hash']}:{i}".encode()).digest())
    timings["RANDOM"] = time.perf_counter()-t
    for order in result.values(): assert set(order) == set(reps) and len(order) == len(reps)
    return result, timings


def assess(row, name, budget, order, timing):
    selected = order[:budget]; good = [i for i in selected if row["scores"][i]["strict_admissible"]]
    available = bool(good); fallback = not available and len(selected) < len(row["reps"])
    considered = row["reps"] if fallback else selected
    admissible = [i for i in considered if row["scores"][i]["strict_admissible"]]
    chosen = None if not admissible else max(admissible, key=lambda i: (float(np.float32(row["scores"][i]["reward"])), -considered.index(i)))
    full_best = min((row["scores"][i]["rho"] for i in row["valid"]), default=None)
    raw_best = min((row["scores"][i]["rho"] for i in good), default=None)
    return {"week": row["week"], "step": row["step"], "state_id": row["id"], "method": name, "budget": budget,
            "full_has_valid": bool(row["valid"]), "shortlist_has_valid": available, "fallback": fallback,
            "expected_unique_public_queries": len(row["reps"]) if fallback else len(selected),
            "near_best_raw": None if full_best is None else bool(raw_best is not None and raw_best-full_best <= .01),
            "raw_rho_gap": None if raw_best is None or full_best is None else raw_best-full_best,
            "delivered_rho_gap": None if chosen is None or full_best is None else row["scores"][chosen]["rho"]-full_best,
            "chosen_pool_id": chosen, "prediction_and_ranking_s": timing, "shared_feature_build_s": row["feature_build_s"], "ordered_shortlist": selected,
            "input_hash": row["input_hash"]}


def main():
    assert not (OUT / "finished.json").exists()
    assert json.loads((ROOT / "outputs/grid19/audit.json").read_text())["passed"]
    assert json.loads((OUT / "preflight.json").read_text())["passed"]
    for name, sha in json.loads((OUT / "code_freeze.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    records = load(D["training_weeks"], "train")
    models = {}; fitted = []; ridge = None
    # Development source records are not opened until every fit/checkpoint ends.
    for kind in ["gnn", "mlp"]:
        for seed in D["seeds"]:
            model, stats = train_model(kind, seed, records); models[f"{kind.upper()}_seed{seed}"] = model; fitted.append(stats)
    ridge, ridge_s = fit_ridge(records)
    write_json(OUT / "training_finished.json", {"fits": fitted, "ridge_fit_s": ridge_s, "rows": len(records), "training_input_ids": [r["id"] for r in records]})
    del records
    dev = load(D["development_weeks"], "development"); results = []
    for row in dev:
        orders, times = orders_for(row, models, ridge)
        write_json(OUT / "orders" / f"{row['id']}.json", {"input_hash": row["input_hash"], "orders": orders, "timings": times})
        for name, order in orders.items():
            for budget in D["budgets"]: results.append(assess(row, name, budget, order, times[name]))
        write_json(OUT / "per_state.json", results); check_cap()
    write_json(OUT / "finished.json", {"passed_engineering": True, "fits": fitted, "ridge_fit_s": ridge_s,
                 "development_states": len(dev), "model_fits": len(fitted)+1, "new_forecasts": 0, "new_physical_steps": 0,
                 "wall_s": time.perf_counter()-START, "scope": "One metadata-selected development split, two fitting seeds, no sealed final-test or closed-loop learned-policy rollout."})
    print(json.dumps({"fits": len(fitted)+1, "development_states": len(dev), "rows": len(results), "wall_s": time.perf_counter()-START}))


try:
    if __name__ == "__main__": main()
except Exception:
    write_json(OUT / "failure.json", {"traceback": traceback.format_exc(), "wall_s": time.perf_counter()-START})
    raise
finally:
    env.close()
