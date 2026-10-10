"""Saved-input/label arithmetic and independent prediction replay, not RF/NN refits."""
import hashlib
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid20")]
from common import np, torch, make_env, ResidualControl, digest, write_json
from components import CandidateScorer, public_inputs, fingerprint, tensors
OUT = ROOT / "outputs/grid20"; D = json.loads((OUT / "design.json").read_text())
assert json.loads((OUT / "finished.json").read_text())["passed_engineering"]
for name, sha in json.loads((OUT / "code_freeze.json").read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
assert not set(D["training_weeks"]) & set(D["development_weeks"])
env = make_env(); obs0 = env.reset(); ctl = ResidualControl(env, obs0)
library = ctl.base.topo_n1_unsafe.topo_act_list
subids = [int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0]) for a in library]
plans = np.stack([a.set_bus for a in library]); initial_count = int(env.nb_highres_called)
records = {}; targets = []; flats = []; permission = []; checked_descriptors = 0


def flat(f):
    e = np.full((2,2*env.n_line),-1.,np.float32); e[:,:f["edges"].shape[1]] = f["edges"].astype(np.float32)/np.float32(2*env.n_sub)
    attrs = np.zeros((2*env.n_line,4),np.float32); attrs[:len(f["edge_attr"])] = f["edge_attr"]
    return np.concatenate((f["x"].ravel(),e.ravel(),attrs.ravel(),f["glob"],f["base"],f["plans"].ravel()))


try:
    for split, weeks in [("train",D["training_weeks"]),("development",D["development_weeks"])]:
        expected = []
        for week in weeks:
            stage = "grid18" if week in D["existing_training_weeks"] else "grid19"
            expected += [f"{week}__{p.stem}" for p in sorted((ROOT / f"outputs/{stage}/runs/{week}").glob("call[0-9][0-9][0-9].json"))]
        actual = sorted(p.stem for p in (OUT / "inputs" / split).glob("*.npz"))
        assert sorted(expected) == actual
        for identifier in expected:
            meta = json.loads((OUT / "inputs" / split / f"{identifier}.json").read_text())
            assert meta["week"] in weeks
            source = ROOT / meta["source"]; assert hashlib.sha256(source.read_bytes()).hexdigest() == meta["source_sha256"]
            c = json.loads(source.read_text()); scores = dict(zip(c["pool_ids"],c["outcomes"]))
            with np.load(OUT / "inputs" / split / f"{identifier}.npz") as f:
                saved = {k:f[k].copy() for k in f.files}
            features = {k:saved[k] for k in ["x","edges","edge_attr","glob","base","plans"]}
            assert fingerprint(features) == meta["input_hash"]
            with np.load(source.with_name(source.stem+"_state.npz")) as f:
                obs = obs0.copy(); obs.from_vect(f["observation"])
                base = env.action_space(); base.from_vect(f["base_action"])
            assert digest(obs.to_vect()) == c["state_hash"]
            eligible = [i for i,s in enumerate(subids) if obs.time_before_cooldown_sub[s] == 0]
            assert eligible == c["pool_ids"]
            # Independently reconstruct combined pure-topology plans.
            for i in eligible:
                a = library[i].copy(); buses = a.set_bus.copy()
                for l in np.flatnonzero(~obs.line_status):
                    positions = [int(env.line_or_pos_topo_vect[l]),int(env.line_ex_pos_topo_vect[l])]
                    intent = base.line_set_status[l]>0 or base.line_change_status[l] or any(base.set_bus[p]>0 for p in positions)
                    if not intent: buses[positions] = 0
                a.set_bus = buses
                assert np.array_equal((a+base).set_bus,features["plans"][i]*2)
                checked_descriptors += 1
            first = {}; representatives = []
            for i in eligible:
                key = features["plans"][i].tobytes()
                if key not in first: first[key] = i; representatives.append(i)
                rep = first[key]
                assert scores[i]["combined_action_hash"] == scores[rep]["combined_action_hash"]
                assert all(scores[i][key] == scores[rep][key] for key in ["done","illegal","ambiguous","strict_admissible"])
                assert abs(scores[i]["rho"]-scores[rep]["rho"]) < 1e-6
            assert representatives == saved["representatives"].tolist()
            expected_mask = np.zeros(len(library),bool); expected_mask[representatives] = True
            assert np.array_equal(expected_mask,saved["mask"])
            valid = [i for i in representatives if scores[i]["strict_admissible"]]
            target = np.zeros(len(library),np.float32)
            if valid:
                best = min(scores[i]["rho"] for i in valid)
                vals = [math.exp(-(scores[i]["rho"]-best)/D["temperature"]) for i in valid]
                den = math.fsum(vals)
                target[valid] = [v/den for v in vals]
            assert np.allclose(target,saved["target"],atol=1e-7,rtol=1e-6)
            records[identifier] = {"meta":meta,"features":features,"representatives":representatives,"scores":scores,"valid":valid,"obs":obs,"base":base,"target":saved["target"],"mask":saved["mask"]}
            if split == "train" and valid: targets.append(saved["target"]); flats.append(flat(features))
            if split == "development" and not any(p["week"] == meta["week"] for p in permission):
                # Label side table is altered only in memory; features and
                # predictions must still derive entirely from visible channels.
                altered = json.loads(json.dumps(c))
                for r in altered["outcomes"]:
                    r["rho"] += 1000; r["reward"] *= -1; r["strict_admissible"] = not r["strict_admissible"]
                assert altered != c
                assert fingerprint(public_inputs(obs,base,env,plans)) == fingerprint(features)
                permission.append({"week":meta["week"],"input_unchanged":True})
    fitted = json.loads((OUT / "training_finished.json").read_text())
    train_ids = [identifier for identifier,r in records.items() if r["meta"]["week"] in D["training_weeks"]]
    assert fitted["training_input_ids"] == train_ids
    assert all(len(json.loads((OUT / "models" / f"{kind}_seed{s}" / "curve.json").read_text())) == D["epochs"] for kind in ["gnn","mlp"] for s in D["seeds"])
    freq = np.load(OUT / "frequency.npy"); target_table = np.stack(targets).astype(np.float64)
    assert np.allclose(freq,target_table.mean(0),atol=1e-12)
    with np.load(OUT / "ridge.npz") as f: ridge = {k:f[k].copy() for k in f.files}
    matrix = np.stack(flats); selected = ridge["selected"]
    mean = matrix.mean(0,dtype=np.float64); scale = matrix.std(0,dtype=np.float64)
    assert np.array_equal(selected,np.flatnonzero(scale>1e-8))
    assert np.allclose(mean[selected],ridge["mean"],atol=1e-12) and np.allclose(scale[selected],ridge["scale"],atol=1e-12)
    z = (matrix[:,selected]-mean[selected])/scale[selected]/math.sqrt(len(selected))
    assert np.allclose(z,ridge["z"],atol=1e-12)
    assert np.allclose((z@z.T+np.eye(len(z))*D["ridge_alpha"])@ridge["weights"],target_table-freq,atol=1e-8)
    del matrix, flats, z
    models = {}; weight_changes = []
    for kind in ["gnn","mlp"]:
        for seed in D["seeds"]:
            path = OUT / "models" / f"{kind}_seed{seed}"
            initial_state = torch.load(path / "initial.pt",map_location="cpu",weights_only=True)
            final_state = torch.load(path / "final.pt",map_location="cpu",weights_only=True)
            changed_keys = [key for key in initial_state if not torch.equal(initial_state[key],final_state[key])]
            assert changed_keys and all(torch.isfinite(value).all() for value in final_state.values())
            weight_changes.append({"kind":kind,"seed":seed,"changed_tensors":len(changed_keys)})
            model = CandidateScorer(kind,2*env.n_sub,2*env.n_line,len(next(iter(records.values()))["features"]["base"]),env.dim_topo,subids)
            model.load_state_dict(final_state); model.eval()
            models[f"{kind.upper()}_seed{seed}"] = model
    fit_diagnostics = []
    for name,model in models.items():
        entropy = []; cross_entropy = []; retention = []
        for identifier in train_ids:
            r = records[identifier]
            if not r["valid"]: continue
            target = torch.as_tensor(r["target"]); mask = torch.as_tensor(r["mask"])
            with torch.no_grad(): logits = model(**tensors(r["features"]))[0]
            logs = torch.log_softmax(logits.masked_fill(~mask,-1e9),dim=0)
            ce = float(-(target*logs).sum()); nonzero = target>0
            ent = float(-(target[nonzero]*target[nonzero].log()).sum())
            assert math.isfinite(ce) and ce-ent>-1e-5
            cross_entropy.append(ce); entropy.append(ent)
            selected_ids = sorted(r["representatives"],key=lambda i:(-float(logits[i]),i))[:32]
            valid_ids = [i for i in selected_ids if r["scores"][i]["strict_admissible"]]
            best = min(r["scores"][i]["rho"] for i in r["valid"])
            retention.append(bool(valid_ids and min(r["scores"][i]["rho"] for i in valid_ids)-best<=.01))
        fit_diagnostics.append({"method":name,"train_preference_states":len(entropy),"mean_final_cross_entropy":float(np.mean(cross_entropy)),
                                "mean_target_entropy":float(np.mean(entropy)),"mean_final_KL":float(np.mean(np.asarray(cross_entropy)-entropy)),
                                "train_near_best32":float(np.mean(retention))})
    write_json(OUT / "training_fit_diagnostics.json", fit_diagnostics)
    per_state = json.loads((OUT / "per_state.json").read_text()); predictions = arithmetic = 0
    for identifier,r in records.items():
        if r["meta"]["week"] not in D["development_weeks"]: continue
        stored = json.loads((OUT / "orders" / f"{identifier}.json").read_text())
        reps = r["representatives"]
        for name,model in models.items():
            with torch.no_grad(): values = model(**tensors(r["features"]))[0].numpy()
            assert np.isfinite(values).all()
            assert sorted(reps,key=lambda i:(-float(values[i]),i)) == stored["orders"][name]
            predictions += 1
        assert sorted(reps,key=lambda i:(-float(freq[i]),i)) == stored["orders"]["FREQUENCY"]
        query = (flat(r["features"])[selected]-ridge["mean"])/ridge["scale"]/math.sqrt(len(selected))
        vals = np.maximum(0,(query@ridge["z"].T)@ridge["weights"]+freq)
        if not np.any(vals[reps]>0): vals = freq
        assert sorted(reps,key=lambda i:(-float(vals[i]),i)) == stored["orders"]["RIDGE"]
        assert stored["orders"]["PREFIX"] == reps
        for row in [p for p in per_state if p["state_id"] == identifier]:
            order = stored["orders"][row["method"]]
            assert set(order) == set(reps) and len(order) == len(reps)
            chosen_set = order[:row["budget"]]; strict = [i for i in chosen_set if r["scores"][i]["strict_admissible"]]
            assert row["ordered_shortlist"] == chosen_set
            assert row["shortlist_has_valid"] == bool(strict)
            fallback = not strict and len(chosen_set)<len(reps)
            assert row["fallback"] == fallback
            considered = reps if fallback else chosen_set
            assert row["expected_unique_public_queries"] == len(considered)
            allowed = [i for i in considered if r["scores"][i]["strict_admissible"]]
            chosen = max(allowed,key=lambda i:(float(np.float32(r["scores"][i]["reward"])), -considered.index(i))) if allowed else None
            assert chosen == row["chosen_pool_id"]
            full = min([r["scores"][i]["rho"] for i in r["valid"]],default=None)
            best = min([r["scores"][i]["rho"] for i in strict],default=None)
            near = None if full is None else bool(best is not None and best-full<=.01)
            assert row["near_best_raw"] == near
            gap = None if chosen is None else r["scores"][chosen]["rho"]-full
            assert gap == row["delivered_rho_gap"]
            assert row["input_hash"] == r["meta"]["input_hash"]
            arithmetic += 1
    assert int(env.nb_highres_called) == initial_count
finally:
    env.close()
write_json(OUT / "audit.json", {"passed":True,"input_states":len(records),"candidate_descriptors":checked_descriptors,"weight_changes":weight_changes,"fit_diagnostics":fit_diagnostics,
           "prediction_replays":predictions,"delivery_rows":arithmetic,"permission_checks":permission,"new_forecasts":0,"refits":0,
           "scope":"Independent source/descriptor/target/budget arithmetic plus saved-network/ridge prediction consistency. No independent neural refit or second AC solver; feature producer reused only in permissions check."})
print(json.dumps({"passed":True,"inputs":len(records),"predictions":predictions,"rows":arithmetic}))
