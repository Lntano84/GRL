"""Independent saved-data action, ranking, legality and cost ledger checks."""
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid20"), str(ROOT / "work/grid21")]
from common import np, torch, make_env, ResidualControl, action_vector, digest, write_json
from components import public_inputs, fingerprint, tensors
from prior import ResidualScorer, public_prior
OUT = ROOT / "outputs/grid25"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["pilot", "all"], default="pilot")
phase = parser.parse_args().phase
D = json.loads((OUT / "design.json").read_text())
F = json.loads((OUT / ("finished.json" if phase == "all" else "pilot_finished.json")).read_text())
if phase == "pilot": D["execution_order"] = [x for x in D["execution_order"] if x["phase"] == "pilot"]
assert F["passed_engineering"]
for name, sha in json.loads((OUT / "code_freeze.json").read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
resume_checks = 0
interrupted_prefix_checks = 0
if F.get("resumed"):
    for name, sha in json.loads((OUT / "resume_code_freeze.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
        resume_checks += 1
    recovery = json.loads((OUT / "resume_evidence.json").read_text())
    for name, sha in recovery["completed_file_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
        resume_checks += 1
    for name, sha in recovery["archived_file_hashes"].items():
        p = ROOT / recovery["archive"] / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, name
        resume_checks += 1
    archived_rows = []
    try:
        with gzip.open(ROOT / recovery["archive"] / "steps.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                archived_rows.append(json.loads(line))
    except (EOFError, OSError, json.JSONDecodeError):
        pass  # Its incomplete stream was disclosed before restarting.
    assert len(archived_rows) == recovery["partial_readable_rows"]
    replay_path = OUT / "runs/2035-09-03_4__FULL/steps.jsonl.gz"
    with gzip.open(replay_path, "rt", encoding="utf-8") as f:
        replay_rows = [json.loads(line) for line in f]
    for old, repeated in zip(archived_rows, replay_rows):
        for key in ["step", "before_hash", "action_hash", "after_hash", "raw_cost", "done", "complete"]:
            assert old[key] == repeated[key], (old["step"], key)
        interrupted_prefix_checks += 1


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as f: return [json.loads(line) for line in f]


def total(a): return math.fsum(float(v) for v in np.asarray(a).ravel())


env = make_env(); obs0 = env.reset(); ctl = ResidualControl(env, obs0)
library = ctl.base.topo_n1_unsafe.topo_act_list
subids = [int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0]) for a in library]
plans = np.stack([a.set_bus for a in library]); models = {}
initial_queries = int(env.nb_highres_called)
physical = forecasts = calls_count = labels_count = prediction_replays = masked = 0
max_cost_error = 0.
summaries = {(s["repeat"], s["week"], s["method"]): s for s in F["summaries"]}
assert len(summaries) == len(D["execution_order"])
try:
    for item in D["execution_order"]:
        week, method, rule = item["week"], item["method"], item["rule"]
        s = summaries[(item["repeat"], week, method)]; path = OUT / "runs" / f"r{item['repeat']}__{week}__{method}"
        meta = json.loads((path / "metadata.json").read_text())
        rows = read_rows(path / "steps.jsonl.gz")
        assert meta["library_hashes"] == [digest(action_vector(a)) for a in library]
        if item["checkpoint"]:
            checkpoint = ROOT / item["checkpoint"]
            assert meta["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if rule not in models:
                f = public_inputs(obs0, env.action_space(), env, plans)
                model = ResidualScorer(rule.split("_")[0].lower(), 2*env.n_sub, 2*env.n_line, len(f["base"]), env.dim_topo, subids)
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)); model.eval(); models[rule] = model
        with np.load(path / "vectors.npz") as f: vectors = {k: f[k].copy() for k in f.files}
        assert len(rows) == len(vectors["observation"]) == len(vectors["action"]) == s["steps"]
        assert rows[-1]["complete"] == s["complete"] and rows[-1]["done"]
        assert abs(math.fsum(r["raw_cost"] for r in rows)-s["cost"]) < 1e-6
        assert sum(r["simulations"] for r in rows) == s["public_forecasts"]
        assert abs(math.fsum(r["decision_s"] for r in rows)-s["controller_s"]) < 1e-7
        assert abs(math.fsum(r["decision_cpu_s"] for r in rows)-s["controller_cpu_s"]) < 1e-7
        assert abs(math.fsum(r["all_forecast_s"] for r in rows)-s["all_forecast_s"]) < 1e-7
        for r in rows:
            assert 0 <= r["all_forecast_s"] <= r["decision_s"]
            assert r["all_forecast_calls"] == r["simulations"]
        previous = None
        for i, row in enumerate(rows):
            a = env.action_space(); a.from_vect(vectors["action"][i]); assert digest(action_vector(a)) == row["action_hash"]
            obs = obs0.copy(); obs.from_vect(vectors["observation"][i]); assert digest(obs.to_vect()) == row["after_hash"]
            if i: assert row["before_hash"] == rows[i-1]["after_hash"]
            if row["ledger"]:
                prev_curt = row["ledger"]["previous_curtailed_mw"]
                if previous is not None: assert prev_curt == total(previous.curtailment_mw)
                price = max(float(p) for p, mw in zip(meta["cost_per_MW"], obs.gen_p) if mw > 0)
                cost = price*meta["dt_hours"]*(total(obs.gen_p)-total(obs.load_p)+total(np.abs(obs.actual_dispatch))+total(np.abs(obs.storage_power))+total(obs.curtailment_mw)-prev_curt)
                residual = abs(cost-row["raw_cost"]); max_cost_error = max(max_cost_error, residual)
                assert residual <= row["ledger_tolerance"]
            previous = obs
        if rule == "FULL":
            historical = read_rows(ROOT / f"outputs/grid19/runs/{week}/steps.jsonl.gz")
            assert len(historical) == len(rows)
            for a, b in zip(rows, historical):
                for key in ["step", "before_hash", "action_hash", "after_hash", "raw_cost", "done", "complete"]: assert a[key] == b[key]
        source_files = sorted(path.glob("call[0-9][0-9][0-9].json"))
        assert len(source_files) == s["n1_calls"]
        run_queries = run_fallbacks = 0
        for file in source_files:
            c = json.loads(file.read_text())
            with np.load(file.with_name(file.stem+"_state.npz")) as f:
                obs = obs0.copy(); obs.from_vect(f["observation"])
                base = env.action_space(); base.from_vect(f["base_action"])
            assert digest(obs.to_vect()) == c["state_hash"] == rows[c["step"]-1]["before_hash"]
            eligible = [i for i, sub in enumerate(subids) if obs.time_before_cooldown_sub[sub] == 0]
            assert eligible == c["eligible"]
            reps = []; aliases = {}; first = {}; actions = {}; changed = []
            # Independent endpoint patch and full-vector alias construction.
            for i in eligible:
                a = library[i].copy(); buses = a.set_bus.copy(); any_change = False
                for line in np.flatnonzero(~obs.line_status):
                    positions = [int(env.line_or_pos_topo_vect[line]), int(env.line_ex_pos_topo_vect[line])]
                    intent = base.line_set_status[line]>0 or base.line_change_status[line] or any(base.set_bus[p]>0 for p in positions)
                    if not intent:
                        any_change |= bool(np.any(buses[positions] != 0)); buses[positions] = 0
                if any_change: a.set_bus = buses; changed.append(i)
                combined = a+base; key = digest(action_vector(combined))
                if key not in first: first[key] = i; reps.append(i); actions[i] = a
                aliases[i] = first[key]
            assert reps == c["representatives"] and aliases == {int(k): v for k, v in c["aliases"].items()} and changed == c["masked_ids"]
            order = reps.copy()
            if rule != "FULL":
                prior, order = public_prior(obs, env, ctl, eligible, aliases)
                assert digest(prior) == c["prior_hash"]
                if item["checkpoint"]:
                    with np.load(file.with_name(file.stem+"_inputs.npz")) as f:
                        saved = {k: f[k].copy() for k in f.files}
                    features = {k: saved[k] for k in ["x", "edges", "edge_attr", "glob", "base", "plans"]}
                    # Producer reused only for permissions/input fidelity; action
                    # legality and descriptor reconstruction above are independent.
                    assert fingerprint(public_inputs(obs, base, env, plans)) == fingerprint(features) == c["input_hash"]
                    assert np.array_equal(prior, saved["prior"])
                    with torch.no_grad(): values = models[rule](**tensors(features))[0].numpy()+prior
                    order = sorted(reps, key=lambda i: (-float(values[i]), i))
                    prediction_replays += 1
            assert c["order"] == order
            shortlist = order if rule == "FULL" else order[:item["budget"]]
            assert c["shortlist"] == shortlist
            limit = c["rho_limit"]; assert limit <= float(obs.rho.max())
            delivered = {}
            for label in c["outcomes"]:
                i = label["pool_id"]; assert i in reps and i not in delivered
                assert label["combined_hash"] == digest(action_vector(actions[i]+base))
                valid = bool(0 < label["rho"] < limit and not label["done"] and not label["exceptions"] and not label["illegal"] and not label["ambiguous"])
                assert valid == label["strict_admissible"]
                delivered[i] = label
                labels_count += 1
            short_valid = [i for i in shortlist if delivered[i]["strict_admissible"]]
            short_choice = max(short_valid, key=lambda i: (float(np.float32(delivered[i]["reward"])), -i)) if short_valid else None
            assert c["shortlist_choice"] == short_choice and c["guard_rho"] == .9
            reason = ("no_admissible_shortlist" if short_choice is None else
                      "shortlist_above_author_safe" if delivered[short_choice]["rho"] >= .9 else
                      "shortlist_below_author_safe")
            assert c["guard_reason"] == reason
            fallback = bool(rule != "FULL" and len(shortlist) < len(reps) and
                            (short_choice is None or delivered[short_choice]["rho"] >= .9))
            assert c["fallback"] == fallback
            dispatch_order = shortlist + ([i for i in reps if i not in set(shortlist)] if fallback else [])
            assert [r["pool_id"] for r in c["outcomes"]] == dispatch_order
            assert len(delivered) == c["public_queries"] == len(dispatch_order)
            for j, r in enumerate(c["outcomes"]): assert r["phase"] == ("shortlist" if j<len(shortlist) else "fallback")
            considered = reps if fallback else shortlist
            valid_ids = [i for i in considered if delivered[i]["strict_admissible"]]
            chosen = max(valid_ids, key=lambda i: (float(np.float32(delivered[i]["reward"])), -i)) if valid_ids else None
            assert chosen == c["chosen_pool_id"]
            assert c["returned_action_hash"] == (None if chosen is None else digest(action_vector(actions[chosen])))
            if chosen is not None:
                physical_action = env.action_space(); physical_action.from_vect(vectors["action"][c["step"]-1])
                # Continuous optimizer may add dispatch, but must retain chosen
                # topology. This verifies the auxiliary's choice reaches execution.
                assert np.array_equal(physical_action.set_bus, (actions[chosen]+base).set_bus)
            assert c["forecast_s"] <= c["source_s"]
            run_queries += c["public_queries"]; run_fallbacks += fallback; calls_count += 1; masked += len(changed)
        assert run_queries == s["n1_forecasts"] and run_fallbacks == s["fallbacks"]
        physical += len(rows); forecasts += s["public_forecasts"]
    assert physical == F["counts"]["physical_steps"] and forecasts == F["counts"]["public_forecasts"]
    assert F["serial_workers"] == 1
    assert int(env.nb_highres_called) == initial_queries
finally:
    env.close()
write_json(OUT / ("audit.json" if phase == "all" else "pilot_audit.json"), {"passed": True, "runs": len(summaries), "physical_rows": physical, "source_calls": calls_count,
           "candidate_feedback": labels_count, "masked_candidates": masked, "neural_prediction_replays": prediction_replays,
           "max_independent_cost_residual": max_cost_error, "new_forecasts": 0, "new_physical_steps": 0, "new_model_fits": 0,
           "resume_preservation_hash_checks": resume_checks,
           "interrupted_prefix_exact_physical_checks": interrupted_prefix_checks,
           "scope": "Independent saved-data mask/alias/admission/cost/choice arithmetic, repeat keys, timing sums and checkpoint rank replay. Whole path equals sealed FULL. No independent AC replay/refit; prior code reused."})
print(json.dumps({"passed": True, "runs": len(summaries), "physical_rows": physical, "source_calls": calls_count, "labels": labels_count}))
