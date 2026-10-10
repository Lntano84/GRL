"""Read-only trajectory arithmetic, original-path checks and source-cache audit."""
import ast
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08")]
from common import np, make_env, ResidualControl, action_vector, digest, write_json, torch
OUT = ROOT / "outputs/grid16"; START = time.perf_counter()
D = json.loads((OUT / "design.json").read_text())
F = json.loads((OUT / "finished.json").read_text()); assert F["passed_engineering"]
for manifest in [ROOT / "outputs/grid14/delivery_manifest.json", ROOT / "outputs/grid15/delivery_manifest.json", OUT / "code_freeze.json"]:
    for name, sha in json.loads(manifest.read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
static = json.loads((ROOT / "outputs/grid15/public_static.json").read_text())
tree = ast.parse((ROOT / "work/grid15/analyse.py").read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rebuild")
scope = {"STATIC": static, "np": np, "hashlib": hashlib}
exec(compile(ast.Module(body=[function], type_ignores=[]), "independent_grid15_rule_reconstruction", "exec"), scope)
rebuild = scope["rebuild"]


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f: return [json.loads(line) for line in f]


def total(x): return math.fsum(float(v) for v in np.asarray(x).ravel())


physical = 0; source_records = 0; cache_hits = 0; max_cost_error = 0.; summaries = {}
env = make_env(); env.seed(0); env.set_id(D["weeks"][0]); initial = env.reset()
ctl = ResidualControl(env, initial)
library = ctl.base.topo_n1_unsafe.topo_act_list
all_keys = [digest(action_vector(a)) for a in library]
subids = [int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0]) for a in library]
neural = ctl.base.topo_12_unsafe
nn_hashes = [digest(action_vector(a)) for a in neural.gym_env.action_space.topo_actions_list]
assert len(nn_hashes) == len(set(nn_hashes))
key_to_pool = {key: i for i, key in enumerate(all_keys)}
nn_to_pool = {i: key_to_pool[key] for i, key in enumerate(nn_hashes) if key in key_to_pool}
try:
    for summary in F["summaries"]:
        week, rule = summary["week"], summary["rule"]; path = OUT / "runs" / f"{week}__{rule}"
        rows = read(path / "steps.jsonl.gz"); meta = json.loads((path / "metadata.json").read_text())
        assert len(rows) == summary["steps"] and rows[-1]["complete"] == summary["complete"]
        assert abs(math.fsum(row["raw_cost"] for row in rows)-summary["cost"]) < 1e-6
        assert sum(row["simulations"] for row in rows) == summary["public_forecasts"]
        assert abs(math.fsum(row["decision_s"] for row in rows)-summary["controller_s"]) < 1e-7
        assert meta["action_library_hashes"] == all_keys
        with np.load(path / "vectors.npz") as compressed: vectors = {k: compressed[k] for k in compressed.files}
        for i, row in enumerate(rows):
            action = env.action_space(); action.from_vect(vectors["action"][i])
            assert digest(action_vector(action)) == row["action_hash"]
            obs = initial.copy(); obs.from_vect(vectors["observation"][i])
            assert digest(obs.to_vect()) == row["after_hash"]
            if row["ledger"]:
                previous = row["ledger"]["previous_curtailed_mw"]
                if i: assert previous == total(vectors["curtailment_mw"][i-1])
                price = max(float(p) for p, mw in zip(meta["cost_per_MW"], vectors["gen_p"][i]) if mw > 0)
                j = price * meta["dt_hours"] * (total(vectors["gen_p"][i])-total(vectors["load_p"][i])+
                    total(np.abs(vectors["actual_dispatch"][i]))+total(np.abs(vectors["storage_power"][i]))+
                    total(vectors["curtailment_mw"][i])-previous)
                gap = abs(j-row["raw_cost"]); max_cost_error = max(max_cost_error, gap)
                assert gap <= row["ledger_tolerance"], (week, rule, i, gap)
            if i: assert row["before_hash"] == rows[i-1]["after_hash"]
        if rule == "FULL":
            old = read(ROOT / f"outputs/grid15/runs/{week}/steps.jsonl.gz")
            assert len(old) == len(rows)
            for a, b in zip(old, rows):
                for key in ["step", "before_hash", "action_hash", "after_hash", "raw_cost", "done", "complete"]: assert a[key] == b[key]
        with (path / "source_calls.jsonl").open(encoding="utf-8") as source: calls = [json.loads(line) for line in source]
        assert len(calls) == summary["n1_search_calls"]
        assert sum(c["public_queries"] for c in calls) == summary["n1_forecasts"]
        assert sum(c["cache_hits"] for c in calls) == summary["cache_hits"]
        for c in calls:
            assert c["state_hash"] == rows[c["step"]-1]["before_hash"]
            # Search does not occur at initial safe reset in this frozen scope.
            assert c["step"] > 1
            before = initial.copy(); before.from_vect(vectors["observation"][c["step"]-2])
            assert digest(before.to_vect()) == c["state_hash"]
            ids = [i for i, sub in enumerate(subids) if before.time_before_cooldown_sub[sub] == 0]
            assert ids == c["pool_ids"] and len(ids) == c["full_size"]
            subsets = rebuild(before.rho, before.line_status, ids, [subids[i] for i in ids], c["state_hash"])
            if rule == "FULL": expected = list(range(len(ids)))
            elif rule == "OLDNN128":
                with torch.no_grad(): order = neural.get_top_k(neural.gym_env.observation_space.to_gym(before), len(nn_hashes))
                pos = {i: j for j, i in enumerate(ids)}; expected = []
                for j in order:
                    i = nn_to_pool.get(int(j))
                    if i in pos: expected.append(pos[i])
                occupied = set(expected); expected += [j for j in subsets["LOCAL128"] if j not in occupied]; expected = expected[:128]
            else: expected = subsets[rule]
            assert expected == c["shortlist_positions"] and len(expected) == len(set(expected))
            assert len(expected) == c["shortlist_size"]
            dispatched = expected + ([j for j in range(len(ids)) if j not in set(expected)] if c["fallback"] else [])
            assert len(dispatched) == c["public_queries"] == len(c["labels"])
            assert len(set(label["combined_action_hash"] for label in c["labels"])) == len(c["labels"])
            assert c["cache_hits"] == (len(expected) if c["fallback"] else 0)
            scores = {}
            for position, label in zip(dispatched, c["labels"]):
                admitted = 0 < label["rho"] < float(before.rho.max()) and not label["done"] and not label["exceptions"]
                scores[position] = float(np.float32(label["reward"])) if admitted else -100.
            if c["fallback"]:
                assert not expected or max(scores[j] for j in expected) <= -100.
                assert len(dispatched) == len(ids)
            considered = list(range(len(ids))) if c["fallback"] else expected
            valid = [j for j in considered if scores[j] > -100.]
            chosen = None if not valid else ids[max(considered, key=lambda j: (scores[j], -considered.index(j)))]
            assert chosen == c["chosen_pool_id"]
            assert c["forecast_s"] <= c["source_s"]
            cache_hits += c["cache_hits"]; source_records += 1
        physical += len(rows); summaries[(week, rule)] = summary
    assert physical == F["counts"]["physical_steps"]
    assert sum(s["public_forecasts"] for s in F["summaries"]) == F["counts"]["public_forecasts"]
finally:
    env.close()

comparisons = []; aggregate = []
for rule in D["rules"][1:]:
    paired = []; lost = []; earlier = []
    for week in D["weeks"]:
        base, alt = summaries[(week, "FULL")], summaries[(week, rule)]
        joint = base["complete"] and alt["complete"]
        if base["complete"] and not alt["complete"]: lost.append(week)
        if not base["complete"] and not alt["complete"] and alt["steps"] < base["steps"]-1: earlier.append(week)
        result = {"week": week, "rule": rule, "full_complete": base["complete"], "alternative_complete": alt["complete"],
           "full_steps": base["steps"], "alternative_steps": alt["steps"], "joint_cost_eligible": joint,
           "cost_gain": (base["cost"]-alt["cost"])/abs(base["cost"]) if joint else None,
           "controller_time_reduction": (base["controller_s"]-alt["controller_s"])/base["controller_s"] if joint else None,
           "n1_query_reduction": 1-alt["n1_forecasts"]/base["n1_forecasts"] if joint else None,
           "fallbacks": alt["fallbacks"]}
        comparisons.append(result)
        if joint: paired.append(result)
    metrics = {"rule": rule, "joint_weeks": len(paired), "lost_complete_weeks": lost, "earlier_failures": earlier,
         "mean_cost_gain_joint": math.fsum(r["cost_gain"] for r in paired)/max(1,len(paired)),
         "mean_controller_time_reduction_joint": math.fsum(r["controller_time_reduction"] for r in paired)/max(1,len(paired)),
         "mean_n1_query_reduction_joint": math.fsum(r["n1_query_reduction"] for r in paired)/max(1,len(paired))}
    metrics["simple_qualified"] = not lost and not earlier and bool(paired) and metrics["mean_cost_gain_joint"] >= -.01 and metrics["mean_controller_time_reduction_joint"] >= .1 and metrics["mean_n1_query_reduction_joint"] >= .3
    aggregate.append(metrics)
write_json(OUT / "comparison.json", {"per_week": comparisons, "aggregate": aggregate,
    "scope": "Four reused training weeks, one seed, complete-first development screen. Instrumented single wall timings, not statistical speed guarantees; cached predictions not independently recomputed by an AC solver."})
write_json(OUT / "audit.json", {"passed": True, "physical_rows": physical, "source_calls": source_records,
    "cache_reuses_checked": cache_hits, "max_independent_cost_residual": max_cost_error,
    "model_fits": 0, "neural_baseline_forward_replays": sum(s["n1_search_calls"] for s in F["summaries"] if s["rule"] == "OLDNN128"),
    "extra_forecasts": 0, "extra_physical_steps": 0, "wall_s": time.perf_counter()-START})
print(json.dumps({"aggregate": aggregate, "audit_rows": physical, "source_calls": source_records}, ensure_ascii=False))
