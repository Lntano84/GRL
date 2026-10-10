"""Separate offline rule reconstruction, stored-record audit, and narrow pilot verdict."""
import csv
import gzip
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid15"
START = time.perf_counter()
D = json.loads((OUT / "design.json").read_text())
F = json.loads((OUT / "finished.json").read_text()); assert F["passed"]
STATIC = json.loads((OUT / "public_static.json").read_text())


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def rebuild(rho, live, ids, subs, key):
    # Independent layer-expansion BFS; do not import production rules.py.
    u, v = STATIC["line_or"], STATIC["line_ex"]
    areas = {int(k): int(value) for k, value in STATIC["areas"].items()}
    top = sorted((int(i) for i in np.flatnonzero(live)), key=lambda i: (-float(rho[i]), i))[:3]
    layer = {int(node) for i in top for node in (u[i], v[i])}
    dist = {node: 0 for node in layer}; edges = [(u[i], v[i]) for i in np.flatnonzero(live)]
    depth = 0
    while layer:
        next_layer = set()
        for a, b in edges:
            if a in layer and b not in dist: next_layer.add(b)
            if b in layer and a not in dist: next_layer.add(a)
        depth += 1
        for node in next_layer: dist[node] = depth
        layer = next_layer
    pressure = {node: 0. for node in range(STATIC["n_sub"])}
    for i in np.flatnonzero(live):
        for node in (u[i], v[i]): pressure[node] = max(pressure[node], float(rho[i]))
    order = sorted(range(len(ids)), key=lambda i: (dist.get(subs[i], STATIC["n_sub"]+1), -pressure[subs[i]], ids[i]))
    rand = sorted(range(len(ids)), key=lambda i: hashlib.sha256(f"grid15-seed0:{key}:{ids[i]}".encode()).digest())
    zone_area = areas[u[int(np.argmax(rho))]]
    result = {"ALL": list(range(len(ids))), "ZONE": [i for i, node in enumerate(subs) if areas[node] == zone_area]}
    for k in (32, 128):
        result[f"LOCAL{k}"] = order[:k]; result[f"RANDOM{k}"] = rand[:k]
        result[f"PREFIX{k}"] = list(range(min(k, len(ids))))
    return result


with np.load(OUT / "action_library.npz") as lib:
    all_actions = lib["actions"]; all_subs = lib["subids"]
assert [hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in all_actions] == STATIC["action_hashes"]
physical_rows = 0; feedback_rows = 0; permission_checks = 0; call_records = []; metric_rows = []
time_rows = []
for summary in F["summaries"]:
    week = summary["week"]; path = OUT / "runs" / week
    new = rows(path / "steps.jsonl.gz")
    old = rows(ROOT / f"outputs/grid14/runs/{week}__AUTHOR/steps.jsonl.gz")
    assert len(old) == len(new) == summary["steps"]
    for a, b in zip(old, new):
        for key in ["step", "before_hash", "action_hash", "after_hash", "raw_cost", "done", "complete"]:
            assert a[key] == b[key], (week, key, a["step"])
    assert abs(math.fsum(row["raw_cost"] for row in new)-summary["cost"]) < 1e-6
    assert sum(row["simulations"] for row in new) == summary["public_forecasts"]
    physical_rows += len(new)
    for pointer in summary["calls"]:
        c = json.loads((path / f"call{pointer['call_id']:03d}.json").read_text())
        with np.load(path / f"call{pointer['call_id']:03d}_state.npz") as state:
            obs = state["observation"]; rho = state["rho"]; live = state["live"]; cooldown = state["cooldown_sub"]
        assert hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest() == c["state_hash"]
        assert c["state_hash"] == new[c["action_step"]-1]["before_hash"]
        ids = c["pool_ids"]; subs = c["subids"]
        assert len(set(ids)) == len(ids) and all_subs[ids].tolist() == subs
        assert np.all(cooldown[subs] == 0)
        eligible_library = [i for i, sub in enumerate(all_subs) if cooldown[sub] == 0]
        assert ids == eligible_library
        expected_rules = rebuild(rho, live, ids, subs, c["state_hash"])
        assert expected_rules == c["rules"]
        # Alter every hidden result: reconstructing from the public channel still
        # gives exactly the same sets. Outcome data is not an input argument.
        altered = dict(c); altered["outcomes"] = [{"rho": 9999., "reward": -9999.} for _ in ids]
        assert rebuild(rho, live, altered["pool_ids"], altered["subids"], altered["state_hash"]) == expected_rules
        permission_checks += 1
        assert len(c["outcomes"]) == len(ids)
        forecast_s = math.fsum(r["forecast_s"] for r in c["outcomes"])
        assert abs(forecast_s-c["forecast_s"]) < 1e-7
        for r in c["outcomes"]:
            admitted = 0 < r["rho"] < c["rho_limit"] and not r["exceptions"] and not r["done"]
            assert admitted == r["source_admissible"]
            assert (admitted and not r["illegal"] and not r["ambiguous"]) == r["strict_admissible"]
            expected_score = r["reward"] if admitted else -100.
            assert float(np.float32(expected_score)) == r["source_score"]
            if not r["exceptions"] and not r["illegal"] and not r["ambiguous"]:
                assert abs(r["reward"]-(2-r["rho"])) < 3e-7
        feedback_rows += len(ids)
        good = [i for i, r in enumerate(c["outcomes"]) if r["strict_admissible"]]
        best_rho = min((c["outcomes"][i]["rho"] for i in good), default=None)
        best_score = max((r["source_score"] for r in c["outcomes"]), default=-100.)
        for rule, chosen in c["rules"].items():
            available = [i for i in chosen if c["outcomes"][i]["strict_admissible"]]
            predicted_best = min((c["outcomes"][i]["rho"] for i in available), default=None)
            source_scores = [c["outcomes"][i]["source_score"] for i in chosen]
            metric_rows.append({"week": week, "action_step": c["action_step"], "rule": rule,
                 "pool_size": len(ids), "queries": len(chosen), "full_has_valid": best_rho is not None,
                 "has_valid": predicted_best is not None,
                 "full_best_rho": best_rho, "shortlist_best_rho": predicted_best,
                 "rho_excess": None if best_rho is None or predicted_best is None else predicted_best-best_rho,
                 "near_best": best_rho is not None and predicted_best is not None and predicted_best <= best_rho+.01,
                 "reward_best_recall": best_score > -100. and bool(source_scores) and max(source_scores) >= best_score-1e-6,
                 "source_forecast_s_subset": math.fsum(c["outcomes"][i]["forecast_s"] for i in chosen),
                 "fallback_full_queries_if_none": len(ids) if best_rho is not None and predicted_best is None else len(chosen)})
        call_records.append(c)
    source_fraction = summary["n1_public_forecast_s"]/summary["total_decision_s_instrumented"]
    time_rows.append({"week": week, "n1_calls": len(summary["calls"]),
         "n1_public_forecast_s": summary["n1_public_forecast_s"],
         "controller_s_instrumented": summary["total_decision_s_instrumented"],
         "n1_forecast_fraction_of_instrumented_controller_s": source_fraction,
         "n1_record_bookkeeping_s": summary["forecast_record_bookkeeping_s"],
         "source_calls": sum(r["candidates"] for r in summary["calls"])})

assert physical_rows == F["counts"]["physical_steps"]
assert len(call_records) == F["counts"]["n1_calls"]
aggregates = []
for rule in D["rules"]:
    subset = [r for r in metric_rows if r["rule"] == rule]
    relevant = [r for r in subset if r["full_has_valid"]]
    aggregates.append({"rule": rule, "calls": len(subset), "full_valid_calls": len(relevant),
         "mean_queries": sum(r["queries"] for r in subset)/max(1,len(subset)),
         "source_query_reduction": 1-sum(r["queries"] for r in subset)/max(1,sum(r["pool_size"] for r in subset)),
         "near_best_retention": sum(r["near_best"] for r in relevant)/max(1,len(relevant)),
         "best_reward_recall": sum(r["reward_best_recall"] for r in relevant)/max(1,len(relevant)),
         "lost_valid_calls": sum(not r["has_valid"] for r in relevant),
         "mean_rho_excess_when_valid": math.fsum(r["rho_excess"] for r in relevant if r["rho_excess"] is not None)/max(1,sum(r["rho_excess"] is not None for r in relevant)),
         "fallback_full_query_reduction_if_none": 1-sum(r["fallback_full_queries_if_none"] for r in subset)/max(1,sum(r["pool_size"] for r in subset)),
         "by_week_near_best": {week: [sum(r["near_best"] for r in relevant if r["week"] == week), sum(r["week"] == week for r in relevant)] for week in D["weeks"]}})
qualifying_simple = [a["rule"] for a in aggregates if a["rule"] != "ALL" and a["mean_queries"] <= 128 and a["near_best_retention"] >= .95 and a["lost_valid_calls"] <= 1]
expensive_weeks = [r["week"] for r in time_rows if r["n1_forecast_fraction_of_instrumented_controller_s"] >= .2]
pilot_gate = len(expensive_weeks) >= 2 and not qualifying_simple and feedback_rows > 0
save("analysis.json", {"rules": aggregates, "time": time_rows,
     "qualifying_simple_rules": qualifying_simple, "expensive_weeks": expensive_weeks,
     "supervised_pilot_qualification": pilot_gate,
     "warning": "Same full-search physical path. Public candidate prediction retention is not closed-loop survivability, total operating cost, achieved speedup, or learning advantage. States nested in four reused training weeks."})
save("audit.json", {"passed": True, "physical_control_rows_exact": physical_rows,
     "candidate_feedback_rows": feedback_rows, "permission_reconstruction_checks": permission_checks,
     "new_physical_steps": 0, "new_forecasts": 0, "model_fits": 0,
     "wall_s": time.perf_counter()-START,
     "scope": "Independent stored-record arithmetic, rule reconstruction, source-selection checks and exact archived path comparison; no independent AC solver."})
with (OUT / "per_call_rules.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)
print(json.dumps({"rules": aggregates, "time": time_rows, "pilot_gate": pilot_gate, "qualifying_simple": qualifying_simple}, ensure_ascii=False), flush=True)
