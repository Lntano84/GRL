"""Reconcile the GRID320 saved ledgers after the one manager has exited."""
from collections import defaultdict
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import time

R = Path("D:/Codex/GRL研究")
I = R / "work/GRL-results-integration"
P = I / "research_archive/power-grid-2026/protocols"
A = P / "GRID320_load_guard_closed_loop"
O = R / "outputs/grid320_load_guard_closed_loop"
OLD = P / "GRID312_frame_208_261_pilot"
HISTORY = R / "outputs/grid312_frame_208_261_pilot"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt = load("grid320_saved_runtime", A / "runtime.py")
audit = load("grid320_existing_saved_checks", OLD / "results/close_grid312.py")
audit.O = O
audit.B = R / "archives/GRID320_load_guard_closed_loop/raw_records"
read, save, sha = audit.read, audit.save, audit.sha
FIELDS = ("controller_s_change_pct", "candidate_queries_change_pct", "total_forecasts_change_pct",
          "all_p95_change_pct", "all_p99_change_pct", "search_p95_change_pct", "search_p99_change_pct",
          "cost_delta_pct")


def paid_guard_events(folder, s):
    events, checked = [], 0
    for path in sorted(folder.glob("call[0-9][0-9][0-9].json")):
        call = read(path)
        amount = rt.load_guard_receipt(call)
        checked += 1
        if amount:
            assert s["arm"] != "FRAME_GUARD"
            event = call["load_guard"]
            events.append({"step": call["step"], "state_hash": call["state_hash"],
                           "guarded_choice": call["chosen_pool_id"], "guard_queries": amount, **event})
    assert checked == s["n1_calls"]
    return events


def physical_steps(folder):
    with gzip.open(folder / "steps.jsonl.gz", "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def divergence(a, b):
    for left, right in zip(a, b):
        assert left["step"] == right["step"]
        if any(left[k] != right[k] for k in ("before_hash", "action_hash", "after_hash", "raw_cost", "done", "complete")):
            return {"step": left["step"], "same_input": left["before_hash"] == right["before_hash"],
                    "same_action": left["action_hash"] == right["action_hash"],
                    "raw_cost_delta": left["raw_cost"] - right["raw_cost"]}
    return None


def close():
    assert not (O / "paired_results.json").exists()
    assert not (O / "active_worker.json").exists() or not read(O / "active_worker.json")["active"]
    process_count = subprocess.check_output(["powershell", "-NoProfile", "-Command",
        "@(Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
        "Where-Object CommandLine -Match 'GRID320_load_guard_closed_loop[/\\\\]runtime\\.py').Count"]).decode().strip()
    assert process_count == "0", process_count
    start = time.perf_counter()
    finished, error = O / "finished.json", O / "collection_error.json"
    assert finished.exists() != error.exists()
    end = read(finished if finished.exists() else error)
    design = read(A / "design.json")
    refs = {k: read(R / path) for k, path in design["full_references"].items()}
    manifest = read(O / "manifest.json") if (O / "manifest.json").exists() else {"records": [], "steps": 0, "forecasts": 0}
    assert len(manifest["records"]) == end["accepted"]
    assert manifest["steps"] <= design["caps"]["physical_steps"]
    assert manifest["forecasts"] <= design["caps"]["public_forecasts"]
    if finished.exists():
        assert end["accepted"] == 24 and end["steps"] == manifest["steps"] and end["forecasts"] == manifest["forecasts"]
    for source in ("authorization.json", "design.json", "source_freeze.json", "offline_review.json"):
        assert (A / source).read_bytes() == (O / source).read_bytes()
    rows, trajectories = [], {}
    for record, item in zip(manifest["records"], design["execution_order"]):
        assert time.perf_counter() - start < 600
        row = audit.checked(record, item, refs[item["week"]], design["quality_threshold"])
        summary = record["summary"]
        folder = O / "runs" / f"r{summary['repeat']}__{summary['week']}__{summary['method']}"
        row["guard_events"] = paid_guard_events(folder, summary)
        row["guard_extra_queries"] = sum(e["guard_queries"] for e in row["guard_events"])
        key = row["week"], row["repeat"], row["arm"]
        trajectories[key] = physical_steps(folder)
        rows.append(row)
    pairs = audit.compare(rows)
    for pair in pairs:
        k = pair["week"], pair["repeat"]
        pair["first_difference"] = divergence(trajectories[(*k, pair["arm"])],
                                              trajectories[(*k, pair["reference"])])
    history = {(r["week"], r["repeat"], r["arm"]): r for r in read(OLD / "results/paired_results.json")["records"]}
    old_design = {(r["week"], r["repeat"], r["arm"]): r for r in read(OLD / "design.json")["execution_order"]}
    history_comparisons = []
    for row in rows:
        k = row["week"], row["repeat"], row["arm"]
        prior = history[k]
        assert row["initial_hash"] == prior["initial_hash"]
        same = row["steps"] == prior["steps"] and row["complete"] == prior["complete"]
        entry = {name: row[name] for name in ("week", "repeat", "arm", "steps", "complete")}
        old_method = old_design[k]["method"]
        old_folder = HISTORY / "runs" / f"r{row['repeat']}__{row['week']}__{old_method}"
        entry.update(same_horizon=same, same_trajectory=row["signature"] == prior["signature"],
                     candidate_delta=row["candidate_queries"] - prior["candidate_queries"] if same else None,
                     cost_delta=row["cost"] - prior["cost"] if row["complete"] and prior["complete"] else None,
                     cost_delta_pct=100*(row["cost"]/prior["cost"]-1) if row["complete"] and prior["complete"] else None,
                     first_difference=divergence(trajectories[k], physical_steps(old_folder)))
        history_comparisons.append(entry)
    by = defaultdict(list)
    for row in rows:
        by[row["week"], row["arm"]].append(row["signature"])
        del row["library_hashes"]
    repeat_checks = [dict(week=k[0], arm=k[1], same_trajectory=v[0] == v[1])
                     for k, v in sorted(by.items()) if len(v) == 2]
    averages = {}
    for x, y in audit.COMPARISONS:
        part = [p for p in pairs if p["arm"] == x and p["reference"] == y]
        groups = {"all": part, "full": [p for p in part if p["complete_both"]],
                  "repeat0": [p for p in part if p["repeat"] == 0],
                  "repeat1": [p for p in part if p["repeat"] == 1]}
        averages[x + "_vs_" + y] = {key: {"pairs": len(group), **{f: audit.average(group, f) for f in FIELDS}}
                                    for key, group in groups.items()}
    save(O / "paired_results.json", {"records": rows, "pairs": pairs, "averages": averages,
                                    "repeat_checks": repeat_checks, "history_descriptive_no_timing": history_comparisons,
                                    "new_predictions_in_readout": 0, "new_physics_in_readout": 0, "new_fits": 0})
    raw = audit.archive()
    save(O / "closeout.json", {"status": "FINISHED" if finished.exists() else "STOPPED", "accepted": len(rows),
                              "physical_steps": manifest["steps"], "counted_forecasts": manifest["forecasts"],
                              "wall_s": end["wall_s"], "raw_records_manifest_sha256": sha(O / "raw_records_manifest.json"),
                              "raw_logical_files": raw["logical_files"]})
    print(json.dumps({"accepted": len(rows), "averages": averages, "repeat_checks": repeat_checks}, ensure_ascii=False))


if __name__ == "__main__":
    close()
