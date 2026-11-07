"""Compare the GRID322 saved replays and the immutable GRID309 baseline."""
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

R = Path("D:/Codex/GRL研究")
I = R / "work/GRL-results-integration"
P = I / "research_archive/power-grid-2026/protocols"
A = P / "GRID322_new_family_load_guard"
O = R / "outputs/grid322_new_family_load_guard"
B = R / "archives/GRID322_new_family_load_guard/raw_records"
PREV = P / "GRID320_load_guard_closed_loop"
OLD = R / "outputs/grid309_action_opportunity_preparation"
FIELDS = ("step", "before_hash", "action_hash", "after_hash", "raw_cost", "done", "complete")
COMPARISONS = (("PARENT208_GUARD", "PARENT208"), ("BOUNDED261_GUARD", "BOUNDED261_SAVED"),
               ("BOUNDED261_GUARD", "PARENT208_GUARD"), ("PARENT208", "FRAME_GUARD"),
               ("PARENT208_GUARD", "FRAME_GUARD"), ("BOUNDED261_GUARD", "FRAME_GUARD"))
TIMING_FIELDS = ("controller_s", "all_p95", "all_p99", "search_p95", "search_p99")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


sys.path.insert(0, str(A))
import runtime as rt

prior = load("grid322_base_audit", P / "GRID312_frame_208_261_pilot/results/close_grid312.py")
prior.O, prior.B = O, B
read, save, sha = prior.read, prior.save, prior.sha


def steps(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def trajectory(rows):
    return hashlib.sha256(json.dumps([[r[k] for k in FIELDS] for r in rows],
                                     separators=(",", ":")).encode()).hexdigest()


def divergence(a, b):
    for left, right in zip(a, b):
        assert left["step"] == right["step"]
        if any(left[k] != right[k] for k in FIELDS):
            return {"step": left["step"], "same_input": left["before_hash"] == right["before_hash"],
                    "same_action": left["action_hash"] == right["action_hash"],
                    "cost_delta_at_step": left["raw_cost"] - right["raw_cost"]}
    return None


def guard_events(folder, item, s):
    calls = [read(p) for p in sorted(folder.glob("call[0-9][0-9][0-9].json"))]
    assert len(calls) == s["n1_calls"]
    events = []
    for call in calls:
        if not item["load_guard"]:
            assert "load_guard" not in call
            continue
        q = rt.previous.load_guard_receipt(call)
        if q:
            event = call["load_guard"]
            events.append(dict(step=call["step"], state_hash=call["state_hash"],
                candidate=call["chosen_pool_id"], extra_queries=q, **event))
    return events


def paired(left, right, left_rows, right_rows):
    assert left["initial_hash"] == right["initial_hash"]
    for k in ("library_hashes", "dt_hours", "cost_per_MW"):
        assert left[k] == right[k]
    same = left["steps"] == right["steps"] and left["complete"] == right["complete"]
    full = left["complete"] and right["complete"]
    fresh = right["arm"] != "BOUNDED261_SAVED"
    p = dict(week=left["week"], arm=left["arm"], reference=right["arm"], fresh_pair=fresh,
             same_horizon=same, complete_both=full,
             same_trajectory=left["signature"] == right["signature"],
             first_difference=divergence(left_rows, right_rows),
             candidate_delta=left["candidate_queries"]-right["candidate_queries"] if same else None,
             total_forecast_delta=left["total_forecasts"]-right["total_forecasts"] if same else None,
             cost_delta=left["cost"]-right["cost"] if full else None,
             cost_delta_pct=100*(left["cost"]/right["cost"]-1) if full else None)
    for k in ("candidate_queries", "total_forecasts", *TIMING_FIELDS):
        p[k + "_change_pct"] = (100*(left[k]/right[k]-1)
            if same and (fresh or k not in TIMING_FIELDS) and left[k] is not None and right[k] else None)
    return p


def close():
    assert not (O / "paired_results.json").exists()
    assert not (O / "active_worker.json").exists() or not read(O / "active_worker.json")["active"]
    proc = subprocess.check_output(["powershell", "-NoProfile", "-Command",
        "@(Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
        "Where-Object CommandLine -Match 'GRID322_new_family_load_guard[/\\\\]runtime\\.py').Count"]).decode().strip()
    assert proc == "0"
    start = time.perf_counter()
    finish, error = O / "finished.json", O / "collection_error.json"
    assert finish.exists() != error.exists()
    end = read(finish if finish.exists() else error)
    design = read(A / "design.json")
    caps = design["caps"]
    for name in ("authorization.json", "design.json", "source_freeze.json", "offline_review.json"):
        assert (A / name).read_bytes() == (O / name).read_bytes()
    manifest = read(O / "manifest.json") if (O / "manifest.json").exists() else {"records": [], "steps": 0, "forecasts": 0}
    assert len(manifest["records"]) == end["accepted"]
    assert manifest["steps"] <= caps["physical_steps"] and manifest["forecasts"] <= caps["public_forecasts"]
    if finish.exists():
        assert end["accepted"] == 16 and end["steps"] == manifest["steps"] and end["forecasts"] == manifest["forecasts"]
    records, series = [], {}
    for record, item in zip(manifest["records"], design["execution_order"]):
        assert time.perf_counter() - start < 600
        row = prior.checked(record, item, {"complete":False}, .005)
        s = record["summary"]
        folder = O / "runs" / f"r{s['repeat']}__{s['week']}__{s['method']}"
        row.update(arm=item["evaluation_arm"], model_arm=s["arm"], quality="NO_FULL_REFERENCE",
                   guard_events=guard_events(folder, item, s))
        row["guard_extra_queries"] = sum(e["extra_queries"] for e in row["guard_events"])
        series[row["week"],row["arm"]] = steps(folder / "steps.jsonl.gz")
        records.append(row)
    baselines = []
    baseline_manifest = read(OLD / "baseline_manifest.json")
    old_index = {x["summary"]["week"]:x for x in baseline_manifest["records"]}
    old_raw = read(OLD / "baseline_raw_manifest.json")
    for member in design["saved_unprotected261"]:
        folder = R / member["folder"]
        s = old_index[member["week"]]["summary"]
        assert all(sha(R / p)==h for p,h in member["files"].items())
        assert s == read(folder / "summary.json") and s["steps"]==member["steps"]
        metadata = folder / "metadata.json"
        assert sha(metadata) == old_raw[metadata.relative_to(R).as_posix()]["sha256"]
        meta = read(metadata)
        ledger = steps(folder / "steps.jsonl.gz")
        assert len(ledger)==s["steps"] and ledger[-1]["complete"]==s["complete"]
        assert all(l["after_hash"]==r["before_hash"] and not l["done"] for l,r in zip(ledger,ledger[1:]))
        assert abs(sum(r["raw_cost"] for r in ledger)-s["cost"]) < 1e-6
        assert sum(r["simulations"] for r in ledger)==s["public_forecasts"]
        row = {k: s[k] for k in ("week","steps","complete","cost","illegal","ambiguous","exception_steps")}
        row.update({k:meta[k] for k in ("library_hashes","dt_hours","cost_per_MW")})
        row.update(arm="BOUNDED261_SAVED", initial_hash=ledger[0]["before_hash"], signature=trajectory(ledger),
                   candidate_queries=s["n1_forecasts"], total_forecasts=s["public_forecasts"],
                   historical_runtime_not_used=True)
        series[row["week"],row["arm"]] = ledger
        baselines.append(row)
    index = {(r["week"],r["arm"]):r for r in records + baselines}
    pairs = []
    for left, right in COMPARISONS:
        for week in sorted({r["week"] for r in records}):
            a,b = (week,left),(week,right)
            if a in index and b in index:
                pairs.append(paired(index[a],index[b],series[a],series[b]))
    for r in records+baselines:
        del r["library_hashes"]
    save(O / "paired_results.json", {"records": records, "historical_baselines": baselines, "pairs": pairs,
        "new_physics_in_readout": 0,"new_predictions_in_readout":0,"new_fits":0})
    raw = prior.archive()
    save(O / "closeout.json", {"status":"FINISHED" if finish.exists() else "STOPPED",
        "accepted":len(records),"physical_steps":manifest["steps"],"counted_forecasts":manifest["forecasts"],
        "wall_s":end["wall_s"],"raw_logical_files":raw["logical_files"],
        "raw_records_manifest_sha256":sha(O / "raw_records_manifest.json")})
    print(json.dumps({"accepted":len(records),"pairs":len(pairs),"archive_files":raw["logical_files"]}))


if __name__ == "__main__":
    close()
