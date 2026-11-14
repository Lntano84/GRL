"""Read only finished GRID330 records and preserve the raw run artifacts."""
import gzip
import json
from pathlib import Path
import runpy
import subprocess
import time

ROOT = Path("D:/Codex/GRL研究")
REPO = ROOT / "work/GRL-results-integration"
P = REPO / "research_archive/power-grid-2026/protocols"
A = P / "GRID330_remaining19"
O = ROOT / "outputs/grid330_remaining19"
B = ROOT / "archives/GRID330_remaining19/raw_records"
FIELDS = ("step", "before_hash", "action_hash", "after_hash", "raw_cost", "done", "complete")
COMPONENTS = ("n1_source_s", "n1_prepare_s", "dc_export_s", "dc_rank_s", "n1_forecast_s",
              "feature_s", "local_model_inference_s", "dc_ranked_calls", "dc_bypassed_calls")


def steps(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def first_difference(a, b):
    for left, right in zip(a, b):
        assert left["step"] == right["step"]
        if any(left[k] != right[k] for k in FIELDS):
            return {"step": left["step"], "same_input": left["before_hash"] == right["before_hash"],
                    "same_action": left["action_hash"] == right["action_hash"],
                    "cost_delta_at_step": left["raw_cost"] - right["raw_cost"]}
    return None


def main():
    audit = runpy.run_path(str(P / "GRID312_frame_208_261_pilot/results/close_grid312.py"))
    checked, archive, compare = (audit[key] for key in ("checked", "archive", "compare"))
    read, save, sha = (audit[key] for key in ("read", "save", "sha"))
    checked.__globals__["O"] = O
    archive.__globals__["B"] = B
    design = read(A / "design.json")
    compare.__globals__["COMPARISONS"] = [tuple(x) for x in design["comparisons"].values()]
    assert not (O / "active_worker.json").exists() or not read(O / "active_worker.json")["active"]
    proc = subprocess.check_output(["powershell", "-NoProfile", "-Command",
        "@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and "
        "$_.CommandLine -match 'GRID330_remaining19[/\\\\]runtime\\.py'}).Count"]).decode().strip()
    assert proc == "0"
    end, error = O / "finished.json", O / "collection_error.json"
    assert end.exists() != error.exists()
    close = read(end if end.exists() else error)
    manifest = read(O / "manifest.json") if (O / "manifest.json").exists() else {"records": [], "steps": 0, "forecasts": 0}
    assert len(manifest["records"]) == close["accepted"]
    assert manifest["steps"] <= design["caps"]["physical_steps"]
    assert manifest["forecasts"] <= design["caps"]["public_forecasts"]
    if end.exists():
        assert close["accepted"] == len(design["execution_order"]) == 19
        assert close["steps"] == manifest["steps"] and close["forecasts"] == manifest["forecasts"]
    for name in ("authorization.json", "design.json", "source_freeze.json", "offline_review.json"):
        local = O / name
        if not local.exists():
            local.write_bytes((A / name).read_bytes())
        assert local.read_bytes() == (A / name).read_bytes()
    receipts = runpy.run_path(str(P / "GRID320_load_guard_closed_loop/runtime.py"))
    load_guard = receipts["load_guard_receipt"]
    records, series = [], {}
    start = time.monotonic()
    for record, item in zip(manifest["records"], design["execution_order"]):
        assert time.monotonic() - start < 600
        row = checked(record, item, {"complete": False}, .005)
        summary = record["summary"]
        folder = O / "runs" / f"r{item['repeat']}__{item['week']}__{item['method']}"

        logs = [read(p) for p in sorted(folder.glob("call[0-9][0-9][0-9].json"))]
        mixed, bypassed = [], []
        for call in logs:
            n = load_guard(call) if item["load_guard"] else 0
            row["guard_queries"] = row.get("guard_queries", 0) + n
            receipt = call.get("dc_mix_receipt")
            assert (receipt is not None) == item["dc_mix"]
            if receipt is None:
                continue
            if receipt["status"] == "ranked":
                original = sorted(call["representatives"])
                order = call["order"]
                assert order[:128] == receipt["mix128"] and order[:64] == original[:64]
                assert order[128:] == [i for i in original if i not in set(order[:128])]
                assert len(order) == len(set(order)) == len(original)
                assert set(order) == set(original)
                assert receipt["scored"] + sum(receipt["fallback"].values()) == len(order)
                mixed.append(receipt)
            else:
                assert receipt["status"] == "bypass" and call["order"] == sorted(call["representatives"])
                bypassed.append(receipt)
        assert len(mixed) == summary["dc_ranked_calls"] and len(bypassed) == summary["dc_bypassed_calls"]
        row.update({k: summary[k] for k in COMPONENTS})
        row.update(model_arm=item["arm"], arm=item["evaluation_arm"], quality="NO_FULL_REFERENCE",
                   guard_queries=row.get("guard_queries", 0),
                   dc_bypass_reasons=sorted({r["reason"] for r in bypassed}))
        series[(row["week"], row["arm"])] = steps(folder / "steps.jsonl.gz")
        records.append(row)
    history = ROOT / "outputs/grid330_dc_mix_recovery01"
    history_receipt = read(A / "historical_receipt.json")
    assert all(sha(ROOT / path) == expected for path, expected in history_receipt["files"].items())
    historic, = read(history / "manifest.json")["records"]
    historic_item = read(history / "design.json")["execution_order"][0]
    checked.__globals__["O"] = history
    old = checked(historic, historic_item, {"complete": False}, .005)
    checked.__globals__["O"] = O
    old.update({k: historic["summary"][k] for k in COMPONENTS})
    old.update(quality="NO_FULL_REFERENCE", timing_group="historical", model_arm=old["arm"],
               guard_queries=0, dc_bypass_reasons=[])
    for record in records:
        record["timing_group"] = "fresh"
    series[(old["week"], old["arm"])] = steps(ROOT / history_receipt["baseline_folder"] / "steps.jsonl.gz")

    pairs = compare(records + [old])
    historic_step_times = ("controller_s", "all_p95", "all_p99", "search_p95", "search_p99")
    for pair in pairs:
        historical = pair["week"] == old["week"] and pair["reference"] == old["arm"]
        pair["fresh_timing"] = not historical
        if historical:
            for field in historic_step_times:
                pair[field + "_change_pct"] = None
    for pair in pairs:
        a = series[(pair["week"], pair["arm"])]
        b = series[(pair["week"], pair["reference"])]
        pair["first_difference"] = first_difference(a, b)
    for record in records + [old]:
        del record["library_hashes"]
    save(O / "paired_results.json", {"records": records, "historical_baseline": old, "pairs": pairs,
         "new_grid2op_simulation_in_readout": 0, "new_model_forward_in_readout": 0})
    raw = archive()
    save(O / "closeout.json", {"status": "FINISHED" if end.exists() else "STOPPED",
         "accepted": len(records), "planned": 19, "physical_steps": manifest["steps"],
         "counted_forecasts": manifest["forecasts"], "wall_s": close["wall_s"],
         "raw_logical_files": raw["logical_files"], "raw_bytes": raw["logical_bytes"],
         "raw_records_manifest_sha256": sha(O / "raw_records_manifest.json")})


if __name__ == "__main__":
    main()

