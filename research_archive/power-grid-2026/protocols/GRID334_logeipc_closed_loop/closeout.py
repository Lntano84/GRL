"""Audit saved matched-horizon metrics and archive every started GRID334 record."""
import json
import runpy
import time

import runtime as rt

BASE = rt.P / "GRID312_frame_208_261_pilot/results/close_grid312.py"


def first_difference(rows_a, rows_b):
    for a, b in zip(rows_a, rows_b):
        assert a["step"] == b["step"]
        if a["action_hash"] != b["action_hash"] or a["after_hash"] != b["after_hash"]:
            return dict(step=a["step"], same_before=a["before_hash"] == b["before_hash"],
                        same_action=a["action_hash"] == b["action_hash"])
    return None


def check_logeipc(call):
    record = call["logeipc"]
    trace = record["trace"]
    paid = call["local_queries"]
    prefix_end = len(call["outcomes"]) - paid
    assert len(trace) == paid and not call["learned_decisions"] and "load_guard" not in call
    ids = [event["id"] for event in trace]
    assert ids == call["extra_selected_ids"] == [row["pool_id"] for row in call["outcomes"][prefix_end:]]
    assert set(ids) <= set(call["completion_candidate_ids"])
    assert all(event["available"] and event["log_ei_over_fee"] > 0 for event in trace)
    if record["reason"] == "ei_not_above_fee":
        assert record["final_log_ei_over_fee"] is None or record["final_log_ei_over_fee"] <= 0
        assert paid < len(call["completion_candidate_ids"])
    elif record["reason"] == "pool_exhausted":
        assert paid == len(call["completion_candidate_ids"]) > 0
    else:
        assert record["reason"] == "no_pending" and not call["completion_candidate_ids"] and paid == 0


def main():
    import gzip
    assert not any(path.exists() for path in (rt.O / "closeout.json", rt.O / "paired_results.json"))
    assert not rt.read(rt.O / "active_worker.json")["active"]
    end, error = rt.O / "finished.json", rt.O / "collection_error.json"
    assert end.exists() != error.exists()
    close = rt.read(end if end.exists() else error)
    design = rt.read(rt.A / "design.json")
    manifest_path = rt.O / "manifest.json"
    manifest = rt.read(manifest_path) if manifest_path.exists() else dict(records=[], steps=0, forecasts=0)
    assert len(manifest["records"]) == close["accepted"]
    assert manifest["steps"] <= design["caps"]["physical_steps"]
    assert manifest["forecasts"] <= design["caps"]["public_forecasts"]
    if end.exists():
        assert close["accepted"] == len(design["execution_order"])
        assert close["steps"] == manifest["steps"] and close["forecasts"] == manifest["forecasts"]
    audit = runpy.run_path(str(BASE))
    checked, compare, archive = (audit[k] for k in ("checked", "compare", "archive"))
    target = rt.R / "archives/GRID334_logeipc_closed_loop/raw_records"
    checked.__globals__["O"] = rt.O
    archive.__globals__.update(O=rt.O, B=target)
    compare.__globals__["COMPARISONS"] = ((rt.COMPARATOR, rt.REFERENCE),)
    guard = rt.parent.parent.load_guard_receipt
    records, trajectories = [], {}
    start = time.monotonic()
    for record, item in zip(manifest["records"], design["execution_order"]):
        assert time.monotonic() - start < 600
        row = checked(record, item, {"complete": False}, .005)
        folder = rt.runner.folder(item)
        with gzip.open(folder / "steps.jsonl.gz", "rt", encoding="utf-8") as stream:
            series = [json.loads(line) for line in stream]
        logeipc_stops, guarded_queries = 0, 0
        for path in sorted(folder.glob("call[0-9][0-9][0-9].json")):
            call = rt.read(path)
            assert "dc_mix_receipt" in call and call["order_mode"] == "STATIC64_DC64"
            if item["evaluation_arm"] == rt.COMPARATOR:
                check_logeipc(call)
                logeipc_stops += call["logeipc"]["reason"] == "ei_not_above_fee"
            else:
                assert "logeipc" not in call
                guarded_queries += guard(call)
        s = record["summary"]
        row.update(arm=item["evaluation_arm"], quality="NO_FULL_REFERENCE",
                   guard_queries=guarded_queries, logeipc_fee_stops=logeipc_stops,
                   **{k: s[k] for k in ("n1_forecast_s", "local_model_inference_s", "dc_export_s", "dc_rank_s")})
        records.append(row)
        trajectories[(row["week"], row["arm"])] = series
    pairs = compare(records)
    for row in pairs:
        week = row["week"]
        row["first_difference"] = first_difference(trajectories[(week, rt.COMPARATOR)],
                                                  trajectories[(week, rt.REFERENCE)])
    for record in records:
        del record["library_hashes"]
    raw = archive()
    rt.save(rt.O / "paired_results.json", {"records": records, "pairs": pairs})
    rt.save(rt.O / "closeout.json", dict(status="FINISHED" if end.exists() else "STOPPED",
        accepted=len(records), planned=8, steps=manifest["steps"], forecasts=manifest["forecasts"],
        archive_manifest_sha256=rt.sha(rt.O / "raw_records_manifest.json"),
        archived_logical_files=raw["logical_files"], archived_bytes=raw["logical_bytes"]))


if __name__ == "__main__":
    main()

