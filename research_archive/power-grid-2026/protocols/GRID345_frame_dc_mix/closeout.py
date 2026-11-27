"""Check the four new ledgers against the frozen historical GRID330 comparator."""
import runpy

import runtime as rt

OLD = rt.R / "outputs/grid330_remaining19"


def main():
    finished, error = (rt.O / name for name in ("finished.json", "collection_error.json"))
    assert finished.exists() != error.exists()
    active = rt.O / "active_worker.json"
    assert not active.exists() or rt.read(active)["active"] is False
    assert not (rt.O / "closeout.json").exists()
    status = rt.read(finished if finished.exists() else error)
    manifest_path = rt.O / "manifest.json"
    manifest = rt.read(manifest_path) if manifest_path.exists() else dict(records=[], steps=0, forecasts=0)
    design = rt.read(rt.A / "design.json")
    assert len(manifest["records"]) == status["accepted"]
    assert manifest["steps"] <= rt.CAPS["physical_steps"] and manifest["forecasts"] <= rt.CAPS["public_forecasts"]
    shared = runpy.run_path(str(rt.P / "GRID312_frame_208_261_pilot/results/close_grid312.py"))
    prior = runpy.run_path(str(rt.P / "GRID330_remaining19/results/closeout.py"))
    checked, compare, archive = (shared[key] for key in ("checked", "compare", "archive"))
    params = checked.__globals__
    params.update(O=rt.O, B=rt.R / "archives/GRID345_frame_dc_mix/raw_records",
                  COMPARISONS=((rt.ARM, rt.REFERENCE),))
    history = {r["week"]: r for r in rt.read(OLD / "paired_results.json")["records"]
               if r["arm"] == rt.REFERENCE}
    records, reference, details = [], [], {}
    for saved, item in zip(manifest["records"], design["execution_order"], strict=finished.exists()):
        row = checked(saved, item, {"complete": False}, .005)
        row["arm"] = rt.ARM
        row.pop("quality")
        row.pop("full_cost_ratio")
        folder = rt.runner.folder(item)
        logs = [rt.read(path) for path in sorted(folder.glob("call[0-9][0-9][0-9].json"))]
        for call in logs:
            original = sorted(call["representatives"])
            order = call["order"]
            dc = call["dc_mix_receipt"]
            assert call["order_mode"] == "STATIC64_DC64" and not call["learned_decisions"]
            assert len(order) == len(set(order)) == len(original) and set(order) == set(original)
            if dc["status"] == "ranked":
                assert order[:64] == original[:64] and order[:128] == dc["mix128"]
                prefix = set(order[:128])
                assert order[128:] == [i for i in original if i not in prefix]
            else:
                assert dc["status"] == "bypass" and order == original
        summary = saved["summary"]
        assert summary["local_model_inference_s"] == 0
        records.append(row)
        frozen = next(value for value in design["comparisons"] if value["week"] == row["week"])
        expected = rt.read(rt.A / "source_freeze.json")
        assert all(rt.sha(rt.R / p) == expected[p] for p in frozen["files"])
        meta, steps = (rt.R / p for p in frozen["files"][1:])
        ref = {**history[row["week"]], "library_hashes": rt.read(meta)["library_hashes"]}
        reference.append(ref)
        details[row["week"]] = prior["first_difference"](
            prior["steps"](folder / "steps.jsonl.gz"), prior["steps"](steps))
    pairs = compare(records + reference)
    for row in pairs:
        row.update(first_difference=details[row["week"]], fresh_timing=False)
    for row in records + reference:
        row.pop("library_hashes")
    raw = archive()
    rt.save(rt.O / "paired_results.json", {"records": records, "reference": reference, "pairs": pairs})
    rt.save(rt.O / "closeout.json", dict(status="FINISHED" if finished.exists() else "STOPPED",
            accepted=len(records), planned=rt.RUNS, steps=manifest["steps"], forecasts=manifest["forecasts"],
            archive_manifest_sha256=rt.sha(rt.O / "raw_records_manifest.json"), archived_files=raw["logical_files"]))
    print(f"{len(records)}/{rt.RUNS} audited; {len(pairs)} paired comparisons")


if __name__ == "__main__":
    main()