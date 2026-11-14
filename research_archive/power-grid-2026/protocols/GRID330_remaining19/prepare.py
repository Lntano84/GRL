"""Preserve the historic baseline and validate the exact offline launch inputs."""
import argparse
import ast
import gzip
import math
import subprocess
import sys

import runtime as rt

OLD_REC = rt.R / "outputs/grid330_dc_mix_recovery01"
OLD = rt.parent.O
PRIOR_REC = rt.A.parent / "GRID330_dc_mix_recovery01"


def refs(paths):
    return {p.relative_to(rt.R).as_posix(): rt.sha(p) for p in paths}


def freeze():
    manifest = rt.read(OLD_REC / "manifest.json")
    accepted, = manifest["records"]
    baseline = accepted["summary"]
    original = rt.read(rt.PARENT / "design.json")
    assert rt.read(PRIOR_REC / "design.json")["execution_order"] == original["execution_order"]
    assert baseline["index"] == 0 and baseline["evaluation_arm"] == "FRAME_GUARD"
    assert all(baseline[k] == v for k, v in original["execution_order"][0].items())
    assert rt.read(OLD_REC / "pipeline_state.json") == {"phase": "stopped", "accepted": 1}
    assert rt.read(OLD_REC / "active_worker.json")["active"] is False
    assert "model_references.json" in (OLD_REC / "replay_01.log").read_text(encoding="utf8")
    assert sorted(p.stem for p in (OLD_REC / "claims").glob("*.started")) == ["00", "01"]
    folder = OLD_REC / "runs" / ("r0__" + baseline["week"] + "__" + baseline["method"])
    assert list((OLD_REC / "runs").iterdir()) == [folder]
    assert rt.read(folder / "summary.json") == baseline == rt.read(OLD_REC / "worker_result_00.json")
    assert rt.read(folder / "startup_forecast_receipt.json")["forecasts"] == 0
    with gzip.open(folder / "steps.jsonl.gz", "rt", encoding="utf8") as stream:
        import json
        rows = [json.loads(line) for line in stream]
    assert len(rows) == baseline["steps"] == 2017 and rows[-1]["complete"]
    assert all(a["after_hash"] == b["before_hash"] and not a["done"] for a, b in zip(rows, rows[1:]))
    assert abs(math.fsum(r["raw_cost"] for r in rows) - baseline["cost"]) < 1e-6
    assert sum(r["simulations"] for r in rows) == accepted["counted_forecasts"] == 7084
    prior_files = [OLD_REC / name for name in ("manifest.json", "collection_error.json", "worker_result_00.json",
                      "replay_01.log", "pipeline_state.json", "active_worker.json")]
    baseline_files = [folder / name for name in ("summary.json", "steps.jsonl.gz", "metadata.json",
                                                "startup_forecast_receipt.json")]
    rt.save(rt.A / "historical_receipt.json", {
        "completed_original_index": 0, "steps": 2017, "counted_forecasts": 7084,
        "baseline_folder": folder.relative_to(rt.R).as_posix(),
        "failed_pre_activity_original_index": 1, "never_started": list(range(2, 20)),
        "files": refs(prior_files + baseline_files),
        "baseline_timing": "Recorded on November 9. Keep separate from fresh timing comparisons.",
    })
    items = [dict(item, index=i, original_index=item["index"])
             for i, item in enumerate(original["execution_order"][1:])]
    assert [r["original_index"] for r in items] == list(range(1, 20))
    design = dict(original, id=rt.SCOPE, caps=rt.CAPS, execution_order=items,
                  timing="One serial continuation. Preserve the November 9 FRAME baseline and exclude its timing from fresh paired timing.",
                  historical_baseline="historical_receipt.json")
    rt.save(rt.A / "design.json", design)
    (rt.O / "design.json").write_bytes((rt.A / "design.json").read_bytes())
    previous = rt.read(rt.PARENT / "source_freeze.json")
    freeze = {p: h for p, h in previous.items() if not p.startswith(OLD.relative_to(rt.R).as_posix() + "/")}
    models = OLD / "model_references.json"
    assert rt.sha(models) == previous[models.relative_to(rt.R).as_posix()]
    (rt.O / models.name).write_bytes(models.read_bytes())
    model, = rt.read(rt.O / models.name)
    assert model["pool"] == design["pool"] == "f0_s0" and len(model["hashes"]) == 7
    assert all(freeze[path] == expected == rt.sha(rt.R / path) for path, expected in model["hashes"].items())
    freeze.update(refs([*rt.A.glob("*.py"), rt.A / "design.json", rt.A / "historical_receipt.json",
                       rt.PARENT / "source_freeze.json", models, rt.O / models.name, rt.O / "design.json",
                       *prior_files, *baseline_files]))
    for p, expected in freeze.items():
        assert rt.sha(rt.R / p) == expected, p
    rt.save(rt.O / "code_freeze.json", freeze)
    rt.save(rt.A / "source_freeze.json", {**freeze, **refs([rt.O / "code_freeze.json"])})


def literal_keys(tree, var):
    return {n.slice.value for n in ast.walk(tree)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == var
            and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}


def qualify():
    design = rt.check_local()
    worker = rt.runner.instrument_worker((rt.R / "work/grid55/worker.py").read_text(encoding="utf8"))
    launch = rt.assemble()
    entry, = [n for n in ast.parse(worker).body if isinstance(n, ast.FunctionDef) and n.name == "worker"]
    assert literal_keys(entry, "design") <= design.keys()
    assert literal_keys(ast.parse(launch), "D") <= design.keys()
    assert all((literal_keys(entry, "item") | literal_keys(ast.parse(launch), "item")) <= item.keys()
               for item in design["execution_order"])
    sources = {"worker": worker, "launcher": launch}
    for item in design["execution_order"]:
        sources[item["evaluation_arm"]] = rt.parent.adapter_source(item)
    assert len(sources) == 7
    for name, text in sources.items():
        compile(text, name, "exec")
    assert rt.runner.O == rt.O and "grid330_remaining19" in launch
    assert rt.read(rt.O / "model_references.json") == rt.read(OLD / "model_references.json")
    assert not (rt.O / "launch_context.json").exists() and not (rt.O / "runs").exists()
    for args in (["run"], ["worker", "0"]):
        p = subprocess.run([sys.executable, "-B", "-X", "utf8", str(rt.A / "runtime.py"), *args],
                           capture_output=True, text=True, encoding="utf8", timeout=15)
        assert p.returncode == 2 and "authorization absent before live imports" in p.stderr
    assert not any(k in sys.modules for k in ("torch", "common", "grid2op"))
    rt.save(rt.A / "offline_review.json", {"passed": True, "compiled": list(sources),
        "frozen_inputs": len(rt.read(rt.A / "source_freeze.json")), "items": len(design["execution_order"]),
        "model_reference_files": 7, "missing_authorization_entries_rejected": 2,
        "model_loads": 0, "physical_steps": 0, "predictions": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "qualify"))
    {"freeze": freeze, "qualify": qualify}[parser.parse_args().stage]()

