"""Freeze and qualify the exact GRID330 metadata repair; never start an environment."""
import argparse
import ast
from datetime import datetime, timezone
import subprocess
import sys

import runtime as rt


def refs(*paths):
    return {p.relative_to(rt.R).as_posix(): rt.sha(p) for p in paths}


def freeze():
    head = rt.git("rev-parse", "HEAD").decode().strip()
    for path in (rt.BASE / "authorization.json", rt.BASE / "design.json", rt.BASE / "source_freeze.json",
                 rt.BASE / "offline_review.json", rt.BASE / "runtime.py"):
        rt.original.tracked(path, head)
    start = rt.read(rt.OLD / "launch_context.json")
    error = rt.read(rt.OLD / "collection_error.json")
    log = (rt.OLD / "replay_00.log").read_text(encoding="utf8")
    assert "KeyError: 'timing'" in log and error["accepted"] == error["steps_accepted"] == 0
    assert rt.read(rt.OLD / "worker_setup_000.json")["pool"] is None
    assert not any(p.is_file() for p in (rt.OLD / "runs").rglob("*"))
    assert not (rt.OLD / "manifest.json").exists()
    files = ["launch_context.json", "collection_error.json", "pipeline_state.json", "active_worker.json",
             "manager_stderr.log", "replay_00.log", "worker_setup_000.json"]
    rt.save(rt.A / "recovery_source.json", {
        "original_authorization": start["authorization_commit"], "original_error": "KeyError: timing",
        "original_files": {name: rt.sha(rt.OLD / name) for name in files},
        "accepted_before": 0, "environment_initialized_before": True, "physical_steps_before": 0,
        "candidate_queries_before": 0, "startup_forecast_count_measured": False,
        "forecast_reserve": 1000, "original_failed_wall_s": error["wall_s"],
        "original_deadline_utc": datetime.fromtimestamp(start["start_utc"] + 3600, timezone.utc).isoformat(),
        "repair": "Add the worker metadata field timing. Preserve original execution_order and numerical sources. Use separate output and claims. Count startup forecasts before metadata, keep the original clock and debit the failed startup from first worker cap.",
    })
    rt.anchor()
    design = rt.read(rt.BASE / "design.json")
    design.update(id=rt.SCOPE, caps=rt.CAPS,
                  timing="Fresh serial workers; arm start rotated by date; compare equal-horizon ctl.propose timings.")
    rt.save(rt.A / "design.json", design)
    (rt.O / "design.json").write_bytes((rt.A / "design.json").read_bytes())
    old = rt.read(rt.BASE / "source_freeze.json")
    data = {rel: val for rel, val in old.items() if not rel.startswith(rt.OLD.relative_to(rt.R).as_posix() + "/")}
    data.update(refs(rt.BASE / "source_freeze.json", rt.BASE / "authorization.json",
        rt.A / "runtime.py", rt.A / "prepare.py", rt.A / "design.json", rt.A / "recovery_source.json",
        rt.O / "design.json", *(rt.OLD / name for name in files)))
    for p, expected in data.items():
        assert rt.sha(rt.R / p) == expected, p
    rt.save(rt.O / "code_freeze.json", data)
    rt.save(rt.A / "source_freeze.json", {**data, **refs(rt.O / "code_freeze.json")})


def literal_keys(text, variable):
    return {n.slice.value for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name) and n.value.id == variable
            and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}


def qualify():
    head = rt.git("rev-parse", "HEAD").decode().strip()
    for name in ("runtime.py", "prepare.py", "design.json", "recovery_source.json", "source_freeze.json"):
        rt.original.tracked(rt.A / name, head)
    original = rt.read(rt.BASE / "design.json")
    design = rt.read(rt.A / "design.json")
    assert design["execution_order"] == original["execution_order"]
    assert {k for k in design if design[k] != original.get(k)} == {"id", "caps", "timing"}
    assert design["caps"] == {**original["caps"], "public_forecasts": 249000}
    worker = rt.instrument_worker((rt.R / "work/grid55/worker.py").read_text(encoding="utf8"))
    launcher = rt.assemble()
    manager = rt.manager_source()
    assert literal_keys(worker, "design") <= design.keys() and "timing" in literal_keys(worker, "design")
    assert literal_keys(launcher, "D") <= design.keys()
    required_items = literal_keys(worker, "item") | literal_keys(launcher, "item")
    assert all(required_items <= item.keys() for item in design["execution_order"])
    assert "'startup_forecast_receipt.json'" in worker and "failed[\"wall_s\"]" in manager
    for name, src in (("worker", worker), ("manager", manager), ("launcher", launcher)):
        compile(src, "grid330_recovery_" + name, "exec")
    assert not (rt.O / "launch_context.json").exists() and not (rt.O / "runs").exists()
    assert not (rt.A / "authorization.json").exists()
    for arguments in (["run"], ["worker", "0"]):
        p = subprocess.run([sys.executable, "-B", "-X", "utf8", str(rt.ENTRY), *arguments],
                           cwd=rt.R, capture_output=True, text=True, encoding="utf8", timeout=15)
        assert p.returncode == 2 and "recovery authorization absent before live imports" in p.stderr
    assert not any(x in sys.modules for x in ("torch", "common", "grid2op"))
    rt.save(rt.A / "offline_review.json", {"passed": True, "source_commit": head,
        "compiled": ["worker", "manager", "launcher"],
        "worker_design_fields": sorted(literal_keys(worker, "design")),
        "item_fields": sorted(required_items), "items_checked": 20, "entries_denied_without_receipt": 2,
        "model_loads": 0, "predictions": 0, "physical_steps": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "qualify"))
    {"freeze": freeze, "qualify": qualify}[parser.parse_args().stage]()

