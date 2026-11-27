"""Freeze existing inputs and qualify the new combination before replay."""
import ast
from pathlib import Path
import subprocess
import sys
import tokenize

import runtime as rt

PREVIOUS = rt.P / "GRID330_remaining19"
OLD = rt.R / "outputs/grid330_remaining19"


def relative(path):
    return path.relative_to(rt.R).as_posix()


def references(paths):
    return {relative(path): rt.sha(path) for path in paths}


def literal_keys(tree, variable):
    return {node.slice.value for node in ast.walk(tree)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == variable and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)}


def prepare():
    if rt.O.exists() or (rt.A / "authorization.json").exists():
        raise FileExistsError("Preserve existing GRID345 preparation or run")
    previous = rt.read(PREVIOUS / "design.json")
    original = rt.read(rt.parent.A / "design.json")
    finished = rt.read(OLD / "finished.json")
    assert finished["accepted"] == 19
    old = rt.read(OLD / "paired_results.json")["records"]
    manifest = rt.read(OLD / "manifest.json")["records"]
    templates = {r["week"]: r for r in original["execution_order"] if r["evaluation_arm"] == "FRAME_GUARD"}
    comparator = {r["week"]: r for r in old if r["arm"] == rt.REFERENCE}
    items, input_paths, comparison = [], [], []
    for day in previous["calendar"]:
        week = day["scenario"]
        assert day["environment_seed"] == 0 and day["scenario_id"] == 100
        item = dict(templates[week], index=len(items), method=rt.ARM, evaluation_arm=rt.ARM, dc_mix=True)
        assert (item["arm"], item["load_guard"], item["dc_mix"]) == rt.parent.ARMS[rt.ARM]
        assert item["pool"] is item["model_seed"] is None and item["within_pair"] == 0
        saved = comparator[week]
        record = manifest[saved["index"]]
        summary = record["summary"]
        assert summary["week"] == week and summary["evaluation_arm"] == rt.REFERENCE
        folder = OLD / "runs" / f"r{summary['repeat']}__{week}__{summary['method']}"
        paths = [folder / name for name in ("summary.json", "metadata.json", "steps.jsonl.gz")]
        assert rt.read(paths[0]) == summary
        input_paths.extend(paths)
        comparison.append({"week": week, "saved_index": saved["index"], "files": [relative(p) for p in paths]})
        items.append(item)
    assert len(items) == len(comparator) == rt.RUNS
    assert len({item["week"] for item in items}) == rt.RUNS
    frozen = {path: expected for path, expected in rt.read(PREVIOUS / "source_freeze.json").items()
              if not path.startswith(("outputs/grid330_remaining19/", "outputs/grid330_dc_mix_recovery01/"))}
    new_refs = references([*rt.A.glob("*.py"), PREVIOUS / "source_freeze.json", *input_paths,
                           OLD / "paired_results.json", OLD / "finished.json", OLD / "manifest.json"])
    frozen.update(new_refs)
    for path, expected in frozen.items():
        actual = rt.R / path
        if not actual.is_file() or rt.sha(actual) != expected:
            raise ValueError(f"Frozen input differs: {path}")
    design = dict(id=rt.SCOPE, caps=rt.CAPS, execution_order=items, comparisons=comparison,
                  environment=previous["environment"], environment_seed=0, calendar=previous["calendar"],
                  method="Frozen GRID330 64+64 global order with original FRAME_GUARD completion.",
                  timing="Historical 261 timing is descriptive. Compare matching horizons only.",
                  new_model_fits=0, new_scenarios=0, no_retry=True)
    launcher = rt.assemble()
    worker_path = rt.R / "work/grid55/worker.py"
    with tokenize.open(worker_path) as stream:
        worker = rt.runner.instrument_worker(stream.read())
    nodes = [node for node in ast.parse(worker).body if isinstance(node, ast.FunctionDef)
             and node.name == "worker"]
    entry, = nodes
    assert literal_keys(ast.parse(launcher), "D") | literal_keys(entry, "design") <= design.keys()
    for item in items:
        assert literal_keys(ast.parse(launcher), "item") | literal_keys(entry, "item") <= item.keys()
    adapter = rt.parent.adapter_source(items[0])
    assert "provider=None if item['arm']=='FRAME_GUARD'" in adapter
    assert "N.CompletionPolicy('FRAME_GUARD')" in adapter and "GuardedPolicyNamespace" not in adapter
    assert "STATIC64_DC64" in adapter
    assert "None if item['arm']=='FRAME_GUARD' else models297.FrozenModels" in launcher
    texts = {"launcher": launcher, "worker": worker, "adapter": adapter}
    for name, source in texts.items():
        compile(source, name, "exec")
    denied = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", str(rt.A / "runtime.py"), "worker", "0"],
        check=False, timeout=20, capture_output=True)
    assert denied.returncode == 2 and b"authorization absent before live imports" in denied.stderr
    assert not {"common", "grid2op", "torch"} & sys.modules.keys()

    rt.O.mkdir()
    rt.save(rt.A / "design.json", design)
    (rt.O / "design.json").write_bytes((rt.A / "design.json").read_bytes())
    frozen.update(references([rt.A / "design.json", rt.O / "design.json"]))
    rt.save(rt.O / "code_freeze.json", frozen)
    rt.save(rt.A / "source_freeze.json", {**frozen, **references([rt.O / "code_freeze.json"])})
    rt.save(rt.A / "offline_review.json", dict(passed=True, frozen_inputs=len(frozen),
            compiled=list(texts), missing_authorization_refused=True, model_loads=0, forecasts=0, physical_steps=0))
    rt.save(rt.A / "authorization.json", dict(**rt.receipt_fields(), approved_by_user=True,
            user_instruction="来做一次",
            user_message_reference="Direct user reply to the immediately preceding four-replay proposal"))
    print(f"Four replay entries and {len(frozen)} frozen inputs verified")


if __name__ == "__main__":
    prepare()