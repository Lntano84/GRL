"""Run the forty GRID301 replay indices that were never started."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback
import uuid


R = Path("D:/Codex/GRL研究")
I = R / "work/GRL-results-integration"
O = R / "outputs/grid301_remaining_collection"
OUT = R / "outputs/grid302_unstarted_opposites"
W = R / "work/grid301_remaining_collection/recovery03"
SOURCE = Path(__file__).resolve().parent
APPROVAL = SOURCE / "approval.json"
PLAN = SOURCE / "remaining_scope.json"
OLD_BOOT_FILES = ("design.json", "runtime_design.json", "code_freeze.json", "model_references.json")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old_guard = load_module("grid301_base_guard", W.parent / "guard.py")
read, save, sha, git = old_guard.read, old_guard.save, old_guard.sha, old_guard.git
verify_generated = old_guard.verify_generated
verify_stage = old_guard.verify_stage


def check(context):
    launch = read(OUT / "launch_context.json")
    for field in ("boot_id", "start_monotonic", "deadline_monotonic", "deadline_utc_epoch",
                  "authorization_sha256", "manager_pid", "token"):
        if launch[field] != context[field]:
            raise PermissionError(f"Launch context differs on {field}")
    if old_guard.boot_id() != launch["boot_id"] or (OUT / "STOP_REQUEST.json").exists():
        raise RuntimeError("New scope stopped or boot changed")
    if time.time() >= launch["deadline_utc_epoch"] or time.perf_counter() >= min(
        launch["deadline_monotonic"], context.get("worker_deadline_monotonic", float("inf"))
    ):
        raise TimeoutError("GRID302 time cap reached")


def verify_publication():
    approval = read(APPROVAL)
    plan = read(PLAN)
    if (
        approval["user_message_reference"] != "request_user_input_async/call_1207b1b02071415ca6637279c6a8a23c/0"
        or approval["user_reply"] != "批准40条回放，限时90分钟"
        or approval["scope_sha256"] != sha(PLAN)
        or plan["proposed_wall_cap_s"] != 5400
        or [item["index"] for item in plan["execution_order"]] != list(range(337, 377))
    ):
        raise PermissionError("GRID302 approval or scope differs")
    head = git("rev-parse", "HEAD").decode().strip()
    if git("status", "--porcelain").strip():
        raise PermissionError("Integration worktree must be clean")
    if git("ls-remote", "lntano-src", "refs/heads/main").decode().split()[0] != head:
        raise PermissionError("Remote does not match local HEAD")
    for p in (APPROVAL, PLAN, Path(__file__).resolve()):
        if git("show", f"{head}:{p.relative_to(I).as_posix()}") != p.read_bytes():
            raise PermissionError(f"Publication differs for {p.name}")
    return plan, head


def verify_inputs(plan):
    for name, expected in plan["source_sha256"].items():
        if sha(O / name) != expected:
            raise PermissionError(f"GRID301 source changed: {name}")
    old_guard.verify_frozen()
    for name, expected in read(O / "recovery03_source_freeze.json").items():
        if sha(R / name) != expected:
            raise PermissionError(f"Prior runtime dependency changed: {name}")
    verify_stage("baseline")
    for name in OLD_BOOT_FILES:
        if sha(O / name) != sha(OUT / name):
            raise PermissionError(f"Mirrored metadata differs: {name}")


def build_in_new_directory():
    builder = load_module("source_builder", W / "source_builder.py")
    original_build = builder.build

    def build():
        source = original_build()
        marker = "OUT=ROOT/'outputs/grid301_remaining_collection'"
        if source.count(marker) != 1:
            raise ValueError("Frozen output marker differs")
        return source.replace(marker, "OUT=ROOT/'outputs/grid302_unstarted_opposites'", 1)

    builder.build = build
    return builder


def run_folder(root, item):
    return root / "runs" / f"r{item['repeat']}__{item['week']}__{item['method']}"


def ensure_unstarted(item):
    index = item["index"]
    for root in (O, OUT):
        if (root / f"worker_result_{index:02d}.json").exists() or run_folder(root, item).exists():
            raise RuntimeError(f"Existing result or trajectory at index {index}")
        if any(root.glob(f"claims*/replay_{index:03d}.*")):
            raise RuntimeError(f"Existing claim at index {index}")


def claim(kind, index):
    plan, _ = verify_publication()
    if kind != "replay" or index not in {item["index"] for item in plan["execution_order"]}:
        raise PermissionError("Worker outside the approved forty")
    context = read(OUT / "run_context.json")
    file = OUT / "claims" / f"replay_{index:03d}.json"
    expected = read(file)
    if (expected != context or context["index"] != index or context["kind"] != kind
        or os.environ.get("GRL_GRID300_TOKEN") != context["token"]
        or not old_guard.ancestor(context["manager_pid"])):
        raise PermissionError("Worker claim or manager identity differs")
    check(context)
    with file.with_suffix(".started").open("x", encoding="utf8") as out:
        out.write(str(os.getpid()))
    return context


def worker(index):
    # The successful GRID301 worker performs the original paired_adapter registration.
    sys.path.insert(0, str(W))
    sys.modules["guard"] = sys.modules[__name__]
    sys.modules["source_builder"] = build_in_new_directory()
    sys.argv = [str(W / "worker.py"), "--index", str(index)]
    runpy.run_path(str(W / "worker.py"), run_name="__main__")


def launch_worker(context, item):
    index = item["index"]
    log_path = OUT / f"replay_{index:03d}.log"
    env = {**os.environ, "GRL_GRID300_TOKEN": context["token"],
           "PYTHONDONTWRITEBYTECODE": "1", "TEMP": str(R / "tmp/grid100"),
           "TMP": str(R / "tmp/grid100")}
    with log_path.open("x", encoding="utf8") as log:
        child = subprocess.Popen(
            [sys.executable, "-B", "-X", "utf8", "-u", str(Path(__file__).resolve()),
             "worker", "--index", str(index)],
            cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        save(OUT / "active_worker.json", {"active": True, "pid": child.pid, "index": index})
        try:
            while child.poll() is None:
                check(context)
                time.sleep(0.5)
        except BaseException:
            if child.poll() is None:
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                               stdout=log, stderr=subprocess.STDOUT, check=False)
                child.wait()
            raise
        finally:
            save(OUT / "active_worker.json", {"active": False, "index": index})
    if child.returncode:
        raise RuntimeError(f"Replay {index} exited with {child.returncode}")


def run():
    plan, head = verify_publication()
    boot_id = old_guard.boot_id()
    started = time.perf_counter()
    launch = {
        "start_monotonic": started, "deadline_monotonic": started + 5400,
        "deadline_utc_epoch": time.time() + 5400, "boot_id": boot_id,
        "authorization_sha256": sha(APPROVAL), "authorization_commit": head,
        "manager_pid": os.getpid(), "token": uuid.uuid4().hex,
    }
    with (OUT / "launch_context.json").open("x", encoding="utf8") as file:
        json.dump(launch, file)
    records = []
    steps = forecasts = 0
    current_index = None
    try:
        verify_inputs(plan)
        check(launch)
        for item in plan["execution_order"]:
            check(launch)
            ensure_unstarted(item)
            current_index = item["index"]
            context = {
                **launch, "kind": "replay", "index": current_index,
                "worker_deadline_monotonic": time.perf_counter() + plan["per_run_cap_s"],
                "physical_remaining": plan["new_physical_step_cap"] - steps,
                "forecast_remaining": plan["forecast_cap_remaining_under_original_count_limit"] - forecasts,
            }
            if min(context["physical_remaining"], context["forecast_remaining"]) <= 0:
                raise TimeoutError("GRID302 resource cap reached")
            save(OUT / "pipeline_state.json", {"phase": "running", "index": current_index,
                                              "new_accepted": len(records)})
            claim_file = OUT / "claims" / f"replay_{current_index:03d}.json"
            claim_file.parent.mkdir(exist_ok=True)
            with claim_file.open("x", encoding="utf8") as file:
                json.dump(context, file)
            save(OUT / "run_context.json", context)
            launch_worker(context, item)
            directory = run_folder(OUT, item)
            summary = read(directory / "summary.json")
            source = read(directory / "intervention.json")
            assert summary == read(OUT / f"worker_result_{current_index:02d}.json")
            assert summary["steps"] == source["steps"] and source["prefix_matched"]
            assert source["event"]["common"] == item["target"]["common"]
            assert summary["new_model_fits"] == source["new_cost_labels"] == 0
            assert summary["steps"] <= context["physical_remaining"]
            assert source["forecasts"] <= context["forecast_remaining"]
            steps += summary["steps"]
            forecasts += source["forecasts"]
            records.append({"index": current_index, "summary": summary, "source": source})
            save(OUT / "manifest.json", {"records": records, "new_accepted": len(records),
                                        "physical_steps_new": steps, "forecasts_new": forecasts})
        save(OUT / "finished.json", {"new_accepted": len(records), "physical_steps_new": steps,
                                     "forecasts_new": forecasts, "wall_s": time.perf_counter() - started,
                                     "new_fits": 0, "cost_targets_derived": 0})
        save(OUT / "pipeline_state.json", {"phase": "finished", "new_accepted": len(records)})
    except BaseException:
        save(OUT / "collection_error.json", {"traceback": traceback.format_exc(),
                                             "new_accepted": len(records), "last_index": current_index,
                                             "physical_steps_new": steps, "forecasts_new": forecasts,
                                             "wall_s": time.perf_counter() - started})
        raise


def verify_prepared():
    plan = read(PLAN)
    for item in plan["execution_order"]:
        ensure_unstarted(item)
    for name in OLD_BOOT_FILES:
        assert sha(O / name) == sha(OUT / name)
    source = build_in_new_directory().build()
    compile(source, str(R / "work/grid55/run.py"), "exec")
    assert "outputs/grid302_unstarted_opposites" in source
    assert "outputs/grid301_remaining_collection" not in source
    assert read(O / "load_only_review_recovery03.json")["passed"]
    assert read(O / "baseline_publication.json")["passed"]
    print("Prepared 40 unclaimed original indices and unchanged worker dependencies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "worker", "verify-prepared"])
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.index)
    elif args.mode == "run":
        run()
    else:
        verify_prepared()
