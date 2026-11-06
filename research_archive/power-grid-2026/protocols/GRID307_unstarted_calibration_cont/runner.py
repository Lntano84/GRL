"""Resume only the 16 never-started calibration CONT replays."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid


from bindings import (
    CAPS, GRID305, GRID305_BLOB, I, INDICES, LEGACY, OUT, R, SOURCE,
    assert_prior_unstarted, build, git, guard,
    module, read, save, sha, validate_event,
)

APPROVAL = SOURCE / "approval.json"
PLAN = SOURCE / "execution_order.json"
old_guard = guard
O = OUT


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
        raise TimeoutError("GRID307 time cap reached")


def verify_publication():
    approval = read(APPROVAL)
    execution = read(PLAN)
    if (
        approval["user_reply"] != "那继续补呗" or approval["grid305_scope_git_blob"] != GRID305_BLOB
        or approval["caps"] != CAPS or approval["runs"] != 16 or approval["no_retry"] is not True
        or [item["index"] for item in execution] != INDICES
    ):
        raise PermissionError("GRID307 approval or scope differs")
    head = git("rev-parse", "HEAD").decode().strip()
    if git("status", "--porcelain").strip():
        raise PermissionError("Integration worktree must be clean")
    if git("ls-remote", "lntano-src", "refs/heads/main").decode().split()[0] != head:
        raise PermissionError("Remote does not match local HEAD")
    for p in (APPROVAL, PLAN, SOURCE/"offline_review.json", *SOURCE.glob("*.py")):
        if git("show", f"{head}:{p.relative_to(I).as_posix()}") != p.read_bytes():
            raise PermissionError(f"Publication differs for {p.name}")
    return execution, head


def verify_inputs(execution):
    for name, expected in read(APPROVAL)["source_sha256"].items():
        if sha(R / name) != expected:
            raise PermissionError(f"Source changed: {name}")
    if read(GRID305) != json.loads(git("cat-file", "blob", GRID305_BLOB)):
        raise PermissionError("Approved GRID305 scope changed")
    old_guard.verify_frozen()
    for name, expected in read(old_guard.O / "recovery03_source_freeze.json").items():
        if sha(R / name) != expected:
            raise PermissionError(f"Prior runtime dependency changed: {name}")
    old_guard.verify_stage("baseline")
    old_guard.verify_generated(next(item["week"] for item in execution if item["year"] == 2038))
    guard37 = module("grid306_legacy_guard", LEGACY / "guard.py")
    guard37.verify_frozen()


def run_folder(root, item):
    return root / "runs" / f"r{item['repeat']}__{item['week']}__{item['method']}"


def ensure_unstarted(item):
    assert_prior_unstarted(item)
    index = item["index"]
    if (OUT / f"worker_result_{index:02d}.json").exists() or run_folder(OUT, item).exists():
        raise RuntimeError(f"Existing result or trajectory at index {index}")
    if any(OUT.glob(f"claims/replay_{index:03d}.*")):
        raise RuntimeError(f"Existing claim at index {index}")


def claim(kind, index):
    execution, _ = verify_publication()
    if kind != "replay" or index not in {item["index"] for item in execution}:
        raise PermissionError("Worker outside the approved sixteen")
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
    context = claim("replay", index)
    item = next(item for item in read(PLAN) if item["index"] == index)
    assert index == item["index"]
    year = item["year"]
    assert read(OUT / f"runtime_design_{year}.json")["execution_order"][index] == item
    sys.modules["guard"] = sys.modules[__name__]
    sys.modules["source_builder"] = build(year)
    sys.modules["paired_adapter"] = module("paired_adapter", SOURCE / "bridge.py")
    entry = str(R / "work/grid55/run.py")
    check(context)
    sys.argv = [entry, "--worker", str(index)]
    exec(compile(sys.modules["source_builder"].build(), entry, "exec"), {
        "__file__": entry, "__name__": "__main__", "run_ctx": context,
        "check_runtime_budget": lambda: check(context),
    })


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
    execution, head = verify_publication()
    boot_id = old_guard.boot_id()
    started = time.perf_counter()
    launch = {
        "start_monotonic": started, "deadline_monotonic": started + CAPS["wall_s"],
        "deadline_utc_epoch": time.time() + CAPS["wall_s"], "boot_id": boot_id,
        "authorization_sha256": sha(APPROVAL), "authorization_commit": head,
        "manager_pid": os.getpid(), "token": uuid.uuid4().hex,
    }
    with (OUT / "launch_context.json").open("x", encoding="utf8") as file:
        json.dump(launch, file)
    records = []
    steps = forecasts = 0
    current_index = None
    try:
        verify_inputs(execution)
        check(launch)
        for item in execution:
            check(launch)
            ensure_unstarted(item)
            current_index = item["index"]
            context = {
                **launch, "kind": "replay", "index": current_index,
                "worker_deadline_monotonic": time.perf_counter() + CAPS["episode_s"],
                "physical_remaining": CAPS["physical_steps"] - steps,
                "forecast_remaining": CAPS["public_forecasts"] - forecasts,
            }
            if min(context["physical_remaining"], context["forecast_remaining"]) <= 0:
                raise TimeoutError("GRID307 resource cap reached")
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
            validate_event(source["event"], item["target"])
            assert source["event"]["forced_continue"] and source["event"]["delivered"]
            assert summary["new_model_fits"] == source["new_cost_labels"] == 0
            assert summary["steps"] <= context["physical_remaining"]
            assert source["forecasts"] <= context["forecast_remaining"]
            steps += summary["steps"]
            forecasts += source["forecasts"]
            records.append({"index": current_index, "summary": summary, "event": source["event"],
                            "forecasts": source["forecasts"]})
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
        save(OUT / "pipeline_state.json", {"phase": "stopped", "new_accepted": len(records)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "worker"])
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.index)
    else:
        run()
