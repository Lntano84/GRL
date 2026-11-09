"""One explicitly authorized engineering continuation of GRID330 on its original clock."""
import argparse
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import time

A = Path(__file__).resolve().parent
BASE = A.parent / "GRID330_dc_mix_closed_loop"
spec = importlib.util.spec_from_file_location("_grid330_original_runtime", BASE / "runtime.py")
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
runner, R, I = original.runner, original.R, original.I
OLD = original.O
O = R / "outputs/grid330_dc_mix_recovery01"
ENTRY = A / "runtime.py"
SCOPE = "GRID330_RECOVERY01_ORIGINAL_CLOCK"
CAPS = dict(original.CAPS, public_forecasts=249000)
RUNS = original.REPLAYS
read, save, sha, git = original.read, original.save, original.sha, original.git
old_worker = runner.instrument_worker
old_launch = inspect.getsource(runner.run)
replace_once = original.replace_once


def assemble():
    return replace_once(original.assemble(), "OUT=ROOT/'outputs/grid330_dc_mix_closed_loop'",
                        "OUT=ROOT/'outputs/grid330_dc_mix_recovery01'")


def instrument_worker(text):
    return replace_once(old_worker(text), "    metadata = {**item,",
        "    write_json(path / 'startup_forecast_receipt.json', _intervention.record())\n    metadata = {**item,")


def anchor():
    source = read(A / "recovery_source.json")
    start = read(OLD / "launch_context.json")
    error = read(OLD / "collection_error.json")
    if start["boot_id"] != runner.prior.boot_id():
        raise PermissionError("GRID330 original boot changed")
    for name, expected in source["original_files"].items():
        if sha(OLD / name) != expected:
            raise PermissionError("GRID330 original stop record differs: " + name)
    if (error["accepted"] != 0 or error["steps_accepted"] != 0 or not error["no_retry"]
            or read(OLD / "active_worker.json")["active"] or (OLD / "manifest.json").exists()):
        raise PermissionError("GRID330 original attempt has accepted or active work")
    old_runs = OLD / "runs"
    if any(p.is_file() for p in old_runs.rglob("*")):
        raise PermissionError("GRID330 original run directory contains unexpected artifacts")
    now = time.perf_counter()
    if now < start["start_monotonic"] or now >= start["start_monotonic"] + CAPS["wall_s"] or (
            time.time() >= start["start_utc"] + CAPS["wall_s"]):
        raise PermissionError("GRID330 original deadline expired")
    return start, error


def receipt_fields():
    return {"scope_id": SCOPE, "runs": RUNS, "caps": CAPS,
            "design_sha256": sha(A / "design.json"), "source_sha256": sha(A / "source_freeze.json"),
            "preparation_sha256": sha(A / "offline_review.json"),
            "original_start_sha256": sha(OLD / "launch_context.json")}


def require_publication():
    receipt = A / "authorization.json"
    if not receipt.is_file():
        raise PermissionError("GRID330 recovery authorization absent before live imports")
    approval = read(receipt)
    if (approval.get("approved_by_user") is not True or any(
        not isinstance(approval.get(k), str) or not approval[k].strip()
        for k in ("user_message_reference", "user_instruction"))):
        raise PermissionError("Explicit GRID330 recovery approval required")
    for key, value in receipt_fields().items():
        if approval.get(key) != value:
            raise PermissionError("GRID330 recovery authorization differs: " + key)
    if not read(A / "offline_review.json")["passed"]:
        raise PermissionError("GRID330 recovery preparation incomplete")
    head = git("rev-parse", "HEAD").decode().strip()
    if git("status", "--porcelain").strip() or git("ls-remote", "lntano-src", "refs/heads/main").decode().split()[0] != head:
        raise PermissionError("GRID330 recovery publication barrier")
    for path in (receipt, ENTRY, A / "prepare.py", A / "design.json", A / "recovery_source.json",
                 A / "source_freeze.json", A / "offline_review.json"):
        original.tracked(path, head)
    design = read(A / "design.json")
    assert design["id"] == SCOPE and design["caps"] == CAPS and len(design["execution_order"]) == RUNS
    assert (O / "design.json").read_bytes() == (A / "design.json").read_bytes()
    anchor()
    return design, head


def manager_source():
    text = replace_once(old_launch, '    launch = {"token":',
                        '    previous, failed = recovery_anchor()\n    launch = {"token":')
    text = replace_once(text, '"start_monotonic":time.perf_counter(),"start_utc":time.time()',
                        '"start_monotonic":previous["start_monotonic"],"start_utc":previous["start_utc"]')
    return replace_once(text, 'time.perf_counter()+CAPS["episode_s"]',
                        'time.perf_counter()+CAPS["episode_s"]-(failed["wall_s"] if j == 0 else 0.)')


runner.A, runner.O, runner.OUT, runner.__file__ = A, O, O, str(ENTRY)
runner.CAPS, runner.TOKEN = CAPS, "GRL_GRID330_RECOVERY01_TOKEN"
runner.assemble, runner.instrument_worker = assemble, instrument_worker
runner.require_publication, runner.recovery_anchor = require_publication, anchor
exec(compile(manager_source(), str(ENTRY), "exec"), runner.__dict__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "worker"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    try:
        if args.mode == "run":
            assert args.index is None
            runner.run()
        else:
            assert 0 <= args.index < RUNS
            runner.worker(args.index)
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

