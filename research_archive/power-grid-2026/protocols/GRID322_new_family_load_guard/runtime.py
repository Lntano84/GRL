"""GRID322 on the existing capped serial runner with fixed policy routing."""
import argparse
import importlib.util
from pathlib import Path
import sys

import family_review as plan

R, I, A, O, PREV = plan.R, plan.I, plan.A, plan.O, plan.PREV
ENTRY = A / "runtime.py"
spec = importlib.util.spec_from_file_location("_grid322_guarded_parent", PREV / "runtime.py")
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
runner = previous.runner
original_install = runner.install
read, sha, git, tracked = plan.read, plan.sha, plan.git, plan.tracked


def assemble():
    return previous.replace_once(previous.assemble(), "OUT=ROOT/'outputs/grid320_load_guard_closed_loop'",
                                 "OUT=ROOT/'outputs/grid322_new_family_load_guard'")


def adapter_for(item):
    expected = {
        "FRAME_GUARD": ("FRAME_GUARD", False),
        "PARENT208": ("PARENT208", False),
        "PARENT208_GUARD": ("PARENT208", True),
        "BOUNDED261_GUARD": ("BOUNDED261", True),
    }
    if expected[item["evaluation_arm"]] != (item["arm"], item["load_guard"]):
        raise ValueError("GRID322 arm binding differs")
    return previous.adapter_source if item["load_guard"] else previous.old_adapter_source


def install(search, item, payload, expected=None):
    adapter = runner.adapter_source
    runner.adapter_source = adapter_for(item)
    try:
        return original_install(search, item, payload, expected)
    finally:
        runner.adapter_source = adapter


def receipt_fields():
    return {"scope_id": plan.SCOPE, "runs": 16, "caps": plan.CAPS,
            "design_sha256": sha(A / "design.json"), "source_sha256": sha(A / "source_freeze.json"),
            "preparation_sha256": sha(A / "offline_review.json")}


def validate_receipt(receipt):
    if (receipt.get("approved_by_user") is not True
            or not str(receipt.get("user_message_reference", "")).strip()
            or not str(receipt.get("user_instruction", "")).strip()):
        raise PermissionError("Explicit GRID322 scope approval required")
    for key, value in receipt_fields().items():
        if receipt.get(key) != value:
            raise PermissionError("GRID322 scope or cap differs: " + key)


def require_publication():
    approval = A / "authorization.json"
    if not approval.is_file():
        raise PermissionError("GRID322 authorization absent before live imports")
    validate_receipt(read(approval))
    head = git("rev-parse", "HEAD").decode().strip()
    if git("status", "--porcelain").strip() or git("ls-remote", "lntano-src", "refs/heads/main").decode().split()[0] != head:
        raise PermissionError("GRID322 publication barrier")
    for path in (approval, A / "design.json", A / "source_freeze.json",
                 A / "offline_review.json", ENTRY, A / "prepare.py", A / "family_review.py"):
        tracked(path, head)
    design = read(A / "design.json")
    assert design["id"] == plan.SCOPE and design["caps"] == plan.CAPS and len(design["execution_order"]) == 16
    assert (A / "design.json").read_bytes() == (O / "design.json").read_bytes()
    assert read(A / "offline_review.json")["passed"]
    assert read(O / "family_review.json")["completed_week_has_trigger"]
    env_dates = {p.name for p in (R / design["environment"] / "chronics").iterdir() if p.is_dir()}
    assert env_dates == {r["scenario"] for r in design["calendar"]}
    return design, head


runner.A, runner.O, runner.OUT = A, O, O
runner.__file__ = str(ENTRY)
runner.CAPS, runner.TOKEN = plan.CAPS, "GRL_GRID322_TOKEN"
runner.assemble, runner.install, runner.require_publication = assemble, install, require_publication

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
            assert 0 <= args.index < 16
            runner.worker(args.index)
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
