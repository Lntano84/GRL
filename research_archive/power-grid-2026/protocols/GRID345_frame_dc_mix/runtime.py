"""Four fresh heuristic replays with the frozen GRID330 global order."""
import argparse
import importlib.util
from pathlib import Path
import sys

A = Path(__file__).resolve().parent
P = A.parent
spec = importlib.util.spec_from_file_location(
    "_grid345_parent", P / "GRID330_dc_mix_closed_loop/runtime.py")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)

runner, R, I = parent.runner, parent.R, parent.I
O = R / "outputs/grid345_frame_dc_mix"
SCOPE = "GRID345_FRAME_DC_MIX_FOUR_WEEKS"
ARM, REFERENCE = "FRAME_GUARD_DC", "BOUNDED261_GUARD_DC"
CAPS = {"wall_s": 1800, "episode_s": 1200, "physical_steps": 8068, "public_forecasts": 100000}
RUNS = 4
read, save, sha, git = parent.read, parent.save, parent.sha, parent.git
parent.ARMS[ARM] = ("FRAME_GUARD", False, True)


def assemble():
    return parent.replace_once(parent.assemble(), "OUT=ROOT/'outputs/grid330_dc_mix_closed_loop'",
                               "OUT=ROOT/'outputs/grid345_frame_dc_mix'")


def receipt_fields():
    return dict(scope_id=SCOPE, runs=RUNS, caps=CAPS,
                **{key: sha(A / name) for key, name in (
                    ("design_sha256", "design.json"),
                    ("source_sha256", "source_freeze.json"),
                    ("preparation_sha256", "offline_review.json"))})


def require_publication():
    receipt = A / "authorization.json"
    if not receipt.is_file():
        raise PermissionError("GRID345 authorization absent before live imports")
    approval = read(receipt)
    if (approval.get("approved_by_user") is not True
            or not approval.get("user_instruction") or not approval.get("user_message_reference")
            or any(approval.get(k) != v for k, v in receipt_fields().items())):
        raise PermissionError("GRID345 authorization scope differs")
    if read(A / "offline_review.json")["passed"] is not True:
        raise PermissionError("GRID345 preparation incomplete")
    head = git("rev-parse", "HEAD").decode().strip()
    remote = git("ls-remote", "lntano-src", "refs/heads/main").decode().split()
    if remote != [head, "refs/heads/main"] or git("status", "--porcelain").strip():
        raise PermissionError("GRID345 publication barrier")
    for path in (*A.glob("*.py"), receipt, A / "design.json",
                 A / "source_freeze.json", A / "offline_review.json"):
        parent.tracked(path, head)
    design = read(A / "design.json")
    assert design["id"] == SCOPE and design["caps"] == CAPS and len(design["execution_order"]) == RUNS
    assert (O / "design.json").read_bytes() == (A / "design.json").read_bytes()
    assert sha(O / "code_freeze.json") == read(A / "source_freeze.json")[
        (O / "code_freeze.json").relative_to(R).as_posix()]
    return design, head


runner.A, runner.O, runner.OUT, runner.__file__ = A, O, O, __file__
runner.CAPS, runner.TOKEN = CAPS, "GRL_GRID345_TOKEN"
runner.assemble, runner.require_publication = assemble, require_publication

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
            assert args.index is not None and 0 <= args.index < RUNS
            runner.worker(args.index)
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)