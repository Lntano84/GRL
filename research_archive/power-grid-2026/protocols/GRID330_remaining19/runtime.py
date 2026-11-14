"""Resume the 19 GRID330 entries without saved physical work."""
import argparse
import importlib.util
from pathlib import Path
import sys

A = Path(__file__).resolve().parent
PARENT = A.parent / "GRID330_dc_mix_closed_loop"
spec = importlib.util.spec_from_file_location("_grid330_remaining_parent", PARENT / "runtime.py")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
runner, R, I = parent.runner, parent.R, parent.I
O = R / "outputs/grid330_remaining19"
SCOPE = "GRID330_REMAINING19"
CAPS = {"wall_s": 3600, "episode_s": 1200, "physical_steps": 38323, "public_forecasts": 241916}
RUNS = 19
read, save, sha, git = parent.read, parent.save, parent.sha, parent.git


def assemble():
    return parent.replace_once(parent.assemble(), "OUT=ROOT/'outputs/grid330_dc_mix_closed_loop'",
                               "OUT=ROOT/'outputs/grid330_remaining19'")


def receipt_fields():
    return {"scope_id": SCOPE, "runs": RUNS, "caps": CAPS,
            **{name: sha(A / file) for name, file in (
                ("design_sha256", "design.json"), ("source_sha256", "source_freeze.json"),
                ("preparation_sha256", "offline_review.json"))}}


def check_local():
    for path, expected in read(A / "source_freeze.json").items():
        if sha(R / path) != expected:
            raise ValueError(f"GRID330 input differs: {path}")
    return read(A / "design.json")


def require_publication():
    receipt = A / "authorization.json"
    if not receipt.is_file():
        raise PermissionError("GRID330 remaining authorization absent before live imports")
    approval = read(receipt)
    if (approval.get("approved_by_user") is not True or not approval.get("user_instruction")
        or not approval.get("user_message_reference")
        or any(approval.get(k) != v for k, v in receipt_fields().items())):
        raise PermissionError("GRID330 remaining receipt scope differs")
    if not read(A / "offline_review.json")["passed"]:
        raise PermissionError("GRID330 offline review incomplete")
    head = git("rev-parse", "HEAD").decode().strip()
    remote = git("ls-remote", "lntano-src", "refs/heads/main").decode().split()
    if not remote or remote[0] != head or git("status", "--porcelain").strip():
        raise PermissionError("GRID330 publication barrier")
    for path in (*A.glob("*.py"), receipt, A / "design.json", A / "source_freeze.json",
                 A / "offline_review.json", A / "historical_receipt.json"):
        parent.tracked(path, head)
    design = read(A / "design.json")
    assert design["id"] == SCOPE and design["caps"] == CAPS and len(design["execution_order"]) == RUNS
    for name in ("design.json", "model_references.json", "code_freeze.json"):
        rel = (O / name).relative_to(R).as_posix()
        assert sha(R / rel) == read(A / "source_freeze.json")[rel]
    return design, head


runner.A, runner.O, runner.OUT, runner.__file__ = A, O, O, str(A / "runtime.py")
runner.CAPS, runner.TOKEN = CAPS, "GRL_GRID330_REMAINING19_TOKEN"
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

