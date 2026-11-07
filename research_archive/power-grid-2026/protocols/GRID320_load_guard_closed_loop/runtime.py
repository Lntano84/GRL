"""New guarded arms on the unchanged GRID312 replay runner."""
import argparse
import importlib.util
from pathlib import Path
import sys

A = Path(__file__).resolve().parent
PREVIOUS = A.parent / "GRID312_frame_208_261_pilot"
ENTRY = A / "runtime.py"
spec = importlib.util.spec_from_file_location("_grid320_runner", PREVIOUS / "runtime.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

R, I, BASE, OLD = runner.R, runner.I, runner.BASE, runner.OLD
O = R / "outputs/grid320_load_guard_closed_loop"
CAPS = dict(runner.CAPS)
SCOPE = "GRID320_FIXED_09_LOAD_GUARD_CLOSED_LOOP"
CUTOFF = 0.9
REPLAYS = 24
read, save, sha, git = runner.read, runner.save, runner.sha, runner.git
replace_once = runner.builder.replace_once
old_assemble, old_adapter_source = runner.assemble, runner.adapter_source


def assemble():
    return replace_once(old_assemble(), "OUT=ROOT/'outputs/grid312_frame_208_261_pilot'",
                        "OUT=ROOT/'outputs/grid320_load_guard_closed_loop'")


def policy_source():
    policy = (OLD / "policy.py").read_text(encoding="utf8")
    rule = """   if not action:
    if not extension or not inc['strict_admissible'] or not inc['rho'] > 0.9:break
    ids=pending.copy()
    result['load_guard']=dict(threshold=0.9,incumbent=inc['pool_id'],rho=inc['rho'],paid_before=len(rows),extra_ids=ids.copy())"""
    policy = replace_once(policy, "   if not action:break", rule)
    return policy


def adapter_source():
    old = "P=load('g291_completion','work/grid291_control_preparation/policy.py')"
    new = ("P=type('GuardedPolicyNamespace', (), {})();"
           "exec(compile(" + repr(policy_source()) + ",'grid320_policy.py','exec'),P.__dict__)")
    return replace_once(old_adapter_source(), old, new)


def tracked(path, head):
    if git("show", head + ":" + path.relative_to(I).as_posix()) != path.read_bytes():
        raise PermissionError("Published bytes differ: " + path.name)


def receipt_fields():
    return {"scope_id": SCOPE, "runs": REPLAYS, "caps": CAPS,
            "design_sha256": sha(A / "design.json"),
            "source_sha256": sha(A / "source_freeze.json"),
            "preparation_sha256": sha(A / "offline_review.json")}


def validate_receipt(receipt):
    if (receipt.get("approved_by_user") is not True
            or not isinstance(receipt.get("user_message_reference"), str)
            or not receipt["user_message_reference"].strip()
            or not isinstance(receipt.get("user_instruction"), str)
            or not receipt["user_instruction"].strip()):
        raise PermissionError("Explicit GRID320 scope approval required")
    for k, expected in receipt_fields().items():
        if receipt.get(k) != expected:
            raise PermissionError("GRID320 scope or budget differs: " + k)


def require_publication():
    authorization = A / "authorization.json"
    if not authorization.is_file():
        raise PermissionError("GRID320 authorization absent before live imports")
    validate_receipt(read(authorization))
    if not read(A / "offline_review.json")["passed"]:
        raise PermissionError("GRID320 preparation incomplete")
    head = git("rev-parse", "HEAD").decode().strip()
    if git("status", "--porcelain").strip() or git("ls-remote", "lntano-src", "refs/heads/main").decode().split()[0] != head:
        raise PermissionError("GRID320 publication barrier")
    for path in (authorization, A / "design.json", A / "source_freeze.json",
                 A / "offline_review.json", ENTRY, A / "prepare.py"):
        tracked(path, head)
    design = read(A / "design.json")
    assert design["id"] == SCOPE and design["caps"] == CAPS and len(design["execution_order"]) == REPLAYS
    assert (O / "design.json").read_bytes() == (A / "design.json").read_bytes()
    return design, head


def load_guard_receipt(call):
    """Reconcile a saved guard event with the original decision and paid queries."""
    decisions = call["learned_decisions"]
    last = decisions[-1] if decisions else None
    event = call.get("load_guard")
    if last is None or not last["optional_extension"] or last["action"]:
        assert event is None
        return 0
    paid = last["paid_queries"]
    outcomes = call["outcomes"]
    import struct
    incumbent = max((r for r in outcomes[:paid] if r["strict_admissible"]),
                    key=lambda r: (struct.unpack("f", struct.pack("f", r["reward"]))[0], -r["pool_id"]))
    if not incumbent["rho"] > CUTOFF:
        assert event is None and paid == len(outcomes)
        return 0
    seen = {r["pool_id"] for r in outcomes[:paid]}
    tail = [i for i in call["completion_candidate_ids"] if i not in seen]
    assert len(tail) == last["remaining"] > 0
    assert [r["pool_id"] for r in outcomes[paid:]] == tail
    assert all(r["phase"] == "local_completion" for r in outcomes[paid:])
    assert event == dict(threshold=CUTOFF, incumbent=incumbent["pool_id"], rho=incumbent["rho"],
                         paid_before=paid, extra_ids=tail)
    return len(tail)


runner.A, runner.O, runner.OUT = A, O, O
runner.__file__ = str(ENTRY)
runner.TOKEN = "GRL_GRID320_TOKEN"
runner.assemble, runner.adapter_source, runner.require_publication = assemble, adapter_source, require_publication

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
            runner.worker(args.index)
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
