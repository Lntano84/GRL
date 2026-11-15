"""Single March capture on the original GRID334 protected 261 runner."""
import argparse
import importlib.util
import sys
from pathlib import Path

A = Path(__file__).resolve().parent
PREVIOUS = A.parent / "GRID334_logeipc_closed_loop"
SCOPE = "GRID339_MARCH261_UNSAFE_CAPTURE"
CAPS = dict(wall_s=600, episode_s=600, physical_steps=2017, public_forecasts=12000)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

rt = module("_grid339_grid334", PREVIOUS / "runtime.py")
runner, R, I = rt.runner, rt.R, rt.I
O = R / "outputs/grid339_march261_unsafe_capture"
read, save, sha, git = rt.read, rt.save, rt.sha, rt.git
old_assemble = rt.assemble

def assemble():
    text = rt.parent.replace_once(old_assemble(), "OUT=ROOT/'outputs/grid334_logeipc_closed_loop'",
                                  "OUT=ROOT/'outputs/grid339_march261_unsafe_capture'")
    marker = "    for name,fn in originals.items():"
    # Only the installation loop uses a colon followed by a newline.
    return rt.parent.replace_once(text, marker + "\n",
        "    import unsafe_capture\n"
        "    unsafe_capture.install(OptimModule, originals, OUT/'snapshots', "
        "lambda: (step, in_control, phase), solvers)\n" + marker + "\n")

def bindings():
    return dict(scope_id=SCOPE, runs=1, caps=CAPS, design_sha256=sha(A / "design.json"),
                source_sha256=sha(A / "source_freeze.json"),
                preparation_sha256=sha(A / "offline_review.json"))

def require_publication():
    path = A / "authorization.json"
    approval = read(path)
    if approval.get("user_instruction") != "那你做啊，我持续推进就是说有想法就做":
        raise PermissionError("Current direct user authorization missing")
    if not approval.get("user_message_reference") or any(approval.get(k) != v for k, v in bindings().items()):
        raise PermissionError("GRID339 receipt scope mismatch")
    head = git("rev-parse", "HEAD").decode().strip()
    if git("status", "--porcelain").strip() or git("ls-remote", "lntano-src", "refs/heads/main").decode().split() != [head, "refs/heads/main"]:
        raise PermissionError("GRID339 publication barrier")
    for path in (*A.glob("*.py"), path, A / "design.json", A / "source_freeze.json", A / "offline_review.json"):
        rt.parent.tracked(path, head)
    if not read(A / "offline_review.json")["passed"]:
        raise PermissionError("GRID339 preparation not passed")
    design = read(A / "design.json")
    assert design["id"] == SCOPE and design["caps"] == CAPS and len(design["execution_order"]) == 1
    assert (O / "design.json").read_bytes() == (A / "design.json").read_bytes()
    return design, head

def configure():
    # The prior guarded worker binds these aliases in its own module.
    sys.path.insert(0, str(A))
    runner.A, runner.O, runner.OUT, runner.__file__ = A, O, O, str(A / "runtime.py")
    runner.CAPS, runner.TOKEN = CAPS, "GRL_GRID339_TOKEN"
    runner.assemble, runner.require_publication = assemble, require_publication

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "worker"))
    parser.add_argument("index", type=int, nargs="?")
    args = parser.parse_args()
    configure()
    if args.mode == "run":
        assert args.index is None
        runner.run()
    else:
        assert args.index == 0
        runner.worker(0)
