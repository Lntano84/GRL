"""GRID330 guarded runner for 20 proposed closed-loop replays."""
import argparse
import importlib.util
from pathlib import Path
import sys

A = Path(__file__).resolve().parent
PREV = A.parent / "GRID320_load_guard_closed_loop"
spec = importlib.util.spec_from_file_location("_grid330_guarded_parent", PREV / "runtime.py")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
runner = parent.runner
R, I, BASE = parent.R, parent.I, parent.BASE
ENTRY = A / "runtime.py"
O = R / "outputs/grid330_dc_mix_closed_loop"
SCOPE = "GRID330_DC_MIX_GUARDED_20"
CAPS = {"wall_s": 3600, "episode_s": 1200, "physical_steps": 40340, "public_forecasts": 250000}
ARMS = {
    "FRAME_GUARD": ("FRAME_GUARD", False, False),
    "PARENT208_GUARD": ("PARENT208", True, False),
    "BOUNDED261_GUARD": ("BOUNDED261", True, False),
    "PARENT208_GUARD_DC": ("PARENT208", True, True),
    "BOUNDED261_GUARD_DC": ("BOUNDED261", True, True),
}
REPLAYS = 20
read, save, sha, git = runner.read, runner.save, runner.sha, runner.git
tracked, replace_once = parent.tracked, parent.replace_once
old_install = runner.install
old_instrument = runner.instrument_worker


def assemble():
    return replace_once(parent.assemble(), "OUT=ROOT/'outputs/grid320_load_guard_closed_loop'",
                        "OUT=ROOT/'outputs/grid330_dc_mix_closed_loop'")


def adapter_source(item):
    if ARMS[item["evaluation_arm"]] != (item["arm"], item["load_guard"], item["dc_mix"]):
        raise ValueError("GRID330 arm binding differs")
    text = parent.adapter_source() if item["load_guard"] else parent.old_adapter_source()
    if not item["dc_mix"]:
        return text
    text = replace_once(text, " from common import flags,graph_features",
        " from common import flags,graph_features\n"
        " ranker=load('grid330_live_rank','" + (A / "ordering.py").relative_to(R).as_posix() + "').LiveRank()")
    old = "start=time.perf_counter();eligible,reps,aliases,actions,changed=self.prepare(obs,base);prepared=time.perf_counter();order=sorted(reps);limit="
    new = ("start=time.perf_counter();eligible,reps,aliases,actions,changed,first=ranker.prepare(self,obs,base);"
           "prepared=time.perf_counter();order,dc=ranker.apply(self,obs,base,sorted(reps),first);del first;limit=")
    text = replace_once(text, old, new)
    text = replace_once(text, "order_mode='STATIC_ID'", "order_mode='STATIC64_DC64',dc_mix_receipt=dc")
    return replace_once(text, "forward_rank_s=0.", "forward_rank_s=dc['rank_s']")


def install(search, item, payload, expected=None):
    adapter = runner.adapter_source
    runner.adapter_source = lambda: adapter_source(item)
    try:
        return old_install(search, item, payload, expected)
    finally:
        runner.adapter_source = adapter


def instrument_worker(text):
    text = old_instrument(text)
    old = '            "n1_prepare_s": sum(c["prepare_s"] for c in calls),'
    new = '''            "dc_export_s": sum(c.get("dc_mix_receipt", {}).get("export_s", 0.) for c in calls),
            "dc_rank_s": sum(c.get("dc_mix_receipt", {}).get("rank_s", 0.) for c in calls),
            "dc_ranked_calls": sum(c.get("dc_mix_receipt", {}).get("status")=="ranked" for c in calls),
            "dc_bypassed_calls": sum(c.get("dc_mix_receipt", {}).get("status")=="bypass" for c in calls),
            "local_model_inference_s": sum(c.get("local_inference_s", 0.) for c in calls),
'''
    return replace_once(text, old, new + old)


def receipt_fields():
    return {"scope_id": SCOPE, "runs": REPLAYS, "caps": CAPS,
            "design_sha256": sha(A / "design.json"), "source_sha256": sha(A / "source_freeze.json"),
            "preparation_sha256": sha(A / "offline_review.json")}


def validate(receipt):
    if receipt.get("approved_by_user") is not True or any(
            not isinstance(receipt.get(key), str) or not receipt[key].strip()
            for key in ("user_message_reference", "user_instruction")):
        raise PermissionError("Explicit GRID330 scope approval required")
    for key, value in receipt_fields().items():
        if receipt.get(key) != value:
            raise PermissionError("GRID330 authorized scope differs: " + key)


def require_publication():
    authorization = A / "authorization.json"
    if not authorization.is_file():
        raise PermissionError("GRID330 authorization absent before live imports")
    validate(read(authorization))
    review = A / "offline_review.json"
    if not read(review)["passed"]:
        raise PermissionError("GRID330 offline review incomplete")
    head = git("rev-parse", "HEAD").decode().strip()
    remote = git("ls-remote", "lntano-src", "refs/heads/main").decode().split()
    if git("status", "--porcelain").strip() or not remote or remote[0] != head:
        raise PermissionError("GRID330 publication barrier")
    for path in (authorization, review, ENTRY, A / "ordering.py", A / "prepare.py",
                 A / "design.json", A / "source_freeze.json"):
        tracked(path, head)
    design = read(A / "design.json")
    assert design["id"] == SCOPE and design["caps"] == CAPS and len(design["execution_order"]) == REPLAYS
    assert (O / "design.json").read_bytes() == (A / "design.json").read_bytes()
    return design, head


runner.A, runner.O, runner.OUT, runner.__file__ = A, O, O, str(ENTRY)
runner.CAPS, runner.TOKEN = CAPS, "GRL_GRID330_TOKEN"
runner.assemble, runner.install = assemble, install
runner.instrument_worker, runner.require_publication = instrument_worker, require_publication

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
            assert 0 <= args.index < REPLAYS
            runner.worker(args.index)
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

