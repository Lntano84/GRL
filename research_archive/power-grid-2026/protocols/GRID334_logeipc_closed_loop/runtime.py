"""GRID334 fresh paired closed-loop test with the frozen GRID332 acquisition."""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import importlib.util

A = Path(__file__).resolve().parent
P = A.parent
spec = importlib.util.spec_from_file_location("_grid334_parent", P / "GRID330_dc_mix_closed_loop/runtime.py")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
runner, R, I = parent.runner, parent.R, parent.I
read, save, sha, git = parent.read, parent.save, parent.sha, parent.git
O = R / "outputs/grid334_logeipc_closed_loop"
BO = P / "GRID332_independent_logeipc"
REFERENCE = "BOUNDED261_GUARD_DC"
COMPARATOR = "LOGEIPC_DC"
ARMS = (REFERENCE, COMPARATOR)
CAPS = {"wall_s": 1800, "episode_s": 1200, "physical_steps": 16136, "public_forecasts": 100000}
SCOPE = "GRID334_LOGEIPC_CLOSED_LOOP_8"
original_install = runner.install


def assemble():
    text = parent.replace_once(parent.assemble(), "OUT=ROOT/'outputs/grid330_dc_mix_closed_loop'",
                               "OUT=ROOT/'outputs/grid334_logeipc_closed_loop'")
    return parent.replace_once(text, "if item['arm']=='FRAME_GUARD' else models297.FrozenModels",
                               "if item['arm'] in ('FRAME_GUARD','LOGEIPC') else models297.FrozenModels")


def policy_class():
    source = runner.module("_grid334_acquisition", BO / "readout.py")
    native = runner.module("_grid334_global", R / "work/grid91_mechanism/policies.py")

    class IndependentPolicy:
        def __init__(self, library, deadline):
            self.deadline = deadline
            with source.np.load(source.LIBRARY, allow_pickle=False) as data:
                vectors = data["action_space"]
            actual = [source.sha(v.tobytes()) for v in vectors]
            assert vectors.shape == (909, 1605) and vectors.dtype == source.np.float32
            assert source.np.isfinite(vectors).all() and list(library) == actual
            self.vectors = vectors[:, source.np.any(vectors != vectors[0], axis=0)].astype(source.np.float64)
            self.fee = read(BO / "fixed_scope.json")["cost_per_query"]

        def run(self, order, reveal, subids, aliases, limit=None, provider=None, context=None):
            result = native.run(order, reveal, "NO_LOCAL", subids, aliases, limit=limit)
            rows = result["outcomes"]
            pending = result["completion_candidate_ids"]
            winner = source.best(rows)
            selection_s, log_ratio, trace = 0.0, None, []
            if pending:
                floor = float(source.np.float32(2 - limit))
                local, forecast_s = [], 0.0

                def observe(cid):
                    nonlocal forecast_s
                    row = reveal(cid, "local_completion")
                    local.append(row)
                    forecast_s += row["forecast_s"]
                    return row

                winner, trace, reason, log_ratio, elapsed = source.acquire(
                    self.vectors, rows, pending, floor, self.fee, observe, self.deadline)
                assert reason in ("ei_not_above_fee", "pool_exhausted")
                assert [row["pool_id"] for row in local] == [row["id"] for row in trace]
                selection_s = max(0.0, elapsed - forecast_s)
                rows.extend(local)
            else:
                reason = "no_pending"
            chosen = None if winner is None else winner["pool_id"]
            result.update(chosen_pool_id=chosen, public_queries=len(rows), local_queries=len(trace),
                          extra_selected_ids=[record["id"] for record in trace],
                          stop_reason=result["stop_reason"] + "_" + reason,
                          learned_decisions=[], local_feature_s=0.0, local_inference_s=selection_s,
                          learning_method="LOGEIPC", engineering_fallback=None,
                          logeipc=dict(reason=reason, trace=trace, final_log_ei_over_fee=log_ratio))
            return result

    return IndependentPolicy


def adapter_source():
    template = dict(evaluation_arm=REFERENCE, arm="BOUNDED261", load_guard=True, dc_mix=True)
    text = parent.adapter_source(template)
    marker = next(line for line in text.splitlines() if "policy=N.CompletionPolicy(" in line)
    prefix = marker.split(";P=", 1)[0]
    assert prefix.startswith(" N=load(")
    return parent.replace_once(text, marker, prefix + ";policy=models")


def install_adapter(search, item, policy, loader=None):
    namespace = {}
    exec(compile(adapter_source(), "grid334_adapter", "exec"), namespace)
    if loader is not None:
        namespace["load"] = loader
    namespace["install"](search, item, policy)


def acquisition_deadline(context):
    replay_deadline = min(context["worker_deadline"], context["start_monotonic"] + CAPS["wall_s"])
    return time.monotonic() + (replay_deadline - time.perf_counter())


def install(search, item, payload, expected=None):
    if item["evaluation_arm"] == REFERENCE:
        return original_install(search, item, payload, expected)
    assert item["evaluation_arm"] == COMPARATOR and item["arm"] == "LOGEIPC"
    assert payload is None and expected is None
    context = read(O / "run_context.json")
    # GRID332 checks monotonic, while the replay runner anchors perf_counter.
    deadline = acquisition_deadline(context)
    # Initialize the public action embedding when the actual Search has constructed
    # its library, outside the measured online decision interval.
    impl = policy_class()
    original_init = search.__init__
    from common import action_vector, digest

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._grid334_policy = impl((digest(action_vector(a)) for a in self.library), deadline)

    class PerSearchPolicy:
        def run(self, *args, **kwargs):
            return self.bound.run(*args, **kwargs)

    proxy = PerSearchPolicy()
    search.__init__ = initialize
    install_adapter(search, item, proxy)
    choose = search.choose

    def bound_choose(self, *args, **kwargs):
        proxy.bound = self._grid334_policy
        return choose(self, *args, **kwargs)

    search.choose = bound_choose

    def accept(row):
        counter.steps += 1

    def record():
        return {"steps": counter.steps, "forecasts": counter.forecasts, "new_model_fits": 0}

    counter = SimpleNamespace(steps=0, forecasts=0, accept=accept, record=record)
    return counter


def receipt_fields():
    return dict(scope_id=SCOPE, runs=8, caps=CAPS,
                **{name: sha(A / filename) for name, filename in (
                    ("design_sha256", "design.json"), ("source_sha256", "source_freeze.json"),
                    ("preparation_sha256", "offline_review.json"))})


def require_publication():
    receipt = A / "authorization.json"
    if not receipt.is_file():
        raise PermissionError("GRID334 scope confirmation absent before live imports")
    approval = read(receipt)
    if (approval.get("approved_by_user") is not True or not approval.get("user_instruction")
        or not approval.get("user_message_reference")
        or any(approval.get(k) != v for k, v in receipt_fields().items())):
        raise PermissionError("GRID334 authorization scope differs")
    if not read(A / "offline_review.json")["passed"]:
        raise PermissionError("GRID334 offline review incomplete")
    head = git("rev-parse", "HEAD").decode().strip()
    remote = git("ls-remote", "lntano-src", "refs/heads/main").decode().split()
    if remote != [head, "refs/heads/main"] or git("status", "--porcelain").strip():
        raise PermissionError("GRID334 publication barrier")
    for path in (*A.glob("*.py"), receipt, A / "design.json", A / "source_freeze.json", A / "offline_review.json"):
        parent.tracked(path, head)
    design = read(A / "design.json")
    assert design["id"] == SCOPE and design["caps"] == CAPS and len(design["execution_order"]) == 8
    for name in ("design.json", "model_references.json", "code_freeze.json"):
        relative = (O / name).relative_to(R).as_posix()
        assert sha(R / relative) == read(A / "source_freeze.json")[relative]
    return design, head


runner.A, runner.O, runner.OUT, runner.__file__ = A, O, O, str(A / "runtime.py")
runner.CAPS, runner.TOKEN = CAPS, "GRL_GRID334_TOKEN"
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
            assert args.index is not None and 0 <= args.index < 8
            runner.worker(args.index)
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

