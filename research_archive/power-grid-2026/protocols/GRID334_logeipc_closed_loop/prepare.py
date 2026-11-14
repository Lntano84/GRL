"""Freeze the two-arm replay and qualify its adapter without live simulation."""
import argparse
import ast
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import runtime as rt

PARENT = rt.P / "GRID330_remaining19"
PREVIOUS_OUTPUT = rt.R / "outputs/grid330_remaining19"


def refs(paths):
    return {path.relative_to(rt.R).as_posix(): rt.sha(path) for path in paths}


def freeze():
    previous = rt.read(PARENT / "design.json")
    finished = rt.read(PARENT / "results/finished.json")
    assert finished["accepted"] == 19 and len(previous["calendar"]) == 4
    templates = {item["week"]: item for item in previous["execution_order"]
                 if item["evaluation_arm"] == rt.REFERENCE}
    items = []
    for date_index, day in enumerate(previous["calendar"]):
        template = templates[day["scenario"]].copy()
        del template["original_index"]
        roles = rt.ARMS if date_index % 2 == 0 else rt.ARMS[::-1]
        for role in roles:
            independent = role == rt.COMPARATOR
            arm = "LOGEIPC" if independent else "BOUNDED261"
            items.append(dict(template, index=len(items), arm=arm, evaluation_arm=role,
                              method=role + ("__f0_s0" if not independent else ""),
                              completion_mode=arm, load_guard=not independent,
                              pool=None if independent else template["pool"],
                              model_seed=None if independent else template["model_seed"]))
    assert len(items) == 8 and all(row["within_pair"] == row["repeat"] == 0 for row in items)
    rt.save(rt.A / "design.json", dict(
        id=rt.SCOPE, environment=previous["environment"], environment_seed=0,
        calendar=previous["calendar"], caps=rt.CAPS, execution_order=items, pool="f0_s0",
        comparisons={"primary": [rt.COMPARATOR, rt.REFERENCE]}, learned_models_trained=0,
        timing="Fresh serial paired runs; alternate first arm by date. Model and command embedding setup outside controller decision time, inside total wall cap. No historical timing merged.",
        protocol="Same GRID330 mixed global shortlist and original locked-station local pool. Protected 261 unchanged. Independent arm only uses unchanged GRID332 model, cost, acquisition and stopping on the local pool. No local 0.9 guard in independent arm.",
        endpoint="Completion and safety reported first. Complete costs only where both complete. Candidate/forecast counts, control sum, all-step and search-step P95/P99 compared only at matched horizons, early termination separately. Record first delivered action divergence.",
        resource={"historical_mean_s": finished["wall_s"]/19, "historical_projection_8_s": finished["wall_s"]*8/19,
                  "includes_new_overhead": False, "no_retry": True},
    ))
    rt.O.mkdir(exist_ok=False)
    (rt.O / "design.json").write_bytes((rt.A / "design.json").read_bytes())
    original = rt.read(PARENT / "source_freeze.json")
    model_path = PREVIOUS_OUTPUT / "model_references.json"
    assert original[model_path.relative_to(rt.R).as_posix()] == rt.sha(model_path)
    (rt.O / "model_references.json").write_bytes(model_path.read_bytes())
    model, = rt.read(model_path)
    assert model["pool"] == "f0_s0" and len(model["hashes"]) == 7
    for path, expected in model["hashes"].items():
        assert original[path] == expected
    sources = {path: expected for path, expected in original.items()
               if not path.startswith(("outputs/grid330_remaining19/", "outputs/grid330_dc_mix_recovery01/"))}
    bo_scope = rt.read(rt.BO / "fixed_scope.json")
    assert bo_scope["source_sha256"] == rt.sha(rt.BO / "readout.py")
    references = refs([*rt.A.glob("*.py"), rt.A / "design.json", rt.O / "design.json",
                       rt.O / "model_references.json", PARENT / "source_freeze.json",
                       PARENT / "results/finished.json", rt.BO / "readout.py",
                       rt.BO / "fixed_scope.json"])
    sources.update({key: value for key, value in bo_scope["references"].items()
                    if key.endswith("action_N1_unsafe.npz")})
    sources.update(references)
    verify(sources)
    rt.save(rt.O / "code_freeze.json", sources)
    rt.save(rt.A / "source_freeze.json", {**sources, **refs([rt.O / "code_freeze.json"])})


def verify(sources):
    for path, expected in sources.items():
        assert rt.sha(rt.R / path) == expected, path


def literal_keys(tree, name):
    return {node.slice.value for node in ast.walk(tree) if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name) and node.value.id == name
            and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)}


def qualify():
    source = rt.read(rt.A / "source_freeze.json")
    verify(source)
    design = rt.read(rt.A / "design.json")
    worker = rt.runner.instrument_worker((rt.R / "work/grid55/worker.py").read_text(encoding="utf8"))
    launch = rt.assemble()
    entry, = [node for node in ast.parse(worker).body if isinstance(node, ast.FunctionDef)
              and node.name == "worker"]
    assert literal_keys(entry, "design") | literal_keys(ast.parse(launch), "D") <= design.keys()
    assert all(literal_keys(entry, "item") | literal_keys(ast.parse(launch), "item") <= item.keys()
               for item in design["execution_order"])
    texts = {"worker": worker, "launcher": launch, rt.COMPARATOR: rt.adapter_source()}
    for item in design["execution_order"]:
        if item["evaluation_arm"] == rt.REFERENCE:
            texts[rt.REFERENCE] = rt.parent.adapter_source(item)
    texts["closeout"] = (rt.A / "closeout.py").read_text(encoding="utf8")
    for name, text in texts.items():
        compile(text, name, "exec")
    assert rt.runner.O == rt.O and "grid334_logeipc_closed_loop" in launch and "'LOGEIPC') else" in launch
    assert ";policy=models" in texts[rt.COMPARATOR] and rt.COMPARATOR not in texts[rt.REFERENCE]
    assert not (rt.O / "runs").exists() and not (rt.O / "launch_context.json").exists()

    bo = rt.runner.module("_grid334_qualify_acquisition", rt.BO / "readout.py")
    # The same acquisition gets controlled observations for both stop and query cases.
    vectors = bo.np.asarray([[0.], [1.], [2.]])
    baseline = dict(pool_id=0, strict_admissible=True, reward=0.9)
    history = [baseline, dict(baseline, pool_id=1, reward=0.8)]
    calls = []
    def observe(cid):
        calls.append(cid)
        return dict(pool_id=cid, strict_admissible=True, reward=1.0)
    deadline = time.monotonic() + 10
    got = bo.acquire(vectors, history, [2], 0.0, 1.0, observe, deadline)
    assert got[2] == "ei_not_above_fee" and not calls
    got = bo.acquire(vectors, history, [2], 0.0, 1e-6, observe, deadline)
    assert got[0]["pool_id"] == 2 and got[1][0]["id"] == calls[0] == 2
    assert all(action["available"] for action in got[1])

    # Execute the same install, Search construction, and generated choose used
    # in the worker. Only Grid2Op objects and the DC backend are synthetic.
    implementation = rt.policy_class()
    class FixturePolicy(implementation):
        def __init__(self, library, deadline):
            assert list(library) == list(range(196))
            self.vectors = bo.np.arange(196, dtype=float)[:, None]
            self.deadline, self.fee = deadline, 1e-6

    class Ranker:
        def prepare(self, search, obs, base):
            ids = list(search.library)
            return ids, ids, dict(zip(ids, ids)), ids, [], None
        def apply(self, search, obs, base, order, first):
            return order, dict(status="bypass", rank_s=0.0)

    class Search:
        def __init__(self):
            self.library = list(range(196))
            self.subids = [0] * 196
            self.rule, self.policy = "STATIC_ID", "FRAME_GUARD"

    class Observation:
        rho = bo.np.asarray([1.2])
        def simulate(self, action):
            calls.append(action)
            reward = 1.11 if action == 128 else 1.15 if action >= 192 else (1.0 if action % 2 else 0.85)
            return SimpleNamespace(rho=bo.np.asarray([2 - reward])), reward, False, {}

    common = SimpleNamespace(action_vector=lambda action: action, digest=lambda action: action,
        flags=lambda info: dict(exceptions=[], illegal=False, ambiguous=False),
        graph_features=lambda *args: (None, None, None))
    public = SimpleNamespace(PublicState=lambda *args: None)
    ranking = SimpleNamespace(LiveRank=Ranker)
    modules = {"g291_public_runtime": public, "grid330_live_rank": ranking}
    real_install = rt.install_adapter
    def fixture_install(search, item, policy):
        real_install(search, item, policy, loader=lambda name, path: modules[name])

    stamp = time.perf_counter()
    item = next(item for item in design["execution_order"] if item["evaluation_arm"] == rt.COMPARATOR)
    with patch.dict(sys.modules, {"common": common}), patch.object(rt, "read", return_value={
            "worker_deadline": stamp + 10, "start_monotonic": stamp}), patch.object(
            rt, "policy_class", return_value=FixturePolicy), patch.object(rt, "install_adapter", fixture_install):
        rt.install(Search, item, None)
        search = Search()
        search.env = object()
        calls.clear()
        selected, log, _, _ = search.choose(Observation(), 0, 0)
        assert selected >= 192 and log["learning_method"] == "LOGEIPC"
        assert log["local_queries"] > 0 and len(calls) == len(set(calls)) == log["public_queries"]
        assert all(r["log_ei_over_fee"] > 0 for r in log["logeipc"]["trace"])
        search._grid334_policy.fee = 1.0
        calls.clear()
        selected, log, _, _ = search.choose(Observation(), 0, 0)
        assert selected == 128 and log["local_queries"] == 0
        assert log["logeipc"]["reason"] == "ei_not_above_fee"

    for argv in (["run"], ["worker", "0"]):
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(rt.A / "runtime.py"), *argv],
                                capture_output=True, text=True, encoding="utf8", timeout=15)
        assert result.returncode == 2 and "scope confirmation absent" in result.stderr
    assert not any(name in sys.modules for name in ("torch", "grid2op", "common"))
    rt.save(rt.A / "offline_review.json", dict(passed=True, compiled=list(texts), frozen_inputs=len(source),
        items=8, model_references=7, synthetic_acquisition_cases=2, actual_choose_fixture_cases=2, rejected_without_scope=2,
        physical_steps=0, true_forecasts=0, true_model_loads=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("freeze", "qualify"))
    {"freeze": freeze, "qualify": qualify}[parser.parse_args().step]()

