"""Freeze the two-arm replay and qualify its adapter without live simulation."""
import argparse
import ast
import subprocess
import sys
import time

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
    for name, text in texts.items():
        compile(text, name, "exec")
    assert rt.runner.O == rt.O and "grid333_logeipc_closed_loop" in launch and "'LOGEIPC') else" in launch
    assert " policy=models" in texts[rt.COMPARATOR] and rt.COMPARATOR not in texts[rt.REFERENCE]
    assert not (rt.O / "runs").exists() and not (rt.O / "launch_context.json").exists()

    bo = rt.runner.module("_grid333_qualify_acquisition", rt.BO / "readout.py")
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

    # Exercise the live policy wrapper with a tiny public fixture and injected
    # global prefix. No environment, neural model, or real candidate feedback.
    implementation = rt.policy_class()
    native = implementation.run.__closure__
    bindings = dict(zip(implementation.run.__code__.co_freevars, (cell.cell_contents for cell in native)))
    original = bindings["native"].run
    policy = implementation.__new__(implementation)
    policy.vectors, policy.fee, policy.deadline = vectors, 1e-6, deadline
    calls.clear()
    def global_only(order, reveal, mode, subids, aliases, limit):
        assert mode == "NO_LOCAL"
        return dict(outcomes=[dict(r, forecast_s=0.0) for r in history], completion_candidate_ids=[2],
                    stop_reason="fixture")
    bindings["native"].run = global_only
    result = policy.run([0, 1, 2], lambda cid, phase: dict(observe(cid), forecast_s=0.0),
                        [], {}, limit=1.2)
    bindings["native"].run = original
    assert result["learning_method"] == "LOGEIPC" and result["chosen_pool_id"] == 2
    assert result["public_queries"] == result["local_queries"] + 2
    for argv in (["run"], ["worker", "0"]):
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(rt.A / "runtime.py"), *argv],
                                capture_output=True, text=True, encoding="utf8", timeout=15)
        assert result.returncode == 2 and "scope confirmation absent" in result.stderr
    assert not any(name in sys.modules for name in ("torch", "grid2op", "common"))
    rt.save(rt.A / "offline_review.json", dict(passed=True, compiled=list(texts), frozen_inputs=len(source),
        items=8, model_references=7, synthetic_policy_cases=3, rejected_without_scope=2,
        physical_steps=0, true_forecasts=0, true_model_loads=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("freeze", "qualify"))
    {"freeze": freeze, "qualify": qualify}[parser.parse_args().step]()

