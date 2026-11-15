"""Freeze the single capture and exercise its hooks before live imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import numpy as np

import runtime as rt
from unsafe_capture import install, PARAMETERS

def freeze():
    design = rt.read(rt.PREVIOUS / "design.json")
    row, = [item for item in design["execution_order"] if item["date"] == "2039-03-14"
           and item["evaluation_arm"] == rt.rt.REFERENCE]
    assert row["index"] == 0
    design.update(id=rt.SCOPE, caps=rt.CAPS, execution_order=[row],
                  calendar=[d for d in design["calendar"] if d["date"] == row["date"]],
                  comparisons={}, timing="Capture only. Wall time includes instrumentation.",
                  protocol="Original protected 261 with the GRID334 mixed order. Capture all in-control unsafe QP inputs, outputs and associated action conversion. No changed solver or action.",
                  endpoint="Compare saved GRID334 trajectory and cost at identical steps; verify input/output snapshot pairs.",
                  resource=dict(historical_single_run_s=95.2361354, no_retry=True))
    rt.O.mkdir(exist_ok=False)
    rt.save(rt.A / "design.json", design)
    (rt.O / "design.json").write_bytes((rt.A / "design.json").read_bytes())
    old_out = rt.rt.O
    (rt.O / "model_references.json").write_bytes((old_out / "model_references.json").read_bytes())
    sources = rt.read(rt.PREVIOUS / "source_freeze.json")
    sources.update({path.relative_to(rt.R).as_posix(): rt.sha(path) for path in
                     (*rt.A.glob("*.py"), rt.A / "design.json", rt.O / "design.json",
                      rt.O / "model_references.json", rt.PREVIOUS / "results/raw_records_manifest.json")})
    for path, sha in sources.items():
        assert rt.sha(rt.R / path) == sha, path
    rt.save(rt.O / "code_freeze.json", sources)
    sources[(rt.O / "code_freeze.json").relative_to(rt.R).as_posix()] = rt.sha(rt.O / "code_freeze.json")
    rt.save(rt.A / "source_freeze.json", sources)
    return design, len(sources)

def keys(tree, variable):
    return {node.slice.value for node in ast.walk(tree) if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name) and node.value.id == variable
            and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)}

def test_hooks():
    solvers = []
    location = [5, True, "compute_optimum_unsafe"]
    original_calls = []

    class Optim:
        def to_grid2op(self, obs, c, s, r, base, safe):
            original_calls.append("convert")
            return base

    def solve(mod):
        original_calls.append("solve")
        solvers.append(dict(step=5, phase="compute_optimum_unsafe", solver="OSQP", status="optimal"))
        return (np.ones(2), np.zeros(2), np.zeros(2))

    mod = Optim()
    for name in PARAMETERS:
        setattr(mod, name, SimpleNamespace(value=np.zeros(2)))
    mod.env = SimpleNamespace(gen_renewable=[True], gen_max_ramp_up=[0.], gen_max_ramp_down=[0.])
    mod.nb_max_bus, mod.n_gen, mod.max_iter = 2, 1, 100
    mod.gen_to_subid, mod.flow_computed, mod.margin_sparse = np.zeros(1), np.zeros(1), .01
    obs = SimpleNamespace(gen_renewable=np.ones(1), gen_p=np.ones(1), gen_redispatchable=np.ones(1),
                          gen_margin_up=np.zeros(1), gen_margin_down=np.zeros(1), n_gen=1,
                          to_vect=lambda: np.ones(1))
    action = SimpleNamespace(to_vect=lambda: np.arange(2))
    originals = {"compute_optimum_unsafe": solve}
    with TemporaryDirectory(dir=rt.O) as folder:
        output = Path(folder) / "captures"
        install(Optim, originals, output, lambda: tuple(location), solvers)
        result = originals["compute_optimum_unsafe"](mod)
        location[2] = "other"
        assert mod.to_grid2op(obs, *result, action, False) is action
        assert original_calls == ["solve", "convert"]
        assert len(list(output.iterdir())) == 4
        with np.load(output / "000_action.npz", allow_pickle=False) as data:
            assert np.array_equal(data["converted"], data["base"])
    return len(original_calls)

if __name__ == "__main__":
    design, count = freeze()
    launch = rt.assemble()
    worker = rt.runner.instrument_worker((rt.R / "work/grid55/worker.py").read_text(encoding="utf8"))
    adapter = rt.rt.parent.adapter_source(design["execution_order"][0])
    trees = [ast.parse(text) for text in (launch, worker, adapter)]
    for tree in trees:
        compile(tree, "<prepared>", "exec")
    worker_fn = next(n for n in trees[1].body if isinstance(n, ast.FunctionDef) and n.name == "worker")
    assert keys(trees[0], "D") | keys(worker_fn, "design") <= design.keys()
    assert all(keys(trees[0], "item") | keys(worker_fn, "item") <= item.keys() for item in design["execution_order"])
    assert test_hooks() == 2
    result = dict(passed=True, references=count, generated_sources_compiled=3, fixture_original_calls=2,
                  new_forecasts=0, new_steps=0, new_fits=0)
    rt.save(rt.A / "offline_review.json", result)
    print(result)
