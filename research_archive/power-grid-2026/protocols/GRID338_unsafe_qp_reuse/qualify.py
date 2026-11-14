"""Offline unsafe-QP reuse qualification. Never import the live controller."""
import ast
from collections import Counter, OrderedDict
from hashlib import sha256
import json
import logging
from pathlib import Path
import runpy
import time
from types import SimpleNamespace
import warnings

import cvxpy as cp
import numpy as np
import osqp


ROOT = Path(__file__).resolve().parents[6]
ARCHIVE = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/grid338_unsafe_qp_reuse"
OPTIM = ROOT / "work/grid02/ljn_nn/modules/convex_optim.py"
KEY_SOURCE = ROOT / "work/grid29/cache_dc.py"
SCALARS = (
    "_alpha_por_error", "_margin_th_limit", "_penalty_curtailment_unsafe",
    "_penalty_storage_unsafe", "_penalty_redispatching_unsafe",
)
CAPACITY = 32


def read_pinned(path, digest):
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected source bytes: {path}")
    return data


def load_original():
    """Copy only the original pure method bodies from the pinned source."""
    data = read_pinned(OPTIM, "f7518a44a5e529bd398e07d21a9bef299309663d523c649a113fa72cc3d7756f")
    tree = ast.parse(data)
    wanted = {
        "_integer_index", "_aux_compute_kcl", "_mask_theta_zero", "_solve_problem",
        "compute_optimum_unsafe",
    }
    methods = {node.name: node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name in wanted}
    if methods.keys() != wanted:
        raise RuntimeError("Frozen function inventory changed")
    namespace = {
        "cp": cp, "np": np, "warnings": warnings, "logger": logging.getLogger("frozen_unsafe"),
    }
    exec(compile(ast.Module(body=list(methods.values()), type_ignores=[]),
                 str(OPTIM), "exec"), namespace)
    return methods["compute_optimum_unsafe"], namespace


def cached_method(original, namespace):
    """Separate the original graph construction from its unchanged solve and result path."""
    split, = [index for index, stmt in enumerate(original.body)
              if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
              and stmt.targets[0].id == "prob"]
    graph_inputs = {"_powerlines_x": "reactance", **{key: key for key in SCALARS}}
    replacements = Counter()

    class FreezeCoefficients(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                if node.attr in graph_inputs:
                    replacements[node.attr] += 1
                    return ast.copy_location(ast.Name(graph_inputs[node.attr], ast.Load()), node)
            return self.generic_visit(node)

    graph = FreezeCoefficients().visit(
        ast.Module(body=original.body[:split + 1], type_ignores=[])
    )
    if replacements != Counter(graph_inputs.keys()):
        raise RuntimeError("Unexpected static coefficient use")
    build = compile(ast.fix_missing_locations(graph), str(OPTIM), "exec")
    tail = compile(ast.fix_missing_locations(ast.Module(body=[ast.FunctionDef(
        name="_cached_finish",
        args=ast.parse("def f(self, prob, f_or, curtailment_mw, storage, redispatching): pass").body[0].args,
        body=original.body[split + 1:], decorator_list=[],
    )], type_ignores=[])), str(OPTIM), "exec")
    exec(tail, namespace)
    read_pinned(KEY_SOURCE, "235996780a5a4d869ccbe4a0a4cbd2926626bb960bf9e9906e4fe5eea84b33e7")
    key_for = runpy.run_path(str(KEY_SOURCE))["key_for"]

    def signature(value):
        array = np.asarray(value)
        return array.shape, array.dtype.str, array.tobytes()

    def solve(mod):
        reactance = np.asarray(mod._powerlines_x.value)
        coefficients = {name: float(getattr(mod, name).value) for name in SCALARS}
        if not np.isfinite(reactance).all() or np.any(reactance == 0):
            mod.cache_event = "fallback"
            return namespace["compute_optimum_unsafe"](mod)
        key = (
            key_for(mod), mod.nb_max_bus, mod.n_gen, signature(reactance),
            *(signature(value) for value in coefficients.values()),
            *(signature(value) for value in (mod.gen_to_subid, mod.env.gen_renewable,
                 mod.env.gen_max_ramp_up, mod.env.gen_max_ramp_down)),
        )
        cache = mod.unsafe_programs
        if key in cache:
            mod.cache_event = "hit"
            cache.move_to_end(key)
        else:
            mod.cache_event = "miss"
            inputs = dict(self=mod, reactance=reactance, **coefficients)
            exec(build, namespace, inputs)
            problem = inputs["prob"]
            if not problem.is_dcp(dpp=True):
                mod.cache_event = "fallback"
                return namespace["compute_optimum_unsafe"](mod)
            cache[key] = tuple(inputs[name] for name in
                               ("prob", "f_or", "curtailment_mw", "storage", "redispatching"))
            if len(cache) > CAPACITY:
                cache.popitem(last=False)
        return namespace["_cached_finish"](mod, *cache[key])

    return solve


def read_saved_baseline():
    path = ARCHIVE.parent / "GRID334_logeipc_closed_loop/results/raw_records_manifest.json"
    manifest = json.loads(path.read_bytes())
    rows = []
    for date in ("2039-03-14", "2039-06-13", "2039-09-12", "2039-12-12"):
        for arm in ("BOUNDED261_GUARD_DC__f0_s0", "LOGEIPC_DC"):
            prefix = f"outputs/grid334_logeipc_closed_loop/runs/r0__{date}_100__{arm}"
            expected = {f"{prefix}/components.json", f"{prefix}/summary.json"}
            receipts = {
                entry["path"].rsplit("/", 1)[-1]:
                json.loads(read_pinned(ROOT / entry["path"], entry["sha256"]))
                for entry in manifest["entries"] if entry["path"] in expected
            }
            timings = receipts["components.json"]
            summary = receipts["summary.json"]

            def unsafe(records):
                return [row for row in records if row["in_control"]
                        and row["phase"] == "compute_optimum_unsafe"]

            calls = unsafe(timings["calls"])
            solves = unsafe(timings["solver_records"])
            def identities(records):
                return Counter((row["step"], row["owner"]) for row in records)

            if identities(calls) != identities(solves) or any(s["solver"] != "OSQP" for s in solves):
                raise RuntimeError(f"Call and solver records disagree: {prefix}")
            rows.append(dict(date=date, arm=arm, complete=summary["complete"], steps=summary["steps"],
                             controller_s=summary["controller_s"], calls=len(calls),
                             total_unsafe_s=sum(c["wall_s"] for c in calls),
                             compilation_s=sum(s["compilation_s"] for s in solves),
                             backend_s=sum(s["backend_s"] for s in solves),
                             statuses=dict(Counter(s["status"] for s in solves))))
    return rows


def fixture(namespace):
    def param(value, **options):
        return cp.Parameter(shape=np.shape(value), value=value, **options)

    class Optimizer:
        _aux_compute_kcl = namespace["_aux_compute_kcl"]
        _mask_theta_zero = namespace["_mask_theta_zero"]

        def _solve_problem(self, prob, solver_type):
            result = namespace["_solve_problem"](self, prob, solver_type)
            stats = prob.solver_stats
            self.last_solve = dict(converged=result, status=prob.status,
                                   iterations=stats.num_iters if stats is not None else None,
                                   compilation_s=prob.compilation_time)
            return result

    mod = Optimizer()
    mod.nb_max_bus = 3
    mod.n_gen = 2
    mod.gen_to_subid = np.array([0, 1])
    mod.env = SimpleNamespace(gen_renewable=np.array([False, True]),
                              gen_max_ramp_up=np.array([30., 1.]),
                              gen_max_ramp_down=np.array([30., 1.]))
    for name, value in {
        "bus_or": np.array([0]), "bus_ex": np.array([1]),
        "bus_load": np.array([1]), "bus_gen": np.array([0, 1]),
        "bus_storage": np.array([1]), "_powerlines_x": np.array([2.]),
        "_prev_por_error": np.array([-0.25]), "_th_lim_mw": np.array([8.]),
        "load_per_bus": np.array([0., 24., 0.]), "gen_per_bus": np.array([20., 4., 0.]),
        "redisp_up": np.array([30., 0., 0.]), "redisp_down": np.array([30., 0., 0.]),
        "curtail_up": np.array([0., 4., 0.]), "curtail_down": np.array([0., 0., 0.]),
        "storage_up": np.array([0., 8., 0.]), "storage_down": np.array([0., 8., 0.]),
        "_storage_power_obs": 0.0,
    }.items():
        setattr(mod, name, param(value))
    for name, value in zip(SCALARS, (0.81, 0.9, 1., 0.2, 0.5), strict=True):
        setattr(mod, name, param(value, nonneg=True))
    mod.flow_computed = np.empty(1)
    mod.unsafe_programs = OrderedDict()
    mod.max_iter = 100_000
    return mod


def qualify(original, candidate, namespace):
    def inputs(mod, case):
        mod.load_per_bus.value = np.array([0., case["load"], 0.])
        mod._prev_por_error.value = np.array([case["error"]])
        mod._storage_power_obs.value = case["observed_storage"]
        mod._th_lim_mw.value = np.array([case["thermal_limit"]])
        mod.bus_or.value = np.array([case["origin"]])
        mod.bus_ex.value = np.array([1 - case["origin"]])
        mod._margin_th_limit.value = case["margin"]
        flexible = float(case["flexible"])
        mod.redisp_up.value = np.array([30., 0., 0.]) * flexible
        mod.redisp_down.value = np.array([30., 0., 0.]) * flexible
        mod.curtail_up.value = np.array([0., 4., 0.]) * flexible
        mod.storage_up.value = np.array([0., 8., 0.]) * flexible
        mod.storage_down.value = np.array([0., 8., 0.]) * flexible
        mod.max_iter = case["max_iter"]

    defaults = dict(load=24., error=-.25, observed_storage=0., origin=0, thermal_limit=8.,
                    margin=.9, max_iter=100_000, flexible=True)
    cases = (
        ("cold", "miss", {}),
        ("changed_injections", "hit", {"load": 24.25, "observed_storage": .25}),
        ("changed_flow_error_and_limit", "hit", {"error": .35, "thermal_limit": 7.7}),
        ("changed_constant", "miss", {"margin": .92}),
        ("changed_topology", "miss", {"origin": 1}),
        ("restored_topology", "hit", {}),
        ("limited_iterations", "hit", {"max_iter": 1}),
        ("infeasible", "hit", {"load": 25., "flexible": False}),
        ("after_infeasible", "hit", {}),
    )
    reference, reused = fixture(namespace), fixture(namespace)
    rows = []
    for name, expected, changed in cases:
        case = defaults | changed
        outputs = []
        receipts = []
        for mod, solve in ((reference, original), (reused, candidate)):
            inputs(mod, case)
            start = time.perf_counter()
            out = solve(mod)
            elapsed = time.perf_counter() - start
            outputs.append(tuple(np.asarray(value).copy() for value in
                                 (*out, mod.flow_computed, mod._storage_power_obs.value)))
            receipts.append(dict(mod.last_solve, wall_s=elapsed))
        first, second = outputs
        identical = all(a.tobytes() == b.tobytes() for a, b in zip(first, second, strict=True))
        close = all(np.allclose(a, b, rtol=0, atol=1e-7, equal_nan=True)
                    for a, b in zip(first, second, strict=True))
        finite_deltas = [np.abs(a[np.isfinite(a) & np.isfinite(b)] - b[np.isfinite(a) & np.isfinite(b)])
                         for a, b in zip(first, second, strict=True)]
        statuses_match = all(receipts[0][key] == receipts[1][key]
                             for key in ("status", "converged", "iterations"))
        rows.append(dict(case=name, parameters=case, reference=receipts[0], cached=receipts[1],
                         expected_cache=expected, observed_cache=reused.cache_event,
                         bitwise=identical, within_atol_1e_7=close, statuses_match=statuses_match,
                         max_finite_error=max(float(np.max(value, initial=0.)) for value in finite_deltas)))
    return rows


def main():
    OUTPUT.mkdir(exist_ok=True)
    out = OUTPUT / "result.json"
    if out.exists():
        raise RuntimeError(f"Receipt already exists: {out}")
    source, namespace = load_original()
    candidate = cached_method(source, namespace)
    rows = read_saved_baseline()
    cases = qualify(namespace["compute_optimum_unsafe"], candidate, namespace)
    result = dict(numerical_probe_passed=all(row["bitwise"] and row["statuses_match"]
                  and row["expected_cache"] == row["observed_cache"] for row in cases),
                  cvxpy=cp.__version__, osqp=osqp.__version__, numpy=np.__version__,
                  saved=rows, synthetic=cases, simulated_steps=0, fits=0)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
