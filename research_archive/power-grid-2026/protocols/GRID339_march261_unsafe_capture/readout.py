"""Read one captured episode and test the GRID338 candidate without a simulator."""
import gzip
import hashlib
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import cvxpy as cp
import numpy as np

A = Path(__file__).resolve().parent
ROOT = A.parents[5]
O = ROOT / "outputs/grid339_march261_unsafe_capture"
PREV = ROOT / "outputs/grid334_logeipc_closed_loop"
ORIGINAL_DIR = A.parent / "GRID338_unsafe_qp_reuse"
RUN = "r0__2039-03-14_100__BOUNDED261_GUARD_DC__f0_s0"
SCALARS = ("nb_max_bus", "n_gen", "max_iter")
ENV_FIELDS = ("gen_renewable", "gen_max_ramp_up", "gen_max_ramp_down")
INTEGER = {"bus_or", "bus_ex", "bus_load", "bus_gen", "bus_storage"}
NONNEG = {"load_per_bus", "gen_per_bus", "redisp_up", "redisp_down", "curtail_up",
          "curtail_down", "storage_up", "storage_down", "_th_lim_mw"}

def read(path):
    return json.loads(path.read_bytes())

def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def row_list(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def same(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (left.shape, left.dtype.str, left.tobytes()) == (right.shape, right.dtype.str, right.tobytes())

def main():
    active = read(O / "active_worker.json")
    finish = read(O / "finished.json")
    if active["active"] or finish["accepted"] != 1:
        raise RuntimeError("Expected one completed capture worker")
    out = O / "offline_rows.jsonl"
    archive = A.parent / "GRID334_logeipc_closed_loop/results/raw_records_manifest.json"
    pinned = {row["path"]: row["sha256"] for row in read(archive)["entries"]}
    hist_root, fresh_root = PREV / "runs" / RUN, O / "runs" / RUN
    for name in ("steps.jsonl.gz", "components.json", "summary.json"):
        historical_file = hist_root / name
        assert sha(historical_file) == pinned[historical_file.relative_to(ROOT).as_posix()]
    historical, fresh = [row_list(folder / "steps.jsonl.gz") for folder in (hist_root, fresh_root)]
    keys = ("step", "before_hash", "after_hash", "action_hash", "raw_cost", "simulations",
            "done", "complete", "illegal", "ambiguous", "exceptions")
    before_after = [all(row.get(key) == old.get(key) for key in keys)
                    for row, old in zip(fresh, historical, strict=True)]
    if not all(before_after):
        raise RuntimeError(f"Capture trajectory differs at steps {[i+1 for i, ok in enumerate(before_after) if not ok][:10]}")
    historical_summary, summary = [read(folder / "summary.json") for folder in (hist_root, fresh_root)]
    assert summary["complete"] == historical_summary["complete"] is True
    for key in ("steps", "cost", "public_forecasts"):
        assert summary[key] == historical_summary[key], key
    paired = [folder / "components.json" for folder in (hist_root, fresh_root)]
    select = lambda path: [row for row in read(path)["solver_records"]
                           if row["in_control"] and row["phase"] == "compute_optimum_unsafe"]
    previous, records = map(select, paired)
    assert [[v[key] for key in ("step", "owner", "status", "iterations")] for v in previous] == [
           [v[key] for key in ("step", "owner", "status", "iterations")] for v in records]
    assert len(records) == 23
    snapshots = O / "snapshots"
    assert len(list(snapshots.iterdir())) == 4 * len(records)
    hashes = {file.name: sha(file) for file in sorted(snapshots.iterdir())}
    spec = importlib.util.spec_from_file_location("_grid338_offline", ORIGINAL_DIR / "qualify.py")
    qualifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)
    method, namespace = qualifier.load_original()
    cached = qualifier.cached_method(method, namespace)

    class Optim:
        _aux_compute_kcl = namespace["_aux_compute_kcl"]
        _mask_theta_zero = namespace["_mask_theta_zero"]

        def __init__(self, arrays):
            self.env = SimpleNamespace()
            self.unsafe_programs = OrderedDict()
            self.bus_storage = None
            self.params = set()
            for key, value in arrays.items():
                if key not in SCALARS + ENV_FIELDS + ("gen_to_subid", "flow_computed"):
                    attr = {}
                    if key in INTEGER:
                        attr["integer"] = True
                    elif key == "_powerlines_x":
                        attr["pos"] = True
                    elif key in NONNEG or key in qualifier.SCALARS:
                        attr["nonneg"] = True
                    setattr(self, key, cp.Parameter(shape=value.shape, **attr))
                    self.params.add(key)
            self.update(arrays)

        def update(self, arrays):
            updated = set(arrays) - set(SCALARS + ENV_FIELDS + ("gen_to_subid", "flow_computed"))
            assert updated == self.params
            for key in self.params:
                # Original writes values in place, including possible bound roundoff.
                getattr(self, key).save_value(arrays[key].copy())
            for key in SCALARS:
                setattr(self, key, int(arrays[key]))
            for key in ENV_FIELDS:
                setattr(self.env, key, arrays[key].copy())
            self.gen_to_subid = arrays["gen_to_subid"].copy()
            self.flow_computed = arrays["flow_computed"].copy()

        def _solve_problem(self, problem, solver_type):
            ok = namespace["_solve_problem"](self, problem, solver_type)
            self.status = problem.status
            self.iterations = problem.solver_stats.num_iters
            self.compilation_s = problem.compilation_time
            return ok

    owners = {}
    rows = []
    with out.open("x", encoding="utf8") as log:
        for index, record in enumerate(records):
            stem = f"{index:03d}"
            assert record == read(snapshots / f"{stem}_solve.json")
            inputs = npz(snapshots / f"{stem}_input.npz")
            original = npz(snapshots / f"{stem}_result.npz")
            action = npz(snapshots / f"{stem}_action.npz")
            assert {"converted", "obs", "gen_renewable", "gen_p", "gen_redispatchable",
                    "gen_margin_up", "gen_margin_down", "n_gen", "margin_sparse"} <= action.keys()
            owner = record["owner"]
            if owner not in owners:
                owners[owner] = Optim(inputs)
            mod = owners[owner]
            mod.update(inputs)
            output = dict(zip(("curtailment", "storage", "redispatching"), cached(mod), strict=True))
            output.update(flow=mod.flow_computed, storage_state=mod._storage_power_obs.value)
            exact = {name: same(original[name], output[name]) for name in original}
            errors = {}
            for name in original:
                expected, actual = np.asarray(original[name]), np.asarray(output[name])
                finite = np.isfinite(expected) & np.isfinite(actual)
                errors[name] = float(np.max(np.abs(expected[finite] - actual[finite]), initial=0.))
            row = dict(index=index, step=record["step"], owner=owner,
                       saved_status=record["status"], cached_status=mod.status,
                       saved_iterations=record["iterations"], cached_iterations=mod.iterations,
                       cache_event=mod.cache_event, exact=exact, max_finite_error=errors,
                       capture_compilation_s=record["compilation_s"],
                       cached_compilation_s=mod.compilation_s)
            rows.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
    result = dict(passed_exact_gate=all(all(row["exact"].values())
                  and row["saved_status"] == row["cached_status"]
                  and row["saved_iterations"] == row["cached_iterations"] for row in rows),
                  steps=summary["steps"], complete=summary["complete"], original_action_ledger_exact=True,
                  snapshots=hashes, captured_solves=len(rows),
                  hits=sum(row["cache_event"] == "hit" for row in rows),
                  misses=sum(row["cache_event"] == "miss" for row in rows),
                  capture_compilation_s=sum(row["capture_compilation_s"] for row in rows),
                  cached_compilation_s=sum(row["cached_compilation_s"] for row in rows),
                  recorded_cost=summary["cost"], recorded_forecasts=finish["forecasts"],
                  offline_osqp_solves=len(rows), new_model_fits=0)
    (O / "offline_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                          encoding="utf8")
    print({key: value for key, value in result.items() if key != "snapshots"})

if __name__ == "__main__":
    main()
