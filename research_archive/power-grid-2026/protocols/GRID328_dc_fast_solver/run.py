"""GRID328: batched native Cholesky for the frozen GRID325 DC score."""
import argparse
from collections import Counter
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[5]
OUT = ROOT / "outputs/grid328_dc_fast_solver"
OLD = ARCHIVE.parent / "GRID325_dc_global_priority"
MIX = ARCHIVE.parent / "GRID326_fixed_dc_mix"
READOUT = ARCHIVE.parent / "GRID327_mixed_fallback_readout"
DLL = OUT / "solver.dll"
GPP = Path("C:/mingw64/mingw64/bin/g++.exe")
BUDGET = 128
NEAR = 1e-10
CAP = 600
REFERENCES = {
    "GRID325_dc_global_priority": ("readout.py", "fixed_scope.json", "public_ranks.json"),
    "GRID326_fixed_dc_mix": ("readout.py", "fixed_scope.json", "public_mix.json"),
    "GRID327_mixed_fallback_readout": ("results/rows.json", "results/summary.json"),
}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, payload):
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def freeze():
    refs = {}
    for name, files in REFERENCES.items():
        base = ARCHIVE.parent / name
        delivery = read(base / "results/delivery_hashes.json")
        for file in files:
            actual = sha(base / file)
            assert actual == delivery[file]
            refs[(base / file).relative_to(ROOT).as_posix()] = actual
    save(ARCHIVE / "fixed_scope.json", {
        "parent_commit": "e5dee4dfcd10d153db6e2f660ff4d5dcb7f6c9fd",
        "refs": refs, "source": {p.name: sha(p) for p in (Path(__file__), ARCHIVE / "solver.cpp")},
        "binary_sha256": sha(DLL), "compiler_sha256": sha(GPP),
        "compile": ["g++", "-O3", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                    "-std=c++17", "-shared", "-static", "solver.cpp", "-o", "solver.dll"],
        "method": "Pack all public candidate busbar plans of a call. Assemble the identical grounded DC equations natively, using connected components by union-find and dense LAPACK DPOSV Cholesky (SciPy dposv capsule). Same injections, limits, branch weights and slack. One native batch per call, single BLAS thread.",
        "numerical": "For uncertain native solve, use original sparse computation. For adjacent score gaps <=1e-10*(1+abs(left)+abs(right)), identify contiguous groups touching the supported prefix, then use original sparse scores within the entire group. Stable old ordering for equal reference scores. No score scaling or selection changes.",
        "qualification": "All 305 published STATIC_ID/DC128 and mixed128 lists must be identical in order; candidate scored and fallback counts must agree per call. If any mismatch, do not infer GRID327's estimates for the new solver.",
        "timing": "Measure packing, solve, native fallback, tie resolution, ranking and mix construction in one component interval. Decoder plus integrity checks timed separately. One pass on all 305. Combine qualified same-order timings with archived GRID327 forecast-work deltas only; no new end-to-end measurement.",
        "scope": {"calls": 305, "candidate_rows": 268356, "new_model_forward": 0,
                  "new_grid2op_simulation": 0, "new_fits": 0, "wall_cap_s": CAP},
    })


class NativeSolver:
    def __init__(self):
        from scipy.linalg import cython_lapack

        self.lib = ctypes.CDLL(str(DLL))
        self.score = self.lib.score_batch
        self.score.argtypes = [ctypes.c_int] * 5 + [ctypes.c_void_p] * 12
        self.score.restype = None
        name = ctypes.pythonapi.PyCapsule_GetName
        name.argtypes = [ctypes.py_object]
        name.restype = ctypes.c_char_p
        pointer = ctypes.pythonapi.PyCapsule_GetPointer
        pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        pointer.restype = ctypes.c_void_p
        capsule = cython_lapack.__pyx_capi__["dposv"]
        assert name(capsule) == (b"void (char *, int *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, "
                                b"__pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, int *)")
        self.lapack = pointer(capsule, name(capsule))
        assert self.lapack

    def batch(self, n_sub, positions, weights, limits, plans, unknown, inputs):
        import numpy as np

        injection, slack, shunt = inputs
        scores = np.zeros(len(plans), dtype=np.float64)
        status = np.empty(len(plans), dtype=np.uint8)
        arrays = (plans, unknown, *positions, weights, limits, injection, shunt, scores, status)
        assert all(a.flags.c_contiguous for a in arrays)
        self.score(len(plans), n_sub, plans.shape[1], len(weights), int(slack),
                   *(a.ctypes.data for a in arrays), self.lapack)
        return scores, status


def smoke(native):
    import numpy as np

    positions = (np.array([0, 1, 0, 1, 0, 1, 2, 2], dtype=np.int32),
                 np.array([2, 3, 4], dtype=np.int32), np.array([5, 6, 7], dtype=np.int32))
    plans = np.ones((3, 8), dtype=np.int32)
    plans[1, 5:] = -1
    values = np.ones(3)
    inputs = (np.array([1., 0., 0., 0., 0., 0., 0., -1.]), 0, np.zeros(6))
    plans[1, 6:] = 1
    plans[1, 4] = -1
    plans[1, 3] = -1
    scores, status = native.batch(3, positions, values, values, plans,
                                  np.array([0, 0, 1], dtype=np.uint8), inputs)
    assert status.tolist() == [0, 3, 1] and abs(scores[0] - 2 / 3) < 1e-12, (scores, status)


def prefix_rank(order, plans, unknown, inputs, grid, native, positions):
    import numpy as np

    arr = np.stack([plans[cid] for cid in order]).astype(np.int32, copy=False)
    flags = np.fromiter((bool(unknown[cid]) for cid in order), dtype=np.uint8, count=len(order))
    scores, status = native.batch(grid.ac.n_sub, positions, grid.weights, grid.limits, arr, flags, inputs)
    assert (status <= 4).all()
    fallback_names = {1: "unknown_reclose_bus", 2: "disconnected_injection", 3: "disconnected_dc_grid"}
    fallback = Counter()
    supported = []
    counters = Counter()
    for slot, state in enumerate(status):
        cid = order[slot]
        if state == 0:
            supported.append((slot, cid, float(scores[slot])))
            counters["native_solved"] += 1
        elif state == 4:
            score, reason = grid.score(plans[cid], unknown[cid], *inputs)
            counters["reference_solver_retries"] += 1
            if reason is None:
                supported.append((slot, cid, score))
            else:
                fallback[reason] += 1
        else:
            fallback[fallback_names[int(state)]] += 1
    ranked = sorted(supported, key=lambda row: row[2])
    n = sum(row[0] < BUDGET for row in supported)
    left = 0
    while left < n:
        right = left + 1
        while right < len(ranked):
            a, b = ranked[right - 1][2], ranked[right][2]
            if b - a > NEAR * (1 + abs(a) + abs(b)):
                break
            right += 1
        if right - left > 1:
            for index in range(left, right):
                slot, cid, _ = ranked[index]
                score, reason = grid.score(plans[cid], unknown[cid], *inputs)
                assert reason is None
                ranked[index] = (slot, cid, score)
            counters["reference_near_tie_solves"] += right - left
        left = right
    if counters["reference_near_tie_solves"]:
        ranked.sort(key=lambda row: (row[2], row[0]))
    result = order[:BUDGET]
    for orig, scored in zip(supported[:n], ranked[:n], strict=True):
        result[orig[0]] = scored[1]
    assert len(result) == len(set(result)) == BUDGET
    return result, len(supported), dict(fallback), counters


def timings(values):
    return {"total_s": sum(values), "mean_s": statistics.mean(values),
            "median_s": statistics.median(values),
            "p95_s": sorted(values)[-(-95 * len(values) // 100) - 1]}


def evaluate():
    start = time.monotonic()
    scope = read(ARCHIVE / "fixed_scope.json")
    assert sha(DLL) == scope["binary_sha256"] and sha(GPP) == scope["compiler_sha256"]
    for name, digest in scope["source"].items():
        assert sha(ARCHIVE / name) == digest
    for name, digest in scope["refs"].items():
        assert sha(ROOT / name) == digest
    prior = runpy.run_path(str(OLD / "readout.py"))
    manifest = prior["setup"]()
    mix_fn = runpy.run_path(str(MIX / "readout.py"))["hybrid"]
    sys.path.insert(0, str(ROOT / prior["DECODER"]))
    from public import Decoder
    import numpy as np

    native = NativeSolver()
    smoke(native)
    decoder = Decoder()
    assert all(prior["supported_commands"](action) for action in decoder.library)
    grid = prior["PublicGrid"](decoder.ac, *prior["static_parameters"](decoder.ac))
    positions = tuple(np.ascontiguousarray(a, dtype=np.int32)
                      for a in (grid.subs, grid.ac.line_or_pos_topo_vect, grid.ac.line_ex_pos_topo_vect))
    ranks = read(OLD / "public_ranks.json")["rows"]
    mixed = read(MIX / "public_mix.json")["rows"]
    rows = []
    for member, old, hybrid in zip(manifest["members"], ranks, mixed, strict=True):
        assert member["call_id"] == old["call_id"] == hybrid["call_id"]
        assert time.monotonic() - start < CAP
        before = time.perf_counter()
        call, ob, plans, unknown, _ = decoder.decode(member["source"])
        order = call["order"]
        assert len(order) == member["M"] and set(order) == set(plans)
        assert call["state_hash"] == old["state_hash"] and order[:BUDGET] == old["static128"]
        with np.load(ROOT / member["state"], allow_pickle=False) as z:
            base = decoder.ac.from_vect(z["base_action"])
        decode_s = time.perf_counter() - before

        before = time.perf_counter()
        if prior["supported_commands"](base):
            dc, n, fallback, counters = prefix_rank(order, plans, unknown, grid.state_inputs(ob),
                                                    grid, native, positions)
        else:
            dc, n, fallback, counters = order[:BUDGET], 0, {"unsupported_active_power_command": len(order)}, Counter()
        new_mix = mix_fn(order[:BUDGET], dc)
        component_s = time.perf_counter() - before
        assert n + sum(fallback.values()) == len(order)
        rows.append({**{k: member[k] for k in ("call_id", "partition", "date")},
                     "dc128": dc, "mix128": new_mix, "scored": n, "fallback": fallback,
                     "dc_prefix_exact": dc == old["dc128"], "mix_prefix_exact": new_mix == hybrid["mix128"],
                     "support_counts_exact": n == old["scored"] and fallback == old["fallback"],
                     "component_s": component_s, "decode_integrity_s": decode_s,
                     "old_dc_plus_mix_s": old["dc_and_sort_s"] + hybrid["mix_ns"] / 1e9,
                     "solver_counts": dict(counters)})
    assert len(rows) == 305 and decoder.checks["combined_hashes"] == 268356
    qualified = all(r["dc_prefix_exact"] and r["mix_prefix_exact"] and r["support_counts_exact"] for r in rows)
    proxy = {p: read(READOUT / "results/summary.json")["metrics"][p]["FRAME_GUARD"]
             for p in ("train", "validation")} if qualified else {}
    partitions = {}
    for p in ("train", "validation"):
        cases = [r for r in rows if r["partition"] == p]
        observed = timings([r["component_s"] for r in cases])
        previous = timings([r["old_dc_plus_mix_s"] for r in cases])
        result = {"calls": len(cases), "new_timing": observed, "archived_timing": previous,
                  "component_ratio": previous["total_s"] / observed["total_s"],
                  "decoder_integrity_total_s": sum(r["decode_integrity_s"] for r in cases)}
        if qualified:
            old_work = proxy[p]["all"]["original"]["forecast_work_s"]
            new_work = proxy[p]["all"]["mixed"]["forecast_work_s"]
            result.update(estimated_total_original_s=old_work,
                          estimated_total_mixed_s=new_work + observed["total_s"],
                          estimated_net_delta_s=new_work + observed["total_s"] - old_work,
                          by_date={date: {
                              "component_s": (cost := sum(r["component_s"] for r in cases if r["date"] == date)),
                              "estimated_net_delta_s": d["forecast_work_delta_s"] + cost,
                          } for date, d in proxy[p]["by_date"].items()})
        partitions[p] = result
    save(OUT / "rows.json", rows)
    save(OUT / "summary.json", {"qualified": qualified, "partition_timing": partitions,
         "exact_dc": sum(r["dc_prefix_exact"] for r in rows),
         "exact_mix": sum(r["mix_prefix_exact"] for r in rows),
         "exact_support": sum(r["support_counts_exact"] for r in rows),
         "mismatch_ids": [r["call_id"] for r in rows
             if not all(r[k] for k in ("dc_prefix_exact", "mix_prefix_exact", "support_counts_exact"))],
         "solver_counts": dict(sum((Counter(r["solver_counts"]) for r in rows), Counter())),
         "decoder_checks": dict(decoder.checks), "elapsed_s": time.monotonic() - start,
         "scope_sha256": sha(ARCHIVE / "fixed_scope.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    {"freeze": freeze, "evaluate": evaluate}[parser.parse_args().stage]()

