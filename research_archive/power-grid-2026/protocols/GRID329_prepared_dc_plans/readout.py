"""GRID329: export DC bus plans from Search.prepare's existing canonical bytes."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[5]
OUT = ROOT / "outputs/grid329_prepared_dc_plans"
PREP = "work/grid26/engine.py"
VEC = "work/grid01/trace_agent.py"
PROVIDER = "work/grid183_online_contract/runtime.py"
DECODER = "work/grid111_action_feedback"
GRID291 = "outputs/grid291_control_preparation/code_freeze.json"
GRID100 = "outputs/grid100_data_foundation/bulk_code_freeze.json"
OLD = ARCHIVE.parent / "GRID325_dc_global_priority"
FAST = ARCHIVE.parent / "GRID328_dc_fast_solver"
LIMIT = 600
FIELDS = ("_set_line_status", "_switch_line_status", "_set_topo_vect", "_change_bus_vect",
          "_redispatch", "_storage_power", "_curtail", "_raise_alert")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, obj):
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def freeze():
    old_paths = (OLD / "fixed_scope.json", OLD / "public_ranks.json",
                 FAST / "fixed_scope.json", FAST / "results/summary.json")
    for path in old_paths:
        base = OLD if path.is_relative_to(OLD) else FAST
        assert sha(path) == read(base / "results/delivery_hashes.json")[path.relative_to(base).as_posix()]
    for manifest in (GRID291, GRID100):
        sources = read(ROOT / manifest)
        for ref in (PREP, "work/grid28/engine.py"):
            assert sha(ROOT / ref) == sources[ref]
    assert sha(ROOT / PROVIDER) == read(ROOT / GRID291)[PROVIDER]
    paths = [*old_paths, *(ROOT / p for p in
        (PREP, VEC, PROVIDER, GRID291, GRID100, "work/grid28/engine.py",
         "work/grid291_control_preparation/adapter.py", f"{DECODER}/public.py", f"{DECODER}/common111.py"))]
    save(ARCHIVE / "fixed_scope.json", {
        "parent_commit": "74b201ea0ab3c69e6cabf6089c87bfdfeb265d70",
        "source_sha256": sha(Path(__file__)),
        "source_hashes": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        "method": "Expose the existing first map of canonical serialized combined-action bytes by appending it to the original Search.prepare return. Assemble int32 post-command topology matrix and uint8 unknown-reconnect flags from those vectors in one vectorized pass. No extra Action addition in proposed path.",
        "reference": "Frozen GRID26 Search.prepare plus GRID183 PublicState.plan on serialized GRID100 train/validation states, and exact old metadata, combined hash and plan/candidate identities. No grid or model construction.",
        "timing": "Paired once on each saved call, alternate baseline/enhanced invocation order by fixed row parity. Baseline includes sorted representatives. Enhanced includes sorted representatives and vectorized export. Report direct export separately and paired prep differences. Reuse GRID328 component and GRID327 forecast proxies only for additive accounting. No ranking or replay rerun.",
        "limit": {"calls": 305, "candidate_representatives": 268356, "wall_cap_s": LIMIT,
                  "new_dc_solves": 0, "new_model_forward": 0, "new_fit": 0, "new_simulate": 0},
    })


def compile_node(node, path, symbols):
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    names = dict(symbols)
    exec(compile(module, str(path), "exec"), names)
    return names[node.name]


def preparation_functions():
    import copy
    import numpy as np
    from grid2op.dtypes import dt_float

    vector_source = ROOT / VEC
    node, = [n for n in ast.parse(vector_source.read_text(encoding="utf8")).body
             if isinstance(n, ast.FunctionDef) and n.name == "action_vector"]
    vector = compile_node(node, vector_source, {"np": np, "dt_float": dt_float})
    path = ROOT / PREP
    search, = [n for n in ast.parse(path.read_text(encoding="utf8")).body
               if isinstance(n, ast.ClassDef) and n.name == "Search"]
    node, = [n for n in search.body if isinstance(n, ast.FunctionDef) and n.name == "prepare"]
    symbols = {"np": np, "action_vector": vector}
    original = compile_node(node, path, symbols)
    enhanced = copy.deepcopy(node)
    ret = enhanced.body[-1]
    assert isinstance(ret, ast.Return) and ast.unparse(ret.value) == "(eligible, reps, aliases, actions, changed)"
    ret.value.elts.append(ast.Name(id="first", ctx=ast.Load()))
    return original, compile_node(enhanced, path, symbols), vector


def reference_provider():
    import numpy as np

    path = ROOT / PROVIDER
    nodes = ast.parse(path.read_text(encoding="utf8")).body
    unavailable, = [n for n in nodes if isinstance(n, ast.ClassDef) and n.name == "InputUnavailable"]
    provider, = [n for n in nodes if isinstance(n, ast.ClassDef) and n.name == "PublicState"]
    return compile_node(provider, path, {"np": np, "InputUnavailable": compile_node(unavailable, path, {})})


class PreparedPlans:
    """Build the GRID328 native input using the same bytes already used for aliasing."""

    def __init__(self, action, static):
        import numpy as np

        assert tuple(type(action).attr_list_vect) == FIELDS
        self.slices = {}
        self.width = 0
        for name in FIELDS:
            size = len(action._get_array_from_attr_name(name))
            self.slices[name] = slice(self.width, self.width + size)
            self.width += size
        self.op = np.asarray(static.line_or_pos_topo_vect)
        self.ep = np.asarray(static.line_ex_pos_topo_vect)
        self.static = static

    def from_keys(self, obs, order, first):
        import numpy as np

        assert list(first.values()) == order
        raw = np.frombuffer(b"".join(first), dtype=np.float32).reshape(len(order), self.width)
        fields = self.slices
        cmd = raw[:, fields["_set_topo_vect"]]
        switch = raw[:, fields["_change_bus_vect"]] != 0
        if (switch & (cmd != 0)).any() or not np.isfinite(raw).all():
            raise ValueError("Unsupported or invalid canonical action")
        plans = np.broadcast_to(obs.topo_vect, cmd.shape).astype(np.int32, copy=True)
        np.copyto(plans, cmd, where=cmd != 0, casting="unsafe")
        live = switch & (plans > 0)
        plans[live] = 3 - plans[live]
        status = raw[:, fields["_set_line_status"]]
        flip = raw[:, fields["_switch_line_status"]] != 0
        off = (status < 0) | (flip & obs.line_status)
        on = (status > 0) | (flip & ~obs.line_status)
        plans[:, self.op] = np.where(off, -1, plans[:, self.op])
        plans[:, self.ep] = np.where(off, -1, plans[:, self.ep])
        unknown_or = on & (plans[:, self.op] <= 0)
        unknown_ex = on & (plans[:, self.ep] <= 0)
        flags = np.any(unknown_or | unknown_ex, axis=1).astype(np.uint8)
        assert plans.flags.c_contiguous and flags.flags.c_contiguous
        return plans, flags, unknown_or, unknown_ex


def old_path(prepare, search, obs, base):
    begin = time.perf_counter()
    result = prepare(search, obs, base)
    order = sorted(result[1])
    return result, order, time.perf_counter() - begin


def new_path(prepare, search, obs, base, adapter):
    begin = time.perf_counter()
    result = prepare(search, obs, base)
    order = sorted(result[1])
    before = time.perf_counter()
    arrays = adapter.from_keys(obs, order, result[-1])
    end = time.perf_counter()
    return result, order, arrays, before - begin, end - before


def timing(values):
    return {"total_s": sum(values), "mean_s": statistics.mean(values),
            "median_s": statistics.median(values),
            "p95_s": sorted(values)[-(-95 * len(values) // 100) - 1]}


def evaluate():
    import numpy as np

    start = time.monotonic()
    scope = read(ARCHIVE / "fixed_scope.json")
    assert sha(Path(__file__)) == scope["source_sha256"]
    for path, digest in scope["source_hashes"].items():
        assert sha(ROOT / path) == digest, path
    manifest = read(OLD / "fixed_scope.json")
    for path, digest in manifest["source_hashes"].items():
        assert sha(ROOT / path) == digest, path
    sys.path.insert(0, str(ROOT / DECODER))
    from public import Decoder, digest

    decoder = Decoder()
    search = SimpleNamespace(env=decoder.ac, library=decoder.library, subids=decoder.subs)
    old_prepare, enhanced_prepare, vector = preparation_functions()
    provider_type = reference_provider()
    adapter = PreparedPlans(decoder.library[0], decoder.ac)
    assert all(tuple(type(a).attr_list_vect) == FIELDS for a in search.library)
    public = read(OLD / "public_ranks.json")["rows"]
    assert len(public) == len(manifest["members"]) == 305
    rows = []
    counts = Counter()
    for index, (member, rank) in enumerate(zip(manifest["members"], public, strict=True)):
        assert member["call_id"] == rank["call_id"]
        assert time.monotonic() - start < LIMIT
        path = ROOT / member["source"]
        call = read(path)
        with np.load(ROOT / member["state"], allow_pickle=False) as state:
            obs = decoder.oc.from_vect(state["observation"])
            base = decoder.ac.from_vect(state["base_action"])
            assert np.array_equal(obs.to_vect(), state["observation"])
            assert np.array_equal(vector(base), state["base_action"])
        assert digest(obs.to_vect()) == call["state_hash"] == rank["state_hash"]
        if index % 2 == 0:
            baseline, order, baseline_s = old_path(old_prepare, search, obs, base)
            enhanced, new_order, arrays, modified_s, export_s = new_path(enhanced_prepare, search, obs, base, adapter)
        else:
            enhanced, new_order, arrays, modified_s, export_s = new_path(enhanced_prepare, search, obs, base, adapter)
            baseline, order, baseline_s = old_path(old_prepare, search, obs, base)
        eligible, reps, aliases, actions, changed = baseline
        assert order == new_order == call["order"] and order[:128] == rank["static128"]
        assert eligible == enhanced[0] == call["eligible"] and reps == enhanced[1] == call["representatives"]
        assert aliases == enhanced[2] == {int(k): v for k, v in call["aliases"].items()}
        assert changed == enhanced[4] == call["masked_ids"]
        plans, flags, unknown_or, unknown_ex = arrays
        oracle = provider_type(obs, base, actions, decoder.ac)
        assert [x["pool_id"] for x in call["outcomes"]] == order
        for i, ((key, cid), saved) in enumerate(zip(enhanced[-1].items(), call["outcomes"], strict=True)):
            assert cid == saved["pool_id"] and hashlib.sha256(key).hexdigest() == saved["combined_hash"]
            assert np.array_equal(vector(enhanced[3][cid]), vector(actions[cid]))
            assert np.array_equal(plans[i], oracle.plan(cid))
            unknown = set(np.r_[adapter.op[unknown_or[i]], adapter.ep[unknown_ex[i]]].tolist())
            assert unknown == oracle.unknown[cid] and bool(flags[i]) == bool(unknown)
        counts["candidates"] += len(order)
        counts["unknown"] += int(flags.sum())
        counts["obs_base_roundtrips"] += 2
        rows.append({**{k: member[k] for k in ("call_id", "partition", "date")},
                     "n": len(order), "unknown": int(flags.sum()), "baseline_s": baseline_s,
                     "modified_s": modified_s, "export_s": export_s,
                     "paired_total_delta_s": modified_s + export_s - baseline_s})
    assert len(rows) == 305 and counts["candidates"] == 268356 and counts["unknown"] == 67658
    assert time.monotonic() - start < LIMIT
    old_summary = read(FAST / "results/summary.json")
    assert old_summary["qualified"]
    metrics = {}
    for part in ("train", "validation"):
        split = [r for r in rows if r["partition"] == part]
        result = {field: timing([r[field] for r in split])
                  for field in ("baseline_s", "modified_s", "export_s", "paired_total_delta_s")}
        proxy = old_summary["partition_timing"][part]
        result["accounting"] = {
            "calls": len(split),
            "export_only_plus_previous_net_s": proxy["estimated_net_delta_s"] + result["export_s"]["total_s"],
            "paired_delta_plus_previous_net_s": proxy["estimated_net_delta_s"] + result["paired_total_delta_s"]["total_s"],
            "export_only_net_by_date_s": {day: stats["estimated_net_delta_s"] +
                sum(r["export_s"] for r in split if r["date"] == day)
                for day, stats in proxy["by_date"].items()},
        }
        metrics[part] = result
    save(OUT / "rows.json", rows)
    save(OUT / "summary.json", {"qualified": True, "calls": len(rows), "counts": dict(counts),
         "metrics": metrics, "wall_s": time.monotonic() - start,
         "scope_sha256": sha(ARCHIVE / "fixed_scope.json"), "new_dc_solves": 0,
         "new_simulate": 0, "new_model_forward": 0, "new_fits": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    {"freeze": freeze, "evaluate": evaluate}[parser.parse_args().stage]()

