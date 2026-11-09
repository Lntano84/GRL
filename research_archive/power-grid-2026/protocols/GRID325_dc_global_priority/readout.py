"""GRID325: one fixed DC priority on the complete saved GRID139 population."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[5]
OUT = ROOT / "outputs/grid325_dc_global_priority"
LEDGER = "outputs/grid139_global_feasibility/call_ledger.json"
ORIGINAL = "outputs/grid139_global_feasibility/source_hashes.json"
ENV = "work/grid100_data_foundation/trainval_env"
DESIGN = "outputs/grid100_data_foundation/design.json"
DECODER = "work/grid111_action_feedback"
BUDGET = 128
CAP = 600
BASELINE = "STATIC_ID"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, payload):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def freeze():
    prior = read(ROOT / ORIGINAL)
    original_design = read(ROOT / DESIGN)
    assert original_design["environment"] == ENV
    for name in ("grid.json", "config.py"):
        assert sha(ROOT / ENV / name) == original_design["static_source_sha256"][name]
    members = []
    paths = {LEDGER, ORIGINAL, DESIGN, f"{ENV}/grid.json", f"{ENV}/config.py",
             f"{DECODER}/public.py", f"{DECODER}/common111.py",
             "outputs/grid28/offline.json",
             "outputs/grid98_feature_enrichment/template_metadata.json",
             "work/grid02/ljn_nn/assets/action_N1_unsafe.npz",
             "outputs/grid03/scoring_smoke/agent_episode/dict_action_space.json",
             "outputs/grid03/scoring_smoke/agent_episode/dict_observation_space.json"}
    fields = ("call_id", "source", "partition", "date", "scenario", "step", "M", "teacher_complete")
    for row in read(ROOT / LEDGER):
        source = row["source"]
        assert sha(ROOT / source) == prior[source], source
        state = str(Path(source).with_name(Path(source).stem + "_state.npz")).replace("\\", "/")
        paths.update((source, state, str(Path(source).with_name("metadata.json")).replace("\\", "/")))
        members.append({**{key: row[key] for key in fields}, "state": state})
    assert Counter(row["partition"] for row in members) == {"train": 243, "validation": 62}
    assert len({row["call_id"] for row in members}) == 305
    assert sum(row["M"] for row in members) == 268356
    dates = {part: sorted({row["date"] for row in members if row["partition"] == part})
             for part in ("train", "validation")}
    assert len(dates["train"]) == 16 and len(dates["validation"]) == 4
    assert not set(dates["train"]) & set(dates["validation"])
    save(ARCHIVE / "fixed_scope.json", {
        "members": members, "dates": dates, "source_sha256": sha(Path(__file__)),
        "source_hashes": {p: sha(ROOT / p) for p in sorted(paths)},
        "comparison": "Same full GRID139 STATIC_ID global action pool and the first min(128,M) paid feedback records per arm. No extra action, query, re-lock, or alternative stopping decisions.",
        "proxy": "Solve candidate-conditioned DC balance on post-command busbars, with frozen current public generation, load, storage and shunt active power. Balance losses at highest currently injecting generator (first id on ties). Minimize max absolute line/transformer MW divided by nominal origin kV * sqrt(3) * configured current limit kA. Effective 1/x divided by tap. Same-score ties keep STATIC_ID.",
        "fallback": "Unknown reclose terminal bus, disconnected or singular grid, or unsupported active-power command yields no proxy. Keep unsupported candidates in original slots; sort only supported candidates across supported slots. No candidate deletion.",
        "references": "Report original FRAME choice and legal FULL best separately. Strict-admissible feedback only, float32 score then lower pool id. Report retained/lost legal action separately from numeric regret.",
        "decision": "Offline lead only when validation wins (including legal-action recoveries) exceed losses, wins cover >=2 distinct validation dates, and no validation date has more losses than wins. Retain all 20 dates and both scenario streams. No fitted variant, score/threshold sweep, or follow-up experiment inside this scope.",
        "timing": "Report whole local ranking pass and candidate numerical+ranking time per call. Archive verification, decoding/integrity checks and elapsed seconds separate from DC numerical work. No online control time claim.",
        "per_stage_cap_s": CAP,
        "new_model_forwards": 0, "new_simulate_calls": 0, "new_fits": 0,
    })


def setup():
    scope = read(ARCHIVE / "fixed_scope.json")
    assert sha(Path(__file__)) == scope["source_sha256"]
    for path, expected in scope["source_hashes"].items():
        assert sha(ROOT / path) == expected, path
    return scope


def static_parameters(ac):
    import numpy as np

    grid = read(ROOT / ENV / "grid.json")["_object"]

    def table(name):
        data = json.loads(grid[name]["_object"])
        return {int(i): dict(zip(data["columns"], row, strict=True))
                for i, row in zip(data["index"], data["data"], strict=True)}

    for field in ("switch", "ext_grid", "trafo3w", "impedance", "dcline"):
        assert not table(field), field
    buses = table("bus")
    assert len(buses) == ac.n_sub
    assert set(buses) == set(range(ac.n_sub))
    branch_data = {}
    for row in table("line").values():
        origin, end = row["from_bus"], row["to_bus"]
        kv = buses[origin]["vn_kv"]
        assert kv == buses[end]["vn_kv"] and row["parallel"] == 1 and row["in_service"]
        x = row["x_ohm_per_km"] * row["length_km"] / kv ** 2
        branch_data[row["name"]] = (origin, end, kv, 1 / x)
    for row in table("trafo").values():
        origin, end = row["hv_bus"], row["lv_bus"]
        kv = buses[origin]["vn_kv"]
        assert (kv == row["vn_hv_kv"] and buses[end]["vn_kv"] == row["vn_lv_kv"]
                and row["parallel"] == 1 and row["in_service"]
                and row["shift_degree"] == row["tap_step_degree"] == 0
                and not row["tap_phase_shifter"] and (row["tap_side"] in ("hv", "lv") or row["tap_step_percent"] == 0))
        x = (row["vk_percent"] ** 2 - row["vkr_percent"] ** 2) ** .5 / (100 * row["sn_mva"])
        tap = 1 + (row["tap_pos"] - row["tap_neutral"]) * row["tap_step_percent"] / 100
        assert tap > 0
        branch_data[row["name"]] = (origin, end, kv, (1 / x) / (tap if row["tap_side"] == "hv" else 1 / tap))
    assert len(branch_data) == ac.n_line and set(branch_data) == set(ac.name_line)
    a, b, kv, weights = map(np.asarray, zip(*(branch_data[name] for name in ac.name_line), strict=True))
    assert np.array_equal(a, ac.line_or_to_subid) and np.array_equal(b, ac.line_ex_to_subid)
    source = ast.parse((ROOT / ENV / "config.py").read_text(encoding="utf8"))
    assignment, = [n for n in source.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "th_lim" for t in n.targets)]
    limits = np.asarray(ast.literal_eval(assignment.value), dtype=float) * (3 ** .5 / 1000) * kv
    assert len(limits) == ac.n_line and np.isfinite(limits).all() and (limits > 0).all()
    assert np.isfinite(weights).all() and (weights > 0).all()
    return weights, limits


def dc_max(origin, end, weights, limits, injection, slack):
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import splu

    n = len(injection)
    matrix = coo_matrix((np.r_[weights, weights, -weights, -weights],
                         (np.r_[origin, end, origin, end], np.r_[origin, end, end, origin])),
                        shape=(n, n)).tocsc()
    if n < 2 or connected_components(matrix, directed=False, return_labels=False) != 1:
        return None, "disconnected_dc_grid"
    kept = np.arange(n) != slack
    reduced = matrix[kept, :][:, kept].tocsc()
    try:
        theta = np.zeros(n)
        theta[kept] = splu(reduced).solve(injection[kept])
    except RuntimeError:
        return None, "singular_dc_grid"
    error = np.max(np.abs((reduced @ theta[kept]) - injection[kept]))
    flows = weights * (theta[origin] - theta[end])
    score = np.max(np.abs(flows) / limits)
    if not np.isfinite(score) or error > 1e-6 * max(1, np.max(np.abs(injection))):
        return None, "unreliable_dc_grid"
    return float(score), None


def check_solver():
    import numpy as np

    a, b = np.array([0, 1, 0]), np.array([1, 2, 2])
    values = np.ones(3)
    score, reason = dc_max(a, b, values, values, np.array([1., 0., -1.]), 0)
    assert reason is None and abs(score - 2 / 3) < 1e-12
    score, reason = dc_max(a[:1], b[:1], values[:1], values[:1], values, 0)
    assert score is None and reason == "disconnected_dc_grid"


class PublicGrid:
    def __init__(self, action_space, weights, limits):
        import numpy as np

        self.ac = action_space
        self.weights, self.limits = weights, limits
        self.subs = np.repeat(np.arange(self.ac.n_sub), self.ac.sub_info)

    def state_inputs(self, ob):
        import numpy as np

        ac = self.ac
        injection = np.zeros(len(self.subs))
        injection[ac.gen_pos_topo_vect] = ob.gen_p
        injection[ac.load_pos_topo_vect] = -ob.load_p
        injection[ac.storage_pos_topo_vect] = -ob.storage_power
        slack = ac.gen_pos_topo_vect[int(np.argmax(ob.gen_p))]
        shunt = np.zeros(2 * ac.n_sub)
        shunt_live = ob._shunt_bus > 0
        assert (ob._shunt_bus[shunt_live] <= 2).all()
        shunt_nodes = ac.shunt_to_subid[shunt_live] + ac.n_sub * (ob._shunt_bus[shunt_live] - 1)
        np.add.at(shunt, shunt_nodes, -ob._shunt_p[shunt_live])
        assert np.isfinite(injection).all() and np.isfinite(shunt).all()
        return injection, slack, shunt

    def score(self, plan, unknown, injection, slack, shunt):
        import numpy as np

        if unknown:
            return None, "unknown_reclose_bus"
        if plan[slack] <= 0 or ((plan <= 0) & (injection != 0)).any():
            return None, "disconnected_injection"
        ac = self.ac
        nodes = self.subs + ac.n_sub * (plan - 1)
        a, b = nodes[ac.line_or_pos_topo_vect], nodes[ac.line_ex_pos_topo_vect]
        live = (a >= 0) & (b >= 0)
        a, b = a[live], b[live]
        attached = plan > 0
        p = np.bincount(nodes[attached], weights=injection[attached], minlength=len(shunt)) + shunt
        active = np.unique(np.r_[a, b, nodes[slack], np.flatnonzero(p)])
        lookup = np.full(len(shunt), -1, dtype=int)
        lookup[active] = np.arange(len(active))
        return dc_max(lookup[a], lookup[b], self.weights[live], self.limits[live], p[active], lookup[nodes[slack]])


def supported_commands(action):
    import numpy as np

    return (not np.any(action._redispatch) and not np.any(action._storage_power)
            and np.all(action._curtail == -1))


def rank():
    start = time.monotonic()
    scope = setup()
    sys.path.insert(0, str(ROOT / DECODER))
    from public import Decoder
    import numpy as np
    import scipy
    import grid2op

    check_solver()
    decoder = Decoder()
    assert all(supported_commands(action) for action in decoder.library)
    grid = PublicGrid(decoder.ac, *static_parameters(decoder.ac))
    out = []
    for member in scope["members"]:
        assert time.monotonic() - start < CAP
        before = time.perf_counter()
        call, ob, plans, unknown, _ = decoder.decode(member["source"])
        order = call["order"]
        assert call["order_mode"] == BASELINE and len(order) == member["M"] and set(order) == set(plans)
        assert decoder.checks["combined_hashes"] == sum(r["M"] for r in out) + member["M"]
        with np.load(ROOT / member["state"], allow_pickle=False) as state:
            base = decoder.ac.from_vect(state["base_action"])
        decode_s = time.perf_counter() - before

        before = time.perf_counter()
        supported = []
        fallback = Counter()
        score_count = 0
        if supported_commands(base):
            inputs = grid.state_inputs(ob)
            for slot, cid in enumerate(order):
                score, reason = grid.score(plans[cid], unknown[cid], *inputs)
                if reason is None:
                    supported.append((slot, cid, score))
                else:
                    fallback[reason] += 1
            score_count = len(supported)
        else:
            fallback["unsupported_active_power_command"] = len(order)
        ranked = order.copy()
        for slot_row, score_row in zip(supported, sorted(supported, key=lambda row: row[2]), strict=True):
            ranked[slot_row[0]] = score_row[1]
        assert len(set(ranked)) == len(order) == score_count + sum(fallback.values())
        score_s = time.perf_counter() - before
        assert time.monotonic() - start < CAP
        out.append({"call_id": member["call_id"], "M": len(order), "state_hash": call["state_hash"],
                    "static128": order[:BUDGET], "dc128": ranked[:BUDGET],
                    "scored": score_count, "fallback": dict(fallback),
                    "decode_integrity_s": decode_s, "dc_and_sort_s": score_s})
    save(ARCHIVE / "public_ranks.json", {"scope_sha256": sha(ARCHIVE / "fixed_scope.json"),
         "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                      "scipy": scipy.__version__, "grid2op": grid2op.__version__},
         "checks": dict(decoder.checks), "elapsed_s": time.monotonic() - start, "rows": out})


def reward(row):
    return None if row is None else struct.unpack("f", struct.pack("f", row["reward"]))[0]


def choice(row):
    return None if row is None else row["pool_id"]


def best(rows):
    return max((r for r in rows if r["strict_admissible"]),
               key=lambda r: (reward(r), -choice(r)), default=None)


def compare(left, right):
    if left is None or right is None:
        return (left is not None) - (right is not None)
    return (left > right) - (left < right)


def summarize(rows):
    return {
        "calls": len(rows), "candidate_queries_each": sum(row["budget"] for row in rows),
        "changed_prefix_sets": sum(row["changed_prefix_set"] for row in rows),
        "wins": sum(row["comparison"] > 0 for row in rows),
        "ties": sum(row["comparison"] == 0 for row in rows),
        "losses": sum(row["comparison"] < 0 for row in rows),
        "scored": sum(row["scored"] for row in rows),
        "fallback": dict(sum((Counter(r["fallback"]) for r in rows), Counter())),
        **{arm: {
            "frame_action_match": sum(r[arm]["frame_action_match"] for r in rows),
            "frame_winner_in_prefix": sum(r[arm]["frame_winner_in_prefix"] for r in rows),
            "full_action_match": sum(r[arm]["full_action_match"] for r in rows),
            "no_legal_with_full_legal": sum(r[arm]["no_legal_with_full_legal"] for r in rows),
            "positive_full_regret": sum((r[arm]["full_regret"] or 0) > 0 for r in rows),
            "mean_comparable_full_regret": statistics.mean(values) if
                (values := [r[arm]["full_regret"] for r in rows if r[arm]["full_regret"] is not None])
                else None,
        } for arm in ("static", "dc")},
        "dc_and_sort_total_s": sum(row["dc_and_sort_s"] for row in rows),
        "dc_and_sort_median_s": statistics.median(row["dc_and_sort_s"] for row in rows),
    }


def evaluate():
    start = time.monotonic()
    scope = setup()
    ranks = read(ARCHIVE / "public_ranks.json")
    assert ranks["scope_sha256"] == sha(ARCHIVE / "fixed_scope.json")
    ledger = {r["call_id"]: r for r in read(ROOT / LEDGER)}
    out = []
    for member, public in zip(scope["members"], ranks["rows"], strict=True):
        assert time.monotonic() - start < CAP
        identity = member["call_id"]
        assert identity == public["call_id"]
        call = read(ROOT / member["source"])
        assert call["state_hash"] == public["state_hash"]
        feedback = {choice(row): row for row in call["outcomes"]}
        assert list(feedback) == call["order"]
        full = best(feedback.values())
        frame = ledger[identity]["frame_choice"]
        assert choice(full) == ledger[identity]["full_choice"]
        assert frame is None or feedback[frame]["strict_admissible"]
        n = min(BUDGET, member["M"])
        row = {key: member[key] for key in ("call_id", "partition", "date", "scenario", "step")}
        row.update(budget=n, frame=frame, full=choice(full), scored=public["scored"],
                   fallback=public["fallback"], dc_and_sort_s=public["dc_and_sort_s"],
                   changed_prefix_set=set(public["static128"]) != set(public["dc128"]))
        for arm in ("static", "dc"):
            prefix = public[arm + "128"]
            assert len(prefix) == len(set(prefix)) == n and set(prefix) <= feedback.keys()
            if arm == "static":
                assert prefix == call["order"][:n]
            picked = best(feedback[i] for i in prefix)
            row[arm] = {"winner": choice(picked), "reward": reward(picked),
                        "frame_action_match": choice(picked) == frame and frame is not None,
                        "frame_winner_in_prefix": frame in prefix,
                        "full_action_match": choice(picked) == choice(full) and full is not None,
                        "no_legal_with_full_legal": picked is None and full is not None,
                        "full_regret": None if picked is None else reward(full) - reward(picked)}
        row["comparison"] = compare(row["dc"]["reward"], row["static"]["reward"])
        out.append(row)
    splits = {}
    for part, dates in scope["dates"].items():
        rows = [r for r in out if r["partition"] == part]
        splits[part] = {"all": summarize(rows),
                        "by_date": {date: summarize([r for r in rows if r["date"] == date]) for date in dates}}
    validation = splits["validation"]
    days = list(validation["by_date"].values())
    keep = (validation["all"]["wins"] > validation["all"]["losses"]
            and sum(day["wins"] > 0 for day in days) >= 2
            and all(day["wins"] >= day["losses"] for day in days))
    save(OUT / "rows.json", out)
    save(OUT / "summary.json", {"metrics": splits,
         "scope_sha256": sha(ARCHIVE / "fixed_scope.json"),
         "public_ranks_sha256": sha(ARCHIVE / "public_ranks.json"),
         "decision": "OFFLINE_LEAD_ONLY" if keep else "CLOSE_FIXED_DC_PRIORITY",
         "new_model_forwards": 0, "new_simulate_calls": 0, "new_fits": 0,
         "rank_elapsed_s": ranks["elapsed_s"], "evaluate_elapsed_s": time.monotonic() - start})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "rank", "evaluate"))
    stage = parser.parse_args().stage
    {"freeze": freeze, "rank": rank, "evaluate": evaluate}[stage]()

