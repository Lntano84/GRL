"""Check whether saved winning forecasts and the immediate caller use the same action."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time

R = Path("D:/Codex/GRL研究")
A = Path(__file__).resolve().parent
O = R / "outputs/grid318_forecast_handoff"
ARCHIVED291 = R / "archives/GRID291_control_replay_20261101/raw_records/records_manifest.json"
G291 = R / "outputs/grid291_control_preparation"
G312 = R / "outputs/grid312_frame_208_261_pilot"
SOURCE312 = A.parent / "GRID312_frame_208_261_pilot/runtime.py"
CAP = 600


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def ref(path):
    return path.relative_to(R).as_posix()


def freeze():
    review291 = G291 / "saved_partial_review.json"
    review312 = G312 / "paired_results.json"
    manifest312 = G312 / "raw_records_manifest.json"
    refs291 = {ref(G291 / k): v["sha256"] for k, v in read(ARCHIVED291)["files"].items()}
    refs312 = {v["path"]: v["sha256"] for v in read(manifest312)["entries"]}
    sources = [
        ("GRID291", G291, review291, refs291, "total_queries"),
        ("GRID312", G312, review312, refs312, "total_forecasts"),
    ]
    runs = []
    for stage, root, review, refs, forecast_count in sources:
        members = defaultdict(list)
        for key in sorted(refs):
            if re.fullmatch(r"call[0-9]{3,}\.json", Path(key).name):
                members[Path(key).parent.as_posix()].append(key)
        for row in read(review)["records"]:
            method = row["arm"] + ("" if row["arm"] == "FRAME_GUARD" else "__" + row["pool"])
            folder = ref(root / "runs" / f"r{row['repeat']}__{row['week']}__{method}")
            calls = []
            for path in members[folder]:
                state = path[:-5] + "_state.npz"
                assert (R / path).is_file() and (R / state).is_file()
                calls.append(dict(path=path, sha256=refs[path], state=state, state_sha256=refs[state]))
            runs.append(dict(stage=stage, folder=folder, total_forecasts=row[forecast_count],
                             **{k: row[k] for k in ("index", "arm", "pool", "repeat", "week", "steps",
                                                    "complete", "controller_s", "candidate_queries")},
                             calls=calls))
    assert len(runs) == 210 and sum(r["stage"] == "GRID291" for r in runs) == 186
    deps = [review291, review312, ARCHIVED291, manifest312, SOURCE312, G291 / "code_freeze.json",
            G312 / "code_freeze.json", R / "work/grid291_control_preparation/adapter.py",
            R / "work/grid291_control_preparation/source_builder.py", R / "work/grid55/worker.py",
            R / "work/grid08/common.py", R / "work/grid02/ljn_nn/LJNAgent.py",
            R / "work/grid26/engine.py", R / "work/grid01/trace_agent.py",
            R / "work/grid111_action_feedback/public.py", R / "work/grid111_action_feedback/common111.py",
            R / "outputs/grid03/scoring_smoke/agent_episode/dict_action_space.json",
            R / "outputs/grid03/scoring_smoke/agent_episode/dict_observation_space.json",
            R / "outputs/grid28/offline.json", R / "outputs/grid98_feature_enrichment/template_metadata.json",
            R / "work/grid02/ljn_nn/assets/action_N1_unsafe.npz"]
    save(A / "fixed_scope.json", dict(
        runs=runs, source_hashes={ref(p): sha(p) for p in deps},
        rule="All accepted GRID291 and GRID312 trajectories. For each paid selected action, compare candidate+base with original caller base.copy(); base+=candidate. Count at most the one immediate next-step check. No reuse across time steps or controllers, no subsequent optimizer check.",
        outcome="Prove saved vector identity only. Forecast-count opportunity ceiling, with time of the already-paid winning forecast as a rough reference, not the unrecorded duplicate duration or measured speedup. Keep arms separate.",
        limits=dict(wall_s=CAP, new_predictions=0, new_physics=0, new_fits=0)))


def audit():
    start = time.monotonic()
    scope = read(A / "fixed_scope.json")
    for path, expected in scope["source_hashes"].items():
        assert sha(R / path) == expected, path
    sys.path.insert(0, str(R / "work/grid111_action_feedback"))
    import numpy as np
    from public import Decoder, av, digest
    d = Decoder()
    rows, run_rows = [], []
    for run in scope["runs"]:
        assert time.monotonic() - start < CAP
        candidate_queries = 0
        run_rows_start = len(rows)
        for member in run["calls"]:
            path = R / member["path"]
            assert sha(path) == member["sha256"]
            call = read(path)
            candidate_queries += call["public_queries"]
            selected = call["chosen_pool_id"]
            row = {k: run[k] for k in ("stage", "arm", "week", "repeat", "pool", "index")}
            row.update(path=member["path"], step=call["step"], selected=selected,
                       paid_winner_s=None, exact_vector_match=False)
            if selected is not None:
                assert selected in call["representatives"] and selected in call["eligible"]
                paid, = [x for x in call["outcomes"] if x["pool_id"] == selected]
                assert paid["strict_admissible"]
                state = R / member["state"]
                assert sha(state) == member["state_sha256"]
                with np.load(state, allow_pickle=False) as data:
                    ob = d.oc.from_vect(data["observation"])
                    base = d.ac.from_vect(data["base_action"])
                    assert digest(ob.to_vect()) == call["state_hash"]
                    assert av(base).tobytes() == data["base_action"].tobytes()
                ac = d.ac
                op, ep = ac.line_or_pos_topo_vect, ac.line_ex_pos_topo_vect
                intent = (base.line_set_status > 0) | base.line_change_status
                intent |= (base.set_bus[op] > 0) | (base.set_bus[ep] > 0)
                disconnected = ~ob.line_status & ~intent
                positions = np.r_[op[disconnected], ep[disconnected]]
                action = d.library[selected].copy()
                buses = action.set_bus.copy()
                changed = positions[buses[positions] != 0]
                assert bool(len(changed)) == (selected in call["masked_ids"])
                if len(changed):
                    buses[changed] = 0
                    action.set_bus = buses
                assert digest(av(action)) == call["returned_action_hash"]
                paid_vector = av(action + base)
                assert digest(paid_vector) == paid["combined_hash"]
                delivered = base.copy()
                delivered += action
                delivered_vector = av(delivered)
                row.update(paid_winner_s=paid["forecast_s"], exact_vector_match=(
                    delivered_vector.tobytes() == paid_vector.tobytes()
                ), paid_hash=paid["combined_hash"], immediate_check_action_hash=digest(delivered_vector),
                           winner_rho=paid["rho"])
            rows.append(row)
        assert candidate_queries == run["candidate_queries"]
        group = rows[run_rows_start:]
        exact = [r for r in group if r["exact_vector_match"]]
        run_rows.append({**{k: v for k, v in run.items() if k != "calls"},
                         "calls": len(group), "selected": sum(r["selected"] is not None for r in group),
                         "exact_vector_matches": len(exact),
                         "reference_winner_forecast_s": sum(r["paid_winner_s"] for r in exact)})
    grouped = defaultdict(list)
    for run in run_rows:
        grouped[(run["stage"], run["arm"])].append(run)
    totals = {}
    for (stage, arm), group in grouped.items():
        t = {k: sum(r[k] for r in group) for k in ("calls", "selected", "exact_vector_matches",
                                                   "total_forecasts", "controller_s", "reference_winner_forecast_s")}
        totals.setdefault(stage, {})[arm] = dict(
            runs=len(group), **t,
            forecast_share_pct=100 * t["exact_vector_matches"] / t["total_forecasts"],
            reference_winner_controller_share_pct=100 * t["reference_winner_forecast_s"] / t["controller_s"],
        )
    O.mkdir(exist_ok=False)
    save(O / "calls.json", rows)
    save(O / "runs.json", run_rows)
    save(O / "summary.json", dict(totals=totals, elapsed_s=time.monotonic()-start,
                                   new_physics=0, new_predictions=0, new_fits=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "audit"))
    {"freeze": freeze, "audit": audit}[parser.parse_args().stage]()
