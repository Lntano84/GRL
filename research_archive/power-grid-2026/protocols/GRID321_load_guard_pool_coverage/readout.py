"""Frozen-trajectory coverage of the already-fixed GRID320 load guard."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import time

R = Path("D:/Codex/GRL研究")
A = Path(__file__).resolve().parent
I = R / "work/GRL-results-integration"
O = R / "outputs/grid321_load_guard_pool_coverage"
OLD = R / "outputs/grid291_control_preparation"
REF = R / "outputs/grid320_load_guard_closed_loop"
AR291 = A.parent / "GRID291_control_preparation/budget_limited_results"
AR320 = A.parent / "GRID320_load_guard_closed_loop"
MAN291 = R / "archives/GRID291_control_replay_20261101/raw_records/records_manifest.json"
CAP = 600
CUTOFF = .9
PUBLIC = ("eligible", "masked_ids", "order", "aliases", "representatives", "rho_limit",
          "completion_candidate_ids", "local_substation", "parameters")
FEEDBACK = ("pool_id", "rho", "reward", "done", "illegal", "ambiguous", "exceptions",
            "strict_admissible", "phase", "combined_hash")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, obj):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def path_key(path):
    return path.relative_to(R).as_posix()


def tracked(path, head):
    data = subprocess.check_output(["git", "-C", str(I), "show", head + ":" + path.relative_to(I).as_posix()])
    assert data == path.read_bytes(), path


def members(root, rows, design, refs):
    by_folder = defaultdict(list)
    for path, digest in sorted(refs.items()):
        if re.fullmatch(r"call[0-9]{3,}\.json", Path(path).name):
            state = path[:-5] + "_state.npz"
            assert (R / path).is_file() and (R / state).is_file()
            by_folder[Path(path).parent.as_posix()].append(
                {"call": path, "call_hash": digest, "state": state, "state_hash": refs[state]})
    entries = []
    for row in rows:
        item = design[row["index"]]
        folder = root / "runs" / f"r{item['repeat']}__{item['week']}__{item['method']}"
        assert all(item[k] == row[k] for k in ("week", "pool", "arm", "repeat"))
        entries.append({**{k: row[k] for k in ("index", "week", "arm", "pool", "repeat", "candidate_queries")},
                        "calls": by_folder[path_key(folder)]})
    return entries


def freeze():
    old_results = AR291 / "saved_partial_review.json"
    decisions = AR291 / "saved_decisions.json"
    ref_results = AR320 / "results/closed_loop/paired_results.json"
    ref_manifest = AR320 / "results/closed_loop/raw_records_manifest.json"
    head = subprocess.check_output(["git", "-C", str(I), "rev-parse", "HEAD"]).decode().strip()
    tracked_files = (old_results, decisions, ref_results, ref_manifest, AR320 / "design.json",
                     AR320 / "runtime.py", AR291 / "raw_archive_review.json")
    for path in tracked_files:
        tracked(path, head)
    old = read(old_results)
    assert old["passed_saved_data"] and old["accepted_runs"] == 186
    ref = read(ref_results)["records"]
    assert len(ref) == 24
    scope_ref = read(AR320 / "design.json")
    assert scope_ref["load_guard_threshold"] == CUTOFF
    old_refs = {path_key(OLD / path): row["sha256"] for path, row in read(MAN291)["files"].items()}
    ref_refs = {row["path"]: row["sha256"] for row in read(ref_manifest)["entries"]}
    old_design = read(OLD / "design.json")
    old_design_archived = A.parent / "GRID291_control_preparation/source_launch_recovery02/design.json"
    tracked(old_design_archived, head)
    assert (OLD / "design.json").read_bytes() == old_design_archived.read_bytes()
    old_runs = members(OLD, old["records"], old_design["execution_order"], old_refs)
    ref_runs = members(REF, [r for r in ref if r["arm"] == "FRAME_GUARD"],
                       scope_ref["execution_order"], ref_refs)
    assert len(old_runs) == 186 and len(ref_runs) == 8
    assert len({r["pool"] for r in old_runs}) == 12
    save(A / "fixed_scope.json", {
        "learned_members": old_runs, "frame_members": ref_runs,
        "source_hashes": {path_key(p): sha(p.read_bytes())
                          for p in (*tracked_files, MAN291, OLD / "design.json", old_design_archived)},
        "rule": "For every archived final learned STOP with pending candidates, use the saved strict-admissible incumbent forecast rho strictly greater than 0.9.",
        "matching": "Match FRAME on date, step, exact observation and base action. Require all public setup, identical paid prefix feedback and exact unpaid locked-station suffix. Both FRAME repeats must agree if both are eligible. No extrapolation on unmatched rows.",
        "interpretation": "Sum pending queries only at original saved states. Tabulate action changes only in exactly matched cases. Keep repeats, KEEP, all dates, and unavailable comparisons. This does not execute a policy or estimate whole-week counterfactuals.",
        "limits": {"wall_s": CAP, "new_fits": 0, "new_predictions": 0, "new_physics": 0}
    })


def verified(path, expected):
    data = (R / path).read_bytes()
    assert sha(data) == expected, path
    return data


def call_and_state(entry, call):
    import io
    import numpy as np
    with np.load(io.BytesIO(verified(entry["state"], entry["state_hash"])), allow_pickle=False) as z:
        ob = sha(z["observation"].tobytes())
        base = sha(z["base_action"].tobytes())
    assert ob == call["state_hash"]
    return ob, base


def reward(row):
    return struct.unpack("f", struct.pack("f", row["reward"]))[0]


def best(rows):
    return max((r for r in rows if r["strict_admissible"]), key=lambda r: (reward(r), -r["pool_id"]))


def compare(call, frame, tail):
    if any(call[k] != frame[k] for k in PUBLIC):
        return "public_setup_differs", None
    prior = call["outcomes"]
    other = frame["outcomes"]
    if [r["pool_id"] for r in prior] != [r["pool_id"] for r in other[:len(prior)]]:
        return "paid_prefix_differs", None
    if any(any(a[k] != b[k] for k in FEEDBACK) for a, b in zip(prior, other)):
        return "paid_feedback_differs", None
    suffix = other[len(prior):]
    if [r["pool_id"] for r in suffix] != tail or not all(r["phase"] == "local_completion" for r in suffix):
        return "unpaid_suffix_differs", None
    chosen = best(other)
    assert chosen["pool_id"] == frame["chosen_pool_id"]
    return "matched", chosen


def aggregate(rows):
    triggered = [r for r in rows if r["triggered"]]
    matched = [r for r in rows if r["match"] == "matched"]
    covered = [r for r in matched if r["triggered"]]
    return {
        "stops": len(rows), "triggered": len(triggered),
        "pending_if_applied_to_saved_states": sum(r["remaining"] for r in triggered),
        "matched": len(matched), "matched_triggers": len(covered),
        "matched_trigger_pending": sum(r["remaining"] for r in covered),
        "matched_trigger_winner_changes": sum(r["original_winner"] != r["frame_winner"] for r in covered),
        "matched_trigger_strict_gains": sum(r["frame_reward_gain"] > 0 for r in covered),
        "matched_nontrigger_misses": sum(r["original_winner"] != r["frame_winner"] and not r["triggered"] for r in matched),
        "triggered_residual_stops": sum(r["parent_cont"] for r in triggered),
        "matching_status": dict(Counter(r["match"] for r in rows))
    }


def audit():
    start = time.monotonic()
    assert not O.exists()
    scope = read(A / "fixed_scope.json")
    for path, digest in scope["source_hashes"].items():
        verified(path, digest)
    frames = defaultdict(list)
    frame_steps = set()
    for run in scope["frame_members"]:
        total = 0
        for entry in run["calls"]:
            c = json.loads(verified(entry["call"], entry["call_hash"]))
            total += c["public_queries"]
            ob, base = call_and_state(entry, c)
            frames[run["week"], c["step"], ob, base].append((run["repeat"], c))
            frame_steps.add((run["week"], c["step"]))
        assert total == run["candidate_queries"]
    rows = []
    for run in scope["learned_members"]:
        assert time.monotonic() - start < CAP
        total = 0
        for entry in run["calls"]:
            c = json.loads(verified(entry["call"], entry["call_hash"]))
            total += c["public_queries"]
            d = c["learned_decisions"][-1] if c["learned_decisions"] else None
            if d is None or not d["optional_extension"] or d["action"]:
                continue
            outcomes = c["outcomes"]
            assert d["paid_queries"] == len(outcomes) == c["public_queries"]
            inc = best(outcomes)
            assert inc["pool_id"] == c["chosen_pool_id"]
            seen = {r["pool_id"] for r in outcomes}
            tail = [i for i in c["completion_candidate_ids"] if i not in seen]
            assert len(tail) == d["remaining"] > 0 and d["next_ids"] == tail[:64]
            ob, base = call_and_state(entry, c)
            key = (run["week"], c["step"], ob, base)
            refs = frames.get(key, [])
            comparisons = [(repeat, *compare(c, frame, tail)) for repeat, frame in refs]
            winners = [(repeat, winner) for repeat, status, winner in comparisons if status == "matched"]
            status = ("matched" if winners else comparisons[0][1] if comparisons else
                      "observation_or_base_differs" if key[:2] in frame_steps else "no_frame_call_at_step")
            if winners:
                _, winner = winners[0]
                assert all(other["pool_id"] == winner["pool_id"] and reward(other) == reward(winner)
                           for _, other in winners)
            else:
                winner = None
            rows.append({**{k: run[k] for k in ("index", "week", "arm", "pool", "repeat")},
                         "step": c["step"], "state_hash": ob, "base_hash": base, "original_winner": inc["pool_id"],
                         "incumbent_rho": inc["rho"], "parent_cont": d["parent_CONT"], "batch": d["batch"],
                         "remaining": len(tail), "triggered": inc["rho"] > CUTOFF,
                         "source_call": entry["call"], "match": status,
                         "ref_repeat_status": {repeat: status for repeat, status, _ in comparisons},
                         "frame_winner": None if winner is None else winner["pool_id"],
                         "frame_reward_gain": None if winner is None else reward(winner) - reward(inc)})
        assert total == run["candidate_queries"]
    saved = read(AR291 / "saved_decisions.json")
    key = lambda x: (x["index"], x["step"], x["batch"], x["parent_cont"])
    assert Counter(key(r) for r in rows) == Counter(key({**d, "parent_cont": d["parent_CONT"]})
        for d in saved if d["optional_extension"] and d["action"] == 0)
    summary = {}
    for arm in sorted({r["arm"] for r in rows}):
        arm_rows = [r for r in rows if r["arm"] == arm]
        summary[arm] = {"all": aggregate(arm_rows),
                        "balanced_repeat0": aggregate([r for r in arm_rows if r["repeat"] == 0]),
                        "by_pool": {pool: aggregate([r for r in arm_rows if r["pool"] == pool])
                                    for pool in sorted({r["pool"] for r in scope["learned_members"]})},
                        "by_week": {week: aggregate([r for r in arm_rows if r["week"] == week])
                                    for week in sorted({r["week"] for r in scope["learned_members"]})}}
    O.mkdir(exist_ok=False)
    save(O / "rows.json", rows)
    save(O / "summary.json", {"by_arm": summary, "accepted_learned_runs": len(scope["learned_members"]),
                             "frame_runs": len(scope["frame_members"]), "elapsed_s": time.monotonic()-start,
                             "new_physics": 0, "new_fits": 0, "new_predictions": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "audit"))
    {"freeze": freeze, "audit": audit}[parser.parse_args().stage]()
