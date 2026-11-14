"""One saved-feedback comparison of staged verification at GRID330's load guard."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
ARCHIVE = HERE.parent / "GRID330_remaining19"
OUTPUT = ROOT / "outputs/grid331_staged_verification"
INPUT = ROOT / "outputs/grid330_remaining19"
ARM = "BOUNDED261_GUARD_DC"
PARENT = "40f3bb60"
SAFE = 0.9
FIRST_BATCH = 8
DAYS = ("2039-03-14", "2039-06-13", "2039-09-12", "2039-12-12")


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return hash_bytes(path.read_bytes())


def restore(path, expected):
    data = path.read_bytes()
    if hash_bytes(data) != expected:
        raise ValueError(f"Input hash mismatch: {path}")
    return json.loads(data)


def freeze():
    manifest_path = ARCHIVE / "results/raw_records_manifest.json"
    archived = read(manifest_path)
    runs = read(INPUT / "manifest.json")["records"]
    assert archived["passed"] and archived["independent_object_readback"]
    hashes = {entry["path"]: entry["sha256"] for entry in archived["entries"]}
    members = []
    calendar = []
    for record in runs:
        info = record["summary"]
        if info["evaluation_arm"] != ARM:
            continue
        assert info["pool"] == "f0_s0" and info["load_guard"] and info["dc_mix"]
        prefix = (INPUT / "runs" / f"r{info['repeat']}__{info['week']}__{info['method']}"
                  ).relative_to(ROOT).as_posix()
        summary = restore(ROOT / prefix / "summary.json", hashes[f"{prefix}/summary.json"])
        assert all(summary[k] == info[k] for k in ("index", "method", "week", "n1_calls", "steps", "complete"))
        files = sorted(path for path in hashes if path.startswith(prefix + "/")
                       and Path(path).match("call[0-9][0-9][0-9].json"))
        assert len(files) == info["n1_calls"]
        calendar.append(dict(date=info["date"], calls=len(files), steps=info["steps"],
                             complete=info["complete"]))
        members.extend(dict(date=info["date"], path=path, sha256=hashes[path]) for path in files)
    assert tuple(sorted(day["date"] for day in calendar)) == DAYS and len(members) == 21
    save(HERE / "fixed_scope.json", {
        "parent_commit": PARENT, "source_sha256": digest(Path(__file__)),
        "parent_archive_manifest_sha256": digest(manifest_path),
        "source_implementation": "GRID320 runtime policy_source and load_guard_receipt, reused by GRID330",
        "original": "Frozen mixed-order protected 261; exact first native STOP plus parent/residual log and complete original 0.9 guard tail.",
        "variant": "At recorded load_guard events only, use original locked-station tail order, verify rounds of 8,16,32,64,... capped by remaining tail, and stop at a round boundary once the current best strictly admissible action has rho <= 0.9; otherwise read the full tail. No reranking, fitting, or decisions outside those events.",
        "method_reference": "ADAP-inspired increasing verification budget and first target hit, with fixed pre-existing candidates and order; not the original generative algorithm.",
        "method_url": "https://arxiv.org/abs/2605.17609v3",
        "comparison": "Same GRID330 recorded pre-guard observations and candidate ordering. Variant queries a prefix of the original paid tail; maximum budget equals original. Compare original strict float32 immediate-reward key and winner, plus verified candidate counts. No new whole-week trajectory or latency estimate.",
        "calendar": calendar, "members": members,
    })


def key(row):
    return struct.unpack("f", struct.pack("f", row["reward"]))[0], -row["pool_id"]


def best(rows):
    return max((row for row in rows if row["strict_admissible"]), default=None, key=key)


def collect_staged(before, pending, reveal):
    incumbent = before
    rounds, paid, batch = [], 0, FIRST_BATCH
    while paid < len(pending):
        ids = pending[paid:paid + batch]
        for cid in ids:
            row = reveal(cid)
            if row["strict_admissible"] and key(row) > key(incumbent):
                incumbent = row
        paid += len(ids)
        rounds.append(dict(ids=ids, paid=paid, winner=incumbent["pool_id"], rho=incumbent["rho"]))
        if incumbent["rho"] <= SAFE:
            break
        batch *= 2
    return incumbent, rounds


def evaluate_member(member):
    call = restore(ROOT / member["path"], member["sha256"])
    assert call["order_mode"] == "STATIC64_DC64" and call["learning_method"] == "BOUNDED261"
    outcomes = call["outcomes"]
    assert len(outcomes) == call["public_queries"] == len({r["pool_id"] for r in outcomes})
    original = best(outcomes)
    assert (None if original is None else original["pool_id"]) == call["chosen_pool_id"]
    row = {**member, "step": call["step"], "state_hash": call["state_hash"],
           "original_total": len(outcomes), "staged_total": len(outcomes),
           "original_winner": call["chosen_pool_id"], "staged_winner": call["chosen_pool_id"],
           "guard_triggered": False, "reward_lost": False, "saved": 0}
    guard = call.get("load_guard")
    if guard is None:
        return row
    cut = guard["paid_before"]
    initial = best(outcomes[:cut])
    seen = {r["pool_id"] for r in outcomes[:cut]}
    pending = [cid for cid in call["completion_candidate_ids"] if cid not in seen]
    bought = outcomes[cut:]
    decision = call["learned_decisions"][-1]
    assert decision["optional_extension"] and not decision["action"]
    assert decision["paid_queries"] == cut and decision["remaining"] == len(pending)
    assert pending and pending == [r["pool_id"] for r in bought] == guard["extra_ids"]
    assert all(r["phase"] == "local_completion" for r in bought)
    assert initial["pool_id"] == guard["incumbent"] and initial["rho"] == guard["rho"] > SAFE
    assert guard["threshold"] == SAFE
    feedback = {r["pool_id"]: r for r in bought}
    winner, rounds = collect_staged(initial, pending, feedback.__getitem__)
    paid = rounds[-1]["paid"]
    assert key(winner) <= key(original) and (paid < len(pending) or key(winner) == key(original))
    row.update(guard_triggered=True, paid_before=cut, original_tail=len(pending), staged_tail=paid,
               staged_total=cut + paid, staged_winner=winner["pool_id"],
               original_reward=key(original)[0], staged_reward=key(winner)[0],
               original_rho=original["rho"], staged_rho=winner["rho"],
               reward_lost=key(winner)[0] < key(original)[0],
               saved=len(pending) - paid, rounds=rounds)
    return row


def summarize(rows):
    return dict(calls=len(rows), guards=sum(r["guard_triggered"] for r in rows),
                calls_with_saving=sum(r["saved"] > 0 for r in rows),
                saved_queries=sum(r["saved"] for r in rows),
                original_queries=sum(r["original_total"] for r in rows),
                staged_queries=sum(r["staged_total"] for r in rows),
                original_guard_queries=sum(r.get("original_tail", 0) for r in rows),
                staged_guard_queries=sum(r.get("staged_tail", 0) for r in rows),
                changed_winners=sum(r["original_winner"] != r["staged_winner"] for r in rows),
                worse_reward=sum(r["reward_lost"] for r in rows))


def evaluate():
    start = time.monotonic()
    scope_path = HERE / "fixed_scope.json"
    scope = read(scope_path)
    assert digest(Path(__file__)) == scope["source_sha256"]
    assert digest(ARCHIVE / "results/raw_records_manifest.json") == scope["parent_archive_manifest_sha256"]
    rows = [evaluate_member(m) for m in scope["members"]]
    assert len(rows) == 21
    summary = dict(all=summarize(rows),
                   by_date={day: summarize([r for r in rows if r["date"] == day]) for day in DAYS},
                   wall_s=time.monotonic() - start, scope_sha256=digest(scope_path),
                   new_fit=0, new_forecasts=0, new_model_predictions=0)
    OUTPUT.mkdir(exist_ok=False)
    save(OUTPUT / "rows.json", rows)
    save(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    {"freeze": freeze, "evaluate": evaluate}[parser.parse_args().stage]()

