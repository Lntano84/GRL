"""Audit label coverage and fixed-budget headroom from saved FULL feedback."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import runpy
import time

A = Path(__file__).resolve().parent
PRIOR = A.parent / "GRID316_candidate_priority_readout/readout.py"
reference = runpy.run_path(str(PRIOR))
ROOT, BUDGET, WALL_CAP = (reference[k] for k in ("ROOT", "BUDGET", "WALL_CAP"))
OUT = ROOT / "outputs/grid317_candidate_gain_coverage"
LEDGER, OLD_HASHES = (reference[k] for k in ("LEDGER", "OLD_HASHES"))
read, save, sha, tail, winner, candidate, reward = (
    reference[k] for k in ("read", "save", "sha", "tail", "winner", "candidate", "reward")
)
SUBS = "outputs/grid28/offline.json"
FIELDS = (*reference["PUBLIC"], "partition", "source")


def freeze():
    old = read(ROOT / OLD_HASHES)
    members = []
    paths = {LEDGER, OLD_HASHES, SUBS, PRIOR.relative_to(ROOT).as_posix()}
    for row in read(ROOT / LEDGER):
        assert row["partition"] in ("train", "validation")
        member = {k: row[k] for k in FIELDS}
        path = ROOT / row["source"]
        assert sha(path) == old[row["source"]], path
        paths.add(row["source"])
        if row["local_queries"] > BUDGET:
            state = path.with_name(path.stem + "_state.npz").relative_to(ROOT).as_posix()
            paths.add(state)
            member["state"] = state
        members.append(member)
    train = [m for m in members if m["partition"] == "train"]
    dev = [m for m in members if m["partition"] == "validation"]
    assert len(train) == 243 and len(dev) == 62
    assert len({m["call_id"] for m in members}) == 305
    fit_dates, dev_dates = ({m["date"] for m in rows} for rows in (train, dev))
    assert len(fit_dates) == 16 and len(dev_dates) == 4 and not fit_dates & dev_dates
    save(A / "fixed_scope.json", {
        "members": members,
        "source_hashes": {p: sha(ROOT / p) for p in sorted(paths)},
        "fit_dates": sorted(fit_dates),
        "dev_dates": sorted(dev_dates),
        "budget": BUDGET,
        "label": "Tail candidate positive iff strict-admissible float32 reward exceeds the same incumbent after the original FRAME global prefix. Keep other valid and inadmissible counts separate.",
        "headroom": "Tail >64 and full locked tail strictly beats the incumbent plus the original first 64. Ties including changed ID without greater reward do not count.",
        "risk": "Static first 64 improves on incumbent and at least 64 eligible-to-query candidates fall strictly below the static best (including inadmissible). This makes regression possible at unchanged budget, not an estimated policy error rate.",
        "grouping": "Original 16 training/4 development dates. Count candidate rows, calls, exact observation and base action within date, and date coverage separately. No fit or model selection on development data.",
        "necessary_gate": "At least two training dates have strict headroom with positive/negative labels present, and at least two original development dates have strict headroom. This is only a minimum for cross-date comparison, not a power guarantee or training authorization.",
        "limits": {"wall_s": WALL_CAP, "new_fits": 0, "new_predictions": 0, "new_physics": 0},
    })


def inspect():
    import hashlib
    import numpy as np
    start = time.monotonic()
    scope = read(A / "fixed_scope.json")
    for path, digest in scope["source_hashes"].items():
        assert sha(ROOT / path) == digest, path
    subs = read(ROOT / SUBS)["public_subids"]
    ledger = {r["call_id"]: r for r in read(ROOT / LEDGER)}
    OUT.mkdir(exist_ok=False)
    rows = []
    for member in scope["members"]:
        assert time.monotonic() - start < WALL_CAP
        call = read(ROOT / member["source"])
        station, pending = tail(call, member, subs)
        feedback = {r["pool_id"]: r for r in call["outcomes"]}
        assert list(feedback) == call["order"]
        incumbent = winner(feedback[i] for i in call["order"][:member["global_n"]])
        assert candidate(incumbent) == member["global_choice"]
        row = dict(member, station=station, observation_hash=call["state_hash"])
        if not pending:
            rows.append(row)
            continue
        assert incumbent is not None
        tail_rows = [feedback[i] for i in pending]
        static = winner([incumbent, *tail_rows[:BUDGET]])
        best = winner([incumbent, *tail_rows])
        assert candidate(best) == ledger[member["call_id"]]["frame_choice"]
        base_reward, static_reward, best_reward = map(reward, (incumbent, static, best))
        counts = defaultdict(int)
        for i, item in enumerate(tail_rows):
            if not item["strict_admissible"]:
                counts["inadmissible"] += 1
                counts["below_static"] += 1
                continue
            score = reward(item)
            counts["below_static"] += score < static_reward
            counts["positive"] += score > base_reward
            counts["legal_nonpositive"] += score <= base_reward
            counts["beats_static"] += score > static_reward
            counts["positive_in_static"] += i < BUDGET and score > base_reward
        assert len(tail_rows) == sum(counts[k] for k in ("inadmissible", "positive", "legal_nonpositive"))
        row.update(
            label_counts={k: counts[k] for k in ("positive", "legal_nonpositive", "inadmissible",
                                                 "positive_in_static", "beats_static")},
            static_choice=candidate(static), full_tail_choice=candidate(best),
            best_reward_minus_incumbent=best_reward - base_reward,
            static_reward_minus_incumbent=static_reward - base_reward,
            strict_headroom=best_reward - static_reward,
            static_regression_possible=static_reward > base_reward and counts["below_static"] >= BUDGET,
        )
        if "state" in member:
            with np.load(ROOT / member["state"], allow_pickle=False) as data:
                for field in ("observation", "base_action"):
                    row[field + "_hash"] = hashlib.sha256(data[field].tobytes()).hexdigest()
            assert row["observation_hash"] == call["state_hash"]
        rows.append(row)
    save(OUT / "rows.json", rows)
    partitions = {}
    for part in ("train", "validation"):
        days = {}
        for date in scope["fit_dates" if part == "train" else "dev_dates"]:
            day = [r for r in rows if r["date"] == date]
            long = [r for r in day if r["local_queries"] > BUDGET]
            possible = [r for r in long if r["strict_headroom"] > 0]
            groups = {(r["observation_hash"], r["base_action_hash"]) for r in long}
            missed_groups = {(r["observation_hash"], r["base_action_hash"]) for r in possible}
            days[date] = dict(
                calls=len(day), long_calls=len(long), exact_long_inputs=len(groups),
                positive=sum(r["label_counts"]["positive"] for r in long),
                legal_nonpositive=sum(r["label_counts"]["legal_nonpositive"] for r in long),
                inadmissible=sum(r["label_counts"]["inadmissible"] for r in long),
                calls_with_positive=sum(r["label_counts"]["positive"] > 0 for r in long),
                calls_with_both_legal_labels=sum(r["label_counts"]["positive"] > 0 and
                                                r["label_counts"]["legal_nonpositive"] > 0 for r in long),
                headroom_calls=len(possible), exact_headroom_inputs=len(missed_groups),
                headroom_candidates=sum(r["label_counts"]["beats_static"] for r in possible),
                regression_risk_calls=sum(r["static_regression_possible"] for r in long),
            )
        totals = {k: sum(d[k] for d in days.values()) for k in next(iter(days.values()))}
        totals["headroom_dates"] = sum(d["headroom_calls"] > 0 for d in days.values())
        totals["headroom_dates_with_both_legal_labels"] = sum(
            d["headroom_calls"] > 0 and d["calls_with_both_legal_labels"] > 0 for d in days.values()
        )
        partitions[part] = dict(totals=totals, days=days)
    gate = (partitions["train"]["totals"]["headroom_dates_with_both_legal_labels"] >= 2
            and partitions["validation"]["totals"]["headroom_dates"] >= 2)
    save(OUT / "summary.json", dict(partitions=partitions, passed_necessary_gate=gate,
                                   new_fits=0, new_predictions=0, new_physics=0,
                                   elapsed_s=time.monotonic() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "inspect"))
    args = parser.parse_args()
    {"freeze": freeze, "inspect": inspect}[args.stage]()
