"""Compare GRID326 mixed prefix with the original order under frozen GRID91 search."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import time

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[5]
OUT = ROOT / "outputs/grid327_mixed_fallback_readout"
OLD = ARCHIVE.parent / "GRID325_dc_global_priority"
MIX = ARCHIVE.parent / "GRID326_fixed_dc_mix"
POLICY = "work/grid91_mechanism/policies.py"
LEDGER = "outputs/grid139_global_feasibility/call_ledger.json"
SUBS = "outputs/grid28/offline.json"
MODES = ("NO_LOCAL", "FRAME_GUARD")
LIMIT = 600


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
    paths = [ROOT / p for p in (POLICY, LEDGER, SUBS)]
    for archive, names in ((OLD, ("fixed_scope.json", "public_ranks.json")),
                           (MIX, ("fixed_scope.json", "public_mix.json", "results/rows.json"))):
        delivery = read(archive / "results/delivery_hashes.json")
        for name in names:
            path = archive / name
            assert sha(path) == delivery[name]
            paths.append(path)
    old = read(OLD / "fixed_scope.json")
    assert sha(ROOT / LEDGER) == old["source_hashes"][LEDGER]
    assert sha(ROOT / SUBS) == old["source_hashes"][SUBS]
    save(ARCHIVE / "fixed_scope.json", {
        "source_sha256": sha(Path(__file__)),
        "parents": {"rank": "e32300e1", "mixed_prefix": "26d3394f", "two_case_note": "cd57941c"},
        "source_hashes": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        "members": "Exactly all 305 calls from GRID325/326, same train and validation groups and candidate pool. No new cases.",
        "original": "Entire original STATIC_ID candidate order.",
        "mixed": "Published GRID326 64 original + 64 DC prefix; append all unqueried candidates in original STATIC_ID order. No new score, ratio, pruning or other variants.",
        "policy": "Use actual frozen GRID91 stopping and station logic. Compare NO_LOCAL (shared global phase and pending local budget only) and FRAME_GUARD (its own full local completion). Neither applies 208 or 261 learned completion.",
        "feedback": "Only saved reveal for requested ID, with original strict admissibility and float32 comparison. No new model or physical calls.",
        "timing": "Reuse archived per-ID forecast_s as a work estimate and GRID325 per-call DC numerical plus sorting time as the paid overhead of mixing, even when no scores are available. Add GRID326 prefix construction time. No hardware normalization, no AC/simulation or end-to-end time measurement.",
        "reports": "Final ID, availability, same-state reward, first acceptable position, exact FULL match, actual searched IDs, global/local and total queries, estimated forecast times and overhead; groups by original date. Keep adverse cases.",
        "limits": {"wall_s": LIMIT, "new_fit": 0, "new_dc_solves": 0, "new_model_forward": 0, "new_simulate": 0},
    })


def setup():
    scope = read(ARCHIVE / "fixed_scope.json")
    assert sha(Path(__file__)) == scope["source_sha256"]
    for name, digest in scope["source_hashes"].items():
        assert sha(ROOT / name) == digest, name
    old = read(OLD / "fixed_scope.json")
    frozen_mix = read(MIX / "public_mix.json")
    public = read(OLD / "public_ranks.json")
    assert public["scope_sha256"] == sha(OLD / "fixed_scope.json")
    assert frozen_mix["scope_sha256"] == sha(MIX / "fixed_scope.json")
    assert frozen_mix["parent_rank_sha256"] == sha(OLD / "public_ranks.json")
    return old, public["rows"], frozen_mix["rows"]


def replay(run, key, order, feedback, subids, aliases, limit, mode):
    reveals = []

    def observe(cid, phase):
        reveals.append((cid, phase))
        return feedback[cid]

    result = run(order, observe, mode, subids, aliases, limit=limit)
    outcomes = result["outcomes"]
    ids = [r["pool_id"] for r in outcomes]
    n = result["public_queries"]
    assert ids == [cid for cid, _ in reveals] and len(ids) == len(set(ids)) == n
    assert sum(phase == "local_completion" for _, phase in reveals) == result["local_queries"]
    durations = [r["forecast_s"] for r in outcomes]
    assert all(0 <= value < float("inf") for value in durations)
    winner = max((r for r in outcomes if r["strict_admissible"]), key=key, default=None)
    picked = None if winner is None else winner["pool_id"]
    assert picked == result["chosen_pool_id"]
    return {
        "queried_ids": ids, "queries": n, "global_queries": n - result["local_queries"],
        "local_queries": result["local_queries"], "pending_at_local_entry": result["completion_budget"],
        "first_admissible": next((i for i, r in enumerate(outcomes, 1) if r["strict_admissible"]), None),
        "prefix_choice": result["shortlist_choice"], "choice": picked,
        "reward": None if winner is None else key(winner)[0],
        "rho": None if winner is None else winner["rho"],
        "forecast_work_s": sum(durations), "reason": result["stop_reason"],
    }


def compare(a, b):
    if a is None or b is None:
        return int(a is not None) - int(b is not None)
    return int(a > b) - int(a < b)


def metrics(rows, mode):
    pairs = [r[mode] for r in rows]
    summaries = {
        "calls": len(rows),
        "outcomes": dict(Counter(p["reward_outcome"] for p in pairs)),
        "same_action": sum(p["original"]["choice"] == p["mixed"]["choice"] for p in pairs),
        "lost_admissible": sum(p["original"]["choice"] is not None and p["mixed"]["choice"] is None for p in pairs),
        "recovered_admissible": sum(p["original"]["choice"] is None and p["mixed"]["choice"] is not None for p in pairs),
        "same_action_query_delta": sum(p["query_delta"] for p in pairs
            if p["original"]["choice"] == p["mixed"]["choice"]),
        "query_delta": sum(p["query_delta"] for p in pairs),
        "more_query_calls": sum(p["query_delta"] > 0 for p in pairs),
        "fewer_query_calls": sum(p["query_delta"] < 0 for p in pairs),
        "forecast_work_delta_s": sum(p["forecast_work_delta_s"] for p in pairs),
        "dc_and_prefix_overhead_s": sum(r["overhead_s"] for r in rows),
        "timing_proxy_delta_s": sum(p["timing_proxy_delta_s"] for p in pairs),
        "timing_proxy_faster_calls": sum(p["timing_proxy_delta_s"] < 0 for p in pairs),
        "timing_proxy_median_delta_s": statistics.median(p["timing_proxy_delta_s"] for p in pairs),
    }
    for arm in ("original", "mixed"):
        summaries[arm] = {field: sum(p[arm][field] for p in pairs)
            for field in ("queries", "global_queries", "local_queries", "pending_at_local_entry", "forecast_work_s")}
        summaries[arm]["full_match"] = sum(p[arm]["choice"] == r["full"] and r["full"] is not None
            for p, r in zip(pairs, rows, strict=True))
    return summaries


def evaluate():
    start = time.monotonic()
    old, public, mixed = setup()
    policy = runpy.run_path(str(ROOT / POLICY))
    assert policy["DEFAULT"] == {"p": 128, "h": 64, "patience": 2, "safe": .9}
    run, key = policy["run"], policy["key"]
    ledger = {r["call_id"]: r for r in read(ROOT / LEDGER)}
    subids = read(ROOT / SUBS)["public_subids"]
    assert len(ledger) == len(old["members"]) == len(public) == len(mixed) == 305
    results = []
    legacy_diffs = Counter()
    for m, p, h in zip(old["members"], public, mixed, strict=True):
        assert time.monotonic() - start < LIMIT
        identity = m["call_id"]
        assert identity == p["call_id"] == h["call_id"]
        path = ROOT / m["source"]
        assert sha(path) == old["source_hashes"][m["source"]]
        call = read(path)
        original_order = call["order"]
        prefix = h["mix128"]
        seen = set(prefix)
        mixed_order = prefix + [cid for cid in original_order if cid not in seen]
        assert original_order[:128] == p["static128"] and len(prefix) == 128
        assert len(mixed_order) == len(original_order) == len(set(mixed_order)) == m["M"]
        assert set(mixed_order) == set(original_order)
        feedback = {r["pool_id"]: r for r in call["outcomes"]}
        assert list(feedback) == original_order and call["state_hash"] == p["state_hash"]
        aliases = {int(k): v for k, v in call["aliases"].items()}
        baseline = ledger[identity]
        row = {k: m[k] for k in ("call_id", "date", "scenario", "step", "partition")}
        row.update(full=baseline["full_choice"], overhead_s=p["dc_and_sort_s"] + h["mix_ns"] / 1e9,
                   original_prefix=p["static128"], mixed_prefix=prefix)
        for mode in MODES:
            a = replay(run, key, original_order, feedback, subids, aliases, call["rho_limit"], mode)
            b = replay(run, key, mixed_order, feedback, subids, aliases, call["rho_limit"], mode)
            delta = b["forecast_work_s"] - a["forecast_work_s"]
            row[mode] = {"original": a, "mixed": b, "query_delta": b["queries"] - a["queries"],
                         "reward_outcome": ("loss", "tie", "win")[compare(b["reward"], a["reward"]) + 1],
                         "forecast_work_delta_s": delta, "timing_proxy_delta_s": delta + row["overhead_s"]}
            if mode == "NO_LOCAL":
                legacy_diffs["global_query_count"] += a["queries"] != baseline["global_n"]
                legacy_diffs["global_choice"] += a["choice"] != baseline["global_choice"]
                legacy_diffs["pending_count"] += a["pending_at_local_entry"] != baseline["local_queries"]
            else:
                legacy_diffs["frame_choice"] += a["choice"] != baseline["frame_choice"]
        assert row["NO_LOCAL"]["original"]["global_queries"] == row["FRAME_GUARD"]["original"]["global_queries"]
        assert row["NO_LOCAL"]["mixed"]["global_queries"] == row["FRAME_GUARD"]["mixed"]["global_queries"]
        results.append(row)
    splits = {}
    for part, dates in old["dates"].items():
        part_rows = [r for r in results if r["partition"] == part]
        splits[part] = {mode: {"all": metrics(part_rows, mode),
            "by_date": {date: metrics([r for r in part_rows if r["date"] == date], mode) for date in dates}}
            for mode in MODES}
    save(OUT / "rows.json", results)
    save(OUT / "summary.json", {
        "metrics": splits, "legacy_static_ledger_mismatches": dict(legacy_diffs),
        "wall_s": time.monotonic() - start, "scope_sha256": sha(ARCHIVE / "fixed_scope.json"),
        "new_fit": 0, "new_dc_solves": 0, "new_model_forward": 0, "new_simulate": 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    {"freeze": freeze, "evaluate": evaluate}[parser.parse_args().stage]()

