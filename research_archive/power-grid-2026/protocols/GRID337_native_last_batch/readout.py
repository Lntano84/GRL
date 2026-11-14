"""Audit saved last native-CONT batches for the fixed GRID337 qualification."""
import hashlib
import json
from pathlib import Path
import re
import struct

A = Path(__file__).resolve().parent
ROOT = A.parents[5]
P = A.parent
O = ROOT / "outputs/grid337_native_last_batch"
BASE = P / "GRID301_remaining_collection/baseline_public"
ROLES = P / "GRID300_new_training_dates/source/pool_roles.json"
DEV = P / "GRID334_logeipc_closed_loop/results"
CALL = re.compile(r"call[0-9]+\.json$")
POOL = "f0_s0"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path, digest=None):
    data = path.read_bytes()
    if digest is not None:
        assert sha(data) == digest, path
    return json.loads(data)


def write(path, obj):
    with path.open("x", encoding="utf8") as stream:
        json.dump(obj, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def key(row):
    return struct.unpack("f", struct.pack("f", row["reward"]))[0], -row["pool_id"]


def best(rows):
    return max((row for row in rows if row["strict_admissible"]), key=key, default=None)


def members():
    source = read(BASE / "baseline_manifest.json")
    assert len(source["records"]) == source["completed"] == 143
    roles, = [row for row in read(ROLES) if row["pool"] == POOL]
    assert len(roles["fit_dates"]) == 8 and len(roles["cal_dates"]) == 4
    by_date = {date: role for role in ("fit", "cal") for date in roles[f"{role}_dates"]}
    assert len(by_date) == 12
    baseline = [row["summary"] for row in source["records"] if row["summary"]["pool"] == POOL]
    assert len({row["date"] for row in baseline}) == len(baseline)
    assert all(row["date"] in by_date for row in baseline)
    hashes = {path: ref["sha256"] for path, ref in read(BASE / "baseline_raw_manifest.json").items()}
    receipt = read(DEV / "raw_records_manifest.json")
    assert receipt["passed"] and receipt["independent_object_readback"]
    hashes.update({row["path"]: row["sha256"] for row in receipt["entries"]})
    result = []
    for row in baseline:
        result.append(dict(date=row["date"], role=by_date[row["date"]], repeat=row["repeat"],
                           week=row["week"], method=row["method"], steps=row["steps"],
                           complete=row["complete"], expected_calls=row["n1_calls"],
                           expected_queries=row["n1_forecasts"]))
    for date in sorted(set(by_date) - {row["date"] for row in baseline}):
        result.append(dict(date=date, role=by_date[date], missing_run=True))
    for row in read(DEV / "paired_results.json")["records"]:
        if row["arm"] == "BOUNDED261_GUARD_DC":
            result.append(dict(date=row["week"][:10], role="dev", repeat=row["repeat"], week=row["week"],
                               method=f"{row['arm']}__{POOL}", steps=row["steps"], complete=row["complete"],
                               expected_queries=row["candidate_queries"]))
    assert sum(row["role"] == "dev" for row in result) == 4
    index = {}
    for path, digest in hashes.items():
        p = Path(path)
        if CALL.fullmatch(p.name):
            index.setdefault(p.parent.name, []).append((path, digest))
    for row in result:
        if not row.get("missing_run"):
            folder = f"r{row['repeat']}__{row['week']}__{row['method']}"
            row["calls"] = dict(sorted(index.get(folder, [])))
    return source, result


def freeze():
    source, runs = members()
    inputs = [ROLES, BASE / "baseline_manifest.json", BASE / "baseline_raw_manifest.json",
              DEV / "raw_records_manifest.json", DEV / "paired_results.json"]
    write(A / "scope.json", dict(source_sha=sha(Path(__file__).read_bytes()), pool=POOL,
        inputs={p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in inputs},
        total_baseline_receipts=source["completed"],
        excluded_partial_index=source["missing_partial_baseline_index"], runs=runs,
        eligibility="Native CONT, 1..64 remaining, next_ids exhaust original pending station pool, incumbent rho <= .9",
        outcome="Compare original float32 reward/candidate-ID winner before decision and final delivered candidate. Verify actual last batch feedback and original selected winner.",
        dedup="Date, physical state hash, ordered paid prefix IDs and canonical action hashes, locked pending IDs, and cutoff. Duplicate margins and labels must agree.",
        gates="Each of fit and cal must have >=2 dates per class. Then use minimum fit must-keep q1-q0, strict less only. Calibration must save queries on >=2 dates and lose no winner. Preserve known 2039-03-14 step 520."))


def inspect(run, path, digest):
    call = read(ROOT / path, digest)
    outcomes = call["outcomes"]
    assert call["public_queries"] == len(outcomes)
    assert len({r["pool_id"] for r in outcomes}) == len(outcomes)
    result = []
    for decision in call["learned_decisions"]:
        q = decision.get("native_q")
        if not q or decision["optional_extension"] or not decision["action"] or not q[1] > q[0]:
            continue
        n = decision["paid_queries"]
        pending = decision["next_ids"]
        if not 1 <= decision["remaining"] == len(pending) <= 64:
            continue
        prefix = outcomes[:n]
        seen = {row["pool_id"] for row in prefix}
        remaining = [cid for cid in call["completion_candidate_ids"] if cid not in seen]
        assert remaining == pending, path
        incumbent = best(prefix)
        assert incumbent is not None
        if incumbent["rho"] > 0.9:
            continue
        assert [row["pool_id"] for row in outcomes[n:]] == pending, path
        winner = best(outcomes)
        assert winner["pool_id"] == call["chosen_pool_id"]
        # Keep the decision margin and public identity independent of the label.
        identity = [run["date"], call["state_hash"],
                    [(row["pool_id"], row["combined_hash"]) for row in prefix], pending,
                    call["rho_limit"]]
        result.append(dict(role=run["role"], date=run["date"], step=call["step"], path=path,
            identity=sha(json.dumps(identity,separators=(",",":")).encode()),
            queries=len(pending), rho=incumbent["rho"], margin=q[1]-q[0],
            from_id=incumbent["pool_id"], to_id=winner["pool_id"],
            must_keep=incumbent["pool_id"] != winner["pool_id"],
            reward_delta=key(winner)[0]-key(incumbent)[0]))
    return call["public_queries"], result


def evaluate():
    scope = read(A / "scope.json")
    assert scope["source_sha"] == sha(Path(__file__).read_bytes())
    for name, digest in scope["inputs"].items():
        assert sha((ROOT / name).read_bytes()) == digest
    rows, runs, unique = [], [], {}
    for run in scope["runs"]:
        receipt = {k: v for k, v in run.items() if k != "calls"}
        if run.get("missing_run"):
            runs.append(receipt)
            continue
        queries, available, missing = 0, 0, []
        for path, digest in run["calls"].items():
            if not (ROOT / path).is_file():
                missing.append(path)
                continue
            count, records = inspect(run, path, digest)
            queries += count
            available += 1
            rows.extend(records)
        expected = run.get("expected_calls", len(run["calls"]))
        if not missing and available == expected:
            assert queries == run["expected_queries"], run
        runs.append(dict(receipt, verified_calls=available, expected_calls=expected, missing_call_files=missing,
                         complete_call_coverage=available==expected and not missing))
    for row in rows:
        original = unique.setdefault((row["role"],row["identity"]), row)
        assert all(row[k] == original[k] for k in ("margin","must_keep","queries","from_id","to_id")), row
    distinct = list(unique.values())
    counts = {}
    for role in ("fit","cal","dev"):
        group = [row for row in distinct if row["role"] == role]
        counts[role] = {}
        for outcome in (False,True):
            chosen = [row for row in group if row["must_keep"] == outcome]
            counts[role]["must_keep" if outcome else "can_skip"] = dict(
                batches=len(chosen), dates=sorted({row["date"] for row in chosen}),
                candidates=sum(row["queries"] for row in chosen))
    march, = [r for r in distinct if r["role"]=="dev" and r["date"]=="2039-03-14" and r["step"]==520]
    assert march["must_keep"] and march["from_id"] == 142 and march["to_id"] == 238
    coverage = all(r.get("complete_call_coverage",False) for r in runs)
    support = all(len(group["dates"]) >= 2 for role in ("fit","cal") for group in counts[role].values())
    threshold, checks = None, None
    if coverage and support:
        threshold = min(r["margin"] for r in distinct if r["role"]=="fit" and r["must_keep"])
        skips = [r for r in distinct if r["role"]=="cal" and r["margin"] < threshold]
        checks=dict(cal_saved_dates=sorted({r["date"] for r in skips}),
                    cal_saved_queries=sum(r["queries"] for r in skips),
                    cal_lost_winners=sum(r["must_keep"] for r in skips),
                    march_preserved=march["margin"] >= threshold)
    qualified = bool(checks and len(checks["cal_saved_dates"]) >= 2
                     and not checks["cal_lost_winners"] and checks["march_preserved"])
    summary = dict(counts=counts, runs=runs, rows=distinct, duplicate_count=len(rows)-len(distinct),
        coverage_complete=coverage, support_passed=support, threshold=threshold, checks=checks,
        decision="QUALIFIED_FOR_ADAPTER" if qualified else "CLOSE_BRANCH_KEEP_261",
        simulated=0, new_model_forwards=0)
    O.mkdir(exist_ok=False)
    write(O / "summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("runs","rows")},indent=2))


if __name__ == "__main__":
    import sys
    {"freeze":freeze,"evaluate":evaluate}[sys.argv[1]]()

