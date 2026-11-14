"""Decompose the actual local query pathways on the four GRID334 261 trajectories."""
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PARENT = HERE.parent / "GRID334_logeipc_closed_loop/results"
RUNS = ROOT / "outputs/grid334_logeipc_closed_loop/runs"
OUTPUT = ROOT / "outputs/grid336_261_intervention_scope"


def load(path, expected=None):
    content = path.read_bytes()
    if expected is not None:
        assert hashlib.sha256(content).hexdigest() == expected, path
    return json.loads(content)


def key(row):
    return struct.unpack("f", struct.pack("f", row["reward"]))[0], -row["pool_id"]


def best(rows):
    return max((r for r in rows if r["strict_admissible"]), key=key)


def main():
    checks = load(PARENT / "delivery_hashes.json")
    manifest = load(PARENT / "raw_records_manifest.json", checks["raw_records_manifest.json"])
    assert manifest["passed"] and manifest["independent_object_readback"]
    digests = {record["path"]: record["sha256"] for record in manifest["entries"]}
    paired = load(PARENT / "paired_results.json", checks["paired_results.json"])
    records = [row for row in paired["records"] if row["arm"] == "BOUNDED261_GUARD_DC"]
    assert len(records) == 4
    batches, stops = [], []
    calls = global_queries = 0
    for record in records:
        date = record["week"][:10]
        folder = RUNS / f"r0__{record['week']}__BOUNDED261_GUARD_DC__f0_s0"
        queries = 0
        for path in sorted(folder.glob("call[0-9][0-9][0-9].json")):
            call = load(path, digests[path.relative_to(ROOT).as_posix()])
            calls += 1
            queries += call["public_queries"]
            global_queries += call["public_queries"] - call["local_queries"]
            outcomes = call["outcomes"]
            candidates = []
            for decision in call["learned_decisions"]:
                optional = decision["optional_extension"]
                if optional:
                    stops.append(dict(date=date, step=call["step"], parent_continue=decision["parent_CONT"],
                                      actual_continue=bool(decision["action"])))
                if decision["action"]:
                    assert "native_q" in decision
                    kind = "parent_extension" if optional else "native_continue"
                    candidates.append((kind, decision["paid_queries"], decision["next_ids"]))
            if guard := call.get("load_guard"):
                candidates.append(("load_guard", guard["paid_before"], guard["extra_ids"]))
            paid = []
            for kind, start, ids in candidates:
                paid.extend(ids)
                end = start + len(ids)
                assert ids == [r["pool_id"] for r in outcomes[start:end]]
                before, after = best(outcomes[:start]), best(outcomes[:end])
                batches.append(dict(date=date, step=call["step"], path=kind, size=len(ids),
                                    from_id=before["pool_id"], to_id=after["pool_id"],
                                    winner_improved=key(after) > key(before),
                                    reward_gain=key(after)[0] - key(before)[0]))
            assert paid == call["extra_selected_ids"] and len(paid) == call["local_queries"]
        assert queries == record["candidate_queries"]
    assert calls == 21 and global_queries == 5568
    totals = {}
    for kind in ("native_continue", "parent_extension", "load_guard"):
        matching = [row for row in batches if row["path"] == kind]
        totals[kind] = dict(batches=len(matching), queries=sum(row["size"] for row in matching),
                            improved=sum(row["winner_improved"] for row in matching))
    assert sum(r["queries"] for r in totals.values()) == 526
    result = dict(parent_commit="4c7cb42a", all_calls=calls, global_queries=global_queries,
                  all_queries=6094, local_query_paths=totals, stops=stops, batches=batches,
                  models_fitted=0, new_forecasts=0, new_replays=0)
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT / "summary.json").open("x", encoding="utf8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()

