"""Freeze and audit saved GRID301/302 pairs using the GRID298 episode checks."""

import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path
import time
import traceback

R = Path("D:/Codex/GRL研究")
I = R / "work/GRL-results-integration"
O = R / "outputs/grid303_saved_suffix_targets"
A = Path(__file__).resolve().parent
OLD = R / "outputs/grid301_remaining_collection"
NEW = R / "outputs/grid302_unstarted_opposites"
PRIOR = R / "work/grid298_saved_suffix_targets/readout.py"
spec = importlib.util.spec_from_file_location("grid298_saved_audit", PRIOR)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
load, save, sha, git = audit.load, audit.save, audit.sha, audit.git
FILES = ("summary.json", "metadata.json", "intervention.json", "steps.jsonl.gz")
ROLES = ("own_training_fit8", "own_training_cal4")


def relative(path):
    return path.relative_to(R).as_posix()


def ref(path):
    data = path.read_bytes()
    return {"sha256": sha(data), "bytes": len(data)}


def folder(root, item):
    return relative(root / "runs" / f"r{item['repeat']}__{item['week']}__{item['method']}")


def archive_refs(path):
    archive = load(path)
    assert archive["passed"] and archive["independent_object_readback"]
    return {r["path"]: {"sha256": r["sha256"], "bytes": r["bytes"]} for r in archive["entries"]}


def prepare():
    assert not (A / "fixed_inputs.json").exists()
    publication = load(OLD / "baseline_publication.json")
    assert publication["passed"]
    baseline_dir = "research_archive/power-grid-2026/protocols/GRID301_remaining_collection/baseline_public/"
    metadata = {}
    for name, expected in publication["exact_published_files"].items():
        saved = (OLD / name).read_bytes()
        assert sha(saved) == expected
        assert git("show", publication["commit"] + ":" + baseline_dir + name) == saved
        metadata[relative(OLD / name)] = ref(OLD / name)
    archive_paths = [
        R / "archives/GRID301_user_stop_20261105/raw_records/raw_records_manifest.json",
        R / "archives/GRID302_unstarted_opposites_20261106/raw_records/raw_records_manifest.json",
    ]
    receipts = {r["index"]: OLD for r in load(OLD / "manifest.json")["records"] if r["index"] >= 144}
    assert len(receipts) == 192
    more = {r["index"]: NEW for r in load(NEW / "manifest.json")["records"]}
    assert sorted(more) == list(range(337, 377)) and not receipts.keys() & more.keys()
    receipts.update(more)
    raw_refs = {OLD: archive_refs(archive_paths[0]), NEW: archive_refs(archive_paths[1])}
    baseline_refs = load(OLD / "baseline_raw_manifest.json")
    runtime = load(OLD / "runtime_design.json")["execution_order"]
    markers = load(OLD / "selected_markers.json")
    baseline_records = {r["index"] for r in load(OLD / "baseline_manifest.json")["records"]}
    assert baseline_records == set(range(144)) - {35}
    assert len(markers["new_opposite_items"]) == 233
    assert len(markers["empty_bins"]) == 339 and markers["missing_baseline_index"] == 35
    pairs, refs = [], {}
    for opposite in markers["new_opposite_items"]:
        index = opposite["index"]
        assert runtime[index] == opposite
        base_index = opposite["baseline_index"]
        assert base_index in baseline_records
        baseline = {**runtime[base_index], "folder": opposite["baseline_folder"]}
        if index not in receipts:
            assert index == 336
            partial = folder(OLD, opposite)
            for name in ("partial_intervention.json", "partial_steps.jsonl"):
                path = partial + "/" + name
                refs[path] = raw_refs[OLD][path]
            pairs.append({"baseline": baseline, "opposite": opposite, "partial": partial})
            continue
        root = receipts[index]
        opposite = {**opposite, "folder": folder(root, opposite)}
        for item, catalog in ((baseline, baseline_refs), (opposite, raw_refs[root])):
            for name in FILES:
                path = item["folder"] + "/" + name
                refs[path] = {k: catalog[path][k] for k in ("bytes", "sha256")}
        pairs.append({"baseline": baseline, "opposite": opposite})
    assert len(pairs) == 233 and sum("partial" in p for p in pairs) == 1
    for path in [*archive_paths, OLD / "manifest.json", NEW / "manifest.json",
                 OLD / "user_stop_complete.json", NEW / "closeout.json",
                 OLD / "pool_roles.json", PRIOR]:
        metadata[relative(path)] = ref(path)
    save(A / "fixed_inputs.json", {
        "baseline_publication_commit": publication["commit"], "members": pairs,
        "metadata_refs": metadata, "input_refs": refs, "empty_bins": markers["empty_bins"],
        "missing_baseline_index": 35,
    })
    print(f"Froze {len(pairs)} candidate pairs and {len(refs)} saved input references")


def audit_pair(base_item, other_item, base, other):
    bs, bm, be, br, _ = base
    os, om, oe, rr, _ = other
    assert all(bs[k] == os[k] for k in ("pool", "date", "week", "partition", "KEEP"))
    assert all(bm[k] == om[k] for k in ("library_hashes", "checkpoint_sha256", "cost_per_MW", "dt_hours", "cadence"))
    marker = other_item["target"]
    common = marker["common"]
    event = oe["event"]
    assert be["status"] == "BASELINE261_SOURCE" and marker in be["candidates"]
    assert oe["prefix_matched"] and oe["new_cost_labels"] == 0 and event["common"] == common
    continuation = not marker["ordinary261_continue"]
    assert event["ordinary261_continue"] == marker["ordinary261_continue"]
    assert event["forced_continue"] == continuation
    assert event["arm"] == other_item["arm"] == ("CONT_NEXT" if continuation else "STOP_NEXT")
    t = common["step"]
    assert 1 <= t <= min(len(br), len(rr)) and 1 <= len(common["next_ids"]) <= 64
    assert all(all(x[k] == y[k] for k in audit.KEYS) for x, y in zip(br[:t-1], rr[:t-1]))
    prefix = [{k: row[k] for k in audit.KEYS} for row in br[:t-1]]
    assert audit.digest(prefix) == common["prefix_signature"]
    assert sum(row["raw_cost"] for row in br[:t-1]) == common["prefix_cost"]
    assert br[t-1]["before_hash"] == rr[t-1]["before_hash"] == common["state_hash"]
    delivered = [*audit.KEYS, "restore"]
    assert event["delivered"] == {k: rr[t-1][k] for k in delivered}
    stop, cont = (br, rr) if continuation else (rr, br)
    full = bs["complete"] and os["complete"]
    clean = all(not row["illegal"] and not row["ambiguous"] and not row["exceptions"] for row in (*br, *rr))
    row = {
        "index": other_item["index"], "baseline_index": base_item["index"],
        "pool": os["pool"], "role": os["partition"], "KEEP": os["KEEP"], "date": os["date"],
        "trigger_step": t, "state_hash": common["state_hash"],
        "quarter": other_item["quarter"], "ordinary261_continue": marker["ordinary261_continue"],
        "known_batch_queries": len(common["next_ids"]), "opposite_arm": other_item["arm"],
        "stop_steps": len(stop), "cont_steps": len(cont),
        "stop_complete": stop[-1]["complete"], "cont_complete": cont[-1]["complete"],
        "clean_both": clean, "trigger_action_changed": br[t-1]["action_hash"] != rr[t-1]["action_hash"],
        "same_saved_trajectory": len(br) == len(rr) and all(
            all(a[k] == b[k] for k in audit.KEYS) for a, b in zip(br, rr)
        ),
        "status": "COMPLETE_MATCHED_COST_TARGET" if full else "INCOMPLETE_NO_COST_TARGET",
        "stop_suffix_cost": None, "cont_suffix_cost": None, "cost_cont_minus_stop": None,
    }
    if full:
        assert len(stop) == len(cont) == 2017
        s = math.fsum(x["raw_cost"] for x in stop[t-1:])
        c = math.fsum(x["raw_cost"] for x in cont[t-1:])
        whole_delta = (os["cost"] - bs["cost"]) * (1 if continuation else -1)
        assert abs(c-s-whole_delta) <= audit.tolerance(br) + audit.tolerance(rr)
        row.update(stop_suffix_cost=s, cont_suffix_cost=c, cost_cont_minus_stop=c-s)
    return row


def state_key(row):
    return row["date"], row["trigger_step"], row["state_hash"]


def aggregate(rows):
    valid = [r for r in rows if r["cost_cont_minus_stop"] is not None]
    lower = [r for r in valid if r["cost_cont_minus_stop"] < 0]
    higher = [r for r in valid if r["cost_cont_minus_stop"] > 0]
    return {
        "rows": len(rows), "valid": len(valid), "clean_valid": sum(r["clean_both"] for r in valid),
        "lower": len(lower), "higher": len(higher), "equal": len(valid)-len(lower)-len(higher),
        "lower_states": len({state_key(r) for r in lower}), "higher_states": len({state_key(r) for r in higher}),
        "lower_dates": sorted({r["date"] for r in lower}), "higher_dates": sorted({r["date"] for r in higher}),
        "lower_at_ordinary_stop": sum(not r["ordinary261_continue"] for r in lower),
        "lower_at_ordinary_stop_states": len({state_key(r) for r in lower if not r["ordinary261_continue"]}),
        "lower_at_ordinary_stop_dates": sorted({r["date"] for r in lower if not r["ordinary261_continue"]}),
    }


def run():
    assert not (O / "started.json").exists()
    assert not git("status", "--porcelain").strip()
    head = git("rev-parse", "HEAD").decode().strip()
    assert git("ls-remote", "lntano-src", "refs/heads/main").decode().split()[0] == head
    for path in (A / "readout.py", A / "fixed_inputs.json", A / "protocol.md"):
        assert git("show", head + ":" + path.relative_to(I).as_posix()) == path.read_bytes()
    fixed = load(A / "fixed_inputs.json")
    for path, expected in fixed["metadata_refs"].items():
        assert ref(R / path) == expected, path
    audit.O = O
    audit.REFS = fixed["input_refs"]
    audit.START = time.perf_counter()
    save(O / "started.json", {"source_commit": head, "wall_cap_s": 600})
    grouped = defaultdict(list)
    for pair in fixed["members"]:
        grouped[pair["baseline"]["index"]].append(pair)
    roles = {r["pool"]: r for r in load(OLD / "pool_roles.json")}
    result = []
    episode_rows = 0
    for group in grouped.values():
        baseline = group[0]["baseline"]
        role = roles[baseline["pool"]]
        key = "fit_dates" if baseline["partition"] == ROLES[0] else "cal_dates"
        assert baseline["date"] in role[key] and baseline["date"] not in role["outer_dates_excluded"]
        assert baseline["KEEP"] == role["KEEP"]
        base = audit.audit_episode(baseline)
        episode_rows += len(base[3])
        for pair in group:
            assert pair["baseline"] == baseline
            item = pair["opposite"]
            if "partial" in pair:
                partial = audit.js(pair["partial"] + "/partial_intervention.json")
                saved_rows = audit.frozen(pair["partial"] + "/partial_steps.jsonl").splitlines()
                assert partial["steps"] == len(saved_rows) == 1079 and partial["prefix_matched"]
                row = {
                    "index": item["index"], "baseline_index": baseline["index"],
                    "pool": item["pool"], "role": item["partition"], "KEEP": item["KEEP"],
                    "date": item["date"], "status": "USER_STOP_PARTIAL_NO_COST_TARGET",
                    "partial_saved_steps": len(saved_rows), "cost_cont_minus_stop": None,
                }
            else:
                other = audit.audit_episode(item)
                episode_rows += len(other[3])
                row = audit_pair(baseline, item, base, other)
            result.append(row)
            save(O / "pair_receipts" / f"{item['index']:03d}.json", row)
    audit.checktime()
    result.sort(key=lambda r: r["index"])
    assert [r["index"] for r in result] == list(range(144, 377))
    save(O / "pairs.json", result)
    fields = list(dict.fromkeys(k for row in result for k in row))
    with (O / "paired_costs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(result)
    stats = {
        "all": aggregate(result), "KEEP": aggregate([r for r in result if r["KEEP"]]),
        "active_fit8": aggregate([r for r in result if not r["KEEP"] and r["role"] == ROLES[0]]),
        "active_cal4": aggregate([r for r in result if not r["KEEP"] and r["role"] == ROLES[1]]),
        "by_pool_role": {pool: {role: aggregate([r for r in result if r["pool"] == pool and r["role"] == role])
                               for role in ROLES} for pool in roles},
        "status_counts": dict(Counter(r["status"] for r in result)),
        "matched_completion": dict(Counter(
            f"STOP={r['stop_complete']},CONT={r['cont_complete']}" for r in result if "stop_complete" in r
        )),
        "baselines_read_once": len(grouped), "accepted_opposites": 232,
        "empty_bins": len(fixed["empty_bins"]), "physical_rows_audited": episode_rows,
        "new_fits": 0, "new_predictions": 0, "new_physics": 0,
        "wall_s": time.perf_counter() - audit.START,
    }
    groups = defaultdict(list)
    for row in result:
        if row["cost_cont_minus_stop"] not in (None, 0):
            groups[state_key(row)].append(row)
    save(O / "nonzero_state_groups.json", [
        {"date": k[0], "step": k[1], "state_hash": k[2], "records": v} for k, v in sorted(groups.items())
    ])
    save(O / "summary.json", stats)
    print(json.dumps({k: stats[k] for k in ("all", "active_fit8", "active_cal4", "status_counts", "wall_s")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        try:
            run()
        except Exception:
            save(O / "readout_error.json", {"traceback": traceback.format_exc()})
            raise
