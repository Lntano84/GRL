"""Inventory existing calibration-only STOP candidates using saved public fields."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

R = Path("D:/Codex/GRL研究")
I = R / "work/GRL-results-integration"
A = Path(__file__).resolve().parent
O = R / "outputs/grid305_saved_calibration_candidates"
ARCHIVE = "research_archive/power-grid-2026/protocols/"
PATHS = {
    "candidates299": ARCHIVE + "GRID299_public_state_coverage/results/public_candidates.json",
    "pairs298": ARCHIVE + "GRID298_saved_suffix_targets/results/pairs.json",
    "baseline301": ARCHIVE + "GRID301_remaining_collection/baseline_public/baseline_manifest.json",
    "markers301": ARCHIVE + "GRID301_remaining_collection/baseline_public/selected_markers.json",
    "raw301": ARCHIVE + "GRID301_remaining_collection/baseline_public/baseline_raw_manifest.json",
    "pairs303": ARCHIVE + "GRID303_saved_suffix_targets/results/pairs.json",
    "roles297": "outputs/grid297_paired_collection_preparation/pool_roles.json",
    "roles301": "outputs/grid301_remaining_collection/pool_roles.json",
    "calendar": "outputs/grid301_remaining_collection/calendar.json",
    "raw297": "archives/GRID297_paired_collection_20261102/raw_records/raw_records_manifest.json",
    "timing299": "outputs/grid299_public_state_coverage/resource_estimates.json",
    "timing302": "outputs/grid302_unstarted_opposites/finished.json",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(I), *args])


def path_for(relative):
    return (I if relative.startswith(ARCHIVE) else R) / relative


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def freeze():
    head = git("rev-parse", "HEAD").decode().strip()
    refs = {}
    for key, relative in PATHS.items():
        data = path_for(relative).read_bytes()
        if relative.startswith(ARCHIVE):
            assert git("show", head + ":" + relative) == data
        refs[key] = {"path": relative, "sha256": sha(data), "bytes": len(data)}
    assert not (A / "inputs.json").exists()
    save(A / "inputs.json", refs)


def cal_only(rows):
    active = [r for r in rows if not r["KEEP"]]
    fit = {date for row in active for date in row["fit_dates"]}
    cal = {date for row in active for date in row["cal_dates"]}
    return cal - fit


def state(date, step, digest):
    return date, step, digest


def run():
    assert not O.exists()
    assert not git("status", "--porcelain").strip()
    head = git("rev-parse", "HEAD").decode().strip()
    assert git("ls-remote", "lntano-src", "refs/heads/main").decode().split()[0] == head
    for name in ("inventory.py", "inputs.json", "protocol.md"):
        path = A / name
        assert git("show", head + ":" + path.relative_to(I).as_posix()) == path.read_bytes()
    files = json.loads((A / "inputs.json").read_text(encoding="utf8"))
    data = {}
    for key, ref in files.items():
        raw = path_for(ref["path"]).read_bytes()
        assert len(raw) == ref["bytes"] and sha(raw) == ref["sha256"]
        data[key] = json.loads(raw)
    dates = {2037: cal_only(data["roles297"]), 2038: cal_only(data["roles301"])}
    families = {new: old for old, new in data["calendar"]["old_to_new_slot_map"].items()}
    paired = {state(r["date"], r["trigger_step"], r["common_before_hash"])
              for r in data["pairs298"] if "trigger_step" in r}
    newer = {r["index"]: r for r in data["pairs303"]}
    partial = set()
    for item in data["markers301"]["new_opposite_items"]:
        marker = item["target"]["common"]
        key = state(item["date"], marker["step"], marker["state_hash"])
        if newer[item["index"]]["status"] == "USER_STOP_PARTIAL_NO_COST_TARGET":
            partial.add(key)
        else:
            paired.add(key)

    candidates = []
    for row in data["candidates299"]:
        if row["KEEP"] or row["role"] != "cal4" or row["date"] not in dates[2037] or row["ordinary261_continue"]:
            continue
        # Read only public identity fields; the old baseline source call is retained.
        names = ("pool", "date", "week", "baseline_folder", "step", "state_hash",
                 "batch", "next_ids", "public_NN_input_hash", "raw41_hash", "source_call",
                 "call_ordinal", "decision_index", "source_pair_index", "existing_first_event_pair")
        candidates.append({"year": 2037, **{k: row[k] for k in names}})
    for record in data["baseline301"]["records"]:
        row = record["summary"]
        if row["KEEP"] or row["partition"] != "own_training_cal4" or row["date"] not in dates[2038]:
            continue
        root = "grid300_new_training_dates" if record["index"] < 35 else "grid301_remaining_collection"
        baseline = f"outputs/{root}/runs/r{row['repeat']}__{row['week']}__{row['method']}"
        for target in record["source"]["candidates"]:
            if target["ordinary261_continue"]:
                continue
            c = target["common"]
            candidates.append({
                "year": 2038, "pool": row["pool"], "date": row["date"], "week": row["week"],
                "baseline_folder": baseline, "baseline_index": record["index"],
                "step": c["step"], "state_hash": c["state_hash"], "batch": c["batch"],
                "next_ids": c["next_ids"], "public_NN_input_hash": c["public_NN_input_hash"],
                "target": target,
            })
    available = defaultdict(list)
    for row in candidates:
        key = state(row["date"], row["step"], row["state_hash"])
        row["family"] = families.get(row["date"], row["date"])
        row["coverage_status"] = (
            "already_paired_state" if key in paired else
            "paid_partial_state" if key in partial else "new_state"
        )
        if row["coverage_status"] == "new_state":
            available[key].append(row)
    old_refs = {r["path"]: r for r in data["raw297"]["entries"]}
    picked = []
    for key, group in sorted(available.items()):
        row = min(group, key=lambda r: (r["pool"], r["baseline_folder"], r["batch"],
                                        r.get("call_ordinal", 0), r.get("decision_index", 0)))
        catalog = old_refs if row["year"] == 2037 else data["raw301"]
        paths = [row["baseline_folder"] + "/steps.jsonl.gz"]
        if row["year"] == 2037:
            paths.append(row["source_call"])
        picked.append({
            **row, "new_arm": "CONT_NEXT", "related_pool_records": len(group),
            "baseline_input_refs": {p: {k: catalog[p][k] for k in ("bytes", "sha256")} for p in paths},
        })
    by_date = Counter(r["date"] for r in picked)
    by_family = Counter(r["family"] for r in picked)
    other_families = sorted(set(by_family) - {"2037-10-05"})
    estimate = sum(
        data["timing299"]["simple_wall_s"] / data["timing299"]["new_episodes"]
        if r["year"] == 2037 else data["timing302"]["wall_s"] / data["timing302"]["new_accepted"]
        for r in picked
    )
    # An engineering envelope for review, rather than a runtime guarantee.
    wall_cap = 900 * math.ceil((1.5 * estimate + 600) / 900)
    forecast_cap = 50000 * math.ceil(2 * len(picked) * max(
        data["timing299"]["simple_forecasts"] / data["timing299"]["new_episodes"],
        data["timing302"]["forecasts_new"] / data["timing302"]["new_accepted"],
    ) / 50000)
    result = {
        "source_commit": head, "cal_only_dates": {str(y): sorted(d) for y, d in dates.items()},
        "candidate_records": len(candidates), "coverage_records": dict(Counter(r["coverage_status"] for r in candidates)),
        "new_states": len(picked), "by_date": dict(sorted(by_date.items())), "by_family": dict(sorted(by_family.items())),
        "new_families_beyond_existing_benefit_family": other_families,
        "decision": "PREPARE_REVIEWABLE_COLLECTION" if len(other_families) >= 2 else "CLOSE_SAVED_DATE_TOPUP",
        "estimated_serial_wall_s": estimate, "proposed_total_wall_s": wall_cap,
        "proposed_per_episode_s": 1200, "proposed_physical_steps": 2017 * len(picked),
        "proposed_forecast_cap": forecast_cap,
        "new_model_predictions": 0, "new_physical_steps": 0, "new_fits": 0,
        "new_cost_labels": 0, "physical_authorization": False,
    }
    O.mkdir()
    save(O / "candidate_inventory.json", candidates)
    save(O / "proposed_scope.json", {"summary": result, "fixed_representatives": picked})
    save(O / "summary.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    else:
        run()
