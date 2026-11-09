"""GRID326 fixed 64+64 mix using GRID325 public ranks and the original FULL ledger."""
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
OUT = ROOT / "outputs/grid326_fixed_dc_mix"
PRIOR = ARCHIVE.parent / "GRID325_dc_global_priority"
PRIOR_FILES = ("readout.py", "fixed_scope.json", "public_ranks.json", "results/delivery_hashes.json")
BASELINE = "static"
PURE_DC = "dc"
MIX = "mix"
ARMS = (BASELINE, PURE_DC, MIX)
BUDGET = 128
QUOTA = BUDGET // 2
CAP = 600


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
    previous = read(PRIOR / "results/delivery_hashes.json")
    for name in PRIOR_FILES[:-1]:
        assert sha(PRIOR / name) == previous[name], name
    save(ARCHIVE / "fixed_scope.json", {
        "previous_release": "915c6d0a2bc08b08332a6603b9c97ba8ca11b579",
        "previous_hashes": {name: sha(PRIOR / name) for name in PRIOR_FILES},
        "source_sha256": sha(Path(__file__)),
        "population": "Same 305 saved FULL calls, same original train/validation dates, no new calls.",
        "rank_rule": "Take first 64 STATIC_ID candidates, then first 64 distinct remaining IDs in frozen GRID325 DC order. Stable ties and unsupported slots inherit GRID325. The published DC128 suffices, since at most 64 of its IDs can overlap the original 64. Both original static128 and pure dc128 are controls.",
        "quota": QUOTA,
        "budget": BUDGET,
        "selection": "Saved strict admissibility and float32 immediate reward. Use lower candidate id for equal reward. A valid action beats none; equal reward with different ID is a tie. FULL winner and original FRAME action are tracked separately.",
        "keep_gate": "Validation: zero static-valid to mix-empty losses and no more than zero such losses in training, mix wins strictly more than it loses on validation, wins on at least two validation dates, and no validation date with more losses than wins. Decision only about the one offline mix, not deployment.",
        "timing": "Mix construction time local offline. The 64+64 mix inherits the GRID325 DC ranking cost; no new DC computations. No online timing.",
        "wall_cap_s_each": CAP, "new_dc_solves": 0, "new_fits": 0, "new_simulation": 0,
    })


def setup():
    scope = read(ARCHIVE / "fixed_scope.json")
    assert sha(Path(__file__)) == scope["source_sha256"]
    for name, expected in scope["previous_hashes"].items():
        assert sha(PRIOR / name) == expected, name
    old = read(PRIOR / "fixed_scope.json")
    public = read(PRIOR / "public_ranks.json")
    assert sha(PRIOR / "fixed_scope.json") == public["scope_sha256"]
    assert len(old["members"]) == len(public["rows"]) == 305
    return old, public


def hybrid(static, dc):
    assert len(static) == len(dc) == len(set(static)) == len(set(dc)) == BUDGET
    first = static[:QUOTA]
    seen = set(first)
    other = [cid for cid in dc if cid not in seen][:QUOTA]
    assert len(other) == QUOTA
    return first + other


def rank():
    start = time.monotonic()
    old, public = setup()
    results = []
    for member, row in zip(old["members"], public["rows"], strict=True):
        assert time.monotonic() - start < CAP
        assert member["call_id"] == row["call_id"] and member["M"] == row["M"] >= BUDGET
        before = time.perf_counter_ns()
        selected = hybrid(row["static128"], row["dc128"])
        elapsed_ns = time.perf_counter_ns() - before
        results.append({"call_id": row["call_id"], "mix128": selected, "mix_ns": elapsed_ns})
    save(ARCHIVE / "public_mix.json", {"scope_sha256": sha(ARCHIVE / "fixed_scope.json"),
         "parent_rank_sha256": sha(PRIOR / "public_ranks.json"), "rows": results})


def metrics(rows):
    result = {"calls": len(rows), "queries_per_arm": BUDGET * len(rows),
              "mix_changed_static_sets": sum(r["mix_changed_static_set"] for r in rows),
              "comparisons": {}}
    for other in (BASELINE, PURE_DC):
        pair = [r for r in rows if r[MIX]["reward"] is not None and r[other]["reward"] is not None]
        result["comparisons"][other] = {
            "wins": sum(r["comparison"][other] > 0 for r in rows),
            "ties": sum(r["comparison"][other] == 0 for r in rows),
            "losses": sum(r["comparison"][other] < 0 for r in rows),
            "lost_admissible": sum(r[MIX]["winner"] is None and r[other]["winner"] is not None for r in rows),
            "recovered_admissible": sum(r[MIX]["winner"] is not None and r[other]["winner"] is None for r in rows),
            "paired_both_admissible": len(pair),
            "paired_regret_reference": statistics.mean(r[other]["regret"] for r in pair) if pair else None,
            "paired_regret_mix": statistics.mean(r[MIX]["regret"] for r in pair) if pair else None,
        }
    for arm in ARMS:
        result[arm] = {
            "full_match": sum(r[arm]["full_match"] for r in rows),
            "frame_match": sum(r[arm]["frame_match"] for r in rows),
            "frame_in_prefix": sum(r[arm]["frame_in_prefix"] for r in rows),
            "missing_with_full_admissible": sum(r[arm]["winner"] is None and r["full"] is not None for r in rows),
        }
    return result


def evaluate():
    start = time.monotonic()
    old, public = setup()
    mix = read(ARCHIVE / "public_mix.json")
    assert mix["scope_sha256"] == sha(ARCHIVE / "fixed_scope.json")
    assert mix["parent_rank_sha256"] == sha(PRIOR / "public_ranks.json")
    parent = runpy.run_path(str(PRIOR / "readout.py"))
    best, choice, reward, compare = (parent[key] for key in ("best", "choice", "reward", "compare"))
    outcome_source = {r["call_id"]: r for r in read(PRIOR / "results/rows.json")}
    assert sha(PRIOR / "results/rows.json") == read(PRIOR / "results/delivery_hashes.json")["results/rows.json"]
    results = []
    for m, p, h in zip(old["members"], public["rows"], mix["rows"], strict=True):
        assert time.monotonic() - start < CAP
        assert m["call_id"] == p["call_id"] == h["call_id"]
        assert sha(ROOT / m["source"]) == old["source_hashes"][m["source"]]
        call = read(ROOT / m["source"])
        feedback = {choice(r): r for r in call["outcomes"]}
        assert list(feedback) == call["order"] and call["state_hash"] == p["state_hash"]
        old_result = outcome_source[m["call_id"]]
        full = best(feedback.values())
        frame = old_result["frame"]
        assert choice(full) == old_result["full"]
        assert h["mix128"] == hybrid(p["static128"], p["dc128"])
        row = {key: m[key] for key in ("call_id", "date", "step", "scenario", "partition")}
        row.update(full=choice(full), frame=frame, mix_changed_static_set=set(p["static128"]) != set(h["mix128"]))
        for arm, prefix in zip(ARMS, (p["static128"], p["dc128"], h["mix128"]), strict=True):
            winner = best(feedback[cid] for cid in prefix)
            if arm != MIX:
                assert choice(winner) == old_result[arm]["winner"] and reward(winner) == old_result[arm]["reward"]
            row[arm] = {"winner": choice(winner), "reward": reward(winner),
                        "frame_in_prefix": frame in prefix,
                        "frame_match": frame is not None and choice(winner) == frame,
                        "full_match": full is not None and choice(winner) == choice(full),
                        "regret": None if winner is None else reward(full) - reward(winner)}
        row["comparison"] = {other: compare(row[MIX]["reward"], row[other]["reward"])
                             for other in (BASELINE, PURE_DC)}
        results.append(row)
    splits = {}
    for partition, dates in old["dates"].items():
        rows = [r for r in results if r["partition"] == partition]
        splits[partition] = {"all": metrics(rows),
            "by_date": {date: metrics([r for r in rows if r["date"] == date]) for date in dates}}
    train = splits["train"]["all"]["comparisons"][BASELINE]
    dev = splits["validation"]["all"]["comparisons"][BASELINE]
    days = [d["comparisons"][BASELINE] for d in splits["validation"]["by_date"].values()]
    keep = (train["lost_admissible"] == dev["lost_admissible"] == 0
            and dev["wins"] > dev["losses"] and sum(d["wins"] > 0 for d in days) >= 2
            and all(d["wins"] >= d["losses"] for d in days))
    save(OUT / "rows.json", results)
    save(OUT / "summary.json", {"metrics": splits,
        "decision": "KEEP_FIXED_MIX_OFFLINE" if keep else "FIXED_MIX_DOES_NOT_CLEAR_GATE",
        "scope_sha256": sha(ARCHIVE / "fixed_scope.json"),
        "mix_sha256": sha(ARCHIVE / "public_mix.json"),
        "construct_mix_median_us": statistics.median(r["mix_ns"] for r in mix["rows"]) / 1000,
        "inherited_dc_sort_median_s": statistics.median(r["dc_and_sort_s"] for r in public["rows"]),
        "new_dc_solves": 0, "new_fits": 0, "new_simulation": 0,
        "evaluation_s": time.monotonic() - start})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "rank", "evaluate"))
    {"freeze": freeze, "rank": rank, "evaluate": evaluate}[parser.parse_args().stage]()

