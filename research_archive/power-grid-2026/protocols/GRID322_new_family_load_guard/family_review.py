"""Freeze all four GRID309 families and inspect their saved unguarded stops."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import time

R = Path("D:/Codex/GRL研究")
I = R / "work/GRL-results-integration"
A = Path(__file__).resolve().parent
O = R / "outputs/grid322_new_family_load_guard"
SOURCE = A.parent / "GRID309_action_opportunity_preparation"
PREV = A.parent / "GRID320_load_guard_closed_loop"
OLD = R / "outputs/grid309_action_opportunity_preparation"
SCOPE = "GRID322_FIXED_09_NEW_FAMILY"
CUTOFF = .9
CAPS = {"wall_s": 2700, "episode_s": 1200, "physical_steps": 32272, "public_forecasts": 200000}
ARMS = ("FRAME_GUARD", "PARENT208", "PARENT208_GUARD", "BOUNDED261_GUARD")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, data):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(I), *args])


def tracked(path, head):
    assert git("show", head + ":" + path.relative_to(I).as_posix()) == path.read_bytes(), path


def freeze():
    head = git("rev-parse", "HEAD").decode().strip()
    assert not O.exists()
    sources = [
        (SOURCE / "design.json", OLD / "design.json"),
        (SOURCE / "generation_public/generated_freeze.json", OLD / "generated_freeze.json"),
        (SOURCE / "baseline_public/baseline_manifest.json", OLD / "baseline_manifest.json"),
        (SOURCE / "baseline_public/baseline_raw_manifest.json", OLD / "baseline_raw_manifest.json"),
        (SOURCE / "baseline_public/runtime_design.json", OLD / "runtime_design.json"),
        (SOURCE / "collection_results/collection_closeout.json", OLD / "collection_closeout.json"),
    ]
    deps = (PREV / "design.json", PREV / "runtime.py", SOURCE / "source_freeze.json",
            SOURCE / "published_model_bindings.json", SOURCE / "collection_results/raw_records_manifest.json")
    for path, local in sources:
        tracked(path, head)
        assert path.read_bytes() == local.read_bytes()
    for path in deps:
        tracked(path, head)
    old = read(OLD / "design.json")
    calendar = old["calendar"]["dates"]
    known = read(PREV / "design.json")
    assert known["load_guard_threshold"] == CUTOFF
    assert len(calendar) == 4 and all(x["model_pool"] == "f0_s0" and x["environment_seed"] == 0 for x in calendar)
    assert {c["scenario"] for c in calendar}.isdisjoint({r["week"] for r in known["execution_order"]})
    manifest = read(OLD / "baseline_manifest.json")
    assert len(manifest["records"]) == 4 and read(OLD / "collection_closeout.json")["CONT_replays_accepted"] == 0
    runtime = read(OLD / "runtime_design.json")
    assert old["environment"] == runtime["environment"]
    baseline_raw = read(OLD / "baseline_raw_manifest.json")
    old_items = {r["week"]: r for r in runtime["execution_order"]}
    receipts = {r["summary"]["week"]: r for r in manifest["records"]}
    baselines, items = [], []
    for j, date in enumerate(calendar):
        week = date["scenario"]
        item = old_items[week]
        summary = receipts[week]["summary"]
        assert item["method"] == summary["method"] and item["pool"] == "f0_s0"
        assert item["target"] is None and item["within_pair"] == 0
        folder = OLD / "runs" / f"r0__{week}__{item['method']}"
        refs = {}
        for rel, meta in baseline_raw.items():
            if Path(rel).parent.as_posix() == folder.relative_to(R).as_posix():
                if re.fullmatch(r"call[0-9]{3,}\.json", Path(rel).name) or Path(rel).name in ("summary.json", "steps.jsonl.gz"):
                    refs[rel] = meta["sha256"]
        assert len(refs) == summary["n1_calls"] + 2
        baselines.append(dict(week=week, folder=folder.relative_to(R).as_posix(), files=refs,
                              steps=summary["steps"], complete=summary["complete"],
                              candidate_queries=summary["n1_forecasts"], wall_s=summary["wall_s"]))
        for arm in ARMS[j:] + ARMS[:j]:
            model_arm = "FRAME_GUARD" if arm == "FRAME_GUARD" else "BOUNDED261" if arm == "BOUNDED261_GUARD" else "PARENT208"
            items.append(dict(index=len(items), week=week, date=date["date"], repeat=0,
                method=arm if arm=="FRAME_GUARD" else arm+"__f0_s0", arm=model_arm, evaluation_arm=arm,
                load_guard=arm.endswith("_GUARD") and arm!="FRAME_GUARD",
                completion_mode=model_arm, model_seed=None if arm=="FRAME_GUARD" else 0,
                pool=None if arm=="FRAME_GUARD" else "f0_s0", within_pair=0, checkpoint=None,
                **{k: item[k] for k in ("policy","rule","budget","cache_dc","cache_safe","order_mode")},
                stage_phase="all", partition="other_family_development"))
    assert len(items) == 16
    design = dict(id=SCOPE, environment=old["environment"], environment_seed=0, pool="f0_s0",
                  load_guard_threshold=CUTOFF, caps=CAPS, calendar=calendar,
                  execution_order=items, saved_unprotected261=baselines,
                  timing="Serial, one pass, cyclic first arm across the four fixed dates. Only fresh equal-horizon timing.",
                  criteria="Freeze all four old GRID309 families regardless of trigger or completion. Before live proposal, require at least one saved guard trigger on a completed unprotected261 week. If absent, close the batch. No replacement dates or threshold changes.",
                  interpretation="Fresh FRAME, unguarded208, guarded208 and guarded261 plus archived unguarded261 for cost/query/trajectory only. Same initial state required. No FULL reference, no independent blinded claim. Complete cost only on complete pairs; safety and completion separate.",
                  source_hashes={p.relative_to(R).as_posix(): sha(p) for p in (*deps, *(p for pair in sources for p in pair))})
    save(A / "design.json", design)


def audit():
    start = time.monotonic()
    d = read(A / "design.json")
    assert d["load_guard_threshold"] == CUTOFF
    for path, digest in d["source_hashes"].items():
        assert sha(R / path) == digest
    rows, day = [], []
    for m in d["saved_unprotected261"]:
        total = stops = triggered = pending = 0
        for path, digest in m["files"].items():
            p = R / path
            assert sha(p) == digest, path
            if not re.fullmatch(r"call[0-9]{3,}\.json", p.name):
                continue
            c = read(p)
            assert c["learning_method"] == "BOUNDED261"
            total += c["public_queries"]
            d0 = c["learned_decisions"][-1] if c["learned_decisions"] else None
            if not d0 or d0["action"] or not d0["optional_extension"]:
                continue
            outcomes = c["outcomes"]
            incumbent = max((x for x in outcomes if x["strict_admissible"]),
                key=lambda x: (struct.unpack("f", struct.pack("f", x["reward"]))[0], -x["pool_id"]))
            assert incumbent["pool_id"] == c["chosen_pool_id"]
            assert d0["paid_queries"] == len(outcomes) == c["public_queries"]
            paid = {x["pool_id"] for x in outcomes}
            tail = [i for i in c["completion_candidate_ids"] if i not in paid]
            assert len(tail) == d0["remaining"] > 0 and tail[:64] == d0["next_ids"]
            fire = incumbent["rho"] > CUTOFF
            stops += 1
            triggered += fire
            pending += len(tail) if fire else 0
            rows.append(dict(week=m["week"], step=c["step"], state_hash=c["state_hash"],
                             incumbent=incumbent["pool_id"], rho=incumbent["rho"], parent_CONT=d0["parent_CONT"],
                             triggered=fire, pending=len(tail), source=path))
        assert total == m["candidate_queries"]
        day.append({k: m[k] for k in ("week","steps","complete","candidate_queries","wall_s")} |
                   dict(stops=stops, triggers=triggered, pending_if_applied_to_saved_states=pending))
    O.mkdir(exist_ok=False)
    save(O / "saved_stops.json", rows)
    save(O / "family_review.json", dict(families=day, proposed_new_runs=16,
        completed_week_has_trigger=any(r["complete"] and r["triggers"] for r in day),
        new_fits=0,new_physics=0,new_predictions=0,elapsed_s=time.monotonic()-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "audit"))
    {"freeze": freeze, "audit": audit}[parser.parse_args().stage]()
