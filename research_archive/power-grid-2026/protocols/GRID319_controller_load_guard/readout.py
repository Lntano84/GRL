"""Fixed 0.9 load guard on previously matched and paid search tails."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import runpy
import time
from zipfile import ZipFile

A = Path(__file__).resolve().parent
PROBE = A.parent / "GRID315_one_batch_probe/readout.py"
api = runpy.run_path(str(PROBE))
R, P, PAYLOAD, read, save, sha, git, verify, best, reward = (
    api[k] for k in ("R", "P", "PAYLOAD", "read", "save", "sha", "git", "verify", "best", "reward")
)
OUT = R / "outputs/grid319_controller_load_guard"
SCOPE = A / "fixed_scope.json"
CONTROLLER = R / "work/grid02/ljn_nn/LJNAgent.py"
CUTOFF = 0.9
STOP = {"parent_STOP_after_native_STOP", "residual_STOP"}


def freeze():
    fixed = P / "fixed_scope.json"
    head = git("rev-parse", "HEAD").decode().strip()
    for path in (fixed, PAYLOAD, PROBE):
        verify(path, head)
    text = CONTROLLER.read_text(encoding="utf8")
    assert "rho_safe: float = 0.9" in text and "if _obs.rho.max() > self.rho_safe" in text
    pairs = read(PAYLOAD)
    counts = Counter()
    cases = []
    for pair in pairs:
        for row in pair["qualified"]:
            counts[row["stop_mechanism"]] += 1
            if row["stop_mechanism"] in STOP and row["saved_queries"] > 0:
                cases.append({**row, **{k: pair[k] for k in ("week", "arm", "repeat")}})
    assert len(pairs) == 16 and len(cases) == 36
    raw = read(fixed)
    wanted = {(r["week"], r["repeat"]) for r in cases}
    sources = [{"week": m["item"]["week"], "repeat": m["item"]["repeat"],
                "calls": {p: m["files"][p] for p in m["calls"]}}
               for m in raw["members"] if m["item"]["arm"] == "FRAME_GUARD"
               and (m["item"]["week"], m["item"]["repeat"]) in wanted]
    assert len(sources) == len(wanted)
    save(SCOPE, {
        "rule": "At an existing learned STOP with a nonempty same-station tail, buy all of that tail in original order iff the strict-admissible incumbent's already observed forecast rho exceeds 0.9. Keep original reward and tie-break. Equality does not reopen.",
        "mechanism_basis": "Use original LJNAgentTopoNN rho_safe default and strict greater-than branch for continuous optimization. No threshold search.",
        "scope": "All GRID313 matched calls attributed to a learned STOP with positive skipped queries, both arms and both repeats. Keep other matched mechanisms and excluded counts visible. Outcomes on unmatched states and later diverging states are outside scope.",
        "matched_mechanism_counts": dict(counts),
        "pair_exclusions": [{k: p[k] for k in ("week", "arm", "repeat", "frame_calls", "learned_calls",
                                             "identical_input_calls", "qualified_calls", "input_match_exclusions")}
                            for p in pairs],
        "cases": cases, "sources": sources, "raw_archive": raw["raw_archive"], "cutoff": CUTOFF,
        "source_hashes": {p.relative_to(R).as_posix(): sha(p.read_bytes())
                          for p in (fixed, PAYLOAD, PROBE, CONTROLLER)},
        "limits": {"wall_s": 600, "new_physics": 0, "new_fits": 0, "new_predictions": 0}
    })


def evaluate(case, call):
    outcomes = call["outcomes"]
    paid, tail = case["learned_queries"], case["saved_queries"]
    assert call["public_queries"] == len(outcomes) == paid + tail == case["frame_queries"]
    seen = {r["pool_id"] for r in outcomes[:paid]}
    remaining = [i for i in call["completion_candidate_ids"] if i not in seen]
    assert [r["pool_id"] for r in outcomes[paid:]] == remaining
    assert all(r["phase"] == "local_completion" for r in outcomes[paid:])
    incumbent, full = best(outcomes[:paid]), best(outcomes)
    assert incumbent["pool_id"] == case["learned_candidate"] and full["pool_id"] == case["frame_candidate"]
    assert reward(full) - reward(incumbent) == case["instant_reward_loss"]
    triggered = incumbent["rho"] > CUTOFF
    winner = full if triggered else incumbent
    return {**case, "incumbent_rho": incumbent["rho"], "frame_winner_rho": full["rho"],
            "triggered": triggered, "guarded_candidate": winner["pool_id"],
            "extra_queries": tail if triggered else 0, "remaining_savings": 0 if triggered else tail,
            "restored_winner": triggered and not case["same_candidate"],
            "remaining_missed_winner": winner["pool_id"] != full["pool_id"],
            "remaining_instant_reward_loss": reward(full) - reward(winner)}


def audit():
    start = time.perf_counter()
    assert not OUT.exists()
    scope = read(SCOPE)
    assert scope["cutoff"] == CUTOFF
    for path, digest in scope["source_hashes"].items():
        assert sha((R / path).read_bytes()) == digest
    cases = defaultdict(list)
    for case in scope["cases"]:
        key = (case["week"], case["repeat"], case["step"], case["state_hash"])
        cases[key].append(case)
    rows = []
    with ZipFile(Path(scope["raw_archive"]) / "raw_objects.zip") as archive:
        for source in scope["sources"]:
            for name, ref in source["calls"].items():
                assert time.perf_counter() - start < scope["limits"]["wall_s"]
                path = R / name
                data = path.read_bytes() if path.is_file() else archive.read(ref["object"])
                assert len(data) == ref["bytes"] and sha(data) == ref["sha256"]
                call = json.loads(data)
                key = (source["week"], source["repeat"], call["step"], call["state_hash"])
                for case in cases.pop(key, []):
                    assert call["learning_method"] == "FRAME_GUARD"
                    rows.append(evaluate(case, call))
    assert not cases and len(rows) == len(scope["cases"])
    canonical = sorted((r for r in rows if r["arm"] == "PARENT208" and r["repeat"] == 0),
                       key=lambda r: (r["week"], r["step"]))
    index = {(r["week"], r["step"], r["state_hash"], r["base_hash"]): r for r in canonical}
    assert len(index) == 9
    for r in rows:
        key = (r["week"], r["step"], r["state_hash"], r["base_hash"])
        assert all(index[key][k] == v for k, v in r.items() if k not in ("arm", "repeat"))
    groups = {week: [r for r in canonical if r["week"] == week]
              for week in sorted({p["week"] for p in scope["pair_exclusions"]})}
    groups["all"] = canonical
    totals = {}
    for group, members in groups.items():
        totals[group] = {"calls": len(members),
                         **{k: sum(r[k] for r in members) for k in
                            ("saved_queries", "extra_queries", "remaining_savings", "triggered",
                             "restored_winner", "remaining_missed_winner", "remaining_instant_reward_loss")},
                         "original_missed_winner": sum(not r["same_candidate"] for r in members)}
        assert totals[group]["saved_queries"] == totals[group]["extra_queries"] + totals[group]["remaining_savings"]
    save(OUT / "rows.json", sorted(rows, key=lambda r: (r["week"], r["step"], r["repeat"], r["arm"])))
    save(OUT / "summary.json", {"by_week": totals, "paired_rows": len(rows), "unique_inputs": len(index),
                                "same_across_arms_and_repeats": True, **scope["limits"],
                                "elapsed_s": time.perf_counter() - start})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "audit"))
    {"freeze": freeze, "audit": audit}[parser.parse_args().stage]()
