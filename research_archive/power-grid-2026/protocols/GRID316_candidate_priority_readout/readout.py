"""Fixed-budget, saved locked-station candidate ranking. No Grid2Op environment."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT = Path("D:/Codex/GRL研究")
ARCHIVE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/grid316_candidate_priority_readout"
LEDGER = "outputs/grid139_global_feasibility/call_ledger.json"
OLD_HASHES = "outputs/grid139_global_feasibility/source_hashes.json"
PROBE = ARCHIVE.parent / "GRID315_one_batch_probe/fixed_scope.json"
BUDGET = 64
WALL_CAP = 600
DECODER = ROOT / "work/grid111_action_feedback"
FEATURES = ROOT / "work/grid173_public_state"
FLOW_FIELDS = (
    ("gen_pos_topo_vect", "gen_p", 1),
    ("load_pos_topo_vect", "load_p", -1),
    ("line_or_pos_topo_vect", "p_or", -1),
    ("line_ex_pos_topo_vect", "p_ex", -1),
    ("storage_pos_topo_vect", "storage_power", -1),
)
PUBLIC = ("call_id", "date", "scenario", "step", "M", "global_n", "global_choice", "local_queries")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    with Path(path).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def freeze():
    prior = read(ROOT / OLD_HASHES)
    members = []
    refs = {LEDGER, OLD_HASHES, PROBE.relative_to(ROOT).as_posix()}
    for row in read(ROOT / LEDGER):
        if row["partition"] != "validation":
            continue
        path = ROOT / row["source"]
        assert sha(path) == prior[row["source"]]
        state = path.with_name(path.stem + "_state.npz")
        refs.update([row["source"], state.relative_to(ROOT).as_posix()])
        if row["local_queries"] > BUDGET:
            refs.add((path.parent / "metadata.json").relative_to(ROOT).as_posix())
        members.append({**{k: row[k] for k in PUBLIC},
                        "source": row["source"], "state": state.relative_to(ROOT).as_posix()})
    assert len(members) == 62 and sum(m["local_queries"] > BUDGET for m in members) == 11
    refs.update(p.relative_to(ROOT).as_posix() for p in (
        DECODER / "public.py", DECODER / "common111.py", FEATURES / "feature_core.py",
        ROOT / "outputs/grid28/offline.json",
        ROOT / "outputs/grid98_feature_enrichment/template_metadata.json",
        ROOT / "work/grid02/ljn_nn/assets/action_N1_unsafe.npz",
        ROOT / "outputs/grid03/scoring_smoke/agent_episode/dict_action_space.json",
        ROOT / "outputs/grid03/scoring_smoke/agent_episode/dict_observation_space.json",
    ))
    save(ARCHIVE / "fixed_scope.json", {
        "source": "GRID139 frozen FRAME_GUARD global prefix and locked station over saved FULL teacher feedback",
        "members": members,
        "source_hashes": {p: sha(ROOT / p) for p in sorted(refs)},
        "score": "Ascending GRID173 imbalance on public station-local post-command bus plan and frozen current terminal flows. Equal scores keep STATIC_ID order.",
        "budget": BUDGET,
        "fallback": "For any unknown station reconnection or nonfinite power, leave entire call in STATIC_ID order. Calls with <=64 pending keep original order.",
        "comparison": "Keep identical original global prefix and incumbent. Compare each first min(64,n) on paid strict-admissible float32 reward, tie smaller pool id. Reference is all saved locked-station candidates plus incumbent. Keep all calls grouped by date and separately show matches to the nine previously exposed GRID315 inputs.",
        "decision_rule": "Retain proxy as an offline lead only if strict reward wins exceed losses, wins span at least two dates, and no date has more losses than wins. No score variants or online replacement.",
        "physical": 0, "new_predictions": 0, "fits": 0, "wall_cap_s_per_stage": WALL_CAP,
    })


def setup():
    scope = read(ARCHIVE / "fixed_scope.json")
    for path, expected in scope["source_hashes"].items():
        assert sha(ROOT / path) == expected, path
    return scope


def tail(call, member, subs):
    order = call["order"]
    assert call["order_mode"] == "STATIC_ID" and len(order) == member["M"]
    if member["local_queries"] == 0:
        return None, []
    station = subs[member["global_choice"]]
    ids = {int(rep) for alias, rep in call["aliases"].items() if subs[int(alias)] == station}
    pending = [i for i in order[member["global_n"]:] if i in ids]
    assert len(pending) == member["local_queries"]
    return station, pending


def rank():
    start = time.monotonic()
    scope = setup()
    OUT.mkdir(exist_ok=False)
    sys.path.insert(0, str(DECODER))
    sys.path.insert(0, str(FEATURES))
    import numpy as np
    from public import Decoder, digest
    from feature_core import imbalance
    decoder = Decoder()
    rows = []
    for member in scope["members"]:
        assert time.monotonic() - start < WALL_CAP
        path = ROOT / member["source"]
        call = read(path)
        station, pending = tail(call, member, decoder.subs)
        new = pending
        scores = None
        fallback = "whole_tail_fits_budget"
        with np.load(ROOT / member["state"], allow_pickle=False) as data:
            state_hash = digest(data["observation"])
            base_hash = digest(data["base_action"])
        assert state_hash == call["state_hash"]
        if len(pending) > BUDGET:
            _, observation, plans, unknown, _ = decoder.decode(member["source"])
            a, b = decoder.bounds[station:station + 2]
            positions = set(range(a, b))
            power = np.zeros(len(observation.topo_vect), dtype=float)
            for pos, field, sign in FLOW_FIELDS:
                power[getattr(decoder.ac, pos)] = sign * getattr(observation, field).astype(float)
            power[observation.topo_vect <= 0] = 0
            power = power[a:b]
            if any(unknown[i] & positions for i in pending):
                fallback = "unknown_reconnected_bus"
            elif not np.isfinite(power).all():
                fallback = "nonfinite_current_power"
            else:
                scores = [imbalance(plans[i][a:b], power) for i in pending]
                assert all(math.isfinite(s) for s in scores)
                new = [pending[k] for k in sorted(range(len(pending)), key=scores.__getitem__)]
                fallback = None
        rows.append({**{k: member[k] for k in PUBLIC}, "state_hash": state_hash, "base_hash": base_hash,
                     "station": station, "static": pending, "ranked": new,
                     "scores_in_static_order": scores, "fallback": fallback})
    save(OUT / "public_ranking.json", {"rows": rows, "decoder_checks": dict(decoder.checks),
                                     "elapsed_s": time.monotonic() - start})


def winner(candidates):
    return max((r for r in candidates if r["strict_admissible"]),
               key=lambda r: (reward(r), -r["pool_id"]), default=None)


def reward(row):
    return struct.unpack("f", struct.pack("f", row["reward"]))[0]


def candidate(row):
    return None if row is None else row["pool_id"]


def evaluate():
    start = time.monotonic()
    scope = setup()
    ranks = read(OUT / "public_ranking.json")["rows"]
    ledger = {r["call_id"]: r for r in read(ROOT / LEDGER) if r["partition"] == "validation"}
    matches = {(c["week"], c["step"], c["state_hash"], c["base_hash"]): c
               for c in read(PROBE)["cases"]}
    rows = []
    for member, rank_row in zip(scope["members"], ranks, strict=True):
        assert time.monotonic() - start < WALL_CAP
        identity = member["call_id"]
        assert identity == rank_row["call_id"]
        call = read(ROOT / member["source"])
        feedback = {r["pool_id"]: r for r in call["outcomes"]}
        assert list(feedback) == call["order"]
        incumbent = winner(feedback[i] for i in call["order"][:member["global_n"]])
        assert candidate(incumbent) == member["global_choice"]
        original = rank_row["static"]
        ranked = rank_row["ranked"]
        assert len(ranked) == len(original) == len(set(ranked)) and set(ranked) == set(original)
        n = min(BUDGET, len(original))

        def choose(ids):
            options = [feedback[i] for i in ids]
            if incumbent is not None:
                options.append(incumbent)
            return winner(options)

        ref = choose(original)
        static = choose(original[:n])
        new = choose(ranked[:n])
        assert candidate(ref) == ledger[identity]["frame_choice"]
        assert (static is None) == (new is None)
        delta = 0.0 if static is None else reward(new) - reward(static)
        probe_key = (member["scenario"], member["step"], rank_row["state_hash"], rank_row["base_hash"])
        probe = matches.get(probe_key)
        if probe is not None:
            comparable = (member["global_n"] == probe["learned_queries"]
                          and candidate(incumbent) == probe["learned_candidate"]
                          and candidate(ref) == probe["frame_candidate"]
                          and len(original) == probe["saved_queries"])
        else:
            comparable = False
        rows.append({**{k: member[k] for k in PUBLIC}, "queries_each": n,
                     "incumbent": candidate(incumbent), "reference": candidate(ref),
                     "static_winner": candidate(static), "ranked_winner": candidate(new), "reward_delta": delta,
                     "static_regret": None if ref is None else reward(ref) - reward(static),
                     "ranked_regret": None if ref is None else reward(ref) - reward(new),
                     "reference_pos_static": original.index(candidate(ref)) + 1 if candidate(ref) in original else None,
                     "reference_pos_ranked": ranked.index(candidate(ref)) + 1 if candidate(ref) in ranked else None,
                     "probe_match": probe is not None, "probe_published_counts_and_winners_agree": comparable,
                     "fallback": rank_row["fallback"]})
    save(OUT / "rows.json", rows)
    groups = {"all": rows}
    for day in sorted({r["date"] for r in rows}):
        groups[day] = [r for r in rows if r["date"] == day]
    metrics = {}
    for group, members in groups.items():
        long = [r for r in members if r["local_queries"] > BUDGET]
        metrics[group] = {
            "calls": len(members), "empty": sum(not r["local_queries"] for r in members),
            "long": len(long), "eligible": sum(r["fallback"] is None for r in long),
            "each_budget_total": sum(r["queries_each"] for r in members),
            "strict_reward_wins": sum(r["reward_delta"] > 0 for r in long),
            "strict_reward_losses": sum(r["reward_delta"] < 0 for r in long),
            "strict_reward_ties": sum(r["reward_delta"] == 0 for r in long),
            "static_ref_matches": sum(r["static_winner"] == r["reference"] for r in long),
            "ranked_ref_matches": sum(r["ranked_winner"] == r["reference"] for r in long),
            "static_ref_reached_reward": sum(r["static_regret"] == 0 for r in long),
            "ranked_ref_reached_reward": sum(r["ranked_regret"] == 0 for r in long),
        }
    days = [v for k, v in metrics.items() if k != "all"]
    retain = (metrics["all"]["strict_reward_wins"] > metrics["all"]["strict_reward_losses"]
              and sum(v["strict_reward_wins"] > 0 for v in days) >= 2
              and all(v["strict_reward_wins"] >= v["strict_reward_losses"] for v in days))
    save(OUT / "summary.json", {"metrics": metrics,
                              "decision": "OFFLINE_LEAD_ONLY" if retain else "CLOSE_FIXED_IMBALANCE_PROXY",
                              "elapsed_s": time.monotonic() - start,
                              "physical": 0, "new_predictions": 0, "fits": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "rank", "evaluate"))
    stage = parser.parse_args().stage
    {"freeze": freeze, "rank": rank, "evaluate": evaluate}[stage]()
