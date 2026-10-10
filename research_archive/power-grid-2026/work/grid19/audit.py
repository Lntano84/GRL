"""Independent masked candidate fields, source choice and physical cost arithmetic."""
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "work/grid08"))
from common import np, make_env, ResidualControl, action_vector, digest, write_json
OUT = ROOT / "outputs/grid19"; F = json.loads((OUT / "finished.json").read_text()); assert F["passed_engineering"]
for name, sha in json.loads((OUT / "code_freeze.json").read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
env = make_env(); initial = env.reset(); ctl = ResidualControl(env, initial); library = ctl.base.topo_n1_unsafe.topo_act_list
keys = [digest(action_vector(a)) for a in library]
steps = forecasts = source_calls = labels = changed = illegal_labels = strict_valid = 0
max_residual = 0.; comparison = []


def rows_at(path):
    with gzip.open(path, "rt", encoding="utf-8") as f: return [json.loads(line) for line in f]


def total(a): return math.fsum(float(v) for v in np.asarray(a).ravel())


try:
    for summary in F["summaries"]:
        week = summary["week"]; path = OUT / "runs" / week; rows = rows_at(path / "steps.jsonl.gz")
        meta = json.loads((path / "metadata.json").read_text()); assert meta["action_hashes"] == keys
        with np.load(path / "vectors.npz") as f: vectors = {k: f[k].copy() for k in f.files}
        assert len(rows) == summary["steps"] and len(vectors["action"]) == len(rows)
        assert abs(math.fsum(r["raw_cost"] for r in rows)-summary["cost"]) < 1e-6
        assert sum(r["simulations"] for r in rows) == summary["public_forecasts"]
        obs_before = None
        for i, row in enumerate(rows):
            a = env.action_space(); a.from_vect(vectors["action"][i]); assert digest(action_vector(a)) == row["action_hash"]
            obs = initial.copy(); obs.from_vect(vectors["observation"][i]); assert digest(obs.to_vect()) == row["after_hash"]
            if i: assert row["before_hash"] == rows[i-1]["after_hash"]
            if row["ledger"]:
                previous = row["ledger"]["previous_curtailed_mw"]
                if obs_before is not None: assert previous == total(obs_before.curtailment_mw)
                price = max(float(p) for p, mw in zip(meta["cost_per_MW"], obs.gen_p) if mw > 0)
                j = price*meta["dt_hours"]*(total(obs.gen_p)-total(obs.load_p)+total(np.abs(obs.actual_dispatch))+total(np.abs(obs.storage_power))+total(obs.curtailment_mw)-previous)
                gap = abs(j-row["raw_cost"]); max_residual = max(max_residual, gap); assert gap <= row["ledger_tolerance"]
            obs_before = obs
        call_forecasts = 0
        for file in sorted(path.glob("call[0-9][0-9][0-9].json")):
            c = json.loads(file.read_text()); source_calls += 1
            with np.load(file.with_name(file.stem+"_state.npz")) as saved:
                before = initial.copy(); before.from_vect(saved["observation"])
                base = env.action_space(); base.from_vect(saved["base_action"])
            assert digest(before.to_vect()) == c["state_hash"] == rows[c["step"]-1]["before_hash"]
            ids = [j for j, a in enumerate(library) if before.time_before_cooldown_sub[int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0])] == 0]
            assert ids == c["pool_ids"] and len(ids) == len(c["outcomes"]) == len(c["changes"])
            scores = []
            for pool_id, change, outcome in zip(ids, c["changes"], c["outcomes"]):
                original = library[pool_id]+base; assert digest(action_vector(original)) == change["original_combined_hash"]
                candidate = library[pool_id].copy(); bus = candidate.set_bus.copy(); expected = []
                for line in range(env.n_line):
                    intentional = base.line_set_status[line] > 0 or base.line_change_status[line] or base.set_bus[env.line_or_pos_topo_vect[line]] > 0 or base.set_bus[env.line_ex_pos_topo_vect[line]] > 0
                    if not before.line_status[line] and not intentional:
                        for pos in [int(env.line_or_pos_topo_vect[line]), int(env.line_ex_pos_topo_vect[line])]:
                            if bus[pos] != 0: expected.append(pos)
                            bus[pos] = 0
                assert set(expected) == set(change["changed_positions"])
                if expected: candidate.set_bus = bus
                assert digest(action_vector(candidate+base)) == outcome["combined_action_hash"]
                admitted = 0 < outcome["rho"] < c["rho_limit"] and not outcome["done"] and not outcome["exceptions"]
                assert admitted == outcome["source_admissible"]
                strict = admitted and not outcome["illegal"] and not outcome["ambiguous"]
                assert strict == outcome["strict_admissible"]
                scores.append(float(np.float32(outcome["reward"])) if admitted else -100.)
                labels += 1; changed += bool(expected); illegal_labels += outcome["illegal"]; strict_valid += strict
            chosen = None if not scores or max(scores) <= -100. else int(np.argmax(scores))
            assert chosen == c["chosen_index"]
            assert c["chosen_pool_id"] == (None if chosen is None else ids[chosen])
            call_forecasts += len(ids)
        assert call_forecasts == summary["n1_forecasts"]
        steps += len(rows); forecasts += summary["public_forecasts"]
finally:
    env.close()
assert steps == F["counts"]["physical_steps"] and forecasts == F["counts"]["public_forecasts"]
write_json(OUT / "audit.json", {"passed": True, "physical_rows": steps, "candidate_labels": labels, "source_calls": source_calls,
                               "masked_candidates": changed, "illegal_candidate_labels": illegal_labels, "strict_improving_labels": strict_valid,
                               "max_independent_cost_residual": max_residual, "extra_forecasts": 0, "model_fits": 0, "comparison": comparison,
                               "scope": "Independent saved-data mask/choice/accounting for new teacher collection; no old-week comparisons, no model fits or independent AC rerun."})
print(json.dumps({"comparison": comparison, "calls": source_calls, "labels": labels, "masked": changed, "illegal": illegal_labels}))
