"""Read-only action semantics audit on already exposed public search states."""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid17")]
from common import np, make_env, ResidualControl, action_vector, digest, write_json
from mask import mask_disconnected
OUT = ROOT / "outputs/grid17"; OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / "qualification.json").exists()
for name, sha in json.loads((ROOT / "outputs/grid15/delivery_manifest.json").read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
env = make_env(); env.seed(0); obs0 = env.reset(); ctl = ResidualControl(env, obs0)
library = ctl.base.topo_n1_unsafe.topo_act_list
rows = []; targets = []; tests = {}
try:
    # Explicitly check that the public setter actually clears a previous value.
    candidate = library[0].copy(); positions = np.flatnonzero(candidate.set_bus)
    probe = candidate.copy(); bus = probe.set_bus.copy(); bus[positions[0]] = 0; probe.set_bus = bus
    assert probe.set_bus[positions[0]] == 0 and candidate.set_bus[positions[0]] != 0
    assert digest(action_vector(probe)) != digest(action_vector(candidate))
    tests["set_bus_zero_removes_assignment"] = True
    # Connected reset is invariant; intentional base reconnect survives masking.
    connected = obs0.copy(); connected.line_status[:] = True
    old, new, changed, _ = mask_disconnected(candidate, env.action_space(), connected, env)
    assert not changed and np.array_equal(action_vector(old), action_vector(new))
    tests["connected_state_identity"] = True
    line = int(env.line_or_to_subid.argmax())
    fake = obs0.copy(); fake.line_status[line] = False
    reco = env.action_space({"set_line_status": [(line, 1)]})
    old, new, _, intent = mask_disconnected(candidate, reco, fake, env)
    assert line in intent and new.line_set_status[line] == 1
    tests["explicit_base_reconnection_retained"] = True
    for week in json.loads((ROOT / "outputs/grid15/design.json").read_text())["weeks"]:
        first_target = None
        for p in sorted((ROOT / f"outputs/grid15/runs/{week}").glob("call[0-9][0-9][0-9].json")):
            call = json.loads(p.read_text())
            with np.load(p.with_name(p.stem + "_state.npz")) as data:
                obs = obs0.copy(); obs.from_vect(data["observation"])
                base = env.action_space(); base.from_vect(data["base_action"])
            assert digest(obs.to_vect()) == call["state_hash"]
            possible = []
            for pool_id, outcome in zip(call["pool_ids"], call["outcomes"]):
                if not outcome["illegal"]: continue
                a = library[pool_id]; before = digest(action_vector(a)); base_before = digest(action_vector(base))
                original, masked, positions, intent = mask_disconnected(a, base, obs, env)
                assert digest(action_vector(original)) == outcome["combined_action_hash"]
                assert before == digest(action_vector(a)) and base_before == digest(action_vector(base))
                assert np.array_equal(original.line_set_status, masked.line_set_status)
                assert np.array_equal(original.line_change_status, masked.line_change_status)
                for name in ["redispatch", "storage_p", "curtail"]:
                    assert np.array_equal(getattr(original, name), getattr(masked, name))
                delta = np.flatnonzero(original.set_bus != masked.set_bus)
                assert delta.tolist() == sorted(positions) or set(delta.tolist()) == set(positions)
                original_lines, _ = original.get_topological_impact(obs.line_status)
                masked_lines, _ = masked.get_topological_impact(obs.line_status)
                blocked_before = np.flatnonzero(original_lines & (obs.time_before_cooldown_line > 0)).tolist()
                blocked_after = np.flatnonzero(masked_lines & (obs.time_before_cooldown_line > 0)).tolist()
                row = {"week": week, "call_file": p.relative_to(ROOT).as_posix(), "step": call["action_step"],
                       "state_hash": call["state_hash"], "pool_id": pool_id, "changed_positions": positions,
                       "intentional_base_lines": intent, "blocked_before": blocked_before, "blocked_after": blocked_after,
                       "recorded_exceptions": outcome["exceptions"], "original_hash": digest(action_vector(original)),
                       "masked_hash": digest(action_vector(masked))}
                rows.append(row)
                if blocked_before and not blocked_after and positions: possible.append(row)
            if first_target is None and possible:
                first_target = {"week": week, "step": call["action_step"], "call_file": p.relative_to(ROOT).as_posix(),
                                "state_hash": call["state_hash"], "candidate_ids": [r["pool_id"] for r in possible[:3]]}
        if first_target is not None: targets.append(first_target)
finally:
    env.close()
write_json(OUT / "qualification.json", {"passed": True, "tests": tests, "recorded_illegal_candidates": len(rows),
    "clear_cooldown_conflicts": sum(bool(r["blocked_before"] and not r["blocked_after"]) for r in rows),
    "rows": rows, "targets": targets, "physical_steps": 0, "public_forecasts": 0, "model_fits": 0,
    "scope": "Known Grid2Op action-semantic repair on exposed training states. Topological-impact checks do not establish powerflow feasibility or benefit. Targets chosen for engineering conflict removal, not policy-quality inference.",
    "reference": "https://arxiv.org/html/2503.15190v2#S4.SS4.SSS2"})
print(json.dumps({"tests": tests, "illegal": len(rows), "clearable": sum(bool(r["blocked_before"] and not r["blocked_after"]) for r in rows), "targets": targets}))
