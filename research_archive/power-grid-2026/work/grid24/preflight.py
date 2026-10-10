"""Existing rank fidelity and observable-risk expansion tests; no AC rollout."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid24")]
from common import np, make_env, ResidualControl, action_vector, digest, write_json
from guard_adapter import Search

OUT = ROOT / "outputs/grid24"
D = json.loads((OUT / "design.json").read_text())
env = make_env()
obs0 = env.reset()
ctl = ResidualControl(env, obs0)
before_count = int(env.nb_highres_called)
checks = []
fake_count = 0
try:
    for wi, week in enumerate(D["weeks"]):
        p = sorted((ROOT / "outputs/grid21/inputs/development").glob(week + "__*.json"))[0]
        meta = json.loads(p.read_text())
        source = ROOT / meta["source"]
        with np.load(source.with_name(source.stem + "_state.npz")) as data:
            obs = obs0.copy(); obs.from_vect(data["observation"])
            base = env.action_space(); base.from_vect(data["base_action"])
        expected = json.loads((ROOT / "outputs/grid21/orders" / (p.stem + ".json")).read_text())["orders"]
        for m in D["methods"]:
            search = Search(env, ctl, m["rule"], 128, ROOT / m["checkpoint"] if m["checkpoint"] else None)
            eligible, reps, aliases, actions, changed = search.prepare(obs, base)
            order, prior, features, times = search.rank(obs, base, eligible, reps, aliases)
            assert order == expected[m["rule"]]
            assert digest(prior) == meta["prior_hash"]
            checks.append({"week": week, "rule": m["rule"], "rank_fidelity": True})
        if wi == 0:
            assert len(reps) > 128 and float(obs.rho.max()) > .91
            lookup = {digest(action_vector(actions[i] + base)): i for i in reps}
            wanted = order[128]
            seen = []
            original = obs.simulate

            def trial(name, mode, winner_rho, expected_expansion, invisible_reward=3.):
                global fake_count
                seen.clear()
                def feedback(action):
                    i = lookup[digest(action_vector(action))]
                    assert i not in seen
                    seen.append(i)
                    if mode == "none": rho = float(obs.rho.max()) + 1.
                    else: rho = .7 if i == wanted else winner_rho
                    illegal = mode == "illegal"
                    rr = invisible_reward if i == wanted else 1.
                    return SimpleNamespace(rho=np.asarray([rho])), rr, False, {
                        "exception": [], "is_illegal": illegal, "is_ambiguous": False}
                obs.simulate = feedback
                result, log, _, _ = search.choose(obs, base, 0.)
                assert log["fallback"] == expected_expansion
                assert len(seen) == (len(reps) if expected_expansion else 128)
                assert seen[:128] == order[:128]
                assert len(set(seen)) == len(seen) == log["public_queries"]
                if expected_expansion and mode not in ["none", "illegal"]:
                    assert log["chosen_pool_id"] == wanted
                if mode in ["none", "illegal"]:
                    assert result is None and log["chosen_pool_id"] is None
                checks.append({"test": name, "passed": True, "unique_fake_queries": len(seen)})
                fake_count += len(seen)
                return log
            try:
                low = trial("safe_shortlist_keeps_budget", "valid", .899, False)
                other = trial("unqueried_feedback_invariance", "valid", .899, False, invisible_reward=900.)
                for key in ["order", "shortlist", "chosen_pool_id", "guard_reason", "public_queries"]:
                    assert low[key] == other[key]
                trial("at_safe_boundary_queries_only_complement", "valid", .9, True)
                trial("above_safe_boundary_queries_only_complement", "valid", .91, True)
                trial("no_admissible_fallback", "none", .91, True)
                trial("illegal_candidates_never_delivered", "illegal", .91, True)
            finally:
                obs.simulate = original
    assert int(env.nb_highres_called) == before_count
    for name, sha in json.loads((OUT / "preservation_hashes.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
finally:
    env.close()
write_json(OUT / "preflight.json", {"passed": True, "checks": checks, "fake_feedback_queries": fake_count,
    "new_ac_forecasts": 0, "new_physical_steps": 0, "new_model_fits": 0,
    "scope": "Saved public-input rank fidelity, cache uniqueness, exact boundary, illegal rejection and unqueried-feedback invariance. No production safety proof."})
freeze_paths = [p for p in (ROOT / "work/grid24").glob("*.py")]
freeze_paths += [ROOT / "work/grid23/canonical_adapter.py", ROOT / "work/grid08/common.py"]
write_json(OUT / "code_freeze.json", {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in freeze_paths})
write_json(OUT / "preflight_counts.json", {"checks": len(checks), "fake_queries": fake_count})
print(json.dumps({"passed": True, "checks": len(checks), "fake_queries": fake_count}))
