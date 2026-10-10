"""Compare legal inputs/ranks to sealed V1, plus fake-feedback fallback tests."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid23")]
from common import np, make_env, ResidualControl, action_vector, digest, write_json
from canonical_adapter import Search
OUT = ROOT / "outputs/grid23"; OUT.mkdir(parents=True, exist_ok=True)
D = json.loads((ROOT / "outputs/grid21/design.json").read_text())
assert json.loads((ROOT / "outputs/grid21/audit.json").read_text())["passed"]
for stage in ["grid20", "grid21"]:
    for name, sha in json.loads((ROOT / f"outputs/{stage}/delivery_manifest.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
env = make_env(); obs0 = env.reset(); ctl = ResidualControl(env, obs0)
initial_queries = int(env.nb_highres_called)
checked = []; fake_queries = 0
try:
    for wi, week in enumerate(D["development_weeks"]):
        meta_path = sorted((ROOT / "outputs/grid21/inputs/development").glob(week + "__*.json"))[0]
        meta = json.loads(meta_path.read_text()); identifier = meta_path.stem
        source = ROOT / meta["source"]
        with np.load(source.with_name(source.stem + "_state.npz")) as f:
            obs = obs0.copy(); obs.from_vect(f["observation"])
            base = env.action_space(); base.from_vect(f["base_action"])
        expected = json.loads((ROOT / "outputs/grid21/orders" / (identifier + ".json")).read_text())["orders"]
        for rule in ["OLDNN", "GNN_seed0", "GNN_seed1", "MLP_seed0", "BIAS_seed0"]:
            checkpoint = None if rule == "OLDNN" else ROOT / "outputs/grid21/models" / rule.lower() / "final.pt"
            search = Search(env, ctl, rule, 128, checkpoint)
            eligible, reps, aliases, actions, changed = search.prepare(obs, base)
            order, prior, features, times = search.rank(obs, base, eligible, reps, aliases)
            assert order == expected[rule]
            assert digest(prior) == meta["prior_hash"]
            if features is not None: assert search.rank(obs, base, eligible, reps, aliases)[0] == order
            checked.append({"week": week, "rule": rule, "candidates": len(reps), "rank_matches": True})
        if wi == 0:
            search = Search(env, ctl, "GNN_seed0", 128, ROOT / "outputs/grid21/models/gnn_seed0/final.pt")
            eligible, reps, aliases, actions, _ = search.prepare(obs, base)
            order, _, _, _ = search.rank(obs, base, eligible, reps, aliases)
            key_to_rep = {digest(action_vector(actions[i]+base)): i for i in reps}
            assert len(key_to_rep) == len(reps) > 128
            wanted = order[128]
            seen = []
            original = obs.simulate
            def feedback(action):
                i = key_to_rep[digest(action_vector(action))]; assert i not in seen; seen.append(i)
                rho = float(obs.rho.max())*.5 if i == wanted else float(obs.rho.max())+1.
                return SimpleNamespace(rho=np.asarray([rho])), 1., False, {"exception": [], "is_illegal": False, "is_ambiguous": False}
            obs.simulate = feedback
            try:
                action, log, _, _ = search.choose(obs, base, 0.)
                assert log["fallback"] and log["chosen_pool_id"] == wanted
                assert len(seen) == len(reps) and log["public_queries"] == len(reps)
                assert seen[:128] == order[:128]
                fake_queries += len(seen)
                seen.clear()
                def invalid(action):
                    i = key_to_rep[digest(action_vector(action))]; assert i not in seen; seen.append(i)
                    return SimpleNamespace(rho=np.asarray([float(obs.rho.max())*.5])), 1., False, {"exception": [], "is_illegal": True, "is_ambiguous": False}
                obs.simulate = invalid
                action, log, _, _ = search.choose(obs, base, 0.)
                assert action is None and log["chosen_pool_id"] is None and log["fallback"]
                assert all(not r["strict_admissible"] for r in log["outcomes"])
                fake_queries += len(seen)
            finally:
                obs.simulate = original
    assert int(env.nb_highres_called) == initial_queries
finally:
    env.close()
write_json(OUT / "preflight.json", {"passed": True, "checks": checked, "new_forecasts": 0, "fake_feedback_queries": fake_queries,
           "scope": "Saved-public-state descriptor/order match plus fake-feedback unique fallback and illegal rejection. No AC physical verification."})
print(json.dumps({"passed": True, "rank_checks": len(checked), "new_forecasts": 0, "fake_feedback": fake_queries}))
