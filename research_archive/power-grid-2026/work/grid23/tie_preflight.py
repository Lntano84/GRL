"""The shared selector must ignore rank order on exact reward ties only."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid23")]
from common import np, make_env, ResidualControl, action_vector, digest, write_json
from canonical_adapter import Search

OUT = ROOT / "outputs/grid23"
env = make_env()
obs = env.reset()
ctl = ResidualControl(env, obs)
before_queries = int(env.nb_highres_called)
checks = []
original = obs.simulate
try:
    base = env.action_space()
    search = Search(env, ctl, "OLDNN", 128)
    eligible, reps, aliases, actions, changed = search.prepare(obs, base)
    assert len(reps) > 128
    reverse = list(reversed(reps))
    search.rank = lambda o, b, e, r, a: (reverse.copy(), None, None,
                         {"feature_s": 0., "prior_s": 0., "forward_rank_s": 0.})
    index = {digest(action_vector(actions[i]+base)): i for i in reps}
    seen = []

    def tied(action):
        i = index[digest(action_vector(action))]
        assert i not in seen
        seen.append(i)
        rho = float(obs.rho.max()) * .5
        return SimpleNamespace(rho=np.asarray([rho])), 2.-rho, False, {
            "exception": [], "is_illegal": False, "is_ambiguous": False}

    obs.simulate = tied
    _, log, _, _ = search.choose(obs, base, 0.)
    expected = min(reverse[:128])
    assert expected != reverse[0]
    assert log["chosen_pool_id"] == expected and not log["fallback"]
    assert len(seen) == 128
    checks.append({"case": "All exact rewards tied in deliberately reversed rank", "chosen": expected,
                   "old_order_tie_choice": reverse[0], "passed": True})
    seen.clear()

    def superior(action):
        i = index[digest(action_vector(action))]
        assert i not in seen
        seen.append(i)
        rho = float(obs.rho.max()) * (.25 if i == reverse[0] else .5)
        return SimpleNamespace(rho=np.asarray([rho])), 2.-rho, False, {
            "exception": [], "is_illegal": False, "is_ambiguous": False}

    obs.simulate = superior
    _, log, _, _ = search.choose(obs, base, 0.)
    assert log["chosen_pool_id"] == reverse[0] and not log["fallback"]
    assert len(seen) == 128
    checks.append({"case": "Strictly greater reward overrides smaller action ID", "chosen": reverse[0], "passed": True})
    assert int(env.nb_highres_called) == before_queries
finally:
    obs.simulate = original
    env.close()

write_json(OUT / "tie_preflight.json", {"passed": True, "checks": checks,
           "mock_queries": 256, "new_native_forecasts": 0, "new_model_fits": 0})
print(json.dumps({"passed": True, "cases": len(checks), "new_native_forecasts": 0}))
