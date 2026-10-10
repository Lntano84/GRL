"""Production-path equivalence on all 26 sealed FULL public states; no AC queries."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid25"), str(ROOT / "work/grid08")]
from common import np, make_env, ResidualControl, action_vector, digest, write_json
from engine import Search

OUT = ROOT / "outputs/grid25"
D = json.loads((OUT / "design.json").read_text())
env = make_env(); obs0 = env.reset(); ctl = ResidualControl(env, obs0)
start_count = int(env.nb_highres_called)
checks = []; fake_queries = 0
try:
    for week in D["weeks"]:
        full = ROOT / "outputs/grid23/runs" / (week + "__FULL")
        for path in sorted(full.glob("call[0-9][0-9][0-9].json")):
            c = json.loads(path.read_text())
            with np.load(path.with_name(path.stem + "_state.npz")) as f:
                obs = obs0.copy(); obs.from_vect(f["observation"])
                base = env.action_space(); base.from_vect(f["base_action"])
            outcomes = {r["combined_hash"]: r for r in c["outcomes"]}
            for m in D["methods"]:
                search = Search(env, ctl, m["rule"], m["budget"], ROOT / m["checkpoint"] if m["checkpoint"] else None)
                elig, reps, aliases, actions, changed = search.prepare(obs, base)
                assert elig == c["eligible"] and reps == c["representatives"]
                assert aliases == {int(k): v for k, v in c["aliases"].items()} and changed == c["masked_ids"]
                seen = []
                original = obs.simulate
                def feedback(action, **kwargs):
                    global fake_queries
                    key = digest(action_vector(action)); r = outcomes[key]
                    assert r["pool_id"] not in seen
                    seen.append(r["pool_id"]); fake_queries += 1
                    return SimpleNamespace(rho=np.asarray([r["rho"]])), r["reward"], r["done"], {
                        "exception": ["saved"] if r["exceptions"] else [],
                        "is_illegal": r["illegal"], "is_ambiguous": r["ambiguous"]}
                obs.simulate = feedback
                try: selected, log, _, _ = search.choose(obs, base, 0.)
                finally: obs.simulate = original
                reference = c if m["rule"] == "FULL" else json.loads((ROOT / "outputs/grid24/runs" / (week + "__" + m["method"]) / path.name).read_text())
                for key in ["order", "shortlist", "fallback", "chosen_pool_id", "public_queries"]:
                    assert log[key] == reference[key], (week, path.name, m["method"], key)
                assert seen == [r["pool_id"] for r in reference["outcomes"]]
                assert (None if selected is None else digest(action_vector(selected))) == reference["returned_action_hash"]
                for r in log["outcomes"]:
                    old = next(x for x in reference["outcomes"] if x["pool_id"] == r["pool_id"])
                    for k in ["rho", "reward", "done", "strict_admissible", "phase"]: assert r[k] == old[k]
                checks.append({"week": week, "call": path.name, "method": m["method"], "passed": True})
    assert int(env.nb_highres_called) == start_count
finally: env.close()
write_json(OUT / "preflight.json", {"passed": True, "checks": checks, "fake_feedback_queries": fake_queries,
    "new_ac_forecasts": 0, "new_physical_steps": 0, "new_model_fits": 0,
    "scope": "26 sealed FULL public states x four methods, exact aliases, ranks, dispatch, validity and choice. Whole-loop equality still required."})
paths = list((ROOT / "work/grid25").glob("*.py")) + [ROOT / "work/grid24/guard_adapter.py", ROOT / "work/grid08/common.py", ROOT / "work/grid21/prior.py", ROOT / "work/grid20/components.py"]
write_json(OUT / "code_freeze.json", {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
print(json.dumps({"passed": True, "checks": len(checks), "fake_queries": fake_queries}))
