"""Input permissions, action descriptors and scorer shapes before any fit."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "work/grid08"), str(ROOT / "work/grid17"), str(ROOT / "work/grid20")]
from common import np, torch, make_env, ResidualControl, write_json
from components import public_inputs, fingerprint, tensors, CandidateScorer, canonical_groups
from mask import mask_disconnected
OUT = ROOT / "outputs/grid20"; OUT.mkdir(parents=True, exist_ok=True)
env = make_env(); env.seed(0); initial = env.reset(); ctl = ResidualControl(env, initial)
library = ctl.base.topo_n1_unsafe.topo_act_list; plans = np.stack([a.set_bus for a in library])
subids = [int(a.as_dict()["set_bus_vect"]["modif_subs_id"][0]) for a in library]
checks = []
try:
    empty = env.action_space()
    for action in library:
        for name in type(action).attr_list_vect:
            if name != "_set_topo_vect":
                assert np.array_equal(action._get_array_from_attr_name(name), empty._get_array_from_attr_name(name), equal_nan=True), name
    fixtures = []
    for week in json.loads((ROOT / "outputs/grid18/design.json").read_text())["weeks"]:
        files = sorted((ROOT / f"outputs/grid18/runs/{week}").glob("call[0-9][0-9][0-9].json"))
        fixtures += [files[0], files[-1]]
    for file in fixtures:
        c = json.loads(file.read_text())
        with np.load(file.with_name(file.stem+"_state.npz")) as saved:
            obs = initial.copy(); obs.from_vect(saved["observation"])
            base = env.action_space(); base.from_vect(saved["base_action"])
        visible = public_inputs(obs, base, env, plans)
        # The whole feature construction is replayed after changing the stored
        # labels. Current observation/base/action geometry stay fixed.
        altered = json.loads(json.dumps(c))
        for r in altered["outcomes"]:
            r["rho"] = 1000*r["rho"]+17; r["reward"] = -r["reward"]
            for key in ["done", "illegal", "ambiguous", "source_admissible", "strict_admissible"]: r[key] = not r[key]
        assert altered != c
        repeated = public_inputs(obs, base, env, plans)
        assert fingerprint(visible) == fingerprint(repeated)
        for j in c["pool_ids"]:
            _, expected, _, _ = mask_disconnected(library[j], base, obs, env)
            assert np.array_equal(visible["plans"][j]*2, expected.set_bus)
        representatives, mapping = canonical_groups(visible, c["pool_ids"])
        labels = dict(zip(c["pool_ids"], c["outcomes"]))
        for j, rep in mapping.items():
            assert labels[j]["combined_action_hash"] == labels[rep]["combined_action_hash"]
            for name in ["rho", "reward"]: assert abs(labels[j][name]-labels[rep][name]) < 1e-6
            for name in ["done", "illegal", "ambiguous", "source_admissible", "strict_admissible"]: assert labels[j][name] == labels[rep][name]
        tensors_ = tensors(visible)
        for kind in ["gnn", "mlp"]:
            torch.manual_seed(0)
            model = CandidateScorer(kind, env.n_sub*2, env.n_line*2, len(visible["base"]), env.dim_topo, subids)
            with torch.no_grad(): result = model(**tensors_)
            assert result.shape == (1, len(library)) and torch.isfinite(result).all()
        checks.append({"call": file.relative_to(ROOT).as_posix(), "plans_checked": len(c["pool_ids"]), "unique_physical_actions": len(representatives), "input_hash": fingerprint(visible), "passed": True})
finally:
    env.close()
write_json(OUT / "preflight.json", {"passed": True, "checks": checks, "models_fitted": 0, "forecasts": 0,
                                   "physical_steps": 0, "note": "Permission check reruns feature-only construction after label mutation, not a claim of independent AC verification."})
print(json.dumps({"passed": True, "fixtures": len(checks), "descriptors_checked": sum(c["plans_checked"] for c in checks)}))
