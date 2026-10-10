"""Independently inspect masked fields and compare saved forecast flags."""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "work/grid08"))
from common import np, make_env, action_vector, digest, write_json
OUT = ROOT / "outputs/grid17"; F = json.loads((OUT / "finished.json").read_text()); assert F["passed_engineering"]
for name, sha in json.loads((OUT / "code_freeze.json").read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
env = make_env(); original_obs = env.reset(); checked = 0; cleared = 0; improved = 0
details = []
try:
    for r in F["results"]:
        target = r["target"]
        for pair in r["pairs"]:
            with np.load(OUT / target["week"] / f"pool{pair['pool_id']:04d}.npz") as f: saved = {k: f[k].copy() for k in f.files}
            before = original_obs.copy(); before.from_vect(saved["before_observation"]); assert digest(before.to_vect()) == target["state_hash"]
            base = env.action_space(); base.from_vect(saved["base_action"])
            acts = {}
            for kind in ["original", "masked"]:
                a = env.action_space(); a.from_vect(saved[kind+"_action"]); acts[kind] = a
                assert digest(action_vector(a)) == pair[kind]["action_hash"]
                # Public forecast vectors and physical observation vectors have
                # different layouts in this environment; never decode as native.
                forecast = saved[kind+"_observation"]; offset = 0; fields = {}
                for field in pair[kind]["forecast_layout"]:
                    fields[field["name"]] = forecast[offset:offset+field["size"]]; offset += field["size"]
                assert offset == len(forecast) and len(fields["rho"]) == env.n_line
                assert abs(float(fields["rho"].max())-pair[kind]["rho"]) < 1e-7
            # Reconstruct the rule from bare persisted arrays, not mask.py.
            intended = set(np.flatnonzero(base.line_set_status > 0)) | set(np.flatnonzero(base.line_change_status))
            for line in range(env.n_line):
                if base.set_bus[env.line_or_pos_topo_vect[line]] > 0 or base.set_bus[env.line_ex_pos_topo_vect[line]] > 0: intended.add(line)
            expected = acts["original"].set_bus.copy()
            for line in range(env.n_line):
                if not before.line_status[line] and line not in intended:
                    expected[env.line_or_pos_topo_vect[line]] = 0; expected[env.line_ex_pos_topo_vect[line]] = 0
            assert np.array_equal(expected, acts["masked"].set_bus)
            bus_field = "_set_topo_vect"
            for name in type(acts["original"]).attr_list_vect:
                if name != bus_field:
                    assert np.array_equal(acts["original"]._get_array_from_attr_name(name), acts["masked"]._get_array_from_attr_name(name)), name
            old, new = pair["original"], pair["masked"]
            ok = not new["illegal"] and not new["ambiguous"] and not new["done"] and not new["exceptions"] and new["rho"] > 0
            clear = bool(old["illegal"] and ok); better = clear and bool(new["valid_improving"])
            cleared += clear; improved += better; checked += 1
            details.append({"week": target["week"], "step": target["step"], "pool_id": pair["pool_id"],
                            "original_illegal": old["illegal"], "masked_strictly_valid": ok, "masked_improving": better,
                            "current_rho": float(before.rho.max()), "masked_rho": new["rho"]})
finally:
    env.close()
assert F["counts"]["public_forecasts"] == checked*2
write_json(OUT / "audit.json", {"passed": True, "pairs_checked": checked, "illegal_to_strictly_valid": cleared,
                               "illegal_to_valid_improving": improved, "details": details, "extra_forecasts": 0, "extra_physical_steps": 0,
                               "scope": "Independent saved-vector action-field reconstruction; no independent AC re-solve or whole-policy superiority."})
print(json.dumps({"pairs": checked, "strictly_valid": cleared, "improving": improved}))
