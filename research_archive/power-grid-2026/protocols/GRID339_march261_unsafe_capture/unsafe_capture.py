"""Read-only snapshots around the original unsafe solver and action conversion."""
import json
from pathlib import Path

import numpy as np

PARAMETERS = (
    "_powerlines_x", "_alpha_por_error", "_prev_por_error", "_margin_th_limit",
    "_th_lim_mw", "_penalty_curtailment_unsafe", "_penalty_storage_unsafe",
    "_penalty_redispatching_unsafe", "load_per_bus", "gen_per_bus",
    "_storage_power_obs", "redisp_up", "redisp_down", "curtail_up", "curtail_down",
    "storage_up", "storage_down", "bus_or", "bus_ex", "bus_load", "bus_gen", "bus_storage",
)
OBSERVATION_FIELDS = ("gen_renewable", "gen_p", "gen_redispatchable",
                      "gen_margin_up", "gen_margin_down")

def packet(path, arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)

def install(cls, originals, output, location, solver_records):
    output = Path(output)
    output.mkdir(exist_ok=False)
    original = originals["compute_optimum_unsafe"]
    convert = cls.to_grid2op
    previous = {}
    serial = 0

    def unsafe(mod):
        nonlocal serial
        step, in_control, phase = location()
        if not in_control:
            return original(mod)
        assert phase == "compute_optimum_unsafe"
        number = serial
        serial += 1
        prior = len(solver_records)
        arrays = {name: np.array(getattr(mod, name).value, copy=True)
                  for name in PARAMETERS if getattr(mod, name) is not None}
        arrays.update({name: np.array(getattr(mod.env, name), copy=True)
                       for name in ("gen_renewable", "gen_max_ramp_up", "gen_max_ramp_down")})
        arrays.update({name: np.array(getattr(mod, name), copy=True)
                       for name in ("nb_max_bus", "n_gen", "max_iter", "gen_to_subid", "flow_computed")})
        packet(output / f"{number:03d}_input.npz", arrays)
        result = original(mod)
        assert len(solver_records) == prior + 1
        record = solver_records[prior]
        assert (record["step"], record["phase"], record["solver"]) == (step, phase, "OSQP")
        packet(output / f"{number:03d}_result.npz",
               dict(zip(("curtailment", "storage", "redispatching"), result, strict=True),
                    flow=mod.flow_computed.copy(), storage_state=mod._storage_power_obs.value))
        (output / f"{number:03d}_solve.json").write_text(
            json.dumps(record, allow_nan=False) + "\n", encoding="utf8")
        previous[id(mod)] = (number, step)
        return result

    def to_grid2op(mod, obs, curtailment, storage, redispatching, base_action=None, safe=False):
        identity = id(mod)
        if safe or identity not in previous:
            return convert(mod, obs, curtailment, storage, redispatching, base_action, safe)
        number, step = previous.pop(identity)
        assert location()[:2] == (step, True)
        arrays = {name: np.array(getattr(obs, name), copy=True) for name in OBSERVATION_FIELDS}
        arrays.update(n_gen=obs.n_gen, margin_sparse=mod.margin_sparse, obs=obs.to_vect().copy())
        if base_action is not None:
            arrays["base"] = base_action.to_vect().copy()
        result = convert(mod, obs, curtailment, storage, redispatching, base_action, safe)
        arrays["converted"] = result.to_vect().copy()
        packet(output / f"{number:03d}_action.npz", arrays)
        return result

    originals["compute_optimum_unsafe"] = unsafe
    cls.to_grid2op = to_grid2op
