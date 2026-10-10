"""Borrowed disconnected-endpoint feasibility repair; never alters frozen assets."""
import numpy as np


def mask_disconnected(candidate, base_action, observation, env):
    original = candidate + base_action
    base_bus = base_action.set_bus
    # A base reconnection already explicitly requested by the original controller
    # is retained. This includes implicit bus-setting reconnections in the base.
    intent = (base_action.line_set_status > 0) | base_action.line_change_status
    intent = intent | (base_bus[env.line_or_pos_topo_vect] > 0) | (base_bus[env.line_ex_pos_topo_vect] > 0)
    disconnected = (~observation.line_status) & (~intent)
    positions = np.concatenate([env.line_or_pos_topo_vect[disconnected], env.line_ex_pos_topo_vect[disconnected]])
    result = original.copy()
    bus = result.set_bus.copy()
    changed = positions[bus[positions] != 0]
    if len(changed):
        bus[positions] = 0
        result.set_bus = bus
    return original, result, changed.tolist(), np.flatnonzero(intent).tolist()
