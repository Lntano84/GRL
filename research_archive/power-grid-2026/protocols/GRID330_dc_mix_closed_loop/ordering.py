"""GRID330 online ordering adapter. Construction is separate from the capped runner."""
from pathlib import Path
import runpy
import time

A = Path(__file__).resolve().parent
FAST = A.parent / "GRID328_dc_fast_solver"
PREP = A.parent / "GRID329_prepared_dc_plans"
MIX = A.parent / "GRID326_fixed_dc_mix"


class LiveRank:
    def __init__(self):
        self.fast = runpy.run_path(str(FAST / "run.py"))
        self.prep = runpy.run_path(str(PREP / "readout.py"))
        self.prior = runpy.run_path(str(A.parent / "GRID325_dc_global_priority/readout.py"))
        self.mix = runpy.run_path(str(MIX / "readout.py"))["hybrid"]
        self.old_prepare, self.prepare, _ = self.prep["preparation_functions"]()
        self.native = self.fast["NativeSolver"]()
        self.static = None

    def bind(self, search):
        import numpy as np

        if self.static is search.env:
            return
        static = search.env
        if not all(self.prior["supported_commands"](a) for a in search.library):
            raise ValueError("Unsupported live candidate library active-power commands")
        self.exporter = self.prep["PreparedPlans"](search.library[0], static)
        self.grid = self.prior["PublicGrid"](static, *self.prior["static_parameters"](static))
        self.positions = tuple(np.ascontiguousarray(a, dtype=np.int32) for a in
            (self.grid.subs, static.line_or_pos_topo_vect, static.line_ex_pos_topo_vect))
        self.static = static

    def apply(self, search, obs, base, order, first):
        self.bind(search)
        stamp = time.perf_counter()
        record = {"status": "bypass", "reason": None, "export_s": 0.0, "rank_s": 0.0,
                  "scored": 0, "fallback": {}, "solver_counts": {}}
        if len(order) <= 128:
            record["reason"] = "pool_fits_first_batch"
            return order, record
        if not self.prior["supported_commands"](base):
            record["reason"] = "unsupported_base_active_power_command"
            return order, record
        try:
            plans, flags, _, _ = self.exporter.from_keys(obs, order, first)
        except ValueError as exc:
            if str(exc) != "Unsupported or invalid canonical action":
                raise
            record.update(reason="unsupported_canonical_action", export_s=time.perf_counter()-stamp)
            return order, record
        if plans.min() < -1 or plans.max() > 2:
            record.update(reason="unsupported_busbar_id", export_s=time.perf_counter()-stamp)
            return order, record
        record["export_s"] = time.perf_counter() - stamp
        stamp = time.perf_counter()
        dc, scored, fallback, stats = self.fast["prefix_rank"](
            order, dict(zip(order, plans, strict=True)), dict(zip(order, flags, strict=True)),
            self.grid.state_inputs(obs), self.grid, self.native, self.positions)
        prefix = self.mix(order[:128], dc)
        seen = set(prefix)
        result = prefix + [cid for cid in order if cid not in seen]
        assert len(result) == len(order) and scored + sum(fallback.values()) == len(order)
        record.update(status="ranked", rank_s=time.perf_counter()-stamp,
                      scored=scored, fallback=fallback, solver_counts=dict(stats),
                      original128=order[:128], dc128=dc, mix128=prefix)
        return result, record

