"""Verify the instrumentation bug with a direct object-lifetime counterexample."""
import gc
import hashlib
import json
import weakref
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid26"


class TestObservation:
    def simulate(self, x): return x + 1


gc.disable()
try:
    first = TestObservation(); ref = weakref.ref(first)
    bound = first.simulate
    first.simulate = bound
    del bound, first
    assert ref() is not None
    gc.collect()
    assert ref() is None
    original = TestObservation.simulate
    seen = []
    def measured(self, x):
        seen.append(x)
        return original(self, x)
    TestObservation.simulate = measured
    second = TestObservation(); ref2 = weakref.ref(second)
    assert second.simulate(7) == 8 and seen == [7]
    assert "simulate" not in second.__dict__
    del second
    assert ref2() is None
    TestObservation.simulate = original
finally: gc.enable()
out = {"passed": True, "old_instance_bound_method_requires_gc": True,
       "new_class_timer_no_instance_cycle": True, "return_value_and_count_unchanged": True,
       "new_ac_forecasts": 0, "new_physical_steps": 0,
       "scope": "Direct structural counterexample. Does not prove this caused prior runtime variation; GRID25 pilot remains a diagnostic."}
(OUT / "timer_preflight.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
(OUT / "analysis_code_freeze.json").write_text(json.dumps({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [ROOT/"work/grid26/audit.py", ROOT/"work/grid26/review.py", ROOT/"work/grid26/timer_preflight.py"]}, indent=2), encoding="utf-8")
print(json.dumps(out))
