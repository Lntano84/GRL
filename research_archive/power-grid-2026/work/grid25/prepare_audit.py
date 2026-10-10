"""Retain the independent GRID24 arithmetic, add repeat keys and timing checks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / "work/grid24/audit.py").read_text(encoding="utf-8")
replacements = [
    ('OUT = ROOT / "outputs/grid24"', 'OUT = ROOT / "outputs/grid25"'),
    ('summaries = {(s["week"], s["method"]): s for s in F["summaries"]}',
     'summaries = {(s["repeat"], s["week"], s["method"]): s for s in F["summaries"]}'),
    ('s = summaries[(week, method)]; path = OUT / "runs" / (week + "__" + method)',
     's = summaries[(item["repeat"], week, method)]; path = OUT / "runs" / f"r{item[\'repeat\']}__{week}__{method}"'),
    ('assert abs(math.fsum(r["decision_s"] for r in rows)-s["controller_s"]) < 1e-7',
     'assert abs(math.fsum(r["decision_s"] for r in rows)-s["controller_s"]) < 1e-7\n'
     '        assert abs(math.fsum(r["decision_cpu_s"] for r in rows)-s["controller_cpu_s"]) < 1e-7\n'
     '        assert abs(math.fsum(r["all_forecast_s"] for r in rows)-s["all_forecast_s"]) < 1e-7\n'
     '        for r in rows:\n'
     '            assert 0 <= r["all_forecast_s"] <= r["decision_s"]\n'
     '            assert r["all_forecast_calls"] == r["simulations"]'),
    ('assert physical == F["counts"]["physical_steps"] and forecasts == F["counts"]["public_forecasts"]',
     'assert physical == F["counts"]["physical_steps"] and forecasts == F["counts"]["public_forecasts"]\n'
     '    assert F["serial_workers"] == 1'),
    ('"scope": "Independent saved-data mask/alias/admission/cost/choice arithmetic and checkpoint rank replay. Full path equals teacher. No independent AC replay or neural refit; public prior code reused for rank consistency."',
     '"scope": "Independent saved-data mask/alias/admission/cost/choice arithmetic, repeat keys, timing sums and checkpoint rank replay. Whole path equals sealed FULL. No independent AC replay/refit; prior code reused."')
]
for before, after in replacements:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
target = ROOT / "work/grid25/audit.py"
assert not target.exists()
target.write_text(source, encoding="utf-8")
print("Independent audit generated; scientific execution code unchanged.")
