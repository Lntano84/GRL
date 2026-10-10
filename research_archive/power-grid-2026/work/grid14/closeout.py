"""Read-only physical-control repeat check and seal GRID14 after all jobs finish."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid14"


def read(path):
    with gzip.open(path / "steps.jsonl.gz", "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


checks = []
for week in ["2035-01-29_4", "2035-10-29_15"]:
    original = ROOT / "outputs/grid08/guard_qualification" / f"{week}__ALWAYS_ONE_STEP"
    current = OUT / "runs" / f"{week}__AUTHOR"
    old_rows, new_rows = read(original), read(current)
    assert len(old_rows) == len(new_rows)
    fields = ["before_hash", "action_hash", "after_hash", "raw_cost"]
    mismatches = {field: [] for field in fields}
    for i, (old, new) in enumerate(zip(old_rows, new_rows), 1):
        for field in fields:
            assert field in old and field in new, (week, i, field)
            if old[field] != new[field]:
                mismatches[field].append(i)
    assert not any(mismatches.values()), (week, mismatches)
    checks.append({"scenario": week, "steps": len(old_rows), "fields": fields,
                   "mismatches": mismatches, "physical_control_exact": True,
                   "scope": "Exact stored row fields; excludes wall-clock and added diagnostic forecasts."})
write(OUT / "baseline_replay_check.json", {"passed": True, "checks": checks,
                                           "new_physical_steps": 0, "new_public_forecasts": 0})

assert json.loads((OUT / "audit.json").read_text())["passed"]
assert not json.loads((OUT / "interpretation.json").read_text())["action_evaluator_gate"]
for relative, expected in json.loads((OUT / "code_freeze.json").read_text()).items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative

paths = list(OUT.rglob("*")) + list((ROOT / "work/grid14").rglob("*"))
paths.append(ROOT / "outputs/GRID_PROGRESS_2026-10-09_R2.md")
sealed = {}
for path in sorted(set(paths)):
    if not path.is_file() or "__pycache__" in path.parts or path.name == "delivery_manifest.json":
        continue
    sealed[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
write(OUT / "delivery_manifest.json", sealed)
for relative, expected in json.loads((OUT / "delivery_manifest.json").read_text()).items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
print(json.dumps({"baseline_repeat_checks": checks, "sealed_files": len(sealed),
                  "formal_audit_passed": True, "training_runs": 0}, ensure_ascii=False))
