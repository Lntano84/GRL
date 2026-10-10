"""Freeze balanced serial timing, without reading any reserved trajectory."""
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid25"
if OUT.exists():
    assert not (OUT / "hardware.json").exists(), "Completed preparation must not be rerun"
OUT.mkdir(parents=True, exist_ok=True)
old = json.loads((ROOT / "outputs/grid24/design.json").read_text())
assert json.loads((ROOT / "outputs/grid24/reconciliation.json").read_text())["passed"]
methods = [{"method": "FULL", "rule": "FULL", "budget": None, "checkpoint": None},
           old["methods"][1], old["methods"][0], old["methods"][2]]
order = []
for repeat in range(3):
    for wi, week in enumerate(old["weeks"]):
        offset = (wi + repeat) % len(methods)
        rotation = methods[offset:] + methods[:offset]
        for method in rotation:
            order.append({"repeat": repeat, "week": week, "phase": "pilot" if len(order) < 4 else "remaining", **method})
design = {"stage": "GRID25_CONTROLLED_SERIAL_TIMING", "weeks": old["weeks"], "methods": methods,
    "repeats": 3, "execution_order": order, "environment_seed": 0,
    "weights": "Unchanged GRID21 V1 seed0/seed1; author's original PPO remains frozen.",
    "control": "Unchanged GRID24 K128 guard0.9, masks, full combined aliases, reward tie, restoration cadence6 and continuous optimizer.",
    "timing": "Single fresh worker process per run. Serial only. Same CPU thread settings. Balanced method positions across weeks in each repeat. No candidate compression, artifact writing, vector hashes or offline ledger reconstruction inside decision time.",
    "primary": "Pooled total controller wall time ratio to FULL within the same week and repeat; report week-macro paired ratios too. Physical simulation and startup separately, plus controller+physical+startup. Search/source and forecast components are secondary.",
    "tails": "All-step decision p95/p99/max; N1 search-call p95 separately. No treating physical steps or repeats as independent date families.",
    "quality": "Every development run must retain GRID24/FULL action-observation-cost path. June failure remains counted, not a cost improvement. Stop if refactor changes decisions.",
    "continuation": "After timing audit, keep protected configuration fixed for reserved-family quality/cost/speed evaluation. A small or mixed speed effect is a diagnosis, not permission to select a winning repeat.",
    "caps": {"wall_s": 14400, "episode_s": 720, "physical_steps": 100000, "public_forecasts": 1500000},
    "new_model_fits": 0, "reserved_family_evaluations": 0,
    "instrumentation": "Light simulation timers and common ResidualControl checks remain equally enabled. No production-service latency claim. Parent startup/file-save is excluded from act timing and separately counted."}
if (OUT / "design.json").exists():
    assert json.loads((OUT / "design.json").read_text()) == design
else:
    (OUT / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
preserved = {}
for stage in ["grid21", "grid23", "grid24", "grid_final_holdout"]:
    for path in (ROOT / "outputs" / stage).rglob("*"):
        if path.is_file() and path.name not in ["audit.log", "pilot_audit.log"]:
            preserved[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
for stage in ["grid08", "grid23", "grid24"]:
    for path in (ROOT / "work" / stage).glob("*.py"):
        preserved[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
if (OUT / "preservation_hashes.json").exists():
    assert json.loads((OUT / "preservation_hashes.json").read_text()) == preserved
else:
    (OUT / "preservation_hashes.json").write_text(json.dumps(preserved, indent=2), encoding="utf-8")
info = subprocess.run(["powershell", "-NoProfile", "-Command",
    "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json"], capture_output=True, text=True, check=True)
power = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, check=True)
(OUT / "hardware.json").write_text(json.dumps({"platform": platform.platform(), "processor": json.loads(info.stdout),
    "power_scheme": power.stdout.decode("utf-8", errors="replace").strip(), "gpu_used": False, "workers": 1,
    "thread_environment": {x: "1" for x in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]}}, indent=2), encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("DESIGN FROZEN. Production-equivalence preflight pending; no new fits or reserved outcomes.\n", encoding="utf-8")
print(json.dumps({"planned_runs": len(order), "preserved_files": len(preserved), "workers": 1}))
