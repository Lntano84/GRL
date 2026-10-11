"""Freeze one schedule and one simple comparator before inspecting their results."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid27"
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / "design.json").exists()
old = json.loads((ROOT / "outputs/grid26/design.json").read_text())
design = {"stage": "GRID27_STAGED_FALLBACK_DEVELOPMENT", "weeks": old["weeks"],
          "rankers": [m for m in old["methods"] if m["rule"] != "FULL"],
          "policies": ["FIXED256", "PATIENCE64"],
          "initial": 128, "fixed_limit": 256, "batch": 64, "minimum": 256, "patience": 2,
          "permission": "Only legal current state/ranking and actually dispatched outcomes; never hidden FULL outcome or global winner rank.",
          "change": "Experimental replacement of compulsory full complement in rho>=0.9 states. Does not lower author thresholds or relax per-action admission, but may miss a better candidate. Not an exact/safety guarantee.",
          "offline": "26 stored development FULL states, one row per ranker/policy/state. No forecasting or fitting. Missing FULL action is a diagnostic, not proof of downstream regret.",
          "pilot": "March GNN0 and PPO with PATIENCE64, plus GNN0 protected controller profiling. No final holdout. No choosing a different schedule based on this screen.",
          "gate": "If offline identifies inadmissible output, duplicate charging or information leak: repair before AC. If >1% state-level rho regret or missing action at prior known March critical state: retain findings and analyze before closed-loop; do not switch schedule to rescue. Otherwise run frozen March pilot even if exact choice differs.",
          "quality": "No earlier/new failure; complete-week cost increase <=0.5% to protected baseline as an investment threshold, not equivalence proof. Report all cost/survival outcomes and exact-action disagreement.",
          "cost": "Same runtime, masks, forecast accounting, thread limits, full controller and cadence6. Includes ranking, decision and optional equal profile wrappers; setup/physical/export separate. Pilot timings only, not formal acceleration proof.",
          "continuation": "If pilot preserves quality, run same two PATIENCE64 rankers on remaining three exposed weeks. If mild deficit, diagnose existing records before further changes; keep all policies/models and failures. No immediate RL.",
          "caps": {"offline_s": 120, "pilot_wall_s": 2700, "episode_s": 1200, "total_wall_s": 7200,
                   "new_fits": 0, "reserved_evaluations": 0}}
(OUT / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
(OUT / "schedule_freeze.json").write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [Path(__file__).resolve(), ROOT / "work/grid27/policy.py"]}, indent=2), encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("DESIGN FROZEN. Offline staged fallback screen pending. No new forecasts/fits/holdout.\n", encoding="utf-8")
print(json.dumps({"frozen": True, "offline_policy_states": 156, "new_fits": 0}))
