"""Freeze a known baseline repair before its first whole-week rollout."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid18"; OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / "design.json").exists()
a = json.loads((ROOT / "outputs/grid17/audit.json").read_text()); assert a["passed"] and a["illegal_to_strictly_valid"] > 0
d = {"stage": "GRID18_KNOWN_MASKED_N1_FULL_BASELINE", "weeks": ["2035-01-29_4", "2035-04-16_2", "2035-07-23_10", "2035-10-29_15"],
     "seed": 0, "variant": "Only N1 full-search candidate set-bus assignments at disconnected endpoints cleared unless incoming base intentionally reconnects. No masking in NN20/reversion/reconnection modules; those remain original.",
     "controller": "Original NN20, full N1, unsafe QP, ALWAYS_RESTORE cadence6. Original library and model assets immutable; modified candidates are copies.",
     "reference": "https://arxiv.org/html/2503.15190v2#S4.SS4.SSS2",
     "measure": "Full survival first; cost only jointly complete against GRID16 FULL. One wall run per cell with different insertion overhead, so no end-to-end speed superiority claim.",
     "training": "None. Store all actual source candidate public forecasts for later legal input audit, without promising any new learned model.",
     "scope": "Known engineering baseline improvement. Cannot claim GNN, RL, or novel algorithm contribution from it.",
     "caps": {"physical_steps": 8500, "public_forecasts": 70000, "wall_s": 1800, "episode_s": 420}}
(OUT / "design.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
files = [ROOT / "work/grid18/collect.py", ROOT / "work/grid18/freeze.py", ROOT / "work/grid17/mask.py", ROOT / "work/grid08/common.py", OUT / "design.json"]
(OUT / "code_freeze.json").write_text(json.dumps({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2), encoding="utf-8")
print("GRID18_BASELINE_REPAIR_FROZEN")
