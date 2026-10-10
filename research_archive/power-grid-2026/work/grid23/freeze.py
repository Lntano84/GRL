"""Freeze the narrow revision only after its real interfaces pass preflight."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid23"
assert json.loads((OUT / "preflight.json").read_text())["passed"]
assert json.loads((OUT / "tie_preflight.json").read_text())["passed"]
assert not (OUT / "code_freeze.json").exists()
d = json.loads((OUT / "design.json").read_text())
assert len(d["execution_order"]) == 24
assert all(m["budget"] == 128 for m in d["methods"] if m["method"] != "FULL")
files = list((ROOT / "work/grid23").glob("*.py"))
files += [ROOT / p for p in ["work/grid21/prior.py", "work/grid20/components.py", "work/grid17/mask.py", "work/grid08/common.py"]]
files += [ROOT / m["checkpoint"] for m in d["methods"] if m["checkpoint"]]
files += [OUT / "design.json", OUT / "preflight.json", OUT / "tie_preflight.json"]
manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(OUT / "code_freeze.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("FROZEN.24 common-tie K128 developmental rollouts; no fits or RL.\n", encoding="utf-8")
print(json.dumps({"frozen": True, "runs": 24, "new_fits": 0}))
