"""Freeze a borrowed feasibility probe after zero-query input qualification."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid17"
assert not (OUT / "design.json").exists()
q = json.loads((OUT / "qualification.json").read_text()); assert q["passed"] and q["targets"]
assert json.loads((ROOT / "outputs/grid16/audit.json").read_text())["passed"]
d = {"stage": "GRID17_BORROWED_DISCONNECTED_ENDPOINT_FEASIBILITY_PROBE", "seed": 0, "targets": q["targets"],
     "selection": "First actual original N1 call per exposed week with a line-cooldown conflict removable by masking; first three eligible original pool IDs. Engineering-selected cases only.",
     "mask": "Clear set-bus assignments at endpoints of currently disconnected lines unless original incoming base explicitly reconnects/changes status or sets either endpoint bus. Preserve all other fields; copy only.",
     "replay": "Use already saved physical actions, assert every observation equals original prefix. No agent search in prefix; paired original/masked public one-step forecasts only, no target action applied.",
     "criteria": "Engineering applicability: at least one original illegal candidate becomes strictly legal/nonterminal. Beneficial applicability separately requires its maximum rho improve current rho. Does not establish closed-loop quality or novelty.",
     "scope": "Implementation of known 2025 paper enhancement, not a new algorithm. No fit, no sealed evaluation, no superiority claim from engineer-selected targets.",
     "reference": "https://arxiv.org/html/2503.15190v2#S4.SS4.SSS2",
     "caps": {"physical_steps": sum(t["step"]-1 for t in q["targets"]), "public_forecasts": 2*sum(len(t["candidate_ids"]) for t in q["targets"]), "wall_s": 180}}
(OUT / "design.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
files = [ROOT / "work/grid17" / n for n in ["mask.py", "qualify.py", "probe.py", "freeze.py"]]
files += [OUT / "design.json", OUT / "qualification.json", ROOT / "work/grid08/common.py"]
(OUT / "code_freeze.json").write_text(json.dumps({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2), encoding="utf-8")
print(json.dumps({"frozen": True, "caps": d["caps"]}))
