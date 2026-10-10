"""Freeze untrained shortlist qualification before collecting any new results."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid15"
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / "design.json").exists()
design = {"stage": "GRID15_N1_SEARCH_ASSISTANCE_QUALIFICATION", "seed": 0,
          "weeks": ["2035-01-29_4", "2035-04-16_2", "2035-07-23_10", "2035-10-29_15"],
          "controller": "Original NN20/full N1/unsafe QP plus ALWAYS_RESTORE cadence6; unchanged physical path",
          "dataset": "All actual original N1 search calls, no outcome-based sampling; reference physical trajectory must match GRID14 AUTHOR exactly",
          "rules": ["ALL", "ZONE", "LOCAL32", "LOCAL128", "RANDOM32", "RANDOM128", "PREFIX32", "PREFIX128"],
          "local_rule": "Substation live-line graph distance from endpoints of three most-loaded connected lines, then descending max incident loading, then original pool id; not electrical busbar graph",
          "zone_rule": "Author zone: line origin substation of current argmax rho; no hidden outcomes",
          "random_rule": "Deterministic SHA256 of state vector hash and original action id, seed0 domain separator",
          "metrics": ["Source valid-action availability", "Near-best source rho within .01", "Pool best reward recall within1e-6", "Counterfactual source forecast calls", "Measured source public forecast wall vs instrumented entire controller wall"],
          "pilot_gate": "Consider tiny supervised pilot only if N1 public forecast time >=20% of instrumented controller time in at least two weeks, and no single tested cheap rule averaging <=128 queries achieves >=95% near-best rho retention with at most one no-valid-action loss when ALL has a valid action. More data/split design still required before training.",
          "scope": "Development search-compression qualification, no closed-loop shortlisting evaluation, no speedup claim, no GNN/RL novelty claim, no training or sealed test use.",
          "caps": {"physical_steps": 8500, "public_forecasts": 100000, "wall_s": 1800, "episode_s": 420},
          "references": ["https://github.com/lajavaness/l2rpn-2023-ljn-agent", "https://arxiv.org/html/2503.15190v2", "https://arxiv.org/html/2604.01830v1"]}
(OUT / "design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")
files = [ROOT / "work/grid15/collect.py", ROOT / "work/grid15/rules.py", ROOT / "work/grid15/freeze.py",
         ROOT / "work/grid08/common.py", ROOT / "work/grid02/ljn_nn/modules/base_module.py", OUT / "design.json"]
sealed = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(OUT / "code_freeze.json").write_text(json.dumps(sealed, indent=2), encoding="utf-8")
print("GRID15_FROZEN")
