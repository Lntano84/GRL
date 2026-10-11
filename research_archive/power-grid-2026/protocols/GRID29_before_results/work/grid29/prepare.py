"""Freeze an engineering qualification, never a learned contribution claim."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid29'
OUT.mkdir(parents=True, exist_ok=False)
design = {
    'stage': 'GRID29_SHARED_DC_PROGRAM_CACHE',
    'scope': 'Shared engineering improvement for complete controller; apply to strong baselines as well as GNN variants.',
    'source': 'LJN OptimModule.run_dc unchanged mathematical program and _solve_problem; only retain compiled program by topology.',
    'cache_capacity': 32,
    'shadow_weeks': ['2035-03-05_0', '2035-06-25_8', '2035-09-03_4', '2035-10-01_5'],
    'shadow_source': 'GRID26 repeat0 FULL saved physical observations; every64th nonterminal observation plus last nonterminal. No further physics rollout during shadow qualification.',
    'shadow_state': 'Each observation compared at identical zero storage-power accumulator; this does not verify all live controller memory states.',
    'shadow_checks': 'Same convergence, finite/NaN status and max flow discrepancy<=1e-6MW; live trajectory validation still necessary.',
    'manual_checks': ['two-bus flow', 'injection refresh on hit', 'topology-key invalidation', 'reactance-key invalidation'],
    'continuation': 'Only if shadow qualifies: balanced serial March protectedGNN0 controller ORIGINAL/CACHED in orders ORIGINAL,CACHED then CACHED,ORIGINAL; preserve exact delivered path and cost. On any difference stop expansion and diagnose.',
    'change_forbidden': ['warm_start', 'solver tolerances', 'iteration caps', 'objective', 'constraints', 'topology schedule', 'GNN/PPO weights', 'restoration cadence'],
    'wall_cap_s': 3600,
    'new_fits': 0,
    'reserved_evaluations': 0,
    'warning': 'Conventional DPP reuse; never attribute shared engineering saving to GNN. Shadow speed is not end-to-end speed.'
}
(OUT / 'design.json').write_text(json.dumps(design, indent=2), encoding='utf-8')
paths = [ROOT / 'work/grid29/cache_dc.py', ROOT / 'work/grid29/shadow.py', Path(__file__).resolve(),
         ROOT / 'work/grid02/ljn_nn/modules/convex_optim.py']
(OUT / 'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2), encoding='utf-8')
(OUT / 'RUN_STATE.md').write_text('PREPARED: no shadow computation or closed-loop run yet. Wait for GRID28 serial worker before numerical work.\n', encoding='utf-8')
print(json.dumps(design, indent=2))
