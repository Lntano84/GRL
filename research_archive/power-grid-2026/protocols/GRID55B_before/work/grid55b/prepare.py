"""Freeze the previously proposed eight-run repair, authorized by user continuation."""
import copy
import hashlib
import json
from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid55b'
assert not OUT.exists()
OUT.mkdir()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
save = lambda p, x: p.write_text(json.dumps(x, indent=2, allow_nan=False), encoding='utf-8')
old = read(ROOT / 'outputs/grid55/design.json')
proposal = read(ROOT / 'outputs/grid55/TIMING_REPAIR_PROPOSAL.json')
assert proposal['new_runs'] == 8 and proposal['new_repeat_id'] == 2
assert read(ROOT / 'outputs/grid55/finished.json')['passed_engineering']
assert read(ROOT / 'outputs/grid54/qualification.json')['passed']
freeze = read(ROOT / 'outputs/grid55/code_freeze.json')
for p, h in freeze.items():
    assert sha(ROOT / p) == h, p
design = copy.deepcopy(old)
design.update(stage='GRID55B_EXTERNAL_STANDBY_REPAIR', weeks=[proposal['week']], families=[old['families'][-1]],
    repeats=1, authorization='User continued on 2026-10-13 after the saved shutdown handoff; execute the concrete eight-arm supplemental plan.',
    scope=proposal['data_rule'], qualification=proposal['qualification'],
    timing='Repeat the original affected repetition order for ALL eight methods, one fresh single-thread process each. Scoped Windows awake request; cold prior cache charged as before.',
    source_proposal_sha256=sha(ROOT / 'outputs/grid55/TIMING_REPAIR_PROPOSAL.json'))
design['execution_order'] = [{**i, 'repeat': 2} for i in old['execution_order'] if i['week'] == proposal['week'] and i['repeat'] == 1]
assert len(design['execution_order']) == 8
assert set(i['method'] for i in design['execution_order']) == set(proposal['implementations'])
design['caps'].update(wall_s=2400, physical_steps=17000, public_forecasts=200000)
design['view_rule'] = 'Qualified128 uses original repeat0/1 for seven dates and repeat0/2 for the affected date, for ALL8methods. Physical source repeat2 is retained in provenance; logical slot1 is only a reporting index. No time subtraction or CPU substitution.'
save(OUT / 'design.json', design)
raw = {}
for name in ['design.json', 'code_freeze.json', 'finished.json', 'manifest.json', 'review.json',
             'version_selection.json', 'GRID55_report.md', 'review_status.json', 'TIMING_REPAIR_PROPOSAL.json']:
    p = ROOT / 'outputs/grid55' / name
    raw[p.relative_to(ROOT).as_posix()] = sha(p)
save(OUT / 'raw_preservation.json', raw)
for p in (ROOT / 'work/grid55b').glob('*.py'):
    py_compile.compile(str(p), doraise=True)
    freeze[p.relative_to(ROOT).as_posix()] = sha(p)
freeze['outputs/grid55b/design.json'] = sha(OUT / 'design.json')
freeze['outputs/grid55b/raw_preservation.json'] = sha(OUT / 'raw_preservation.json')
freeze.update(raw)
save(OUT / 'code_freeze.json', freeze)
save(OUT / 'preflight.json', {'passed': True, 'planned': 8, 'new_fits': 0,
     'all_original_code_checkpoint_hashes_valid': True, 'all_eight_configs_exact_except_repeat': True,
     'worker_source_change': "Only OUT=ROOT/'outputs/grid55' -> outputs/grid55b; frozen original worker and instrumentation used."})
(OUT / 'RUN_STATE.md').write_text('FROZEN: eight-arm timing supplement only; no new fitting or controller changes.\n', encoding='utf-8')
print(json.dumps({'frozen': True, 'runs': 8, 'week': proposal['week'], 'source_hashes': len(freeze)}))
