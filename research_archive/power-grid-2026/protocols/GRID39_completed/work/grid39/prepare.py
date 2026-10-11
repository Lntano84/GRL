"""Freeze a limited closed-loop check of the one saved target revision."""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid39'
HERE = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

assert read(ROOT / 'outputs/grid38/audit.json')['passed']
new = read(ROOT / 'outputs/grid38/review.json')
assert all(r['known_critical_choice_match'] for r in new['summaries'] if r['ranker'].startswith('GNN'))
assert len(read(ROOT / 'outputs/grid38/training_finished.json')['fits']) == 4
old = read(ROOT / 'outputs/grid34/design.json')
methods = [dict(m) for m in old['methods'] if m['method'] in ['FULL', 'LOCAL_GNN0', 'LOCAL_GNN1']]
for m in methods:
    if m['method'] != 'FULL':
        m['method'] = m['method'].replace('LOCAL_', 'REVISED_')
        m['checkpoint'] = m['checkpoint'].replace('outputs/grid21/', 'outputs/grid38/')
order = []
for w, week in enumerate(old['weeks']):
    offset = w % len(methods)
    for m in methods[offset:] + methods[:offset]:
        order.append({**m, 'week': week, 'repeat': 0, 'stage_phase': 'pilot'})
D = {
    'stage': 'GRID39_ONE_REVISION_CLOSED_LOOP', 'weeks': old['weeks'], 'methods': methods,
    'execution_order': order, 'repeats': 1, 'serial_workers': 1,
    'timing': 'One fresh serial single-thread process per arm/week; rotate3arm order across4exposed dates. Same DC+safeQP cache and component instrumentation. All feature/model/forecast/solve/guard time included in controller; setup/export/physical stepping separate. Single timing repeat is a development point estimate.',
    'changes': 'Only load the two GRID38 GNN final checkpoints. All control/search/canonicalization/cadence/solver/guard settings remain unchanged. No new algorithms or rule sweep.',
    'scope': '12development runs, not final confirmation. GRID37 original models remain a historical quality/query reference, not a paired timing opponent. MLP revision already retained in offline ablation; no extra closed-loop ablation batch now.',
    'priorities': ['No new or earlier failure versus fresh FULL', 'Comparable cost increase <=0.5% relative to fresh FULL', 'Report both seeds and all dates; do not choose best seed', 'Read actual quality before interpreting offline ranking; no stable speed claim from one timing repeat'],
    'caps': {'episode_s': 1500, 'wall_s': 5000, 'physical_steps': 26000, 'public_forecasts': 250000, 'new_fits': 0, 'reserved_evaluations': 0},
}

run = (ROOT / 'work/grid34/run.py').read_text(encoding='utf-8')
changes = {
    '"""GRID34 common-cache FULL and all existing rankers; staged quality pilot, resumable."""': '"""GRID39 limited closed-loop check of both revised GNNs and fresh FULL."""',
    "OUT=ROOT/'outputs/grid34'": "OUT=ROOT/'outputs/grid39'",
    'all6arms shareDC+safe cache': 'all3arms shareDC+safe cache',
}
for before, after in changes.items():
    assert run.count(before) == 1, before
    run = run.replace(before, after)
assert not (HERE / 'run.py').exists()
(HERE / 'run.py').write_text(run, encoding='utf-8')

audit = (ROOT / 'work/grid37/audit.py').read_text(encoding='utf-8')
changes = {
    "OUT=ROOT/'outputs/grid37'": "OUT=ROOT/'outputs/grid39'",
    "len(F['summaries'])==32": "len(F['summaries'])==12",
    "for week in D['weeks']:\n    for m in D['methods']:\n        a=(0,week,m['method']);b=(1,week,m['method']);repeats.append({'week':week,'method':m['method'],'same_path':paths[a]==paths[b],'same_queries':index[a]['n1_forecasts']==index[b]['n1_forecasts'],'same_cost':index[a]['cost']==index[b]['cost']})": "# One timing repeat; no nonexistent repeat comparison.",
    "[('STATIC_ID','FULL'),('LOCAL_GNN0','FULL'),('LOCAL_GNN1','FULL'),('LOCAL_GNN0','STATIC_ID'),('LOCAL_GNN1','STATIC_ID')]": "[('REVISED_GNN0','FULL'),('REVISED_GNN1','FULL')]",
    'for repeat in [0,1]:': 'for repeat in [0]:',
    "'runs':32": "'runs':12",
    "'warning':'4 exposed development dates; 2 timing repeats are not independent date families. .5% is an investment quality tolerance, not statistical equivalence. Author base PPO and common caches remain in every arm. Saved-data audit is not independent AC feasibility solving. Unmatched earlier failure cannot count as speed win.'": "'warning':'4 exposed development dates and one timing repeat. Original GRID37 timings are historical, not paired to new models. Both seeds retained; .5% is a quality investment tolerance, not statistical equivalence. Common author PPO/caches remain in all arms. Saved-data audit is not independent AC feasibility solving. Incomparable earlier failure never counts as speed win.'",
    "# GRID37 静态顺序与GNN的公平实际比较": "# GRID39 一版目标修订的实际闭环评价",
    "f'32条；{physical}物理步；{forecasts}公开预测。'": "f'12条；{physical}物理步；{forecasts}公开预测。'",
    "{r['quality_qualified_pairs']}/8": "{r['quality_qualified_pairs']}/4",
    "{r['completed_runs']}/8": "{r['completed_runs']}/4",
    "{c['positive_time_pairs']}/8": "{c['positive_time_pairs']}/4",
    "OUT/'GRID37_report.md'": "OUT/'GRID39_report.md'",
    'COMPLETED AND AUDITED:32 fairstatic comparisons. No new rules/fits/final dates.': 'COMPLETED AND AUDITED:12 revised-model development comparisons. No new fits/final dates.',
}
for before, after in changes.items():
    assert audit.count(before) == 1, (before, audit.count(before))
    audit = audit.replace(before, after)
assert not (HERE / 'audit.py').exists()
(HERE / 'audit.py').write_text(audit, encoding='utf-8')
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / 'design.json').exists()
(OUT / 'design.json').write_text(json.dumps(D, indent=2), encoding='utf-8')
files = [Path(__file__).resolve(), HERE / 'run.py', HERE / 'audit.py']
files += [ROOT / p for p in read(ROOT / 'outputs/grid34/code_freeze.json')]
files += [ROOT / m['checkpoint'] for m in methods if m['checkpoint']]
files = list(dict.fromkeys(files))
(OUT / 'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2), encoding='utf-8')
(OUT / 'RUN_STATE.md').write_text('FROZEN:12 limited revised-model runs, unchanged control/search. Original weights retained; final8dates sealed.\n', encoding='utf-8')
print(json.dumps({'frozen': True, 'runs': len(order), 'new_fits': 0, 'source_control_changes': 0}))
