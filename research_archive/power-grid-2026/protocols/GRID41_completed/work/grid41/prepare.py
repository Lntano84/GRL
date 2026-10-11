"""Fresh matched closed-loop comparison, narrow published baseline scope."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'outputs/grid41'; HERE = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf-8'))
assert read(ROOT / 'outputs/grid40b/audit.json')['configuration_qualified']
assert read(ROOT / 'outputs/grid40b/offline_review.json')['passed_execution']
assert len(read(ROOT / 'outputs/grid40b/training_finished.json')['fits']) == 2
old = read(ROOT / 'outputs/grid34/design.json')
methods = [dict(m) for m in old['methods'] if m['method'] in ['FULL', 'LOCAL_GNN0', 'LOCAL_GNN1']]
for seed in [0, 1]:
    methods.append({'method': f'SOFTGNN{seed}', 'rule': 'SOFTGNN', 'budget': 128,
                    'checkpoint': f'outputs/grid40b/models/softgnn_seed{seed}/final.pt',
                    'policy': 'PATIENCE64', 'cache_dc': True, 'cache_safe': True})
order = []
for w, week in enumerate(old['weeks']):
    offset = w % len(methods)
    for method in methods[offset:] + methods[:offset]:
        order.append({**method, 'week': week, 'repeat': 0, 'stage_phase': 'pilot'})
D = {'stage': 'GRID41_MATCHED_SOFTGNN_PILOT', 'weeks': old['weeks'], 'methods': methods,
     'execution_order': order, 'repeats': 1, 'serial_workers': 1,
     'timing': 'Fresh serial single-thread worker per arm/week; rotate5arm order. Identical DC/safeQP caches, timer wrappers, physicalbackend, original PPO branches, controlcadence. All graph/input/model/simulation/optimization/guard time within controller. Setup, physicalstep and export separately. Single timing repetition is a development point estimate.',
     'changes': 'SoftGNN changes ranker only through imported official core and documented adaptation. Both seeds use one uniform lower-lr configuration qualified solely by training fit; original published-lr fits preserved. Original GNN0/GNN1 checkpoints unchanged; FULL exhaustive. All rankers share exact same masks/canonicalactions/stopping/localcompletion/restoration.',
     'scope': '4already exposed development weeks; no sealed final8dates. Five implementations represent2modelseeds of our method,2seeds of one recent publishedmethod adaptation and1mature exhaustivecontroller. No expanded baseline portfolio.',
     'interpretation': 'Retain every seed and failure; slight/mixed changes are not a whole-project stop. Resource-matched SoftGNN pilot cannot refute official WCCI2022 fulltraining. No earlier failure and comparable cost<=0.5%ofFULL are quality investment screens, not equivalence.',
     'caps': {'episode_s': 1500, 'wall_s': 8500, 'physical_steps': 42000, 'public_forecasts': 450000, 'new_fits': 0, 'reserved_evaluations': 0}}
run = (ROOT / 'work/grid34/run.py').read_text(encoding='utf-8')
changes = {
    '"""GRID34 common-cache FULL and all existing rankers; staged quality pilot, resumable."""': '"""GRID41 matched official-core SoftGNN and original GNN development runs."""',
    "OUT=ROOT/'outputs/grid34'": "OUT=ROOT/'outputs/grid41'",
    'all6arms shareDC+safe cache': 'all5arms shareDC+safe cache',
    "worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker);worker.OUT=OUT":
        "worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker);worker.OUT=OUT\n    sys.path[:0]=[str(ROOT/'work/grid40'),str(ROOT/'work/grid28')]\n    from adapter import install_search_adapter\n    import engine\n    install_search_adapter(engine.Search)",
}
for before, after in changes.items():
    assert run.count(before) == 1, before
    run = run.replace(before, after)
assert not (HERE / 'run.py').exists(); (HERE / 'run.py').write_text(run, encoding='utf-8')
audit = (ROOT / 'work/grid39/audit.py').read_text(encoding='utf-8')
changes = {
    "OUT=ROOT/'outputs/grid39'": "OUT=ROOT/'outputs/grid41'",
    "len(F['summaries'])==12": "len(F['summaries'])==20",
    "[('REVISED_GNN0','FULL'),('REVISED_GNN1','FULL')]":
        "[('LOCAL_GNN0','FULL'),('LOCAL_GNN1','FULL'),('SOFTGNN0','FULL'),('SOFTGNN1','FULL'),('LOCAL_GNN0','SOFTGNN0'),('LOCAL_GNN0','SOFTGNN1'),('LOCAL_GNN1','SOFTGNN0'),('LOCAL_GNN1','SOFTGNN1')]",
    "'runs':12": "'runs':20",
    "'warning':'4 exposed development dates and one timing repeat. Original GRID37 timings are historical, not paired to new models. Both seeds retained; .5% is a quality investment tolerance, not statistical equivalence. Common author PPO/caches remain in all arms. Saved-data audit is not independent AC feasibility solving. Incomparable earlier failure never counts as speed win.'":
        "'warning':'4exposed development dates,1timingrepeat. Fresh originalGNN and SoftGNN adaptation compared under common controller/caches/stop. Both seeds retained. .5%is investment quality tolerance,not equivalence. Official-core adaptation,not original WCCI trained agent. Saved audit is not independent ACsolve/refit. Incomparable failure not speedwin.'",
    '# GRID39 一版目标修订的实际闭环评价': '# GRID41 近期SoftGNN适配与原模型的同条件闭环比较',
    "f'12条；{physical}物理步；{forecasts}公开预测。'": "f'20条；{physical}物理步；{forecasts}公开预测。'",
    "OUT/'GRID39_report.md'": "OUT/'GRID41_report.md'",
    'COMPLETED AND AUDITED:12 revised-model development comparisons. No new fits/final dates.':
        'COMPLETED AND AUDITED:20 matched SoftGNN/original model comparisons. No new fits/final dates.',
}
for before, after in changes.items():
    assert audit.count(before) == 1, (before, audit.count(before))
    audit = audit.replace(before, after)
assert not (HERE / 'audit.py').exists(); (HERE / 'audit.py').write_text(audit, encoding='utf-8')
OUT.mkdir(parents=True, exist_ok=True); assert not (OUT / 'design.json').exists()
(OUT / 'design.json').write_text(json.dumps(D, indent=2), encoding='utf-8')
files = [Path(__file__).resolve(), HERE / 'run.py', HERE / 'audit.py', ROOT / 'work/grid40/adapter.py', ROOT / 'outputs/grid40/design.json', ROOT / 'outputs/grid40/normalization.npz', ROOT / 'outputs/grid40b/design.json', ROOT / 'outputs/grid40b/audit.json']
files += [ROOT / p for p in read(ROOT / 'outputs/grid34/code_freeze.json')]
files += [ROOT / p for p in read(ROOT / 'outputs/grid40/code_freeze.json')]
files += [ROOT / m['checkpoint'] for m in methods if m['checkpoint']]
(OUT / 'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(files)}, indent=2), encoding='utf-8')
(OUT / 'RUN_STATE.md').write_text('FROZEN:20matched development runs, common control/stop/caches; all5arms retained; final8dates sealed.\n', encoding='utf-8')
print(json.dumps({'frozen': True, 'runs': len(order)}))
