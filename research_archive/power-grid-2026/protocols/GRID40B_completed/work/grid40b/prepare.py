"""One training-only small-batch optimization repair; original fits preserved."""
import hashlib
import json
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'outputs/grid40b'; HERE = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf-8'))
A = read(ROOT / 'outputs/grid40/training_audit.json'); assert A['passed_accounting']
assert not (OUT / 'design.json').exists()
OUT.mkdir(parents=True, exist_ok=True)
D = read(ROOT / 'outputs/grid40/design.json'); D['stage'] = 'GRID40B_ONE_SMALL_BATCH_OPTIMIZATION_REPAIR'
D['lr'] = .001
D['training_scope'] = 'One fixed lower-learning-rate repair for documented initial-fit plateau, before any SoftGNN development prediction. Same original initialweights/seeds/shuffle/data/64epochs/batch8/dropout/architecture/KL. No validation/test score or architecture sweep. Original published-config fits retained. Small-batch optimization adaptation,not faithful original training.'
D['configuration_choice'] = 'Use40Bforlive only if both seeds final eval-training KL improve by>=0.05 relative40 and both below frequencyKL. Otherwise diagnose fit before claiming a strong publishedbaseline. No seed/checkpoint/epoch selection by devresult.'
(OUT / 'design.json').write_text(json.dumps(D, indent=2), encoding='utf-8')
for name in ['normalization.npz', 'inputs_manifest.json', 'preflight.json']:
    shutil.copy2(ROOT / 'outputs/grid40' / name, OUT / name)
src = (ROOT / 'work/grid40/train.py').read_text(encoding='utf-8')
replacements = {
    'from adapter import OUT, GAT, read, graph, model_config': "from adapter import OUT as BASE_OUT, GAT, read, graph, model_config\nOUT = ROOT / 'outputs/grid40b'",
    "p = OUT / 'inputs' / f\"{row['id']}.npz\"": "p = BASE_OUT / 'inputs' / f\"{row['id']}.npz\"",
}
for before, after in replacements.items(): assert src.count(before) == 1; src = src.replace(before, after)
assert not (HERE / 'train.py').exists(); (HERE / 'train.py').write_text(src, encoding='utf-8')
files = [Path(__file__).resolve(), HERE / 'train.py', OUT / 'design.json', OUT / 'normalization.npz', OUT / 'inputs_manifest.json']
files += [ROOT / p for p in read(ROOT / 'outputs/grid40/code_freeze.json')]
(OUT / 'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(files)}, indent=2), encoding='utf-8')
(OUT / 'RUN_STATE.md').write_text('FROZEN:one lower-lr fit repair,bothseeds,old64epochfits preserved; noSoftGNNdevprediction yet.\n', encoding='utf-8')
print(json.dumps({'frozen': True, 'seeds': D['seeds'], 'lr': D['lr'], 'epochs': D['epochs']}))
