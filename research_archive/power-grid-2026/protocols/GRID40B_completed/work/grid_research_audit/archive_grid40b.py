"""Preserve one optimization repair, never replace original model/source archives."""
import argparse, hashlib, json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; REPO = ROOT / 'work/GRL-results-integration'
ARCH = REPO / 'research_archive/power-grid-2026'; OUT = ROOT / 'outputs/grid40b'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['before', 'complete'], required=True)
phase = parser.parse_args().phase
if phase == 'before': assert not (OUT / 'training_finished.json').exists()
else: assert read(OUT / 'audit.json')['passed_accounting']
old = {}
for manifest in list(ARCH.glob('*manifest.json')) + list((ARCH / 'protocols').glob('*/manifest.json')):
    for r in read(manifest).get('files', []):
        p = manifest.parent / r['path']
        if not p.exists(): p = ARCH / r['path']
        assert sha(p) == r['sha256']; old[str(p)] = r['sha256']
dest = ARCH / 'protocols' / ('GRID40B_before_training' if phase == 'before' else 'GRID40B_completed')
assert not dest.exists(); dest.mkdir(parents=True)
files = list((ROOT / 'work/grid40b').glob('*.py')) + [Path(__file__).resolve()]
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json', '.md', '.csv']]
files.append(OUT / 'normalization.npz')
if phase == 'complete': files += list((OUT / 'models').rglob('*.json'))
entries = []
for p in sorted(set(files)):
    rel = p.relative_to(ROOT); q = dest / rel; q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q); assert sha(p) == sha(q)
    entries.append({'path': rel.as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size})
(dest / 'manifest.json').write_text(json.dumps({'files': entries, 'historical_preserved': len(old),
    'scope': 'One training-only lower-lr repair; old official-config fits and both seeds retained. No test/validation/modelportfolio search; all checkpoints local with hashes.'}, indent=2), encoding='utf-8')
if phase == 'before':
    note = 'GRID40B仅降低学习率至0.001的两种子拟合已冻结；其余初始模型/数据/64轮/批量8/官方GAT/soft标签不变。用训练拟合平台而非评价质量提出这一版，原论文学习率结果全部保留。'
else:
    a = read(OUT / 'audit.json')
    note = 'GRID40B两种子小批量学习率适配完成；' + '；'.join(
        f"seed{r['seed']}训练KL{r['old_final_training_kl']:.3f}→{r['new_final_training_kl']:.3f}" for r in a['fits'])
    note += f"；统一配置资格{a['configuration_qualified']}。选择仅用训练拟合，尚非闭环质量或近期公开方法性能结论。"
p = REPO / 'docs/RESEARCH_STATE.md'; t = p.read_text(encoding='utf-8')
p.write_text('# Research state — baseline training adaptation\n\n- **最新：** ' + note + '\n\n' + t.split('\n', 1)[1], encoding='utf-8')
for name in ['DECISIONS.md', 'EXPERIMENT_LOG.md']:
    p = REPO / 'docs' / name; p.write_text(p.read_text(encoding='utf-8') + '\n## 2026-10-11 — GRID40B ' + phase + '\n\n' + note + '\n', encoding='utf-8')
for p, h in old.items(): assert sha(Path(p)) == h
print(json.dumps({'archived': phase, 'files': len(entries), 'historical_preserved': len(old)}))
