"""Narrow matched comparison; publish configuration and all method summaries."""
import argparse, hashlib, json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; REPO = ROOT / 'work/GRL-results-integration'
ARCH = REPO / 'research_archive/power-grid-2026'; OUT = ROOT / 'outputs/grid41'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['before', 'complete'], required=True)
phase = parser.parse_args().phase
if phase == 'before': assert not (OUT / 'manifest.json').exists()
else: assert read(OUT / 'review.json')['passed_saved_data'] and read(OUT / 'review.json')['runs'] == 20
old = {}
for manifest in list(ARCH.glob('*manifest.json')) + list((ARCH / 'protocols').glob('*/manifest.json')):
    for r in read(manifest).get('files', []):
        p = manifest.parent / r['path']
        if not p.exists(): p = ARCH / r['path']
        assert sha(p) == r['sha256']; old[str(p)] = r['sha256']
dest = ARCH / 'protocols' / ('GRID41_before_results' if phase == 'before' else 'GRID41_completed')
assert not dest.exists(); dest.mkdir(parents=True)
files = list((ROOT / 'work/grid41').glob('*.py')) + [Path(__file__).resolve()]
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json', '.md', '.csv']]
if phase == 'complete':
    for folder in (OUT / 'runs').iterdir(): files += [folder / name for name in ['summary.json', 'metadata.json', 'components.json']]
entries = []
for p in sorted(set(files)):
    rel = p.relative_to(ROOT); q = dest / rel; q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q); assert sha(p) == sha(q)
    entries.append({'path': rel.as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size})
(dest / 'manifest.json').write_text(json.dumps({'files': entries, 'historical_preserved': len(old),
    'scope': '20fresh matched controller runs,original2GNN and2SoftGNN adapted seeds plusFULL;1timingrepeat/4exposedweeks. All timings include feature/model/control/guard; shared caches for every arm. Not sealed final/faithfulWCCI reproduction.'}, indent=2), encoding='utf-8')
if phase == 'before':
    note = 'GRID41已冻结20条同条件闭环：原GNN两种子、近期SoftGNN适配两种子、成熟FULL。4暴露开发日期、单次轮换计时；共同控制/缓存/停止/保护不变，原版与学习率适配模型全部保留。SoftGNN统一配置由训练拟合确定，不按开发表现择种子。'
    next_text = '1. 完成GRID41串行20条，并核对质量/失败、完整控制耗时与拓扑查询；小差异先解释，不随意终止课题。\n2. 必要时解决工程/适配困难，保留全部改动和模型，不扩主对照清单。\n3. 最终8日期封存，LG01暂停；只在确需人工决定或无法推进时停。\n'
else:
    R = read(OUT / 'review.json')
    note = 'GRID41二十条共同控制比较与保存数据核验完成；' + '；'.join(
        f"{r['method']}质量通过{r['quality_qualified_pairs']}/4、查询{r['queries']}、控制{r['control_s']:.2f}秒" for r in R['summaries'])
    note += '。单次开发计时，近期SoftGNN是明确资源匹配适配；不可视为原论文完整复现或封存泛化。'
    next_text = '1. 读取GRID41全部质量—时间与模型适配结果，定位实际贡献；不选最佳种子，不把共有编译节省归GNN。\n2. 保留成熟控制器与近期SoftGNN适配两项主基线，只做有具体依据的改进。\n3. 最终8日期保持封存，LG01暂停；无需人工决定时继续推进。\n'
p = REPO / 'docs/RESEARCH_STATE.md'; t = p.read_text(encoding='utf-8')
p.write_text('# Research state — matched recent baseline comparison\n\n- **最新：** ' + note + '\n\n' + t.split('\n', 1)[1], encoding='utf-8')
(REPO / 'docs/NEXT_STEPS.md').write_text('# Next steps — focused application contribution\n\n' + next_text, encoding='utf-8')
for name in ['DECISIONS.md', 'EXPERIMENT_LOG.md']:
    p = REPO / 'docs' / name; p.write_text(p.read_text(encoding='utf-8') + '\n## 2026-10-11 — GRID41 ' + phase + '\n\n' + note + '\n', encoding='utf-8')
for p, h in old.items(): assert sha(Path(p)) == h
print(json.dumps({'archived': phase, 'files': len(entries), 'historical_preserved': len(old)}))
