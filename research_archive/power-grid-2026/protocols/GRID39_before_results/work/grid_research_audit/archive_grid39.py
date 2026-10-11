"""Preserve one limited live revision before and after results."""
import argparse, hashlib, json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'work/GRL-results-integration'
ARCH = REPO / 'research_archive/power-grid-2026'
OUT = ROOT / 'outputs/grid39'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['before', 'complete'], required=True)
phase = parser.parse_args().phase
if phase == 'complete':
    review = read(OUT / 'review.json')
    assert review['passed_saved_data'] and review['runs'] == 12
else:
    assert not (OUT / 'manifest.json').exists()
old = {}
for manifest in list(ARCH.glob('*manifest.json')) + list((ARCH / 'protocols').glob('*/manifest.json')):
    for row in read(manifest).get('files', []):
        p = manifest.parent / row['path']
        if not p.exists(): p = ARCH / row['path']
        assert sha(p) == row['sha256'], p
        old[str(p)] = row['sha256']
dest = ARCH / 'protocols' / ('GRID39_before_results' if phase == 'before' else 'GRID39_completed')
assert not dest.exists()
dest.mkdir(parents=True)
files = list((ROOT / 'work/grid39').glob('*.py')) + [Path(__file__).resolve()]
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json', '.csv', '.md']]
if phase == 'complete':
    for folder in (OUT / 'runs').iterdir():
        files += [folder / n for n in ['summary.json', 'metadata.json', 'components.json']]
records = []
for p in sorted(set(files)):
    rel = p.relative_to(ROOT)
    q = dest / rel
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q)
    h = sha(p)
    assert sha(q) == h
    records.append({'path': rel.as_posix(), 'sha256': h, 'bytes': p.stat().st_size})
(dest / 'manifest.json').write_text(json.dumps({'files': records, 'historical_preserved': len(old),
    'scope': 'One unchanged-control revision;12runs/single timing repeat;both GNN seeds and fresh FULL. Original GRID37 is a historical quality/query reference, not paired new-model timing. Final8datefamilies sealed.'}, indent=2), encoding='utf-8')
if phase == 'before':
    note = 'GRID39已冻结12条有限闭环评价：一版目标修改的两个GNN种子与新运行FULL，4个已暴露开发周、单次计时、轮换顺序。控制/停止/守卫/缓存/物理求解不变；原模型和GRID37结果保留，新旧时间不作配对比较。不扩大规则或消融清单，最终8日期仍封存。'
    next_text = '''1. 继续GRID39十二条串行冻结运行，先看实际质量/存活，再核对完整时间、源搜索与查询。各条独立保存，可断点续未完成条目。
2. 不从离线混合结果宣告目标修改成功，也不因小幅差异停掉课题；读全部种子后决定是否保留这版或回到原版。
3. 正式主对照2–3项，优先近期最接近的公开论文及成熟领域控制器；只做必要适配，不要求全面击败竞赛顶尖系统。MLP/静态已作为消融保留，不继续扩展。
4. 最终8日期仍封存，LG01暂停；同质量提速是当前应用主张，GNN+RL组合与共有编译技巧不是新意保证。
'''
else:
    parts = []
    for s in review['summaries']:
        if s['method'] == 'FULL': continue
        c = next(c for c in review['comparisons'] if c['method'] == s['method'])
        speed = c['pooled_control_reduction']
        parts.append(f"{s['method']}质量通过{s['quality_qualified_pairs']}/4、最大可比成本差" +
            ('不可比' if s['max_comparable_cost_regret'] is None else f"{100*s['max_comparable_cost_regret']:.3f}%") +
            '、控制时间减少' + ('不可比' if speed is None else f'{100*speed:.2f}%'))
    note = 'GRID39十二条一版目标修订闭环及保存数据核验完成；' + '；'.join(parts) + '。全部结果保留，仅单次计时开发测量，不把跨批旧/新时间当配对模型提升。'
    next_text = '''1. GRID39已完成，先读取全部质量/查询/组件结果以及原GRID37质量参照，决定这版目标是否有实用改进；不按最快种子择优。
2. 正式基线范围保持有限，接下来围绕近期最接近SoftGNN的必要适配与当前成熟控制器组织实验，不再补廉价规则组合。
3. 没有明确收益时保留原模型与这版失败/混合记录，改进须有具体机制依据，不扫参数或以小差异否定整个课题。
4. 最终8日期仍封存，LG01暂停；目标是合理质量—时间改进，非每项指标第一或全面击败顶尖竞赛系统。
'''
docs = REPO / 'docs'
p = docs / 'RESEARCH_STATE.md'
t = p.read_text(encoding='utf-8')
p.write_text('# Research state — limited target-revision rollout\n\n- **最新：** ' + note + '\n\n' + t.split('\n', 1)[1], encoding='utf-8')
(docs / 'NEXT_STEPS.md').write_text('# Next steps — focused application contribution\n\n' + next_text, encoding='utf-8')
for name in ['DECISIONS.md', 'EXPERIMENT_LOG.md']:
    p = docs / name
    p.write_text(p.read_text(encoding='utf-8') + '\n## 2026-10-11 — GRID39 ' + phase + '\n\n' + note + '\n', encoding='utf-8')
p = ARCH / 'README.md'
link = 'design.json' if phase == 'before' else 'GRID39_report.md'
p.write_text('# 最新：GRID39 ' + phase + '\n\n[方案/报告](protocols/' + dest.name + '/outputs/grid39/' + link + ')\n\n' + note + '\n\n' + p.read_text(encoding='utf-8'), encoding='utf-8')
for p, h in old.items(): assert sha(Path(p)) == h, p
print(json.dumps({'passed': True, 'phase': phase, 'files': len(records), 'historical_preserved': len(old)}))
