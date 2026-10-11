"""Preserve official-core adaptation with source/config/cost and old archive hashes."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'work/GRL-results-integration'; ARCH = REPO / 'research_archive/power-grid-2026'
OUT = ROOT / 'outputs/grid40'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['before', 'fit', 'complete'], required=True)
phase = parser.parse_args().phase
assert read(OUT / 'preflight.json')['passed']
if phase == 'before': assert not (OUT / 'training_finished.json').exists()
elif phase == 'fit': assert read(OUT / 'training_audit.json')['passed_accounting']
else: assert read(OUT / 'offline_review.json')['passed_execution']
old = {}
for manifest in list(ARCH.glob('*manifest.json')) + list((ARCH / 'protocols').glob('*/manifest.json')):
    for row in read(manifest).get('files', []):
        p = manifest.parent / row['path']
        if not p.exists(): p = ARCH / row['path']
        assert sha(p) == row['sha256'], p
        old[str(p)] = row['sha256']
dest = ARCH / 'protocols' / {'before': 'GRID40_before_training', 'fit': 'GRID40_training_complete', 'complete': 'GRID40_completed'}[phase]
assert not dest.exists(); dest.mkdir(parents=True)
files = list((ROOT / 'work/grid40').glob('*.py')) + [Path(__file__).resolve()]
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json', '.csv', '.md']]
files.append(OUT / 'normalization.npz')
if phase != 'before': files += list((OUT / 'models').rglob('*.json'))
entries = []
for p in sorted(set(files)):
    rel = p.relative_to(ROOT); q = dest / rel; q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q); assert sha(q) == sha(p)
    entries.append({'path': rel.as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size})
(dest / 'manifest.json').write_text(json.dumps({'files': entries, 'historical_preserved': len(old),
    'scope': 'Official GAT/converter unchanged, documented resource-matched IDF2023 adaptation; no faithful WCCI reproduction claim. Initial/final models and graph input arrays local with hashes; no final dates.'}, indent=2), encoding='utf-8')
if phase == 'before':
    note = 'GRID40官方SoftGNN核心适配预检通过：167既有状态/148正标签、909动作、27维拓扑对象节点、3843309参数。原始GAT/输入转换源码未改；加入相同公开连续base信息、训练侧标准化、共同合法mask和搜索接口。两个种子固定64epochs/batch8尚未拟合；作者800max/256batch/分布式调参未复现，弱试点不否定公开方法。'
    next_text = '1. 运行GRID40两个固定SoftGNN适配训练，检查训练收敛与26暴露状态排序。\n2. 再以共同控制/缓存/停止在开发日期比较原GNN与适配SoftGNN；保留全部种子和失败。\n3. 主基线仍2–3项，不扩大廉价规则/组合；最终8日期封存，LG01暂停。\n'
elif phase == 'fit':
    note = 'GRID40两个原论文学习率的资源匹配拟合完成并核验目标/训练侧标准化；原模型保留。训练拟合存在平台，需要先做一版小批量优化适配（仅学习率降至0.001，其余初始权重/两种子/64轮/数据/结构不变），尚未读取SoftGNN开发预测，不拿欠拟合当作者方法失败。'
    next_text = '1. 先做固定小批量优化适配，两种子与原始拟合均保留；只按训练侧拟合诊断选择统一配置。\n2. 再做现有26状态与有限同条件闭环评价，不读最终8日期。\n3. 主对照保持近期SoftGNN适配与成熟控制器，不展开超参数或廉价规则清单。\n'
else:
    s = read(OUT / 'offline_review.json')['summaries']
    note = 'GRID40两个固定SoftGNN核心适配拟合及26状态离线复核完成；' + '；'.join(
        f"{r['ranker']}查询{r['queries']}、同FULL选择{r['same_full_choice']}/26、关键1733匹配{r['known_critical_choice_match']}" for r in s)
    note += '。这是公开方法资源匹配适配，不是作者已训练模型或忠实WCCI复现；闭环质量与实际速度尚未确定。'
    next_text = '1. 原模型保持；读GRID40收敛/排序结果后进入有限共同控制闭环比较，不把离线名次当实际收益。\n2. 保留近期SoftGNN适配与成熟完整搜索主对照，所有方法共有优化；不再扩廉价基线。\n3. 最终8日期封存，LG01暂停，不按最优种子筛选。\n'
docs = REPO / 'docs'; p = docs / 'RESEARCH_STATE.md'; t = p.read_text(encoding='utf-8')
p.write_text('# Research state — recent SoftGNN baseline adaptation\n\n- **最新：** ' + note + '\n\n' + t.split('\n', 1)[1], encoding='utf-8')
(docs / 'NEXT_STEPS.md').write_text('# Next steps — focused application contribution\n\n' + next_text, encoding='utf-8')
for name in ['DECISIONS.md', 'EXPERIMENT_LOG.md']:
    p = docs / name; p.write_text(p.read_text(encoding='utf-8') + '\n## 2026-10-11 — GRID40 ' + phase + '\n\n' + note + '\n', encoding='utf-8')
for p, h in old.items(): assert sha(Path(p)) == h
print(json.dumps({'archived': phase, 'files': len(entries), 'historical_preserved': len(old)}))
