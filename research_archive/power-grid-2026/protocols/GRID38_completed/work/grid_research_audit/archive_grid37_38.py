"""Preserve completed comparisons/revision and the focused baseline policy."""
import argparse, hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'work/GRL-results-integration'
ARCH = REPO / 'research_archive/power-grid-2026'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=int, choices=[37, 38], required=True)
stage = parser.parse_args().stage
out = ROOT / f'outputs/grid{stage}'
review = read(out / 'review.json')
assert review.get('passed_saved_data', review.get('passed_saved_screen', False))
if stage == 38:
    assert read(out / 'audit.json')['passed']

old = {}
for manifest in list(ARCH.glob('*manifest.json')) + list((ARCH / 'protocols').glob('*/manifest.json')):
    for row in read(manifest).get('files', []):
        p = manifest.parent / row['path']
        if not p.exists():
            p = ARCH / row['path']
        assert sha(p) == row['sha256'], p
        old[str(p)] = row['sha256']

files = list((ROOT / f'work/grid{stage}').glob('*.py')) + [Path(__file__).resolve()]
files += [p for p in out.iterdir() if p.is_file() and p.suffix in ['.json', '.md', '.csv']]
if stage == 37:
    for folder in (out / 'runs').iterdir():
        files += [folder / name for name in ['summary.json', 'metadata.json', 'components.json']]
    files.append(ROOT / 'outputs/grid_target_audit/PAPER_FOCUS.md')
    scope = 'All32 fair delivered-controller runs. Compact results and component measurements archived; large source/physical vectors remain local. No new fit or final-date evaluation.'
else:
    for folder in (out / 'models').iterdir():
        files += [folder / name for name in ['curve.json', 'summary.json']]
    model_files = sorted((out / 'models').glob('*/*.pt'))
    assert len(model_files) == 8
    (out / 'model_manifest.json').write_text(json.dumps({'files': [
        {'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
        for p in model_files], 'preserved_locally': True,
        'original_weights_unchanged': True}, indent=2), encoding='utf-8')
    files.append(out / 'model_manifest.json')
    scope = 'One target mixture, four matched fits, all104 cached development comparisons. Original/new model binaries preserved locally with hashes. No new teacher forecast or final-date evaluation.'

dest = ARCH / 'protocols' / f'GRID{stage}_completed'
assert not dest.exists(), dest
dest.mkdir(parents=True)
records = []
for p in sorted(set(files)):
    assert p.is_file(), p
    rel = p.relative_to(ROOT)
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, target)
    h = sha(p)
    assert sha(target) == h
    records.append({'path': rel.as_posix(), 'sha256': h, 'bytes': p.stat().st_size})
(dest / 'manifest.json').write_text(json.dumps({'files': records,
    'historical_preserved': len(old), 'scope': scope}, indent=2), encoding='utf-8')

if stage == 37:
    by = {r['method']: r for r in review['summaries']}
    parts = []
    for name in ['LOCAL_GNN0', 'LOCAL_GNN1', 'STATIC_ID']:
        r = by[name]
        cost = r['max_comparable_cost_regret']
        parts.append(f"{name}质量通过{r['quality_qualified_pairs']}/8、最大可比成本差" +
                     ('不可比' if cost is None else f'{100*cost:.3f}%'))
    note = 'GRID37全部32条公平静态/GNN/完整搜索比较及保存数据核验完成；' + '；'.join(parts) + '。两模型种子和两计时重复均保留，详见质量—时间主表。静态顺序属于必要消融，不扩展为方法清单。'
    next_steps = '''1. GRID37已完成，无需重跑。执行已冻结GRID38单一目标修改的四次匹配拟合，保留原初始化、原权重、原训练顺序。
2. 四次拟合保存后完成既有26开发状态的排序检查，再进行实际闭环质量/控制计时。不开启最终8日期，不追加规则筛查。
3. 正式主对照控制2–3项，优先近期相近公开论文方法及领域认可控制器；完整搜索/PPO是同一成熟控制器的版本，MLP/静态仅必要消融。
4. 当前投稿贡献收敛为相近质量下的辅助搜索与完整控制提速；不以全面击败最强竞赛系统或每项指标第一为成立条件，不按结果挑弱对手。LG01仍暂停。
'''
else:
    parts = [f"{r['ranker']}查询{r['old_queries']}→{r['queries']}、同FULL选择{r['old_same_full_choice']}→{r['same_full_choice']}/26、关键动作{r['known_critical_choice_match']}" for r in review['summaries']]
    note = 'GRID38一版目标修订的四次拟合与104项已有反馈排序核验完成；' + '；'.join(parts) + '。原/新模型、全部种子和训练记录保留，未新增教师仿真。离线查询与排序变化尚不是闭环质量或完整提速。'
    next_steps = '''1. GRID38四个新模型均已保存并审计，读取全部种子；不选择最佳checkpoint、不扩参数扫描。
2. 按原共同缓存、停止与控制规则测量一版修订的实际闭环质量及完整控制时间；若已知关键动作回归，先记录并解释，不能原样派发。
3. 正式主对照维持2–3项近期相近论文/领域认可方法，MLP与静态为消融。只做必要可行的移植，不自动重建昂贵竞赛训练平台。
4. 最终8日期仍封存。新目标配方和共有编译优化不是独立创新保证；围绕实际应用改进组织稿件，LG01保持暂停。
'''

docs = REPO / 'docs'
p = docs / 'RESEARCH_STATE.md'
t = p.read_text(encoding='utf-8')
p.write_text(f'# Research state — GRID{stage} completed\n\n- **最新：** {note}\n\n' + t.split('\n', 1)[1], encoding='utf-8')
(docs / 'NEXT_STEPS.md').write_text('# Next steps — focused application contribution\n\n' + next_steps, encoding='utf-8')
for name in ['DECISIONS.md', 'EXPERIMENT_LOG.md']:
    p = docs / name
    p.write_text(p.read_text(encoding='utf-8') + f'\n## 2026-10-11 — GRID{stage} completed\n\n' + note + '\n', encoding='utf-8')
p = ARCH / 'README.md'
p.write_text(f'# 最新：GRID{stage}完成\n\n[完整报告](protocols/GRID{stage}_completed/outputs/grid{stage}/GRID{stage}_report.md)\n\n' + note + '\n\n' + p.read_text(encoding='utf-8'), encoding='utf-8')
for p, h in old.items():
    assert sha(Path(p)) == h, p
print(json.dumps({'passed': True, 'stage': stage, 'archived': len(records), 'historical_preserved': len(old)}))
