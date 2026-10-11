"""Publish audited safe-QP numerical qualification and6-run pilot before outcome."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
R=json.loads((ROOT/'outputs/grid30/audit.json').read_text());assert R['passed']
assert not (ROOT/'outputs/grid31/finished.json').exists()
preserved={}
for p in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for row in json.loads(p.read_text(encoding='utf-8')).get('files',[]):
        target=p.parent/row['path']
        if not target.exists():target=ARCH/row['path']
        assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'],target
        preserved[str(target)]=row['sha256']
def copy(dest,files,scope):
    assert not dest.exists();dest.mkdir(parents=True);records=[]
    for p in sorted(set(files)):
        if not p.is_file():continue
        rel=p.relative_to(ROOT);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
        records.append({'path':rel.as_posix(),'sha256':h,'bytes':p.stat().st_size})
    (dest/'manifest.json').write_text(json.dumps({'files':records,'scope':scope,'historical_preserved':len(preserved)},indent=2),encoding='utf-8')
    return len(records)
files=list((ROOT/'work/grid30').glob('*.py'))+[Path(__file__).resolve()]
files += [p for p in (ROOT/'outputs/grid30').glob('*') if p.is_file() and p.suffix in ['.json','.md']]
files += [ROOT/'outputs/grid30/runs/r0__2035-03-05_0__SAFE_QP_SHADOW'/n for n in ['metadata.json','summary.json']]
finished=copy(ARCH/'protocols/GRID30_qualified',files,'9manual/copy and153live paired safe-QP numerical/action qualification; full vectors/parameters and335step failure fragment local.')
files=list((ROOT/'work/grid31').glob('*.py'))+[ROOT/'outputs/grid31'/n for n in ['design.json','code_freeze.json']]
files += [p for p in (ROOT/'outputs/grid31/revisions').rglob('*') if p.is_file()]
prepared=copy(ARCH/'protocols/GRID31_before_results',files,'6balanced actual original/DC/DC+safe trajectories before first cached safe deployment; prefreeze source-count failure retained.')
note=(f"GRID30安全QP共享编译复用资格通过：9项手工/复制检查、153次真实预求解参数/记忆三向量，最大差{R['max_numerical_error_MW']:.3g}MW；306次动作重建相同，2017步实际路径/成本/查询仍原保护控制。隔离原QP{R['shadow_times_s']['ORIGINAL']:.2f}s、缓存{R['shadow_times_s']['CACHED']:.2f}s，减少{100*R['shadow_reduction']:.2f}%，不构成整套提速。V0原负值setter失败已保留，V1原位原样复制未裁剪。")
docs=REPO/'docs';p=docs/'RESEARCH_STATE.md';text=p.read_text(encoding='utf-8')
p.write_text('# Research state — GRID30 qualified / GRID31 frozen\n\n- **最新：** '+note+' GRID31六条三臂平衡完整计时已冻结，尚无缓存安全QP实际派发成绩。\n\n'+text.split('\n',1)[1],encoding='utf-8')
(docs/'NEXT_STEPS.md').write_text('''# Next steps — GRID31 actual three-arm pilot

1. 跑满固定6条三月实际派发轨迹：原控制/DC复用/DC+安全QP复用，正反两种顺序。所有动作/观测/成本/查询必须逐步等同原FULL；有差异即保存并停止，不放松质量门槛。
2. 全部完整控制/组件/solver状态与迭代计数审计；不将局部影子64%与旧整体6.5%相加，也不只选更快重复。
3. 这是共享工程优化，所有强对照同样受益；GNN自身排序/辅助搜索贡献仍需公平单独核验。GRID28质量取舍与历史统计全保留。
4. 没有新拟合/RL/最终8日期家族评价。用户因额度暂停说继续时，续当前任务而不重新选题或重复求确认。
''',encoding='utf-8')
for name,extra in [('EXPERIMENT_LOG.md','\n## 2026-10-11 — GRID30 qualified, GRID31 actual pilot frozen\n\n'+note+' Saved-data audit reconstructs306actions without new QP/AC solve.6actual delivered-arm profiles frozen before cached safe output; same original numeric/status/warm_start/cadence settings, no new fit/hidden dates.\n'),
    ('DECISIONS.md','''\n## 2026-10-11 — Shared safe-QP cache qualifies; measure rather than add percentages

Isolated safe-QP outputs and converted actions match original at live inputs. This permits a6run three-arm original/DC/DC+safe measured pilot, not an assertion of64%total control speed. Preserve negative parameter copier failure and all previous timings. No numerical clipping, solver loosened tolerance or changed optimization cadence. Treat shared compilation reuse as engineering provided to every strong baseline, not GNN innovation.
''')]:
    p=docs/name;p.write_text(p.read_text(encoding='utf-8')+extra,encoding='utf-8')
p=ARCH/'README.md';p.write_text('# 最新：GRID30安全QP复用资格通过，GRID31完整比较已冻结\n\n[资格报告](protocols/GRID30_qualified/outputs/grid30/GRID30_qualification_report.md) · [完整试点协议](protocols/GRID31_before_results/outputs/grid31/design.json)\n\n'+note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
for p,h in preserved.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
print(json.dumps({'completed_files':finished,'protocol_files':prepared,'historical_preserved':len(preserved),'passed':True}))
