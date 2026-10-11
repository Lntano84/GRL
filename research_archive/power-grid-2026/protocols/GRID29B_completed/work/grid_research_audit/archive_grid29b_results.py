"""Immutable live DC diagnosis plus next full-arm component protocol."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
R=json.loads((ROOT/'outputs/grid29b/review.json').read_text());assert R['audit']['passed']
preserved={}
for p in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for row in json.loads(p.read_text(encoding='utf-8')).get('files',[]):
        target=p.parent/row['path']
        if not target.exists():target=ARCH/row['path']
        assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'],target
        preserved[str(target)]=row['sha256']


def copy_stage(dest,paths,scope):
    assert not dest.exists();dest.mkdir(parents=True);records=[]
    for p in sorted(set(paths)):
        if not p.is_file():continue
        rel=p.relative_to(ROOT);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
        records.append({'path':rel.as_posix(),'sha256':h,'bytes':p.stat().st_size})
    (dest/'manifest.json').write_text(json.dumps({'files':records,'scope':scope,'historical_preserved':len(preserved)},indent=2),encoding='utf-8')
    return len(records)


paths=list((ROOT/'work/grid29b').glob('*.py'))
paths+=[p for p in (ROOT/'outputs/grid29b').glob('*') if p.is_file() and p.suffix in ['.json','.md','.csv','.log']]
paths += [ROOT/'outputs/grid29b/runs/r0__2035-03-05_0__LIVE_DC_PROFILE'/n for n in ['summary.json','metadata.json']]
paths.append(Path(__file__).resolve())
done=copy_stage(ARCH/'protocols/GRID29B_completed',paths,'Actual-memory live DC qualification and old-step decomposition; physical/DC vectors remain local. Different prefix preserves original V0 files.')
prepared=copy_stage(ARCH/'protocols/GRID29C_before_results',list((ROOT/'work/grid29c').glob('*.py'))+
    [ROOT/'outputs/grid29c'/n for n in ['design.json','code_freeze.json']],
    'Four balanced full-arm component profiles before first outcomes; shared DC optimization for every comparator.')
note=(f"GRID29B真实控制记忆下153组DC配对：原{R['live_dc_shadow_s']['ORIGINAL']:.3f}s、缓存{R['live_dc_shadow_s']['CACHED']:.3f}s，减少{100*R['live_dc_reduction']:.2f}%；154组保存三向量最大流量差0、2017步路径完全相同。实际原DC调用编译29.17s/后端0.070s，确认重复编译是这一DC组成的主要开销，不等于完整控制器根因。旧10.93%组恢复交付步骤省25.42s、其余多1.75s；旧0.39%组对应省10.32s、其余多9.62s。旧两组没有逐solver计时，不能事后识别各自DC收益。")
docs=REPO/'docs';p=docs/'RESEARCH_STATE.md';text=p.read_text(encoding='utf-8')
p.write_text('# Research state — 2026-10-11 / GRID29B audited\n\n- **最新诊断：** '+note+' GRID29C四条完整两臂组件计时已冻结，暂未结果；无新训练/封存日期。\n\n'+text.split('\n',1)[1],encoding='utf-8')
(docs/'NEXT_STEPS.md').write_text('''# Next steps — GRID29C frozen

1. 执行4条平衡顺序完整原/缓存两臂，记录每次DC和连续QP的编译/后端/迭代及完整控制时间。重点解释10.93%成功组和0.39%组差异；不得从局部探针机械扣减整体时间。
2. 编译复用是所有基线共享工程优化；不变目标/solver/tolerance/warm_start/cadence。每步动作/观测/成本/查询必须仍相同。
3. GRID28查询和搜索增量继续保留；整套控制8.83%含未解释的非搜索变化，同状态GNN/PPO查询优势5.98%不构成架构贡献。
4. 不加训练/RL/最终8日期家族评价；原失败、权重、两组快慢结果均保留。用户因额度暂停后说继续，应直接延续当前任务。
''',encoding='utf-8')
for name,addition in [('EXPERIMENT_LOG.md','\n## 2026-10-11 — GRID29B real-memory DC audit and GRID29C freeze\n\n'+note+' Saved-data audit154DCtriples/2017physicalrows; no independent solve/new fit/reserved evaluation. GRID29C4balanced delivered-arm component traces frozen before results.\n'),
    ('DECISIONS.md','''\n## 2026-10-11 — Explain the10% pair with measured components

Live-memory isolated DC matches actual flows exactly and identifies compilation savings, but old paired trajectories lack per-solver timing. Do not attribute old total variation to a particular optimizer from residual arithmetic. Freeze full-arm component measurement with identical instrumentation/paths and retained two balanced orders. No model or numeric control change. Quota interruptions are not user cancellation; resume current work without repeated confirmation.
''')]:
    p=docs/name;p.write_text(p.read_text(encoding='utf-8')+addition,encoding='utf-8')
p=ARCH/'README.md';p.write_text('# 最新：GRID29B真实记忆诊断通过，GRID29C待运行\n\n[诊断报告](protocols/GRID29B_completed/outputs/grid29b/GRID29B_report.md) · [下一组件协议](protocols/GRID29C_before_results/outputs/grid29c/design.json)\n\n'+note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
for p,h in preserved.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
print(json.dumps({'archived_completed':done,'archived_protocol':prepared,'preserved':len(preserved),'passed':True}))
