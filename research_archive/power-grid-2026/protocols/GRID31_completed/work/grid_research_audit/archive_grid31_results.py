"""Preserve all6actual shared-cache results and next3-week protocol."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
R=json.loads((ROOT/'outputs/grid31/review.json').read_text());assert R['passed']
assert not (ROOT/'outputs/grid32/finished.json').exists()
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
files=list((ROOT/'work/grid31').glob('*.py'))+[Path(__file__).resolve()]
files += [p for p in (ROOT/'outputs/grid31').glob('*') if p.is_file() and p.suffix in ['.json','.md','.csv']]
for folder in (ROOT/'outputs/grid31/runs').iterdir():files += [folder/n for n in ['summary.json','metadata.json','components.json']]
finished=copy(ARCH/'protocols/GRID31_completed',files,'6actual three-arm exact-path pilot; full component/solver facts. Raw physical vectors remain local.')
prepared=copy(ARCH/'protocols/GRID32_before_results',list((ROOT/'work/grid32').glob('*.py'))+
    [ROOT/'outputs/grid32'/n for n in ['design.json','code_freeze.json']],
    '6original/shared-cache compatibility runs on3other exposed dates before outcomes; no new training/final dates.')
whole=next(p for p in R['pooled'] if p['source']=='ORIGINAL' and p['target']=='DC_AND_SAFE')
note=(f"GRID31六条完整三月回放及审计通过：DC+安全QP共享复用相对原完整控制，两组减少{100*whole['pair_reductions'][0]:.2f}%/{100*whole['pair_reductions'][1]:.2f}%，汇总点估计{100*whole['control_reduction']:.2f}%；12102步动作/观测/成本/67950次预测相同，所有6项配对solver状态/迭代比较无差别。局部数值资格与完整时间分开，原10.93%/0.39%及6.45%/6.51%全部保留。仅三月2重复，不是GNN独有或封存泛化证据。")
docs=REPO/'docs';p=docs/'RESEARCH_STATE.md';text=p.read_text(encoding='utf-8')
p.write_text('# Research state — GRID31 complete / GRID32 frozen\n\n- **最新完整控制器结果：** '+note+' GRID32在其余3个已暴露周冻结6条兼容性比较，暂未结果。\n\n'+text.split('\n',1)[1],encoding='utf-8')
(docs/'NEXT_STEPS.md').write_text('''# Next steps — GRID32 exposed-week compatibility

1. 执行6条其余3周的原/共享双复用配对，逐步质量必须等同相应FULL；六月共同1901步失败保留，不宣称全完成。任何数值/路径差异保留后先诊断。
2. 全部日期结果和组件/solver/cache计量同时审计；单日期只有一次配对，不能保证跨环境/操作系统通用速度。三月22.42%仅2重复点估计，不能作GNN贡献。
3. 整套工程优化确定后，所有强基线共用它，单独验证GNN辅助拓扑搜索的增量，包括原PPO和现成MLP。不追加训练或更改停止常数寻找更大提速。
4. 原模型/失败/全部快慢时间保留；最终8个日期家族未动，LG01跟进仍暂停。因额度暂停后的继续自动沿当前任务。
''',encoding='utf-8')
for name,extra in [('EXPERIMENT_LOG.md','\n## 2026-10-11 — GRID31 actual three-arm cache result\n\n'+note+' Saved-data check only, no independent new solve/refit. GRID32frozen before3exposed-week compatibility outcomes.\n'),
    ('DECISIONS.md','''\n## 2026-10-11 — Retain measured22% common acceleration without graph attribution

Full delivered-arm improvement is observed with exact path/cost/native solver behavior. This supports practical shared engineering, not graph-specific novelty. Before fair learned search comparison, check numerical/path compatibility on other exposed dates, including the known June failure. Do not choose only the10%or23%repeat, add component percentages, train immediately or open final date families.
''')]:
    p=docs/name;p.write_text(p.read_text(encoding='utf-8')+extra,encoding='utf-8')
p=ARCH/'README.md';p.write_text('# 最新：GRID31完整控制器共享提速通过，GRID32已冻结\n\n[完整报告](protocols/GRID31_completed/outputs/grid31/GRID31_report.md) · [跨开发周协议](protocols/GRID32_before_results/outputs/grid32/design.json)\n\n'+note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
for p,h in preserved.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
print(json.dumps({'completed_files':finished,'protocol_files':prepared,'historical_preserved':len(preserved),'passed':True}))
