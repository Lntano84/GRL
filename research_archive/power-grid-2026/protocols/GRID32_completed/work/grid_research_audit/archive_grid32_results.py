"""Immutable cross-week cache results plus next matched existing-ranker protocol."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
R=json.loads((ROOT/'outputs/grid32/review.json').read_text());assert R['passed']
assert (ROOT/'outputs/grid33/design.json').exists() and not (ROOT/'outputs/grid33/review.json').exists()
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
files=list((ROOT/'work/grid32').glob('*.py'))+[Path(__file__).resolve()]
files += [p for p in (ROOT/'outputs/grid32').glob('*') if p.is_file() and p.suffix in ['.json','.md','.csv']]
for folder in (ROOT/'outputs/grid32/runs').iterdir():files += [folder/n for n in ['summary.json','metadata.json','components.json']]
finished=copy(ARCH/'protocols/GRID32_completed',files,'6fresh original/shared-cache exposed-week runs,exact path including June failure. Saved-data audit only. Raw vectors remain local.')
prepared=copy(ARCH/'protocols/GRID33_before_results',list((ROOT/'work/grid33').glob('*.py'))+[ROOT/'outputs/grid33'/n for n in ['design.json','code_freeze.json']],
    'Existing4supervised checkpoints and PPO matched26state diagnostic frozen before inference; no training/new forecast/final date.')
parts=[f"{p['week']}减少{100*p['control_reduction']:.2f}%" for p in R['pairs']]
note='GRID32三组其他已暴露周配对通过：'+ '；'.join(parts)+'。11870步动作/观测/成本逐步相同，六月共同1901步失败保留，未新增/提前失败；每周仅一次配对，不作为独立封存泛化或GNN贡献。'
docs=REPO/'docs';p=docs/'RESEARCH_STATE.md';text=p.read_text(encoding='utf-8')
p.write_text('# Research state — GRID32 complete / GRID33 frozen\n\n- **最新跨开发周兼容性：** '+note+' GRID33仅补现成MLP/PPO/GNN的同状态排序诊断，全部共享工程优化原则，不启动新训练。\n\n'+text.split('\n',1)[1],encoding='utf-8')
(docs/'NEXT_STEPS.md').write_text('''# Next steps — shared engineering versus learned ranking

1. GRID33对26个已暴露保存状态重新推理全部现成GNN0/1、MLP0/1和原PPO。共享候选、输入、守卫、停止/变电站补全规则，不用隐藏结果排序。已知1733关键动作遗漏者不直接进入这套停止闭环。
2. 数值资格后冻结公平的完整闭环比较；每个强对照获得相同DC+安全QP复用。分别报告控制时间、昂贵查询、质量和成功步数，不把共有缓存归于GNN。
3. GRID29旧10.93%/0.39%、GRID29C6.45%/6.51%、GRID31三月22.42%及GRID32所有单周幅度全部保留。旧重复无法恢复缺失的逐solver时间，不能追认热状态等原因。
4. 不追加训练/停止常数/最终8日期；原模型与失败保留，LG01暂停。额度中断后继续当前实验，不重选题或重问授权。
''',encoding='utf-8')
for name,extra in [('EXPERIMENT_LOG.md','\n## 2026-10-11 — GRID32 other exposed-week compatibility\n\n'+note+' No new fits/final dates. GRID33existing-ranker protocol frozen before output.\n'),
 ('DECISIONS.md','\n## 2026-10-11 — Give shared solver reuse to all graph controls\n\nCross-week exact paths support common engineering, with strong magnitude heterogeneity and a preserved June failure. Next test existing MLPs rather than train or conflate common reuse with GNN. Known critical-action offline miss blocks unchanged semantic-stop deployment; it does not reject all possible MLP policies.\n')]:
    p=docs/name;p.write_text(p.read_text(encoding='utf-8')+extra,encoding='utf-8')
p=ARCH/'README.md';p.write_text('# 最新：GRID32其他开发周通过，GRID33现成排序对照已冻结\n\n[结果](protocols/GRID32_completed/outputs/grid32/GRID32_report.md) · [下一步](protocols/GRID33_before_results/outputs/grid33/design.json)\n\n'+note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
for p,h in preserved.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
print(json.dumps({'completed_files':finished,'protocol_files':prepared,'historical_preserved':len(preserved),'passed':True}))
