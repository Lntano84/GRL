"""Preserve whole-arm component explanation and safe-QP qualification plan."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
R=json.loads((ROOT/'outputs/grid29c/review.json').read_text());assert R['passed']
assert not (ROOT/'outputs/grid30/qualification.json').exists()
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
files=list((ROOT/'work/grid29c').glob('*.py'))+[Path(__file__).resolve()]
files += [p for p in (ROOT/'outputs/grid29c').glob('*') if p.is_file() and p.suffix in ['.json','.md','.csv']]
for folder in (ROOT/'outputs/grid29c/runs').iterdir():files += [folder/n for n in ['summary.json','metadata.json','components.json']]
finished=copy(ARCH/'protocols/GRID29C_completed',files,'4balanced full-arm profiles; per-component/solver facts, raw physical vectors kept local.')
prepared=copy(ARCH/'protocols/GRID30_before_results',list((ROOT/'work/grid30').glob('*.py'))+
    [ROOT/'outputs/grid30'/n for n in ['design.json','code_freeze.json']],
    'Manual then isolated safe-QP live qualification before new outcome; shared engineering, unchanged controls.')
note=(f"GRID29C完整组件配对4/4审计通过：两组完整控制提速{100*R['pairs'][0]['controller_reduction']:.2f}%/{100*R['pairs'][1]['controller_reduction']:.2f}%；DC省17.47/18.14s（9.61/9.94完整控制百分点），非DC部分分别多耗5.74/6.27s。682个配对solver调用的状态/迭代零改变，8068步动作/观测/成本/查询相同。重复编译节省机制成立，但旧10.93%/0.39%仅有逐步计时，具体历史波动成因无法完整追溯。")
docs=REPO/'docs';p=docs/'RESEARCH_STATE.md';text=p.read_text(encoding='utf-8')
p.write_text('# Research state — GRID29C audited / GRID30 qualification frozen\n\n- **最新整体控制线：** '+note+' 安全恢复QP另有约34–36s编译开销，GRID30只做同数学问题参数复用的数值资格，暂无部署/结果。\n\n'+text.split('\n',1)[1],encoding='utf-8')
(docs/'NEXT_STEPS.md').write_text('''# Next steps — GRID30 qualification

1. 先做固定手工参数化安全QP检查，再在原控制的实际参数/记忆下进行153次隔离配对。原控制输出继续部署，影子结果不得派发；超过1e-6MW或Grid2Op动作改变则保存失败、不部署。
2. 数值资格通过并经保存数据核验后，才冻结完整原/DC缓存/DC+安全QP缓存比较。每个强基线获得同样工程优化，不计为GNN创新。
3. 旧10.93%/0.39%与新6.45%/6.51%全保留；DC组件9.61/9.94百分点并非最终整体收益，不能相加不同阶段百分比。
4. 原模型/失败记录保留；不追加训练/RL、不开放最终8日期家族。用户额度暂停后继续当前工作，无需再次确认。
''',encoding='utf-8')
for name,extra in [('EXPERIMENT_LOG.md','\n## 2026-10-11 — GRID29C delivered-arm component results\n\n'+note+' Sum-disjoint component savings reconstruct total; null stats retained. Unsafe author path still has13user_limit/3optimal_inaccurate/18optimal, restoration153optimal, both unchanged. GRID30 fixed qualification plan before first outcome.\n'),
    ('DECISIONS.md','''\n## 2026-10-11 — Extend proven compilation reuse only after numerical qualification

Whole-arm DC gains are real and match unchanged output/numeric iteration paths, but total saving is partly offset elsewhere. Historical10% is plausible, not a chosen robust guarantee. Safe restoration shows analogous repeated compilation overhead; examine bounded topology/coefficient keyed exact parameterization, with same solver/status/warm_start/cadence. Manual tests plus actual-memory isolated action comparison precede deployment. Generic optimization is available to all learned/nonlearned comparators.
''')]:
    p=docs/name;p.write_text(p.read_text(encoding='utf-8')+extra,encoding='utf-8')
p=ARCH/'README.md';p.write_text('# 最新：GRID29C组件解释通过，GRID30数值资格已冻结\n\n[报告](protocols/GRID29C_completed/outputs/grid29c/GRID29C_report.md) · [下一资格协议](protocols/GRID30_before_results/outputs/grid30/design.json)\n\n'+note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
for p,h in preserved.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
print(json.dumps({'completed_files':finished,'protocol_files':prepared,'preserved':len(preserved),'passed':True}))
