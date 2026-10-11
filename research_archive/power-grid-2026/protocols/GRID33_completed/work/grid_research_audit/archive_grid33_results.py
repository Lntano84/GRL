"""Preserve every existing-ranker outcome and common-engineering live protocol."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
R=json.loads((ROOT/'outputs/grid33/review.json').read_text());assert json.loads((ROOT/'outputs/grid33/audit.json').read_text())['passed']
assert not (ROOT/'outputs/grid34/manifest.json').exists()
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
finished=copy(ARCH/'protocols/GRID33_completed',list((ROOT/'work/grid33').glob('*.py'))+[Path(__file__).resolve()]+
    [p for p in (ROOT/'outputs/grid33').glob('*') if p.is_file() and p.suffix in ['.json','.md','.csv']],
    'All existing4GNN/MLPcheckpoints and PPO,130matched exposed-state comparisons; saved-order independent stop replay,not livequalityproof.')
prepared=copy(ARCH/'protocols/GRID34_before_results',list((ROOT/'work/grid34').glob('*.py'))+[ROOT/'outputs/grid34'/n for n in ['design.json','code_freeze.json']],
    '12Marchpilotthen36otherweekruns;6arms withcommonDC+safeQPprogramreuse,all existingGNN/MLP/PPO; frozen before actual outcome.')
note='GRID33五个现成排序器130项同状态诊断及独立停止复算通过；查询7358/7530(GNN0/1)、7549/7826(MLP0/1)、7823(PPO)。全部找回1733关键动作；差距仅开发线索，不证明长期质量或图结构必要性。'
docs=REPO/'docs';p=docs/'RESEARCH_STATE.md';text=p.read_text(encoding='utf-8')
p.write_text('# Research state — GRID33 complete / GRID34 frozen\n\n- **最新公平对照：** '+note+' GRID34冻结12条三月试点，全部臂共有缓存；质量通过后36条其他开发周，同一冻结权重/停止规则。\n\n以下保留历史更新，旧的“尚未/暂未”只表示当时状态。\n\n'+text.split('\n',1)[1],encoding='utf-8')
(docs/'NEXT_STEPS.md').write_text('''# Next steps — GRID34 actual fair control pilot

1. 完成12条三月平衡闭环：共有缓存FULL、原PPO、GNN0/1、MLP0/1；逐臂保留所有成本/存活/计时/查询。未知质量不以离线rho近似代替。
2. 保存数据质量门槛通过后自动执行同协议36条其他3开发周；失败则保留记录、先诊断，不删模型或改截止寻找赢家。六月共同失败和同周统计边界保留。
3. 共有程序复用的20%–70%测量不能归于GNN；图模型增量只对相同缓存/停止/输入的PPO及MLP单独比较，报告每个日期/模型种子/重复，不选最好种子。
4. 没有新训练/预留8日期，LG01仍暂停。额度暂停后沿正在执行的阶段恢复，无需重复授权；先检查是否仍有后台worker，再续跑manifest未完成部分。
''',encoding='utf-8')
for name,extra in [('EXPERIMENT_LOG.md','\n## 2026-10-11 — GRID33 existing-ranker matched screen\n\n'+note+' No new fit/simulation/final date. GRID34common-engineering livecomparisonfrozen.\n'),
 ('DECISIONS.md','\n## 2026-10-11 — Keep both MLP seeds and both GNN seeds in fair live comparison\n\nAll existing rankers pass the known critical-action screen; do not omit MLP or select a best checkpoint. Give exhaustiveFULL and everyranker common caches. Reversed6arm paired order per exposeddate, shared quality gate and all components. Pilot cost/survival must be measured before widening. Offline matched-query counts are not full-controller acceleration or an architecture-proof.\n')]:
    p=docs/name;p.write_text(p.read_text(encoding='utf-8')+extra,encoding='utf-8')
p=ARCH/'README.md';p.write_text('# 最新：GRID33现成排序对照通过，GRID34公平闭环已冻结\n\n[同状态结果](protocols/GRID33_completed/outputs/grid33/GRID33_report.md) · [公平闭环协议](protocols/GRID34_before_results/outputs/grid34/design.json)\n\n'+note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
for p,h in preserved.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
print(json.dumps({'completed_files':finished,'protocol_files':prepared,'historical_preserved':len(preserved),'passed':True}))
