"""Archive saved-only magnitude explanation without changing old protocols or timings."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
R=json.loads((ROOT/'outputs/grid_speed_review/facts.json').read_text(encoding='utf-8'))
for p,h in R['input_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
dest=ARCH/'protocols/GRID_common_speed_explanation';assert not dest.exists();dest.mkdir(parents=True);items=[]
for p in [Path(__file__).resolve(),ROOT/'work/grid_research_audit/control_speed_explanation.py',ROOT/'outputs/grid_speed_review/facts.json',ROOT/'outputs/grid_speed_review/WHY_10_PERCENT.md']:
    rel=p.relative_to(ROOT);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    items.append({'path':rel.as_posix(),'sha256':h,'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':items,'scope':'Saved-only historical10%versus0.39%limits and shared-cache cross-week component explanation, no additional solve.'},indent=2),encoding='utf-8')
note='已完成用户要求的幅度诊断：三月可复用DC+安全QP占原控制39.7%、安全QP缓存命中77.8%、未复用unsafe占38.6%；其他三周可复用占73.4%–88.1%、命中90.0%–93.8%、unsafe占3.3%–8.2%，支持22%vs62%–74%的组件解释。旧10.93%/0.39%缺逐solver计时，精确波动原因仍不可恢复。GRID32继承的repeats=2/旧四条文案为元数据残留，实际冻结6条execution_order及审计一贯是3单配对；原JSON不改。'
docs=REPO/'docs';p=docs/'RESEARCH_STATE.md';text=p.read_text(encoding='utf-8');header,body=text.split('\n',1)
p.write_text(header+'\n\n- **幅度机制复核：** '+note+'\n\n'+body,encoding='utf-8')
for name,title in [('EXPERIMENT_LOG.md','Saved-only component heterogeneity review'),('DECISIONS.md','Distinguish measured cache dose from unrecoverable historical noise')]:
    p=docs/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-11 — '+title+'\n\n'+note+' 不把共享工程归于GNN，不推测温度/后台负载；GRID34所有强对照共用相同优化，正在公平闭环试点。\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text('# 控制提速差异的可核查解释\n\n[为什么10%与0.39%差很多](protocols/GRID_common_speed_explanation/outputs/grid_speed_review/WHY_10_PERCENT.md)\n\n'+note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'archived':len(items),'new_solves':0,'original_sources_unchanged':True}))
