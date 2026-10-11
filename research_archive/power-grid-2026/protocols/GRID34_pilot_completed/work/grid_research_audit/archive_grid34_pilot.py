"""Archive completed12pilot and saved-data gate before remaining outcomes."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
P=json.loads((ROOT/'outputs/grid34/pilot_audit.json').read_text());assert P['all_rankers_qualified']
assert not (ROOT/'outputs/grid34/finished.json').exists()
dest=ARCH/'protocols/GRID34_pilot_completed';assert not dest.exists();dest.mkdir(parents=True);records=[]
files=list((ROOT/'work/grid34').glob('*.py'))+[Path(__file__).resolve()]
files += [ROOT/'outputs/grid34'/n for n in ['design.json','code_freeze.json','manifest.json','pilot_audit.json','phase_complete.json','RUN_STATE.md']]
for folder in (ROOT/'outputs/grid34/runs').iterdir():files += [folder/n for n in ['summary.json','metadata.json','components.json']]
for p in sorted(set(files)):
    rel=p.relative_to(ROOT);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    records.append({'path':rel.as_posix(),'sha256':h,'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':records,'scope':'12Marchlivepilot,all6armscommoncache; cost/survival/querysaved-data checks before36remaining. No new fits/final dates.'},indent=2),encoding='utf-8')
note='GRID34三月12条公平试点质量门槛通过：GNN1/MLP0/MLP1两次均逐步等同FULL；GNN0/PPO共同成本增加0.1361%，全部2017步完成。优化后FULL141–142秒，各排序器125–128秒；架构间仅小差距，MLP强对照保留，不能把总体节省全部归于GNN。'
docs=REPO/'docs';p=docs/'RESEARCH_STATE.md';text=p.read_text(encoding='utf-8')
p.write_text('# Research state — GRID34 pilot passed / remaining qualified\n\n- **最新：** '+note+' 按已冻结方案执行36条其他开发周配对；最终8日期未动。\n\n'+text.split('\n',1)[1],encoding='utf-8')
(docs/'NEXT_STEPS.md').write_text('''# Next steps — GRID34 remaining fair paired runs

1. 三月12条质量试点及审计全部通过，自动执行冻结的36条其他3开发周，沿manifest续跑未完成项；先检查后台worker，不重复正在运行的任务。
2. 各臂共用DC+安全QP复用，保留两GNN/两MLP/PPO及FULL全部成本、存活、查询、逐组成计时和完整重复结果；六月共同失败如实计入。
3. 全48条后做保存向量/反馈/账目独立复核和同状态排序对照；图模型小幅增量与共有工程提速分开，不挑模型种子、混淆百分比或用总时间残差解释物理原因。
4. 后36条外层过程墙钟另计，frozenrunner的finished.wall_s只属于最后调用，不能当作前12+后36的总实验耗时。每步控制时钟口径正确。
5. 没有新训练/预留8日期，LG01暂停；额度暂停继续沿当前任务，不重选题或重问授权。
''',encoding='utf-8')
for name in ['EXPERIMENT_LOG.md','DECISIONS.md']:
    p=docs/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-11 — GRID34 fair cached March pilot gate\n\n'+note+' 剩余36条按既定协议继续；相反6臂运行顺序全部保留，不能从一次较快时间识别GNN贡献。\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text('# 最新：GRID34公平试点通过，其他开发周继续\n\n[三月试点审计](protocols/GRID34_pilot_completed/outputs/grid34/pilot_audit.json)\n\n'+note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'archived_files':len(records),'remaining_qualified':True}))
