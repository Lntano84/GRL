"""Preserve cheap-order diagnostics and their frozen actual-quality pilot."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
assert json.loads((ROOT/'outputs/grid35/audit.json').read_text())['passed']
assert json.loads((ROOT/'outputs/grid35b/audit.json').read_text())['passed']
assert not (ROOT/'outputs/grid36/manifest.json').exists()
old={}
for p in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for row in json.loads(p.read_text(encoding='utf-8')).get('files',[]):
        q=p.parent/row['path']
        if not q.exists():q=ARCH/row['path']
        assert hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256'],q
        old[str(q)]=row['sha256']
counts={}
for stage,kind in [('grid35','completed'),('grid35b','completed'),('grid36','before_results')]:
    dest=ARCH/'protocols'/f'{stage.upper()}_{kind}';assert not dest.exists();dest.mkdir(parents=True)
    files=list((ROOT/'work'/stage).glob('*.py'))
    files += [p for p in (ROOT/'outputs'/stage).iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv']]
    if stage=='grid36':files.append(Path(__file__).resolve())
    records=[]
    for p in sorted(set(files)):
        rel=p.relative_to(ROOT);q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        h=hashlib.sha256(q.read_bytes()).hexdigest();assert h==hashlib.sha256(p.read_bytes()).hexdigest()
        records.append({'path':rel.as_posix(),'sha256':h,'bytes':q.stat().st_size})
    (dest/'manifest.json').write_text(json.dumps({'files':records,'historical_preserved':len(old),'scope':stage+' '+kind+'; no newfits/finaldateevaluation.'},indent=2),encoding='utf-8');counts[stage]=len(records)
note='GRID35/35B廉价顺序独立诊断通过：STATIC_ID/LOCAL_STRESS/原既有PHYSICS_TAIL查询7879/6965/6748，同FULL选择21/17/17（共26状态）；均找回1733已知关键动作，但最大rho差更大。便宜规则省查询不等于长期质量，GRID36三条三月闭环质量试点已冻结。'
docs=REPO/'docs';p=docs/'RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text('# Research state — common speed established, cheap order quality pending\n\n- **最新：** '+note+'\n\n'+t.split('\n',1)[1],encoding='utf-8')
(docs/'NEXT_STEPS.md').write_text('''# Next steps — GRID36 cheap-order quality pilot

1. 沿GRID36 manifest执行3条已冻结三月规则试点（STATIC_ID、既有PHYSICS_TAIL、LOCAL_STRESS），共有优化及停止规则不变，先检查后台worker以免重复。
2. 每条保存完整向量/反馈/成本、独立核验；对已有FULL仅作确定性成本/轨迹资格检查，本批单次跨批时间不能认证速度。所有规则保留，逐规则判定0.5%质量筛查，失败配置不原样扩展。
3. GRID34全部48条已完成审计：两GNN相对共有优化FULL少约11.3%–11.8%总时间，但相对PPO/MLP仅0.1%–1.9%。六月共同失败、MLP0九月成本+0.60247%等结果均保留；不得把共享编译或停止规则收益全部归于GNN。
4. 原10.93%vs0.39%逐组件差异已解释至现有证据范围，历史缺逐solver数据不能恢复精确因果；最终8日期仍封存、没有新训练/RL、LG01暂停。
5. 额度暂停不是任务取消，下次继续上一条任务，不重选题、重问授权或重复已完成运行。
''',encoding='utf-8')
for name in ['EXPERIMENT_LOG.md','DECISIONS.md']:
    p=docs/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-11 — Public nonlearning order screens and quality pilot\n\n'+note+' 不从26个暴露状态认定可部署或图结构必需；静态原规则部件也给完整公平机会。\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text('# 最新：共有速度与GNN增量已分开，廉价顺序闭环资格待查\n\n[GRID35结果](protocols/GRID35_completed/outputs/grid35/GRID35_report.md) · [原非学习部分](protocols/GRID35B_completed/outputs/grid35b/GRID35B_report.md) · [实际资格试点](protocols/GRID36_before_results/outputs/grid36/design.json)\n\n'+note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
for p,h in old.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
print(json.dumps({'files':counts,'historical_preserved':len(old),'passed':True}))
