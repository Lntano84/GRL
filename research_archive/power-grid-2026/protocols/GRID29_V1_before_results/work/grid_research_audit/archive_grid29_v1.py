"""Retain non-DPP failure and publish revised qualification before re-running."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration'
ARCH=REPO/'research_archive/power-grid-2026';DEST=ARCH/'protocols/GRID29_V1_before_results'
assert not DEST.exists() and not (ROOT/'outputs/grid29/shadow.json').exists()
files=sorted((ROOT/'work/grid29').glob('*.py'))+[ROOT/'outputs/grid29'/n for n in ['design.json','code_freeze.json','revision_v1_freeze.json']]
files+=sorted((ROOT/'outputs/grid29/revisions/v0_non_dpp').glob('*'))+[Path(__file__).resolve()]
records=[]
for p in files:
    rel=p.relative_to(ROOT);target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    records.append({'path':rel.as_posix(),'sha256':h,'bytes':p.stat().st_size})
(DEST/'manifest.json').write_text(json.dumps({'files':records,'scope':'V0 rejected before cache comparison, V1 numeric reciprocal and real Parameter/triangle tests frozen before rerun.'},indent=2),encoding='utf-8')
(DEST/'README.md').write_text('''# GRID29 V0失败与V1资格修订

V0 ndarray玩具用例通过，但真实LJN的线路电抗是CVXPY Parameter，倒数表达式不满足DPP；首次真实缓存模型构造被断言拒绝，未产生有效缓存比较或闭环。缓存键也错误地将Parameter对象当数组，不能按真实数值失效。旧代码、设计、冻结和失败日志保留。

V1在DSL外计算数值倒数，并按实际电抗数值更新键。每项数学约束/目标及原求解器不变，warm_start=False不变；新增真实Parameter原地变化和三角电网手算，用实际流量变化验证键失效。V1尚未运行；失败不是GNN价值判断，也不是闭环质量损失。

CVXPY官方说明：https://www.cvxpy.org/tutorial/dpp/index.html 。
''',encoding='utf-8')
docs=REPO/'docs'
p=docs/'RESEARCH_STATE.md';b=p.read_text(encoding='utf-8')
b=b.replace('尚未运行资格计算','V0首次真实模型构造未通过DPP，V1数值倒数/真实值键及追加测试已在重跑前冻结；尚无有效资格结果')
p.write_text(b,encoding='utf-8')
for filename,note in [('DECISIONS.md','''\n## 2026-10-11 — Reject the non-DPP cache before rollout

V0 was blocked by actual Parameter division, unlike its ndarray toy; its object-valued reactance cache key was also wrong. Retain V0 and use value-keyed numeric reciprocal following CVXPY DPP documentation. Added real Parameter mutation and triangle-flow tests. This changes symbolic representation only, not mathematical constraints or solver settings. No valid cache shadow result/live trajectory existed at revision time.
'''),('EXPERIMENT_LOG.md','''\n## 2026-10-11 — GRID29 V0 qualification blocked, V1 frozen

First real cached model rejected by is_dcp(dpp=True); no cached comparison/closed trajectory. Actual reactance is a Parameter. Preserve failed code/log and initial protocol; freeze V1 numeric reciprocal and numeric reactance-key invalidation plus real Parameter/triangle checks before rerun. [Revision protocol](../research_archive/power-grid-2026/protocols/GRID29_V1_before_results/README.md).\n''')]:
    p=docs/filename;p.write_text(p.read_text(encoding='utf-8')+note,encoding='utf-8')
print(json.dumps({'archived':len(records),'revision':'V1_before_results'}))
