"""Preserve the generic-controller cache design before its numerical outcomes."""
import hashlib
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration'
ARCH=REPO/'research_archive/power-grid-2026';DEST=ARCH/'protocols/GRID29_before_results'
assert not DEST.exists()
assert not (ROOT/'outputs/grid29/shadow.json').exists()
preserved={}
for path in sorted(ARCH.rglob('*manifest.json')):
    # Only archive inventories; local deliverable manifests intentionally also
    # list logs/data which were never copied into the Git archive.
    if not (path.parent==ARCH or (path.name=='manifest.json' and path.parent.parent==ARCH/'protocols')):
        continue
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or 'files' not in data:continue
    parent=path.parent
    for row in data['files']:
        if 'path' not in row or 'sha256' not in row:continue
        file=parent/row['path']
        if not file.exists():file=ARCH/row['path']
        assert file.exists(),(path,row['path'])
        assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256'],file
        preserved[str(file)]=row['sha256']
files=sorted((ROOT/'work/grid29').glob('*.py'))+[ROOT/'outputs/grid29/design.json',ROOT/'outputs/grid29/code_freeze.json',Path(__file__).resolve()]
records=[]
for p in files:
    relative=p.relative_to(ROOT);dest=DEST/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(dest.read_bytes()).hexdigest()==h
    records.append({'path':relative.as_posix(),'sha256':h,'bytes':p.stat().st_size})
(DEST/'manifest.json').write_text(json.dumps({'files':records,'historical_preserved_checks':len(preserved),
    'scope':'DC compiled-program reuse qualification before numeric or live outcomes. GRID28 closed trajectories remain running; no new result copied here.'},indent=2),encoding='utf-8')
(DEST/'README.md').write_text('''# GRID29 整体控制器工程资格设计

在相同拓扑下复用原DC潮流CVXPY问题，不改变约束、目标、OSQP、迭代上限或warm_start=False。拓扑端点、负荷/机组/储能归属、角度零值掩码和线路电抗均进入实例私有LRU缓存键，最多32项。每次都刷新当前注入；不缓存预测结果。

先对四个已暴露日期的保存物理状态做每64步数值核对，声明影子样本统一零储能功率累积量，不代表所有实时记忆状态。只有数值合格才启动受保护GNN0三月完整闭环，原实现/缓存实现两次平衡次序串行对照，要求与历史动作/观测/成本路径逐步完全相同。原有模型、作者源码不改；这是给所有基线共有的常规工程优化，不作为GNN或论文新意。

本快照在数值与闭环结果前保存。没有新拟合或8个封存日期的评价，不增加训练，不改solver参数以获得速度。CVXPY的DPP参数复用已有官方文档：https://www.cvxpy.org/tutorial/dpp/index.html 。
''',encoding='utf-8')
docs=REPO/'docs'
p=docs/'RESEARCH_STATE.md';body=p.read_text(encoding='utf-8')
body=body.replace('现正跑冻结三月试点及同规则PPO强对照','冻结三月试点及同规则PPO强对照已通过0.5%质量投资筛查（成本均+0.136%，不是零差）；其余3个日期按原规则串行运行')
body=body.replace('尚无GRID28闭环裁决','尚无GRID28全矩阵裁决')
body=body.replace('\n\n- **GRID24','\n\n- **整体控制器线：GRID29已冻结**。只复用原DC参数化问题，先数值资格再完整闭环。尚未运行资格计算，不改变模型、求解器或控制规则；常规工程优化同样提供给强基线，不作为GNN贡献。\n\n- **GRID24',1)
p.write_text(body,encoding='utf-8')
p=docs/'NEXT_STEPS.md';p.write_text('''# Next steps — 2026-10-11 / GRID28 remaining and GRID29 prepared

1. 完成GRID28剩余三日期及保存数据审计。冻结分批/变电站补全规则、同规则PPO及受保护GNN0，不根据结果改常数。共同失败周不拿累计成本下降作收益。
2. 三月试点成本+0.136%，单次搜索少约46%、控制少约5%；均为开发点估计，不称零质量差或稳定端到端速度。
3. 串行结束后执行GRID29的原DC问题复用数值资格；合格才做三月原实现/缓存实现两次平衡闭环。全部强对照同样享有该工程优化，不能归因于GNN。
4. 保留旧模型/守卫/失败；新拟合/RL/8个封存日期评价0。待质量与公平重复时间清楚后再决定下一步，不自动扩模型。LG01自动跟进保持暂停。
''',encoding='utf-8')
for filename,note in [('DECISIONS.md','''
## 2026-10-11 — Qualify shared DC compilation reuse independently

Use per-optimizer topology/reactance-keyed bounded CVXPY/DC-program reuse, retaining the exact mathematical problem and original solve wrapper/warm_start=False. Frozen stored-state qualification precedes a balanced protected-March pilot. No blind direct-solver substitution or changed tolerances. Apply any engineering improvement to strong baselines; do not count it as GNN novelty. GRID28 remaining runs still serial, no reserved-date opening.
'''),('EXPERIMENT_LOG.md','''
## 2026-10-11 — GRID28 March gate and GRID29 frozen qualification

GRID28 March3 runs and saved-data audit pass: all complete2017steps, localGNN0/localPPO cost+0.136116% versus protected, query saving54.15%/53.03%, single-repeat source saving46.22%/45.36%, controller saving4.87%/5.06%. No graph-specific speed attribution; remaining3 dates running. GRID29 generic DC-program cache design/code frozen before shadow/live outcomes, using only exposed states and no new weights. [GRID29 protocol](../research_archive/power-grid-2026/protocols/GRID29_before_results/README.md).
''')]:
    p=docs/filename;p.write_text(p.read_text(encoding='utf-8')+note,encoding='utf-8')
for file,h in preserved.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==h
print(json.dumps({'passed':True,'archived':len(records),'preserved':len(preserved)}))
