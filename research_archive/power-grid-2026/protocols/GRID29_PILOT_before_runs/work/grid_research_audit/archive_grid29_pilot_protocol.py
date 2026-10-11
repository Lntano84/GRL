"""Publish qualified shadow evidence and frozen live comparison before runs."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration'
ARCH=REPO/'research_archive/power-grid-2026';DEST=ARCH/'protocols/GRID29_PILOT_before_runs'
assert not DEST.exists() and not (ROOT/'outputs/grid29/pilot/manifest.json').exists()
S=json.loads((ROOT/'outputs/grid29/shadow.json').read_text());assert S['passed']
files=sorted((ROOT/'work/grid29').glob('*.py'))+[ROOT/'outputs/grid29'/n for n in ['design.json','code_freeze.json','revision_v1_freeze.json','shadow.json']]
files+=[ROOT/'outputs/grid29/pilot'/n for n in ['runtime_design.json','code_freeze.json']]+[Path(__file__).resolve()]
records=[]
for p in files:
    rel=p.relative_to(ROOT);target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    records.append({'path':rel.as_posix(),'sha256':h,'bytes':p.stat().st_size})
(DEST/'manifest.json').write_text(json.dumps({'files':records,'scope':'Completed same-state shadow and four-run pilot before any live cache result.'},indent=2),encoding='utf-8')
(DEST/'README.md').write_text(f'''# GRID29 数值资格与闭环冻结

V1的6项针对性检查及{S['states']}个保存状态均通过，最大流量差{S['max_flow_error_MW']}MW。两路统一零储能功率累积量，不能涵盖所有实时记忆状态。局部计时减少{100*S['time_reduction']:.2f}%，不是整体控制提速。

冻结4条三月受保护GNN0完整周回放：原实现/共用DC缓存，两次平衡次序（原→缓存；缓存→原）。要求逐步完全重现历史动作、观测、成本；任何差异先停扩展诊断。setup、物理步、记录开销单列；完整控制内包括图构造/排序/优化/恢复/公开预测。缓存求解仍warm_start=False，原solver参数、控制规则和模型全部不变。

保留V0失败，不进行新拟合/RL或封存日期评价。给强基线共有的工程优化不能算成GNN新意。本快照无新闭环成绩。
''',encoding='utf-8')
docs=REPO/'docs';p=docs/'RESEARCH_STATE.md';b=p.read_text(encoding='utf-8')
b=b.replace('V0首次真实模型构造未通过DPP，V1数值倒数/真实值键及追加测试已在重跑前冻结；尚无有效资格结果',f"V0未通过DPP已保留；V1的6项检查与{S['states']}个保存状态数值资格通过，流量差0，局部时间减少{100*S['time_reduction']:.2f}%；现进入4条平衡闭环，还不能宣称完整控制提速")
p.write_text(b,encoding='utf-8')
p=docs/'NEXT_STEPS.md';b=p.read_text(encoding='utf-8');b=b.replace('执行GRID29原DC问题复用数值资格；合格才启动三月原实现/缓存实现两次平衡闭环','GRID29数值资格已通过，执行三月原实现/缓存实现两次平衡闭环');p.write_text(b,encoding='utf-8')
p=docs/'EXPERIMENT_LOG.md';p.write_text(p.read_text(encoding='utf-8')+f"\n## 2026-10-11 — GRID29 V1 shadow qualification and live protocol\n\nSix manual checks and{S['states']} stored exposed states pass, same convergence/finite flags and max flow difference0MW. Shadow time reduction{100*S['time_reduction']:.2f}%, no new AC replay/refit; unified zero storage accumulator limits scope. Freeze4 original/cache protectedMarch live trajectories in balanced2-repeat order before runs. No whole-controller claim yet. [Snapshot](../research_archive/power-grid-2026/protocols/GRID29_PILOT_before_runs/README.md).\n",encoding='utf-8')
print(json.dumps({'archived':len(records),'shadow_states':S['states'],'max_flow_error_MW':S['max_flow_error_MW']}))
