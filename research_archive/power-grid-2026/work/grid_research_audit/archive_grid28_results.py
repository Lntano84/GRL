"""Archive completed frozen staged-search evidence; preserve previous hashes."""
import hashlib
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration'
ARCH=REPO/'research_archive/power-grid-2026'
A=json.loads((ROOT/'outputs/grid28/audit.json').read_text());assert A['passed'] and A['runs']==12
C=json.loads((ROOT/'outputs/grid28/accounting.json').read_text())
assert not (ARCH/'grid28_archive_manifest.json').exists()
preserved={}
for path in sorted(ARCH.rglob('*manifest.json')):
    if not (path.parent==ARCH or (path.name=='manifest.json' and path.parent.parent==ARCH/'protocols')):
        continue
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or 'files' not in data:continue
    for row in data['files']:
        if 'path' not in row or 'sha256' not in row:continue
        p=path.parent/row['path']
        if not p.exists():p=ARCH/row['path']
        assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
        preserved[str(p)]=row['sha256']
files=[]
for stage in ['grid27','grid28']:
    files+=sorted((ROOT/'work'/stage).glob('*.py'))
    files+=[p for p in sorted((ROOT/'outputs'/stage).glob('*')) if p.is_file() and p.suffix in ['.json','.csv','.md']]
files+=[ROOT/'work/grid27/profile_control.py']
files+=[p for p in sorted((ROOT/'outputs/grid27_profile').glob('*')) if p.is_file() and p.suffix in ['.json','.md']]
for stage in ['grid28','grid27_profile']:
    for folder in sorted((ROOT/'outputs'/stage/'runs').iterdir()):
        files+=[folder/name for name in ['summary.json','metadata.json']]
files+=[Path(__file__).resolve()]
records=[]
for p in sorted(set(files)):
    relative=p.relative_to(ROOT);dest=ARCH/relative
    assert not dest.exists(),dest
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(dest.read_bytes()).hexdigest()==h
    records.append({'path':relative.as_posix(),'sha256':h,'bytes':p.stat().st_size})
(ARCH/'grid28_archive_manifest.json').write_text(json.dumps({'files':records,'historical_preserved_checks':len(preserved),
    'scope':'GRID27 stopping screen, GRID28 completed development results and own code, one protected profile. Large physical/candidate vectors stay local; no copied RTE dataset or new weights.'},indent=2),encoding='utf-8')
g=C['groups']['LOCAL_GNN0'];p=C['groups']['LOCAL_PPO']
summary=(f"GRID28完成12/12条开发回放，保存数据审计通过。GNN0查询节省{100*g['pooled_query_reduction']:.2f}%、搜索时间减少{100*g['pooled_source_reduction']:.2f}%、总控制减少{100*g['pooled_controller_reduction']:.2f}%；共同完成周成本平均增幅{100*g['completed_date_cost_regret_mean']:.4f}%、最大{100*g['completed_date_cost_regret_max']:.4f}%。GNN0/PPO均完成{g['complete_dates']}/4周，没有新增/提前失败；PPO同规则查询节省{100*p['pooled_query_reduction']:.2f}%、搜索减少{100*p['pooled_source_reduction']:.2f}%、控制减少{100*p['pooled_controller_reduction']:.2f}%。单次计时、开发数据及不同轨迹，不证明稳定端到端速度或GNN独有贡献；0.5%为投资筛查，不是等价。")
readme=ARCH/'README.md';body=readme.read_text(encoding='utf-8')
readme.write_text('# 最新：GRID28 分批搜索闭环开发结果（2026-10-11）\n\n[完整核对](outputs/grid28/GRID28_accounting_report.md) · [逐日期](outputs/grid28/all_comparison.csv) · [审计](outputs/grid28/audit.json) · [归档清单](grid28_archive_manifest.json)\n\n'+summary+'\n\nGRID27简单停止遗漏关键动作，保留否定筛查；GRID28只加入当前最佳方案所在变电站补全。旧全池守卫仍保留，新规则不具有全池最优选择保证。当前拟合/RL/封存评价0。整体控制器线另进入GRID29原DC模型复用资格，不计为GNN贡献。\n\n'+body,encoding='utf-8')
docs=REPO/'docs';state=docs/'RESEARCH_STATE.md';body=state.read_text(encoding='utf-8')
body=body.replace('现正跑冻结三月试点及同规则PPO强对照','冻结闭环矩阵已完成')
body=body.replace('冻结三月试点及同规则PPO强对照已通过0.5%质量投资筛查（成本均+0.136%，不是零差）；其余3个日期按原规则串行运行','冻结12条闭环矩阵已完成')
body=body.replace('尚无GRID28全矩阵裁决','GRID28全矩阵质量投资筛查已通过' if C['quality_gate_passed'] else 'GRID28全矩阵质量投资筛查未通过')
state.write_text('# Research state — 2026-10-11\n\n- **最新：** '+summary+' [结果](../research_archive/power-grid-2026/outputs/grid28/GRID28_accounting_report.md)。\n\n'+body.split('\n',1)[1],encoding='utf-8')
(docs/'NEXT_STEPS.md').write_text('''# Next steps — 2026-10-11 / GRID28 complete, GRID29 qualification

1. GRID28全矩阵已审计，保留完整成本差/单次计时边界。稳定提速仍需公平重复，PPO使用同一停止补全规则；不能宣称图模型独有贡献。
2. 执行GRID29原DC问题复用数值资格；合格才启动三月原实现/缓存实现两次平衡闭环，任何历史路径差异先诊断。所有强基线同样可复用工程优化。
3. 不开8个封存日期、不调停止常数或挑模型；原模型/守卫/失败均保留。没有新拟合或RL，LG01自动跟进保持暂停。
''',encoding='utf-8')
for filename,note in [('EXPERIMENT_LOG.md','\n## 2026-10-11 — GRID28 frozen closed-loop matrix completed\n\n'+summary+f" Audited physical rows{A['physical_rows']}, source calls{A['source_calls']}, paid feedback{A['candidate_feedback']}; saved-data checks, no independent AC replay/refit. [Report](../research_archive/power-grid-2026/outputs/grid28/GRID28_accounting_report.md).\n"),('DECISIONS.md','''\n## 2026-10-11 — Retain staged-search tradeoff without graph attribution

Keep measured quality cost and one-repeat timing visible. Substation completion addresses the known omitted action, while the original protected guard remains available. Same stopping logic for GNN and PPO is essential; shared improvement is not GNN benefit. Whole-controller engineering qualification proceeds separately and is granted to all baselines. Do not train RL or open final date families from these developmental results.
''')]:
    p=docs/filename;p.write_text(p.read_text(encoding='utf-8')+note,encoding='utf-8')
for file,h in preserved.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==h
print(json.dumps({'passed':True,'archived':len(records),'preserved':len(preserved),'groups':C['groups']},indent=2))
