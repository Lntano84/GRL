"""Preserve full cache pilot, matched ranker review, and next diagnostic freeze."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
R=json.loads((ROOT/'outputs/grid29/comparison.json').read_text());assert R['passed']
P=json.loads((ROOT/'outputs/grid28_review/paired.json').read_text())
assert not (ARCH/'grid29_archive_manifest.json').exists()
preserved={}
for path in ARCH.rglob('*manifest.json'):
    if not(path.parent==ARCH or(path.name=='manifest.json' and path.parent.parent==ARCH/'protocols')):continue
    data=json.loads(path.read_text(encoding='utf-8'))
    for row in data.get('files',[]):
        p=path.parent/row['path']
        if not p.exists():p=ARCH/row['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
        preserved[str(p)]=row['sha256']
files=sorted((ROOT/'work/grid29').glob('*.py'))+sorted((ROOT/'work/grid29b').glob('*.py'))
files+=[ROOT/'work/grid_research_audit/grid28_pair_review.py',Path(__file__).resolve()]
for stage in ['grid29','grid28_review','grid29b']:
    files+=[p for p in (ROOT/'outputs'/stage).glob('*') if p.is_file() and p.suffix in ['.json','.md','.csv']]
files+=list((ROOT/'outputs/grid29/revisions').rglob('*'))
files+=[p for p in (ROOT/'outputs/grid29/pilot').glob('*') if p.is_file() and p.suffix in ['.json','.md']]
for folder in (ROOT/'outputs/grid29/pilot/runs').iterdir():files+=[folder/n for n in ['summary.json','metadata.json']]
records=[]
for p in sorted(set(p for p in files if p.is_file())):
    rel=p.relative_to(ROOT);target=ARCH/rel;assert not target.exists(),target
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    records.append({'path':rel.as_posix(),'sha256':h,'bytes':p.stat().st_size})
(ARCH/'grid29_archive_manifest.json').write_text(json.dumps({'files':records,'historical_preserved_checks':len(preserved),
    'scope':'V0 failure/V1 shadow/four closed pilot rows and own code, GRID28 posthoc matched review, GRID29B diagnostic before results. Raw physical vectors and datasets remain local.'},indent=2),encoding='utf-8')
note=(f"GRID29工程数值资格130状态流量差0；4条三月闭环逐步动作/观测/成本相同，查询不变。两组控制时间减少{100*R['pairs'][0]['controller_reduction']:.2f}%/{100*R['pairs'][1]['controller_reduction']:.2f}%，汇总点估计{100*R['pooled_controller_reduction']:.2f}%，幅度不稳定，不能认定跨日期稳定整体提速或GNN贡献。V0非DPP/键缺陷已保留，V1修订与协议不覆盖历史。")
intro='# 最新：GRID29 共有模型复用试点（2026-10-11）\n\n[报告](outputs/grid29/GRID29_report.md) · [匹配排序诊断](outputs/grid28_review/GRID28_ranker_review.md) · [归档清单](grid29_archive_manifest.json)\n\n'+note+f"\n\n后验匹配GRID28的26次同状态/同候选池/同选择，4周GNN/PPO交付路径也相同；GNN比PPO少{P['ppo_queries']-P['gnn_queries']}次查询（{100*P['query_reduction_against_ppo']:.2f}%），逐调用7少/4多/15平。只有已暴露开发信息，不是图架构或泛化证明。GRID29B冻结一条实际控制状态/记忆下配对DC诊断，未启动新训练或封存日期评价。\n\n"
p=ARCH/'README.md';p.write_text(intro+p.read_text(encoding='utf-8'),encoding='utf-8')
docs=REPO/'docs';p=docs/'RESEARCH_STATE.md';b=p.read_text(encoding='utf-8')
b=b.replace('现进入4条平衡闭环，还不能宣称完整控制提速','4条平衡闭环已审计，质量路径完全相同，整体计时幅度0.39%–10.93%不稳定；进入GRID29B局部配对诊断')
p.write_text('# Research state — 2026-10-11\n\n- **最新整体线：** '+note+'\n\n'+b.split('\n',1)[1],encoding='utf-8')
(docs/'NEXT_STEPS.md').write_text('''# Next steps — 2026-10-11 / GRID29B prepared

1. 执行冻结GRID29B真实控制记忆下的隔离DC配对诊断，输出完整solver类型/编译/后端字段，缺失不算0。实际交付仍原受保护策略，带额外诊断的控制总时间不作速度成绩。
2. GRID28搜索节省43.18%/查询50.90%有开发读数，但总控制8.83%中的非搜索变化未解释。GNN同状态比PPO少5.98%查询，7/4/15方向不均匀，没有MLP等架构对照或封存泛化确认。
3. 后续公平计时要把共用工程优化也给强基线。不能直接相加GRID28/29百分比；不调停止常数、不选更快重复或立即训练。
4. 原模型/守卫/失败保留；8个封存日期未评价，LG01跟进仍暂停。
''',encoding='utf-8')
for filename,note2 in [('EXPERIMENT_LOG.md','\n## 2026-10-11 — GRID29 cache pilot and GRID28 paired ranker review\n\n'+note+f" Audit8068physicalrows/44sourcecalls/26236paidfeedback; no independentAC/refit. GRID28 matched26same-state/same-pool/same-choice calls, GNN query reduction5.98% vsPPO, directions7/4/15, common feedback6178/6178exact. Both branches4/4same delivered paths. GRID29B frozen before run for actual-memory isolated DC probes, not another total-time sweep.\n"),('DECISIONS.md','''\n## 2026-10-11 — Diagnose component speed before more total-time repeats

Cache pilot preserved exact quality, but10.93% versus0.39% paired total gains do not identify stable whole-speed magnitude. Keep both. Next one instrumented actual controller plus isolated same-memory DC pairs; never deploy probe output or label instrumented total as acceleration. Include positional solver arguments and null statistics correctly. GNN/PPO matched-query developmental signal does not identify GNN architecture or independent transfer; common engineering changes go to both.
''')]:
    p=docs/filename;p.write_text(p.read_text(encoding='utf-8')+note2,encoding='utf-8')
for p,h in preserved.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
print(json.dumps({'archived':len(records),'preserved':len(preserved),'passed':True}))
