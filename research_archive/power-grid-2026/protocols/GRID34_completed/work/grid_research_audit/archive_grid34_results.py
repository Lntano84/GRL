"""Archive the complete fair comparison without overwriting any historical protocol."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
R=json.loads((ROOT/'outputs/grid34/review.json').read_text(encoding='utf-8'))
assert R['passed'] and R['runs']==48
preserved={}
for p in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for row in json.loads(p.read_text(encoding='utf-8')).get('files',[]):
        target=p.parent/row['path']
        if not target.exists():target=ARCH/row['path']
        assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'],target
        preserved[str(target)]=row['sha256']
dest=ARCH/'protocols/GRID34_completed';assert not dest.exists();dest.mkdir(parents=True)
files=list((ROOT/'work/grid34').glob('*.py'))+[Path(__file__).resolve()]
files += [p for p in (ROOT/'work/grid34/revisions').glob('*') if p.is_file()]
files += [p for p in (ROOT/'outputs/grid34').iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv']]
for folder in (ROOT/'outputs/grid34/runs').iterdir():
    files += [folder/n for n in ['summary.json','metadata.json','components.json']]
records=[]
for p in sorted(set(files)):
    rel=p.relative_to(ROOT);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    records.append({'path':rel.as_posix(),'sha256':h,'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':records,'historical_preserved':len(preserved),
    'scope':'48 delivered common-cache comparisons, both GNN and MLP seeds plus original PPO and exhaustive FULL. All exposed development dates; raw physical/source vectors local, no new fits or reserved dates.'},indent=2),encoding='utf-8')
cheap=ARCH/'protocols/GRID35_before_results';assert not cheap.exists();cheap.mkdir(parents=True)
cheap_records=[]
for p in sorted(list((ROOT/'work/grid35').glob('*.py'))+[ROOT/'outputs/grid35/design.json',ROOT/'outputs/grid35/code_freeze.json']):
    rel=p.relative_to(ROOT);target=cheap/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    cheap_records.append({'path':rel.as_posix(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
(cheap/'manifest.json').write_text(json.dumps({'files':cheap_records,'scope':'52frozen cheap public order controls, no outcomes at archival time.'},indent=2),encoding='utf-8')
comparison={(c['gnn'],c['opponent']):c for c in R['comparisons']}
parts=[]
for g in ['LOCAL_GNN0','LOCAL_GNN1']:
    vals=[comparison[(g,x)]['pooled_control_reduction']*100 for x in ['FULL','LOCAL_PPO','LOCAL_MLP0','LOCAL_MLP1']]
    parts.append(f"{g}控制时间相对共有优化FULL/PPO/MLP0/MLP1分别减少"+'/'.join(f'{x:.2f}%' for x in vals))
note='GRID34全部48条公平闭环和保存数据审计完成；'+ '；'.join(parts)+'。全部模型种子保留，六月共同1901步失败保留。两GNN均满足质量投资门槛，MLP0九月成本高0.60247%超过门槛但结果保留；共有编译复用不归于GNN，开发日期不作封存泛化。'
docs=REPO/'docs';p=docs/'RESEARCH_STATE.md';old=p.read_text(encoding='utf-8')
p.write_text('# Research state — GRID34 complete, acceleration contributions separated\n\n- **最新：** '+note+'\n\n'+old.split('\n',1)[1],encoding='utf-8')
(docs/'NEXT_STEPS.md').write_text('''# Next steps — separate shared engineering from learned search

1. GRID34全部48条已完成并审计，无需重跑。所有模型/两计时重复/成本/存活/查询/组成结果见最新报告；先读同状态及相同交付路径对照，不能从总时间残差识别图结构作用。
2. 用户要求的10.93%与0.39%差异已有直接组件诊断：旧批缺逐solver时间不能恢复精确因果；相同路径复测DC节省真实，共享DC+safeQP复用改善幅度取决于占比/复用率，所有强对照均已获得。
3. GRID35已冻结52项廉价顺序诊断：STATIC_ID与仅用当前线路负载的LOCAL_STRESS，所有方法共享原停止规则/语义补全。仅使用已存反馈，不新增仿真/训练；已知1733关键动作遗漏阻断该配置闭环，通过只获得资格，不认定长期质量或GNN必要性。
4. 下一步只在明确的新贡献假设下推进，保留原权重及所有修订。不要通过挑模型种子、改评价预算或削弱MLP/PPO制造GNN收益；不自动启动RL。
5. 最终8日期仍封存。额度暂停不等于用户取消；下一次继续沿本任务与已保存阶段，无需重新选题/询问重复授权。LG01保持暂停。
''',encoding='utf-8')
for name in ['EXPERIMENT_LOG.md','DECISIONS.md']:
    p=docs/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-11 — GRID34 completed fair common-cache comparisons\n\n'+note+' 48条重复内状态/反馈/完整向量/成本复核，未新增拟合或最终日期；相同路径子集和同输入查询比较单独记录。finished.wall_s仅最后剩余阶段，不误报为全部实验墙钟。\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text('# 最新：GRID34公平闭环完成，共有提速与模型增量分开\n\n[完整结果](protocols/GRID34_completed/outputs/grid34/GRID34_report.md) · [独立保存数据审计](protocols/GRID34_completed/outputs/grid34/review.json)\n\n'+note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
for p,h in preserved.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
print(json.dumps({'archived_files':len(records),'historical_preserved':len(preserved),'passed':True}))
