"""Preserve the single necessary static ablation as a unique protocol."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';OUT=ROOT/'outputs/grid57';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','complete'],required=True);a=p.parse_args()
if a.phase=='before':assert read(OUT/'preflight.json')['passed'] and not (OUT/'manifest.json').exists()
else:assert read(OUT/'review.json')['passed_saved_data']
dest=ARCH/'protocols'/('GRID57_'+a.phase);assert not dest.exists();dest.mkdir()
files=list((ROOT/'work/grid57').glob('*.py'))+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv']]
if a.phase=='complete':
    for folder in (OUT/'runs').iterdir():files += [folder/'summary.json',folder/'metadata.json']
entries=[]
for p in files:
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries,'scope':'One frozen existing STATIC_ID ablation on eight exposed dates; no refit, no cross-batch speed claims.'},indent=2),encoding='utf-8')
if a.phase=='before':
    note='GRID57仅冻结一条必要廉价顺序消融：既有STATIC_ID在最终八个已暴露日期各一次回放，原控制/停止/合法信息/DC与安全QP缓存不变，排序是冻结的代表ID升序。无新训练/参数/日期/基线家族；主比较完整周质量、存活和查询，历史FULL/GNN只用于离线参照；新批墙钟不与旧批构成公平提速主张。'
    nxt='1. 执行八条冻结STATIC_ID消融，单串行运行器，保留所有早失败/较差记录。\n2. 按成本账目、固定顺序、合法反馈和向量核验，明确最终GNN相对该廉价消融的质量/查询增量。\n3. 不追加便宜规则或新模型；完成后整理论文证据与缺口。\n'
else:
    r=read(OUT/'review.json');note=f"GRID57八条既有STATIC_ID最终配置消融与保存数据审计完成：质量尺度通过{r['quality_pass']}/8，GNN平均相对其查询减少{100*r['mean_gnn_query_reduction_vs_static']:.3f}%，正向{r['positive_query_families']}/8；裁决{r['verdict']}。无新拟合；不作跨批速度或新独立确认。"
    nxt='1. 将GRID57单一廉价消融纳入论文主张边界与最终结果表，不据此调模型/排序。\n2. 原新模型、FULL/PPO/SoftGNN主对照和全部失败均保留。\n3. 整理论文和复现包；如图模型必要性仍未成立，需明确辅助角色而非强称图架构突破。\n'
p=REPO/'docs/RESEARCH_STATE.md';p.write_text('# Research state — final static ablation\n\n- **最新：** '+note+'\n\n'+p.read_text(encoding='utf-8').split('\n',1)[1],encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n'+nxt,encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-13 — GRID57 '+a.phase+'\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID57 '+a.phase+'\n\n'+note+'\n',encoding='utf-8')
print(json.dumps({'archived':a.phase,'files':len(entries)}))
