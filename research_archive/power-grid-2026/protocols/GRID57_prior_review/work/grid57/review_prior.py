"""Identify the source of the cheap order, without training or new simulation."""
import hashlib,json,math,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid57_prior_review';OUT.mkdir(exist_ok=False)
REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
R=read(ROOT/'outputs/grid57/review.json');assert R['passed_saved_data']
asset=ROOT/'work/grid02/ljn_nn/assets/action_N1_unsafe.npz'
with np.load(asset,allow_pickle=True) as z:counts=z['counts'].copy();actions=z['action_space'].copy()
assert len(counts)==len(actions)==909 and np.all(np.diff(counts)<=0)
hashes=[hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in actions]
for s in read(ROOT/'outputs/grid57/finished.json')['summaries']:
    md=read(ROOT/'outputs/grid57/runs'/f"r0__{s['week']}__STATIC_ID/metadata.json")
    assert hashes==md['library_hashes']
rows=R['results'];ownqueries=math.fsum(r['gnn_seed_average_queries'] for r in rows)
S={'passed':True,'library_order_matches_all8_runs':True,'author_counts_nonincreasing':True,
   'top128_author_count_mass':float(counts[:128].sum()/counts.sum()),'top256_author_count_mass':float(counts[:256].sum()/counts.sum()),
   'static_order_is':'Original public author teacher-frequency-descending action library, after current legal masking/canonicalization; not random/anonymous action ID order.',
   'pool_source':'Author README describes offline CurriculumAgent/teacher collection; load_action_to_grid2op returns asset rows in stored order and TopoSearchModule preserves them.',
   'author_library_teacher_dates_known':False,'all_comparators_share_same_library':True,
   'query_macro_gnn_reduction_defined7':R['mean_gnn_query_reduction_vs_static'],
   'query_pooled_gnn_reduction':1-ownqueries/R['static_n1_queries'],
   'gnn_cost_gain_of_full_macro':math.fsum(r['gnn_cost_gain_of_full'] for r in rows)/8,
   'gnn_cost_better_worse_same':[sum(r['gnn_cost_gain_of_full']>1e-9 for r in rows),sum(r['gnn_cost_gain_of_full'] < -1e-9 for r in rows),sum(abs(r['gnn_cost_gain_of_full'])<=1e-9 for r in rows)],
   'sources':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in [asset,ROOT/'work/grid02/ljn_nn/README.md',ROOT/'work/grid02/ljn_nn/modules/utils.py',ROOT/'work/grid02/ljn_nn/modules/topology_heuristic.py',ROOT/'outputs/grid57/review.json']],
   'new_fits':0,'new_simulations':0}
(OUT/'review.json').write_text(json.dumps(S,indent=2),encoding='utf-8')
text=f'''# GRID57解读：固定顺序包含一个强历史先验

STATIC_ID不是随机或匿名编号：909个作者动作的counts严格非增，运行库的909行hash在全部八条中与资产原顺序相同。读取器和TopoSearchModule保持文件行顺序。因此本次顺序等价于作者历史教师频次降序（在当前合法掩码与动作等价类之后）。前128/256行占已保存频次总和的{100*S['top128_author_count_mass']:.2f}%/{100*S['top256_author_count_mass']:.2f}%。README说明动作库来自离线教师采集；具体教师日期/暴露未知，不能宣称该历史先验无知识，也不能指控测试泄漏。所有方法共用这份动作库。

最终消融8/8完成且符合0.5%质量尺度。GNN在有拓扑调用的7日期上，查询宏平均减少{100*S['query_macro_gnn_reduction_defined7']:.3f}%、4/7正、2/7负、1/7平；全段汇总查询反而多{-100*S['query_pooled_gnn_reduction']:.3f}%（17939对16814）。宏平均与汇总的单位不同，均如实报告。无查询日期继续保留质量与成本，不造0/0比例。

GNN相对该顺序的平均成本增益为{100*S['gnn_cost_gain_of_full_macro']:.4f}%ofFULL，即当前GNN成本更高约0.509个FULL归一化百分点；逐日期1更好/6更差/1平。新批次墙钟仅描述，不据此宣布端到端速度支配。

保留冻结MIXED判读，但它不能被用来掩盖强对照：现有GNN独有贡献未建立，约15.75%的完整框架收益不应全归于图网络。此前模型以PPO排名为先验，却没有在训练中保留这条更有竞争力的教师频次先验，这是一个有代码来源支持的诊断线索，不是已证实根因。

若继续一轮模型改进，范围应只改变先验：用同一合法频次顺序作为零残差基准，保持图网络/目标/原始初始化/64轮/两种子/原20训练周不变。先在旧开发周检查是否保留强基准质量和得到额外查询收益，不在这八日期调参数，不增加模型菜单。已有PPO先验版本、频次顺序和全部失败继续保留。该尝试尚未执行。
'''
(OUT/'PRIOR_FINDING.md').write_text(text,encoding='utf-8')
dest=ARCH/'protocols/GRID57_prior_review';assert not dest.exists();dest.mkdir();entries=[]
for p in [Path(__file__).resolve()]+list(OUT.iterdir()):
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries},indent=2),encoding='utf-8')
note='GRID57固定顺序来源已核实：909动作counts非增且全部八运行行hash等同作者资产顺序，等价历史教师频次先验。当前GNN查询宏平均仅少1.90%（7有调用日期），汇总反而多6.69%；成本1好/6差/1平、平均高0.509个FULL归一化百分点。现有图模块增量未成立，整体提速仍保留。只保留“让GNN校正较强频次先验”的单次改进假设，不继续基线扩散。'
p=REPO/'docs/RESEARCH_STATE.md';p.write_text('# Research state — strong frequency prior identified\n\n- **最新：** '+note+'\n\n'+p.read_text(encoding='utf-8').split('\n',1)[1],encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-13 — GRID57 prior review\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID57 频次先验来源\n\n'+note+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'strong_prior_identified':True,'pooled_query_gain':S['query_pooled_gnn_reduction']}))
