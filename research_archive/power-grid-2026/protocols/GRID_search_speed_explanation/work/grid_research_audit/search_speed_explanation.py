"""Saved-only disjoint timing arithmetic, no source query/fit/controller replay."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid_speed_review'
R=json.loads((ROOT/'outputs/grid34/review.json').read_text());assert R['passed']
S={s['method']:s for s in R['summaries']};full=S['FULL'];result=[]
for g in ['LOCAL_GNN0','LOCAL_GNN1']:
    cur=S[g];whole=full['control_s']-cur['control_s'];source=full['n1_source_s']-cur['n1_source_s']
    other=(full['control_s']-full['n1_source_s'])-(cur['control_s']-cur['n1_source_s'])
    assert abs(whole-source-other)<1e-9
    share=full['n1_source_s']/full['control_s'];reduction=1-cur['n1_source_s']/full['n1_source_s']
    result.append({'gnn':g,'full_search_share':share,'search_reduction':reduction,
       'whole_reduction':whole/full['control_s'],'search_saving_s':source,'other_saving_s':other,
       'total_saving_s':whole,'search_contribution_to_whole_percentage_points':100*source/full['control_s'],
       'other_contribution_to_whole_percentage_points':100*other/full['control_s'],
       'check_search_share_times_reduction':share*reduction,
       'warning':'Search and remaining-control partition is additive observed timing,not causal identification of graph architecture. Two June paths differ despite equal exposure; complete-date path/quality separately audited.'})
O={'source':'outputs/grid34/review.json','pooled_time_scope':'4exposed dates x2repeats,including shared June1901step failure; total-time-weighted ratio,not mean of independent percentages.',
   'full_control_s':full['control_s'],'full_search_s':full['n1_source_s'],'rows':result,
   'new_fits':0,'new_forecasts':0,'new_control_replays':0}
(OUT/'GRID34_disjoint_timing.json').write_text(json.dumps(O,indent=2),encoding='utf-8')
lines=['# 两类约10%提速应分开解释','','## 早前DC复用的10.93%与0.39%','',
    '这两对仅改变DC重复编译复用。后续同输入诊断和组件复测确认DC节省真实，但早前两对缺逐solver计时，无法追溯精确波动原因。组件复测DC分别省17.47/18.14秒，其他组成分别多5.74/6.27秒，最终整体省6.45%/6.51%；不把新实验替代旧数据。', '',
    '继续复用安全QP后，同路径的共有工程改进在三月为22.42%（两重复），其余开发周为61.89%–74.46%（每周单配对）；幅度与可复用组成占比/缓存命中率相对应，全部提供给强对照，不归于GNN。', '',
    '## 现在共有工程优化之后，搜索带来的11%–12%','',
    f"GRID34优化FULL汇总控制{full['control_s']:.3f}秒，其中N1搜索{full['n1_source_s']:.3f}秒（{100*full['n1_source_s']/full['control_s']:.2f}%）。两个计时重复、两个模型种子和全部四周保留。",'',
    '| GNN | 搜索时间减少 | 搜索对整体节省贡献 | 其他部分贡献 | 实测整体减少 |', '|---|---:|---:|---:|---:|']
for r in result:lines.append(f"| {r['gnn']} | {100*r['search_reduction']:.2f}% | {r['search_contribution_to_whole_percentage_points']:.3f}百分点 | {r['other_contribution_to_whole_percentage_points']:.3f}百分点 | {100*r['whole_reduction']:.2f}% |")
lines+=['', '组成恒等式逐项核对，不把各阶段百分比相加。当前约11%不等于复现了旧DC那一对10.93%的全部因果；这是不同修改、不同对照。', '',
    '## 不是11%都属于图结构','', '| GNN | 同停止/缓存的对手 | 查询减少 | 总控制减少 | 时间正向配对 |','|---|---|---:|---:|---:|']
for c in R['comparisons']:
    if c['opponent']=='FULL':continue
    lines.append(f"| {c['gnn']} | {c['opponent']} | {100*c['pooled_query_reduction']:.3f}% | {100*c['pooled_control_reduction']:.3f}% | {c['control_positive_pairs']}/8 |")
lines+=['', 'GNN0与PPO八对完整交付路径相同，查询少5.979%，总时间仅少0.117%；这能支持查询方面的小增量，不能认定稳定架构速度优势。GNN1在三个共同完成周逐步与FULL一致；GNN0/PPO三月成本高0.13612%。MLP0九月高0.60247%结果保留；六月均在1901步失败。', '',
    O['pooled_time_scope'],'','这些仍为开发证据，最终8日期未动，没有新训练。后续廉价顺序先过闭环质量门槛，不因查询更少就称更好。']
(OUT/'GRID34_INCREMENT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(O,indent=2))
