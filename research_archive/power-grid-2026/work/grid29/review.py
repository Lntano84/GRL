"""Quality-first paired March engineering result; no architecture attribution."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid29'
F=json.loads((OUT/'pilot/finished.json').read_text())
A=json.loads((OUT/'pilot/audit.json').read_text());assert A['passed']
index={(s['repeat'],s['method']):s for s in F['summaries']};pairs=[]
for repeat in range(2):
    original=index[(repeat,'ORIGINAL')];cached=index[(repeat,'CACHED')]
    assert original['exact_full_trajectory'] and cached['exact_full_trajectory']
    assert original['cost']==cached['cost'] and original['steps']==cached['steps']==2017
    assert original['complete'] and cached['complete']
    assert original['public_forecasts']==cached['public_forecasts'] and original['n1_forecasts']==cached['n1_forecasts']
    pairs.append({'repeat':repeat,'original_controller_s':original['controller_s'],'cached_controller_s':cached['controller_s'],
        'controller_reduction':1-cached['controller_s']/original['controller_s'],
        'original_cpu_s':original['controller_cpu_s'],'cached_cpu_s':cached['controller_cpu_s'],
        'cpu_reduction':1-cached['controller_cpu_s']/original['controller_cpu_s'],
        'cost_difference':cached['cost']-original['cost'],'same_path':True,
        'public_forecasts':original['public_forecasts'],'cache_counts':cached['cache_counts'],
        'original_setup_s':original['setup_s'],'cached_setup_s':cached['setup_s']})
o=math.fsum(p['original_controller_s'] for p in pairs);c=math.fsum(p['cached_controller_s'] for p in pairs)
result={'stage':'GRID29_SHARED_CACHE_PILOT','passed':True,'paired_repeats':2,'independent_date_families':1,
    'same_trajectory_runs':4,'physical_rows':A['physical_rows'],'public_forecasts':F['counts']['public_forecasts'],
    'controller_s':{'ORIGINAL':o,'CACHED':c},'pooled_controller_reduction':1-c/o,
    'all_pairs_faster':all(p['controller_reduction']>0 for p in pairs),'pairs':pairs,
    'new_fits':0,'reserved_evaluations':0,
    'scope':'Complete decision/control time including DC cache lookup and all controls/forecasts. Setup, physical grid steps, archival separate.',
    'warning':'Only exposed March, two balanced repetitions. Same source queries, delivery and cost; generic DPP engineering improvement shared with all baselines. Does not establish all-date robust speed or GNN contribution.'}
(OUT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines=['# GRID29 整套控制器模型复用试点','',
    '| 重复/顺序 | 原控制时间 | 复用控制时间 | 减少 | 成本差 | 交付轨迹 |',
    '|---|---:|---:|---:|---:|---:|']
for p in pairs:
    lines.append(f"| {p['repeat']} / {'原→复用' if p['repeat']==0 else '复用→原'} | {p['original_controller_s']:.3f}s | {p['cached_controller_s']:.3f}s | {100*p['controller_reduction']:.2f}% | {p['cost_difference']:.0f} | 完全相同 |")
lines+=['',f"汇总控制时间减少{100*result['pooled_controller_reduction']:.2f}%；两组均提速：{result['all_pairs_faster']}。动作/观测/成本逐步相同，公开预测次数不变。",'',
    result['scope'],result['warning'],'',
    'V0真实Parameter倒数表达式不满足DPP，资格检查拒绝；旧代码/日志保留。V1在DSL外计算倒数并按实际电抗值使缓存失效，保持原数学问题/solver/warm_start=False。130个保存状态流量差0，实时闭环仍单独核对。',
    '',f"保存数据审计{A['runs']}条回放、{A['physical_rows']}个物理记录，独立复核掩码/别名/模型排序/实际派发/交付成本；没有另做AC重放或独立拟合。",'',
    '下一步将共有工程优化也用于PPO/FULL等强基线，然后再以平衡重复评估拓扑搜索增量。不能把本表减少与GRID28百分比直接相加。']
(OUT/'GRID29_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('COMPLETED AND AUDITED: 130-state shadow and four exact March protected runs, two balanced pairs. V0 failure preserved; no pending workers, new fit or reserved evaluation.\n',encoding='utf-8')
(OUT/'pilot/RUN_STATE.md').write_text('COMPLETED AND AUDITED: four exact protected trajectories; same costs and query counts. Two balanced March repetitions only.\n',encoding='utf-8')
print(json.dumps(result,indent=2))
