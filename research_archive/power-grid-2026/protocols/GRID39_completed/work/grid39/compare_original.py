"""Compare deterministic quality/query outcomes; never pair cross-batch timings."""
import json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid39'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
old = read(ROOT / 'outputs/grid37/review.json')
new = read(OUT / 'review.json')
assert old['passed_saved_data'] and new['passed_saved_data']
old_runs = read(ROOT / 'outputs/grid37/finished.json')['summaries']
new_runs = read(OUT / 'finished.json')['summaries']
index = {(r['repeat'], r['week'], r['method']): r for r in old_runs}
rows = []
for r in new_runs:
    if r['method'] == 'FULL': continue
    name = r['method'].replace('REVISED_', 'LOCAL_')
    a = index[(0, r['week'], name)]
    b = index[(1, r['week'], name)]
    assert a['cost'] == b['cost'] and a['n1_forecasts'] == b['n1_forecasts']
    full = next(s for s in new_runs if s['week'] == r['week'] and s['method'] == 'FULL')
    comparable = a['steps'] == r['steps'] == full['steps']
    rows.append({'week': r['week'], 'model_seed': int(name[-1]),
        'comparable_exposure': comparable, 'old_cost': a['cost'], 'new_cost': r['cost'],
        'cost_change_of_full': (r['cost']-a['cost']) / max(1, abs(full['cost'])) if comparable else None,
        'old_queries': a['n1_forecasts'], 'new_queries': r['n1_forecasts'],
        'old_complete': a['complete'], 'new_complete': r['complete']})
summary = []
for seed in [0, 1]:
    group = [r for r in rows if r['model_seed'] == seed]
    old_q = sum(r['old_queries'] for r in group)
    new_q = sum(r['new_queries'] for r in group)
    summary.append({'model_seed': seed, 'old_queries': old_q, 'new_queries': new_q,
        'query_reduction': 1-new_q/old_q,
        'mean_cost_change_of_full': math.fsum(r['cost_change_of_full'] for r in group)/len(group)
            if all(r['comparable_exposure'] for r in group) else None,
        'positive_query_dates': sum(r['new_queries'] < r['old_queries'] for r in group)})
result = {'summaries': summary, 'rows': rows,
    'timing_comparison': 'Not calculated: GRID37/39 are different timing batches.',
    'decision': 'Do not replace original configuration: mixed query changes, no quality gain, seed1 quality regression. Retain all new models and curves; continue the research, not another mixture sweep.'}
(OUT / 'original_reference.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
text = '\n\n## 与原版的质量/查询参照\n\n'
text += '| 种子 | 原/新查询 | 查询减少 | 成本变化（四周平均，以FULL成本归一化） |\n|---|---:|---:|---:|\n'
for r in summary:
    cost = '不可比' if r['mean_cost_change_of_full'] is None else f"{100*r['mean_cost_change_of_full']:.5f}%"
    text += f"| {r['model_seed']} | {r['old_queries']}/{r['new_queries']} | {100*r['query_reduction']:.3f}% | {cost} |\n"
text += '\n新旧属于不同计时批次，不计算配对速度增量。这版目标未建立稳定改进，暂不替换原配置；全部原/新权重、训练与失败/混合记录保留。下一步转向限定的近期论文对照适配，不继续扫描目标混合比例。\n'
p = OUT / 'GRID39_report.md'
p.write_text(p.read_text(encoding='utf-8') + text, encoding='utf-8')
print(json.dumps(summary, indent=2))
