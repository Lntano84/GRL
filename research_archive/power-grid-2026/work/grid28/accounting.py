"""Aggregate only matched horizons; separate query/search/control and cost."""
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid28'
F=json.loads((OUT/'finished.json').read_text())
A=json.loads((OUT/'audit.json').read_text());assert A['passed']
C=json.loads((OUT/'all_comparison.json').read_text())
groups={}
for method in ['LOCAL_GNN0','LOCAL_PPO']:
    rows=[r for r in C['rows'] if r['method']==method]
    assert len(rows)==4 and all(r['steps']==r['reference_steps'] and r['complete']==r['reference_complete'] for r in rows)
    get=lambda key:math.fsum(float(r[key]) for r in rows)
    completed=[r['cost_regret'] for r in rows if r['cost_regret'] is not None]
    groups[method]={
        'dates':len(rows),'complete_dates':sum(r['complete'] for r in rows),
        'n1_queries':get('n1_queries'),'reference_n1_queries':get('reference_n1_queries'),
        'pooled_query_reduction':1-get('n1_queries')/get('reference_n1_queries'),
        'source_s':get('search_s'),'reference_source_s':get('reference_search_s'),
        'pooled_source_reduction':1-get('search_s')/get('reference_search_s'),
        'controller_s':get('controller_s'),'reference_controller_s':get('reference_controller_s'),
        'pooled_controller_reduction':1-get('controller_s')/get('reference_controller_s'),
        'source_seconds_saved':get('reference_search_s')-get('search_s'),
        'source_saved_as_reference_controller_fraction':(get('reference_search_s')-get('search_s'))/get('reference_controller_s'),
        'other_control_seconds_delta':(get('controller_s')-get('search_s'))-(get('reference_controller_s')-get('reference_search_s')),
        'completed_date_cost_regret_mean':math.fsum(completed)/len(completed),
        'completed_date_cost_regret_max':max(completed),
        'exact_full_dates':sum(r['exact_full_trajectory'] for r in rows),
        'quality_gate_passed':all(r['qualified'] for r in rows),
        'new_earlier_failures':sum(r['earlier_failure'] for r in rows),
        'illegal':sum(r['illegal'] for r in rows),'ambiguous':sum(r['ambiguous'] for r in rows),
        'exceptions':sum(r['exception_steps'] for r in rows),
        'source_fraction_of_protected_control':get('reference_search_s')/get('reference_controller_s')}
result={'stage':'GRID28_ACCOUNTING','groups':groups,'audited_runs':A['runs'],
        'physical_rows':A['physical_rows'],'public_forecasts':F['counts']['public_forecasts'],
        'quality_gate_passed':C['quality_gate_passed'],'timing_repeats':1,
        'source_definition':'Auxiliary N1 callback including preparation, prior, graph/forward/ranking and paid forecasts.',
        'controller_definition':'All control including continuous optimization/restoration, forecasting and source search; physical simulation/setup/archive separate.',
        'warning':'One exposed development repeat; pooled controller changes include different trajectories and non-search variation. Not stable acceleration, exact equivalence or causal GNN benefit.',
        'new_fits':0,'reserved_evaluations':0}
(OUT/'accounting.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines=['# GRID28 查询、搜索、完整控制成本分开核对','',
       '| 方法 | 查询节省 | 搜索时间减少 | 总控制时间减少 | 完成周成本平均增幅/最大 | 完成/日期 |',
       '|---|---:|---:|---:|---:|---:|']
for method,g in groups.items():
    lines.append(f"| {method} | {100*g['pooled_query_reduction']:.2f}% | {100*g['pooled_source_reduction']:.2f}% | {100*g['pooled_controller_reduction']:.2f}% | {100*g['completed_date_cost_regret_mean']:.4f}% / {100*g['completed_date_cost_regret_max']:.4f}% | {g['complete_dates']}/{g['dates']} |")
lines+=['',f"质量投资筛查通过：{result['quality_gate_passed']}。0.5%是开发筛查尺度，不是等价检验、论文质量门槛或电网安全认证。共同失败周不计成本收益。",'',
         result['warning'],'',
         f"GNN0搜索减少的秒数仅占受保护总控制时间{100*groups['LOCAL_GNN0']['source_saved_as_reference_controller_fraction']:.2f}个百分点。总控制点估计的其余变化来自非搜索部分；不同轨迹与运行波动尚未分辨，不能归因于GNN。",
         '',
         '分批停止和变电站补全是所有排序器共有的非学习控制规则。固定权重GNN与原PPO都使用同一规则；不将共同改善归为图学习独有贡献。原先受保护全池补查仍保留，本实验不保留它的全池选择保证。',
         '',f"已审计{A['runs']}条回放、{A['physical_rows']}个物理记录、{A['source_calls']}次辅助搜索、{A['candidate_feedback']}项实际支付反馈。保存数据审计独立复核账目/掩码/别名/派发/选择并复算模型；没有另做AC重放或独立拟合。"]
(OUT/'GRID28_accounting_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
