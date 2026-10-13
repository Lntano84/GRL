"""Post-run decomposition; preserve the frozen primary screen unchanged."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/grid62'
OUT = ROOT / 'outputs/grid62_timing_review'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
mean = lambda x: math.fsum(x) / len(x) if x else None


def main():
    assert not OUT.exists()
    finished = read(SOURCE / 'finished.json')
    review = read(SOURCE / 'review.json')
    assert review['passed_saved_data']
    rows, pairs = [], []
    summaries = finished['summaries']
    for s in summaries:
        folder = SOURCE / 'runs' / f"r{s['repeat']}__{s['week']}__{s['method']}"
        comp = read(folder / 'components.json')
        with gzip.open(folder / 'steps.jsonl.gz', 'rt', encoding='utf-8') as stream:
            steps = [json.loads(line) for line in stream]
        latencies = sorted(x['decision_s'] for x in steps)
        q = (len(latencies) - 1) * .95
        lo, hi = math.floor(q), math.ceil(q)
        p95 = latencies[lo] * (hi - q) + latencies[hi] * (q - lo) if hi != lo else latencies[lo]
        phases = {phase: math.fsum(x['wall_s'] for x in comp['calls'] if x['in_control'] and x['phase'] == phase)
                  for phase in ['run_dc', 'compute_optimum_safe', 'compute_optimum_unsafe']}
        solver = [x for x in comp['solver_records'] if x['in_control']]
        rows.append({'repeat': s['repeat'], 'week': s['week'], 'method': s['method'],
            'control_s': s['controller_s'], 'control_cpu_s': s['controller_cpu_s'], 'n1_source_s': s['n1_source_s'],
            'outside_n1_s': s['controller_s'] - s['n1_source_s'], 'n1_calls': s['n1_calls'], 'n1_queries': s['n1_forecasts'],
            'decision_p95_s': p95, **phases,
            'solver_wall_s': math.fsum(x['wall_s'] for x in solver),
            'solver_iterations': sum(x['iterations'] or 0 for x in solver),
            'solver_user_limits': sum(x['status'] == 'user_limit' for x in solver)})
    for week in read(SOURCE / 'design.json')['weeks']:
        for method in ('FULL', 'FIRST_SAFE_BEST', 'NO_LOCAL', 'FRAME'):
            a = next(x for x in rows if x['week'] == week and x['method'] == method and x['repeat'] == 0)
            b = next(x for x in rows if x['week'] == week and x['method'] == method and x['repeat'] == 1)
            pairs.append({'week': week, 'method': method, 'physical_paths_identical': True,
                'control_ratio_r0_r1': a['control_s'] / b['control_s'],
                'cpu_ratio_r0_r1': a['control_cpu_s'] / b['control_cpu_s'],
                'n1_source_delta_r0_r1_s': a['n1_source_s'] - b['n1_source_s'],
                'outside_n1_delta_r0_r1_s': a['outside_n1_s'] - b['outside_n1_s'],
                'solver_iteration_total_equal': a['solver_iterations'] == b['solver_iterations']})
    active_weeks = [w for w in read(SOURCE / 'design.json')['weeks']
                    if any(s['n1_calls'] for s in summaries if s['week'] == w and s['method'] == 'FULL')]
    inactive = [w for w in read(SOURCE / 'design.json')['weeks'] if w not in active_weeks]
    active = {}
    for method in ('FIRST_SAFE_BEST', 'NO_LOCAL', 'FRAME'):
        selected = [x for x in review['family_rows'] if x['method'] == method and x['week'] in active_weeks]
        active[method] = {'macro_control_reduction_full': mean([x['control_reduction_full'] for x in selected]),
            'macro_query_reduction_full': mean([x['query_reduction_full'] for x in selected])}
    direct = []
    for week in active_weeks:
        a = next(x for x in review['family_rows'] if x['method'] == 'FRAME' and x['week'] == week)
        b = next(x for x in review['family_rows'] if x['method'] == 'FIRST_SAFE_BEST' and x['week'] == week)
        direct.append({'week': week, 'frame_control_reduction_first': 1-a['controller_s']/b['controller_s'],
            'frame_cost_gain_of_full_vs_first': b['cost_delta_full'] - a['cost_delta_full']})
    result = {'frozen_primary_verdict_unchanged': review['verdict'], 'active_families': active_weeks,
        'inactive_families': inactive, 'active_only_descriptive': active,
        'frame_vs_first_active_descriptive': direct, 'repeat_pairs': pairs,
        'scope': 'Post-run diagnostic, not a replacement primary criterion, date deletion, retimed score, or new confirmation.',
        'interpretation': 'The inactive May family has no topology query for any policy, so its time differences cannot be effects of the uncalled search. Same physical paths can have large wall and process-CPU variation in continuous optimization/compilation. No OS power events were observed; load/clock causes were not separately measured. Do not claim exclusive framework superiority over first-safe from the all-date timing ratio.',
        'sources': [{'path': 'outputs/grid62/' + name, 'sha256': hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()}
                    for name in ['review.json', 'finished.json', 'per_family.csv', 'power_events.json']]}
    OUT.mkdir()
    for name, data in [('per_run_timing.csv', rows), ('repeat_variation.csv', pairs), ('active_vs_first.csv', direct)]:
        with (OUT/name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, list(data[0]))
            writer.writeheader(); writer.writerows(data)
    (OUT/'review.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# GRID62 计时与贡献边界补充', '', '原冻结裁决不变：' + review['verdict'] + '。', '',
        '五月所有方法都未调用拓扑搜索，同步747步失败。因此该日期计时差不能归因于没有执行过的搜索策略；不删除它、不重计原成绩。', '',
        '|有实际搜索的两日期（事后描述）|控制减少/FULL|查询减少/FULL|', '|---|---:|---:|']
    for method, x in active.items():
        lines.append(f"|{method}|{x['macro_control_reduction_full']:+.3%}|{x['macro_query_reduction_full']:+.3%}|")
    lines += ['', 'FIRST_SAFE_BEST在三个新日期也全部通过质量尺度，在有搜索的两日期实际控制时间方向均优于FULL；FRAME没有建立对它的稳定优势。四开发日期普通停止的质量问题仍是有效历史证据，但不能代替本轮新日期的比较。', '',
        'FRAME与NO_LOCAL在新四月的成本差仅为FULL的0.007818%，低于0.1%机制尺度；新十一月两者成本与路径相同。新日期没有复现一个强局部补全质量事件，已知三月事件只保留为机制实例。', '',
        '物理路径、成本与查询在重复间完全一致，但FRAME四月控制时间76.28/44.73秒；无搜索五月FIRST_SAFE_BEST为13.43/6.84秒；十一月NO_LOCAL为88.31/62.92秒。时间变异包含连续优化、编译和CPU时间，不能仅归咎操作系统待机或GNN。无电源事件，不代表不存在所有计算环境变化。', '',
        '论文可保留“相对FULL的有限质量—计算改善”，不写“全面胜过成熟停止策略”或“补全普遍必要”。不自动新增重复、改参数或换日期。']
    (OUT/'TIMING_AND_CLAIMS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'active': active, 'inactive': inactive, 'max_same_path_time_ratio': max(x['control_ratio_r0_r1'] for x in pairs)}))


if __name__ == '__main__':
    main()
