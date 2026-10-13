"""Reuse qualified saved-data audit and apply the frozen date-level screen."""
import contextlib
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid62'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
mean = lambda v: math.fsum(v) / len(v) if v else None


def main():
    d = read(OUT / 'design.json')
    assert len(d['weeks']) == 3 and len(d['execution_order']) == 24
    source_path = ROOT / 'work/grid61/audit.py'
    source = source_path.read_text(encoding='utf-8')
    source = source.replace('outputs/grid61', 'outputs/grid62')
    source = source.replace("len(F['summaries'])==32", "len(F['summaries'])==24")
    source = source.replace('len(first_time_ratios)==8', 'len(first_time_ratios)==6')
    source = source.replace("'runs':32", "'runs':24").replace('}/8|', '}/6|')
    source = source.replace('GRID61', 'GRID62').replace('Four olddevelopment families', 'Three fresh families')
    scope = {'__file__': str(source_path), '__name__': 'grid62_inherited_saved_audit'}
    exec(compile(source, str(source_path), 'exec'), scope)
    with contextlib.redirect_stdout(io.StringIO()):
        scope['main']()
    r = read(OUT / 'review.json')
    (OUT / 'inherited_audit.json').write_text(json.dumps(r, indent=2), encoding='utf-8')
    f = read(OUT / 'finished.json')
    ix = {(s['repeat'], s['week'], s['method']): s for s in f['summaries']}
    family_rows = []
    for week in d['weeks']:
        full = [ix[rep, week, 'FULL'] for rep in (0, 1)]
        for method in ('FULL', 'FIRST_SAFE_BEST', 'NO_LOCAL', 'FRAME'):
            runs = [ix[rep, week, method] for rep in (0, 1)]
            equal = all(a['steps'] == b['steps'] and a['complete'] == b['complete'] for a, b in zip(runs, full))
            complete = all(s['complete'] for s in runs)
            full_complete = all(s['complete'] for s in full)
            survival = all(a['complete'] if b['complete'] else a['steps'] >= b['steps'] for a, b in zip(runs, full))
            cost = (mean([s['cost'] for s in runs]) - mean([s['cost'] for s in full])) / abs(mean([s['cost'] for s in full])) if complete and full_complete else None
            controller = mean([s['controller_s'] for s in runs])
            queries = mean([s['n1_forecasts'] for s in runs])
            full_queries = mean([s['n1_forecasts'] for s in full])
            family_rows.append({'week': week, 'method': method, 'complete': complete, 'steps': runs[0]['steps'],
                'no_earlier_failure': survival, 'equal_horizon_full': equal, 'cost': mean([s['cost'] for s in runs]),
                'cost_delta_full': cost, 'quality_pass': bool(survival and (cost is None or cost <= d['quality_threshold'])),
                'controller_s': controller, 'n1_queries': queries, 'n1_source_s': mean([s['n1_source_s'] for s in runs]),
                'control_reduction_full': 1 - controller / mean([s['controller_s'] for s in full]) if equal else None,
                'query_reduction_full': 1 - queries / full_queries if equal and full_queries else None,
                'decision_p95_s': mean([s['decision_p95_s'] for s in runs]) if all('decision_p95_s' in s for s in runs) else None})
    summaries = []
    for method in ('FULL', 'FIRST_SAFE_BEST', 'NO_LOCAL', 'FRAME'):
        a = [x for x in family_rows if x['method'] == method]
        summaries.append({'method': method, 'families': len(a), 'runs': 2 * len(a),
            'completed_families': sum(x['complete'] for x in a), 'completed': sum(ix[rep, w, method]['complete'] for rep in (0, 1) for w in d['weeks']),
            'quality_families': sum(x['quality_pass'] for x in a), 'quality_pass': 2 * sum(x['quality_pass'] for x in a),
            'macro_control_reduction_full': mean([x['control_reduction_full'] for x in a if x['control_reduction_full'] is not None]),
            'macro_query_reduction_full': mean([x['query_reduction_full'] for x in a if x['query_reduction_full'] is not None]),
            'worst_complete_cost_delta_full': max((x['cost_delta_full'] for x in a if x['cost_delta_full'] is not None), default=None),
            'pooled_controller_s': math.fsum(x['controller_s'] for x in a)})
    frames = [x for x in family_rows if x['method'] == 'FRAME']
    frame_summary = next(x for x in summaries if x['method'] == 'FRAME')
    frame_quality = all(x['quality_pass'] for x in frames)
    completed_comparable = sum(x['complete'] and x['equal_horizon_full'] for x in frames)
    positive_dates = sum((x['control_reduction_full'] or 0) > 0 for x in frames)
    speed = frame_summary['macro_control_reduction_full']
    if not frame_quality:
        verdict = 'FRAME_QUALITY_FAILED'
    elif completed_comparable < d['minimum_comparable_dates']:
        verdict = 'INSUFFICIENT_COMPLETE_HORIZONS'
    elif speed is not None and speed >= d['speed_threshold'] and positive_dates >= d['positive_dates_required']:
        verdict = 'FRAMEWORK_CONFIRMED_IN_LIMITED_FRESH_MATRIX'
    else:
        verdict = 'WEAK_OR_MIXED_CONFIRMATION'
    first_ratios = []
    for w in d['weeks']:
        a = next(x for x in family_rows if x['week'] == w and x['method'] == 'FIRST_SAFE_BEST')
        b = next(x for x in family_rows if x['week'] == w and x['method'] == 'FRAME')
        if a['steps'] == b['steps'] and a['complete'] == b['complete']:
            first_ratios.append(a['controller_s'] / b['controller_s'])
    first_competitive = all(x['quality_pass'] for x in family_rows if x['method'] == 'FIRST_SAFE_BEST') and len(first_ratios) == 3 and mean(first_ratios) <= 1.02
    r.update(verdict=verdict, mechanism_verdict=r['verdict'], summaries=summaries, family_rows=family_rows,
        frame_quality_passed=frame_quality, completed_comparable_families=completed_comparable,
        positive_control_families=positive_dates, first_safe_competitive=first_competitive,
        first_safe_mean_time_ratio_frame=mean(first_ratios),
        warning='Three remaining fresh date families on one grid, two timing repeats. No balanced seasonal coverage, independent grids, equivalence test, or independent AC re-solve. Same public frequency ranking and shared caches; no GNN/RL training. First-safe is a strengthened search-policy adaptation, not published-agent reproduction.')
    # All heavy receipts remain local; recording hashes does not rerun physics.
    payloads = []
    for p in sorted((OUT / 'runs').rglob('*')):
        if p.is_file():
            payloads.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    (OUT / 'payload_manifest.json').write_text(json.dumps(payloads, indent=2), encoding='utf-8')
    (OUT / 'review.json').write_text(json.dumps(r, indent=2, allow_nan=False), encoding='utf-8')
    with (OUT / 'per_family.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    lines = ['# GRID62 最后三个封存日期的框架确认', '', '裁决：' + verdict + '。', '',
        r['warning'], '', '|策略|完成日期|质量通过日期|控制减少/FULL|查询减少/FULL|最坏完整周成本变化/FULL|', '|---|---:|---:|---:|---:|---:|']
    fmt = lambda x: '不可比' if x is None else f'{x:+.3%}'
    for x in summaries:
        lines.append(f"|{x['method']}|{x['completed_families']}/3|{x['quality_families']}/3|{fmt(x['macro_control_reduction_full'])}|{fmt(x['macro_query_reduction_full'])}|{fmt(x['worst_complete_cost_delta_full'])}|")
    lines += ['', f"FRAME正向控制日期{positive_dates}/3，完整同视野日期{completed_comparable}/3；首个安全候选参照具有竞争力：{first_competitive}。", '',
        f"核对保存物理向量{r['physical_rows']}步、匹配合法反馈{r['matched_feedback']}项，{r['exact_repeat_paths']}/12重复路径一致。没有独立AC重求解。", '',
        '结果按日期平均两个计时重复后计算比例；重复不是独立日期。0.5%为冻结投资筛查，不是数学等价或电网安全保证。', '',
        '没有按结果调整模型、日程、截止或日期。三日期是全部剩余封存池；不能与旧开发/旧八日期简单合并成新独立样本。全部原始失败、较差和较慢结果可在per_run.csv及per_family.csv查到。']
    (OUT / 'GRID62_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'RUN_STATE.md').write_text('COMPLETE AND AUDITED:' + verdict + '; all frozen runs retained, no new training.\n', encoding='utf-8')
    print(json.dumps({'verdict': verdict, 'runs': 24, 'summaries': summaries, 'first_safe_competitive': first_competitive}))


if __name__ == '__main__':
    main()
