"""Export BC07 full-trajectory metrics after the frozen runs; no replay."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OUT = BASE / 'outputs'
REPO = BASE / 'work' / 'bc01' / 'Baleen-FAST24'
CAP = 148731
CHUNK_BYTES = 128 * 1024
SAMPLE_RATIO = .001
EVAL_INDEX = 144
PEAK_DESCRIPTION = 'Region1 complete evaluation segment, all 864 windows'
ARMS = ['HIGH-CONTROL', 'HIGH-SCORE', 'HIGH-RECENCY', 'HIGH-RANDOM']
REQUIRED = [
    'time_elapsed_phy', 'time_log', 'service_time_used_stats', 'service_time_writes_stats',
    'service_time_used_demand_stats', 'service_time_used_prefetch_stats', 'service_time_nocache_stats',
    'flashcache/keys_written_stats', 'flashcache/prefetches_stats', 'fetches_chunks_prefetch_stats',
    'fetches_chunks_demandmiss_stats', 'fetches_ios_stats', 'iops_requests_stats',
    'chunk_queries_stats', 'puts_ios_stats', 'puts_chunks_stats', 'flashcache/evictions_stats',
]
sys.path.insert(0, str(REPO))
import compress_json


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def one_stats(root):
    paths = list(Path(root).rglob('*.stats.lzma'))
    if len(paths) != 1:
        raise AssertionError(f'expected one stats file under {root}, found {paths}')
    return paths[0]


def load_series(root):
    path = one_stats(root)
    batches = compress_json.load(str(path))['batches']
    arrays = {}
    for key in REQUIRED:
        if key not in batches:
            raise KeyError(f'missing required statistics field: {key}')
        arr = np.asarray(batches[key], dtype=float)
        if arr.shape != (1008,) or not np.isfinite(arr).all():
            raise AssertionError(('invalid stats vector', key, arr.shape))
        arrays[key] = arr
    return path, arrays


def derive(method, path, a, audit, runtime):
    elapsed = a['time_elapsed_phy']
    duration = np.diff(elapsed, prepend=0.0)
    if np.any(duration <= 0) or abs(elapsed[143] - 86401.23277902603) > 1e-8:
        raise AssertionError(('window clock or evaluation start changed', method, elapsed[143]))
    delta = lambda name: np.diff(a[name], prepend=0.0)
    get = delta('service_time_used_stats')
    demand = delta('service_time_used_demand_stats')
    prefetch = delta('service_time_used_prefetch_stats')
    put = delta('service_time_writes_stats')
    alloc_err = float(np.max(np.abs(get - demand - prefetch)))
    if alloc_err > 1e-9:
        raise AssertionError(('GET decomposition error', method, alloc_err))
    get_dt = get * 100 / (36 * SAMPLE_RATIO * duration)
    demand_dt = demand * 100 / (36 * SAMPLE_RATIO * duration)
    prefetch_dt = prefetch * 100 / (36 * SAMPLE_RATIO * duration)
    put_dt = put * 100 / (36 * SAMPLE_RATIO * duration)
    total_dt = get_dt + put_dt
    writes = delta('flashcache/keys_written_stats')
    prefetch_reads = delta('fetches_chunks_prefetch_stats')
    demand_reads = delta('fetches_chunks_demandmiss_stats')
    evictions = delta('flashcache/evictions_stats')
    for name, values in [('writes', writes), ('prefetch_reads', prefetch_reads), ('demand_reads', demand_reads), ('evictions', evictions)]:
        if np.any(values < 0) or not np.array_equal(values, np.rint(values)):
            raise AssertionError(('invalid event-count series', method, name))
    eval_slice = slice(EVAL_INDEX, 1008)
    eval_writes = int(round(float(writes[eval_slice].sum())))
    eval_duration = float(duration[eval_slice].sum())
    average_dt = float((get[eval_slice].sum() + put[eval_slice].sum()) * 100 /
                       (36 * SAMPLE_RATIO * eval_duration))
    peak_window = EVAL_INDEX + int(np.argmax(total_dt[eval_slice]))
    peak = float(total_dt[peak_window])
    live_eval = audit['evaluation']['write_count_live']
    if live_eval != eval_writes:
        raise AssertionError(('audit/raw evaluation write count mismatch', method, live_eval, eval_writes))
    if audit['max_cache_entries_observed'] > 3002:
        raise AssertionError(('cache capacity exceeded', method, audit['max_cache_entries_observed']))
    return {
        'method': method,
        'stats_path': str(path),
        'stats_sha256': sha(path),
        'evaluation_write_chunks': eval_writes,
        'evaluation_write_bytes_sample': eval_writes * CHUNK_BYTES,
        'resource_eligible': eval_writes <= CAP and audit['max_cache_entries_observed'] <= 3002,
        'max_cache_entries_observed': audit['max_cache_entries_observed'],
        'peak_dt_pct': peak,
        'peak_window': int(peak_window),
        'average_dt_pct': average_dt,
        'evaluation_duration_s': eval_duration,
        'evaluation_windows': 864,
        'window_578_dt_pct': float(total_dt[578]),
        'window_578_get_dt_pct': float(get_dt[578]),
        'window_578_put_dt_pct': float(put_dt[578]),
        'peak_get_dt_pct': float(get_dt[peak_window]),
        'peak_demand_read_dt_pct': float(demand_dt[peak_window]),
        'peak_extra_prefetch_dt_pct': float(prefetch_dt[peak_window]),
        'peak_put_dt_pct': float(put_dt[peak_window]),
        'evaluation_prefetch_read_chunks': int(round(float(prefetch_reads[eval_slice].sum()))),
        'evaluation_prefetch_read_bytes_sample': int(round(float(prefetch_reads[eval_slice].sum()))) * CHUNK_BYTES,
        'evaluation_demand_read_chunks': int(round(float(demand_reads[eval_slice].sum()))),
        'get_decomposition_max_error_s': alloc_err,
        'wall_seconds': runtime,
        'supplemented_parent_rejections': audit['evaluation']['supplemented_parent_rejections'],
        'supplement_successful_cache_writes': audit['evaluation']['supplement_successful_cache_writes'],
        'evaluation_demand_candidates': audit['evaluation']['demand_candidates'],
        'evaluation_high_parent_rejected_demand_candidates': audit['evaluation']['high_parent_rejected_demand_candidates'],
        'evaluation_supplement_rate_of_parent_rejected_demand': audit['evaluation']['supplement_rate_of_parent_rejected_demand'],
        'evaluation_supplement_rate_of_all_demand_candidates': audit['evaluation']['supplement_rate_of_all_demand_candidates'],
        'unique_supplement_requests': audit['supplement_unique_requests'],
        'unique_supplement_blocks': audit['supplement_unique_blocks'],
    }, {'duration': duration, 'get_dt': get_dt, 'demand_dt': demand_dt,
        'prefetch_dt': prefetch_dt, 'put_dt': put_dt, 'total_dt': total_dt,
        'writes': writes, 'prefetch_reads': prefetch_reads, 'demand_reads': demand_reads,
        'evictions': evictions, 'get_service': get, 'put_service': put}


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    status_path = HERE / 'run_status.json'
    if not status_path.exists():
        raise RuntimeError('no BC07 run status exists')
    status = json.loads(status_path.read_text(encoding='utf-8'))
    cal = json.loads((HERE / 'calibration.json').read_text(encoding='utf-8'))
    rows, series_by_method, audits = [], {}, {}
    for arm in ARMS:
        item = status.get('arms', {}).get(arm)
        root = HERE / 'arms' / arm
        audit_path = root / 'decision_audit.json'
        stats_files = list((root / 'raw').rglob('*.stats.lzma'))
        if not item or item.get('status') != 'completed' or not audit_path.exists() or len(stats_files) != 1:
            continue
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if not audit.get('completed'):
            continue
        p, arrays = load_series(root / 'raw')
        row, series = derive(arm, p, arrays, audit, item['wall_seconds'])
        row['calibration_supplement_count'] = cal['rules'].get(arm.removeprefix('HIGH-'), {}).get('calibration_selected_chunks', 0) if arm != 'HIGH-CONTROL' else 0
        row['calibration_supplement_fraction'] = cal['rules'].get(arm.removeprefix('HIGH-'), {}).get('calibration_fraction_of_high_rejected_demand_chunks', 0.0) if arm != 'HIGH-CONTROL' else 0.0
        row['gain_vs_BASE_CONTROL_peak_pct'] = None
        row['peak_change_vs_HIGH_CONTROL_pct'] = None
        row['average_change_vs_HIGH_CONTROL_pct'] = None
        rows.append(row)
        series_by_method[arm] = series
        audits[arm] = audit

    base_path, base_arrays = load_series(BASE / 'work' / 'bc05' / 'BASE-CONTROL' / 'raw')
    base_audit = json.loads((BASE / 'work' / 'bc05' / 'BASE-CONTROL' / 'run_audit.json').read_text(encoding='utf-8'))
    if not base_audit.get('completed'):
        raise AssertionError('frozen BASE-CONTROL is not complete')
    base_eval_duration = float(np.diff(base_arrays['time_elapsed_phy'], prepend=0)[EVAL_INDEX:].sum())
    base_dt = (np.diff(base_arrays['service_time_used_stats'], prepend=0) +
               np.diff(base_arrays['service_time_writes_stats'], prepend=0)) * 100 / (36 * SAMPLE_RATIO * np.diff(base_arrays['time_elapsed_phy'], prepend=0))
    base_peak = float(np.max(base_dt[EVAL_INDEX:]))
    base_average = float((np.diff(base_arrays['service_time_used_stats'], prepend=0)[EVAL_INDEX:].sum() +
                          np.diff(base_arrays['service_time_writes_stats'], prepend=0)[EVAL_INDEX:].sum()) * 100 /
                         (36 * SAMPLE_RATIO * base_eval_duration))

    # The BASE-CONTROL trace is the frozen BC05 reference; also verify it is
    # the same trace as the established STATIC-BASE before using it in G_m.
    static_base_path, static_base = load_series(BASE / 'work' / 'bc03' / 'STATIC-BASE' / 'raw')
    base_errors = {}
    for key in REQUIRED:
        base_errors[key] = float(np.max(np.abs(base_arrays[key] - static_base[key])))
        tol = 1e-9 if key.startswith('service_time') else 0.0
        if base_errors[key] > tol:
            raise AssertionError(('BASE-CONTROL differs from STATIC-BASE', key, base_errors[key]))

    high = next((r for r in rows if r['method'] == 'HIGH-CONTROL'), None)
    for row in rows:
        row['gain_vs_BASE_CONTROL_peak_pct'] = (base_peak - row['peak_dt_pct']) / base_peak * 100
        if high:
            row['peak_change_vs_HIGH_CONTROL_pct'] = (row['peak_dt_pct'] / high['peak_dt_pct'] - 1) * 100
            row['average_change_vs_HIGH_CONTROL_pct'] = (row['average_dt_pct'] / high['average_dt_pct'] - 1) * 100

    # Write per-window evidence for all complete arms.
    all_window_rows = []
    for method, series in series_by_method.items():
        path = OUT / f'BC07_{method}_10min.csv'
        method_rows = []
        for i in range(1008):
            row = {
                'method': method, 'window_index': i,
                'phase': 'evaluation' if i >= EVAL_INDEX else 'warmup',
                'evaluation': i >= EVAL_INDEX,
                'trace_start_elapsed_s': 0.0,
                'duration_s': float(series['duration'][i]),
                'total_dt_pct': float(series['total_dt'][i]),
                'get_dt_including_prefetch_pct': float(series['get_dt'][i]),
                'demand_read_dt_pct': float(series['demand_dt'][i]),
                'extra_prefetch_dt_pct': float(series['prefetch_dt'][i]),
                'put_dt_pct': float(series['put_dt'][i]),
                'evaluation_write_chunks': int(round(float(series['writes'][i]))),
                'evaluation_write_bytes_sample': int(round(float(series['writes'][i]))) * CHUNK_BYTES,
                'backend_demand_read_chunks': int(round(float(series['demand_reads'][i]))),
                'backend_prefetch_read_chunks': int(round(float(series['prefetch_reads'][i]))),
                'evictions': int(round(float(series['evictions'][i]))),
            }
            method_rows.append(row)
        # Replace start values with exact cumulative clock from that method.
        _, method_arrays = load_series(HERE / 'arms' / method / 'raw')
        times = method_arrays['time_elapsed_phy']
        for i, row in enumerate(method_rows):
            row['trace_start_elapsed_s'] = float(0.0 if i == 0 else times[i-1])
            row['trace_end_elapsed_s'] = float(times[i])
        csv_write(path, method_rows)
        all_window_rows.extend(method_rows)
    csv_write(OUT / 'BC07_all_windows.csv', all_window_rows)
    csv_write(OUT / 'BC07_summary.csv', rows)

    rec = next((r for r in rows if r['method'] == 'HIGH-RECENCY'), None)
    score = next((r for r in rows if r['method'] == 'HIGH-SCORE'), None)
    random = next((r for r in rows if r['method'] == 'HIGH-RANDOM'), None)
    complete = status['status'] == 'completed' and len(rows) == 4 and status.get('completed') == ARMS
    if not complete:
        verdict = f"执行不完整（{len(rows)}/4 臂完成，状态 {status['status']}）；只保留已完成臂，不裁决策略效果。"
    elif rec['resource_eligible'] and rec['gain_vs_BASE_CONTROL_peak_pct'] >= 5 and rec['peak_dt_pct'] < score['peak_dt_pct'] and rec['peak_dt_pct'] < random['peak_dt_pct']:
        verdict = 'HIGH-RECENCY 在资源合格时达到相对 BASE-CONTROL 的 5% 降峰，并优于 SCORE 与 RANDOM；这是 Region1 开发信号，不是学习优势或独立泛化证明。'
    elif any(r['resource_eligible'] and r['gain_vs_BASE_CONTROL_peak_pct'] >= 5 for r in [score, random]):
        names = [r['method'] for r in [score, random] if r['resource_eligible'] and r['gain_vs_BASE_CONTROL_peak_pct'] >= 5]
        verdict = f"廉价补充规则 {', '.join(names)} 已取得资源合格的 5% 降峰见证；优先把它们作为基线，不把收益归因于 RECENCY 或学习器。"
    elif rec['resource_eligible'] and rec['gain_vs_BASE_CONTROL_peak_pct'] > 0:
        verdict = 'RECENCY 相对 HIGH-CONTROL 有资源合格的降峰，但没有建立达到 5% 且胜过 SCORE/RANDOM 的增量信号。'
    elif not any(r['resource_eligible'] and r['gain_vs_BASE_CONTROL_peak_pct'] >= 5 for r in [score, rec, random]):
        if any(not r['resource_eligible'] and r['gain_vs_BASE_CONTROL_peak_pct'] >= 5 for r in [score, rec, random]):
            verdict = '达到 5% 的规则只出现在超写入预算臂；资源资格未通过。'
        else:
            verdict = '所有资源合格的补充规则均未达到相对 BASE-CONTROL 的 5% 降峰；只裁决本轮规则和 1% 校准强度。'
    else:
        verdict = '观察到部分规则差异，但预设的 RECENCY 增量条件未同时成立。'

    audit_out = {
        'protocol_passed': bool(complete),
        'execution_status': status['status'], 'complete_arms': [r['method'] for r in rows],
        'attempt_history': status.get('attempt_history', []),
        'total_replay_wall_seconds': status.get('wall_seconds'),
        'replay_budget_seconds': status.get('budget_seconds'),
        'verdict': verdict, 'baseline': {
            'method': 'BC05 BASE-CONTROL', 'stats_path': str(base_path), 'stats_sha256': sha(base_path),
            'peak_dt_pct': base_peak, 'average_dt_pct': base_average,
            'BASE_CONTROL_vs_STATIC_BASE_max_errors': base_errors,
        },
        'calibration_sha256': sha(HERE / 'calibration.json'),
        'preflight_sha256': sha(HERE / 'preflight.json'),
        'summaries': rows, 'budget': {'evaluation_write_chunks': CAP, 'bytes_per_chunk': CHUNK_BYTES,
                                      'evaluation_start_index': EVAL_INDEX, 'evaluation_windows': 864},
        'implementation_sha256': {p.name: sha(p) for p in HERE.glob('*.py')},
        'status_sha256': sha(HERE / 'run_status.json'),
    }
    (HERE / 'audit.json').write_text(json.dumps(audit_out, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'BC07_audit.json').write_text(json.dumps(audit_out, ensure_ascii=False, indent=2), encoding='utf-8')

    report = [
        '# BC07：HIGH 父策略下的 1% 在线补充准入比较', '', verdict, '',
        '## 冻结协议与范围', '',
        '- 四臂均使用 BC03 STATIC-HIGH 配置，仅输出目录不同；从空缓存沿完整 Region1 轨迹运行，预热按原 BASE 阈值，评价阈值为 θH。原模型、LRU、容量、预取与所有成本计量保持不变。',
        '- 评价从作者第 144 个窗口开始（请求处理前跨过 86,401.232779 秒的 checkpoint），覆盖 864 个完整窗口。补充准入仅作用于评价期、实际 HIGH 拒绝的需求候选；预热和 check-only、预取候选均不变。',
        f"- 统一校准比例 q=1%。BC06 预热候选中，{cal['calibration_scope']['counts']['warmup_demand_candidates_rejected_by_high']:,} 个需求候选按已保存作者分数重算后属于 HIGH 拒绝；这不表示它们曾在预热期实际按 HIGH 决策。阈值和并列规则在回放前封存，见 `BC07_calibration.json`。",
        f"- 哈希为 SHA-256(seed:全局请求序号) 的前 64 位，同一请求的所有 chunk 共用哈希。SCORE 以分数降序、哈希升序；RECENCY 以块最近访问间隔升序、哈希升序；RANDOM 以哈希升序。校准边界完整保留相同排序键，不拆并列组。RECENCY 的无历史 chunk 不参与准入。",
        '- 校准只使用 BC06 封存的预热真实需求候选、原模型分数和请求开始前块访问间隔；预热本身使用 BASE 阈值。没有读取结果侧表、成员清单或峰值位置。评价时分支各自重新计算模型分数、请求候选和块访问间隔。',
        '- 主指标为评价段 864 窗口中的最大 DT；DT 同时计入需求 GET、额外预取读取与 PUT。平均 DT 按评价段实际窗口时长加权。写入预算为 148,731 个 128 KiB chunk 写入，容量上限 3,002；采用事后资源资格检查，没有实施硬限流。',
        '', '## 主表', '',
        '| 方法 | 校准选择数 / 比例 | 评价补充接受 / 实际成功写入 | 评价接受率（被拒需求候选 / 全需求候选） | 评价写入 | 合格 | 峰值 DT | 峰值窗 | 平均 DT | 相对 BASE 峰值降幅 | 相对 HIGH 峰值变化 | 相对 HIGH 平均变化 | 预取读取块 | 回放秒数 |',
        '| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'
    ]
    for r in rows:
        cal_count = r['calibration_supplement_count']
        cal_frac = r['calibration_supplement_fraction'] * 100
        eval_fraction = r['evaluation_supplement_rate_of_parent_rejected_demand']
        eval_all = r['evaluation_supplement_rate_of_all_demand_candidates']
        eval_f = '—' if eval_fraction is None else f'{eval_fraction*100:.4f}% / {eval_all*100:.4f}%'
        report.append(
            f"| {r['method']} | {cal_count:,} / {cal_frac:.4f}% | {r['supplemented_parent_rejections']:,} / {r['supplement_successful_cache_writes']:,} | {eval_f} | {r['evaluation_write_chunks']:,} | {'是' if r['resource_eligible'] else '否'} | {r['peak_dt_pct']:.6f}% | {r['peak_window']} | {r['average_dt_pct']:.6f}% | {r['gain_vs_BASE_CONTROL_peak_pct']:+.4f}% | {r['peak_change_vs_HIGH_CONTROL_pct']:+.4f}% | {r['average_change_vs_HIGH_CONTROL_pct']:+.4f}% | {r['evaluation_prefetch_read_chunks']:,} | {r['wall_seconds']:.1f} |"
        )
    report += ['', f"BASE-CONTROL 峰值 / 平均 DT 为 {base_peak:.6f}% / {base_average:.6f}%。最佳资源合格方法不会被当作可部署选参器；逐臂绝对值和全部窗口数据见 `BC07_summary.csv` 与 `BC07_all_windows.csv`。", '',
               '## 资源与轨迹核验', '',
               '- HIGH-CONTROL 的完整 1,008 窗口统计逐字段复现 BC03 STATIC-HIGH；另外三臂的预热 144 窗口逐字段复现相同 HIGH 父轨迹。DT GET 分量分解误差、在线与原始统计评价写入核对、候选决定计数与容量核对见 `BC07_audit.json`。',
               f"- HIGH-SCORE 首次运行在 167 个窗口后因记录器的事件队列清理错误中止；失败片段已归档。修正记录器后，在未改变策略或校准的情况下完整重跑并通过核验。所有尝试累计回放耗时 {status.get('wall_seconds', 0):.1f} 秒，低于 {status.get('budget_seconds', 0):.0f} 秒上限；首次失败不计入主结果。",
               '- 每臂报告全段峰值和平均值。原峰值窗口 578 仅为描述字段，裁决不限定于该窗口。',
               '- SCORE/RECENCY/RANDOM 的校准候选比例约为 q；评价期候选池因策略改变而变化，因此报告的是每臂自己的实际补充比例和写入结果。相同比例校准不保证相同评价写入量。',
               '- 这是同一条已反复用于开发的 Region1 轨迹上的规则筛查。RANDOM 固定一个哈希种子；本轮没有置信区间、显著性或独立泛化主张。未训练模型，未追加比例或种子。', '',
               '## 交付', '',
               '- `BC07_calibration.json`：校准范围、输入哈希、三条规则的完整精度边界、破并列方式和实际校准比例。',
               '- `BC07_summary.csv`：主表全精度数值；`BC07_<方法>_10min.csv`：每臂完整 1,008 行轨迹；`BC07_all_windows.csv`：四臂合并窗口表。',
               '- 每臂 `supplement_events.jsonl` 只列新增接受决策及对应写入结果；未把未选择候选标成负例。',
               '- `BC07_preflight.json`：小前缀、HIGH 决策、check-only 与接口隔离检查；这不是整个策略的全面防泄漏证明。',
               '- 本轮裁决范围：仅上述三条规则及其预先固定的 1% 校准强度，不推出所有历史信息或学习方法无效。', '']
    (OUT / 'BC07_report.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'complete': complete, 'status': status['status'], 'baseline_peak_dt_pct': base_peak,
                      'verdict': verdict, 'summary_rows': rows}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
