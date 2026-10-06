"""Reconcile BC05 stats and export the four-arm auditable result."""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
WORK = HERE
OUT = BASE/'outputs'
REPO = BASE/'work/bc01/Baleen-FAST24'
sys.path.insert(0, str(REPO))
import compress_json

CHUNK = 128*1024
SAMPLE_RATIO = .1
EVAL_START_INDEX = 144
WRITE_CAP = 148731
CAPACITY = 3002
ARMS = ['BASE-CONTROL', 'BASE-CF', 'HIGH-CONTROL', 'HIGH-CF']
FIELDS = ['time_elapsed_phy', 'time_log', 'service_time_used_stats', 'service_time_writes_stats',
          'service_time_used_demand_stats', 'service_time_used_prefetch_stats', 'service_time_nocache_stats',
          'flashcache/keys_written_stats', 'flashcache/prefetches_stats', 'fetches_chunks_prefetch_stats',
          'fetches_chunks_demandmiss_stats', 'fetches_ios_stats', 'iops_requests_stats',
          'chunk_queries_stats', 'puts_ios_stats', 'puts_chunks_stats']
ST_TOL = 1e-9
DT_TOL = 1e-8


def required(batches, key):
    if key not in batches:
        raise KeyError('mandatory stats field missing: '+key)
    arr = np.asarray(batches[key], dtype=float)
    assert arr.shape == (1008,) and np.isfinite(arr).all(), key
    return arr


def delta(a):
    return np.diff(a, prepend=0.)


def dump_csv(path, rows):
    if not rows:
        raise ValueError(f'no rows for {path}')
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        fields = list(rows[0])
        for row in rows[1:]:
            for key in row:
                if key not in fields:
                    fields.append(key)
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def one_stats(arm):
    found = list((WORK/arm/'raw').rglob('*.stats.lzma'))
    assert len(found) == 1, (arm, found)
    return found[0]


def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((WORK/'frozen_manifest.json').read_text(encoding='utf-8'))
    preflight = json.loads((WORK/'preflight.json').read_text(encoding='utf-8'))
    status = json.loads((WORK/'run_status.json').read_text(encoding='utf-8'))
    verification_path = WORK/'deliverable_verification.json'
    if verification_path.exists():
        verification = json.loads(verification_path.read_text(encoding='utf-8'))
        assert verification['passed']
    else:
        verification = None
    assert preflight['passed'] and status['status'] == 'completed' and status.get('completed') == ARMS
    assert status['budget_seconds'] == 3600 and status['wall_seconds'] <= 3600
    assert manifest['targets_sha256'] == status['frozen_targets_sha256']
    targets = json.loads((WORK/'frozen_targets.json').read_text(encoding='utf-8'))
    assert len(targets) == manifest['targets_count'] == 1489

    windows = []
    summaries = {}
    audits = {}
    arrays_by_arm = {}
    baseline = None
    invariant_keys = ['time_elapsed_phy', 'time_log', 'service_time_writes_stats',
                      'service_time_nocache_stats', 'iops_requests_stats', 'chunk_queries_stats',
                      'puts_ios_stats', 'puts_chunks_stats']
    for arm in ARMS:
        path = one_stats(arm)
        batches = compress_json.load(str(path))['batches']
        arrays = {key: required(batches, key) for key in FIELDS}
        arrays_by_arm[arm] = arrays
        assert path.with_name(path.name.replace('.stats.lzma', '.lzma')).exists()
        raw = compress_json.load(str(path.with_name(path.name.replace('.stats.lzma', '.lzma'))))
        assert raw['sampleRatio'] == SAMPLE_RATIO and raw['chunkSize'] == CHUNK and raw['results']['NumCacheElems'] == CAPACITY
        elapsed = arrays['time_elapsed_phy']
        duration = delta(elapsed)
        assert np.all(duration > 0) and elapsed[EVAL_START_INDEX-1] == 86401.23277902603
        gets = delta(arrays['service_time_used_stats'])
        puts = delta(arrays['service_time_writes_stats'])
        demand = delta(arrays['service_time_used_demand_stats'])
        prefetch_st = delta(arrays['service_time_used_prefetch_stats'])
        allocation_error = float(np.max(np.abs(gets-demand-prefetch_st)))
        assert allocation_error <= ST_TOL, (arm, allocation_error)
        total_dt = (gets+puts)/(36*.001*duration)*100
        writes = delta(arrays['flashcache/keys_written_stats'])
        pf_writes = delta(arrays['flashcache/prefetches_stats'])
        pf_reads = delta(arrays['fetches_chunks_prefetch_stats'])
        demand_reads = delta(arrays['fetches_chunks_demandmiss_stats'])
        for name, arr in [('writes', writes), ('pf_writes', pf_writes), ('pf_reads', pf_reads), ('demand_reads', demand_reads)]:
            assert np.all(arr >= 0) and np.array_equal(arr, np.rint(arr)), (arm, name)
        assert np.all(pf_writes <= writes)
        peak_i = EVAL_START_INDEX + int(np.argmax(total_dt[EVAL_START_INDEX:]))
        eval_writes = int(writes[EVAL_START_INDEX:].sum())
        avg_dt = float((gets[EVAL_START_INDEX:].sum()+puts[EVAL_START_INDEX:].sum()) /
                       (36*.001*duration[EVAL_START_INDEX:].sum())*100)
        intervention = json.loads((WORK/arm/'intervention_audit.json').read_text(encoding='utf-8'))
        run_audit = json.loads((WORK/arm/'run_audit.json').read_text(encoding='utf-8'))
        assert run_audit['completed'] and intervention['target_request_sequences_seen'] == len(set(t['reject_request_sequence'] for t in targets))
        assert intervention['max_cache_entries_observed_at_request_boundaries'] <= CAPACITY
        target_events = intervention['target_events']
        assert len(target_events) == len(targets)
        target_statuses = Counter(e['status'] for e in target_events)
        assert sum(target_statuses.values()) == 1489
        assert intervention['effective_actual_accepts'] == run_audit.get('effective_actual_accepts', intervention['effective_actual_accepts'])
        assert intervention['parent_actual_accepts'] + intervention['parent_actual_rejects'] > 0
        assert intervention['effective_actual_accepts'] + intervention['effective_actual_rejects'] == \
               intervention['parent_actual_accepts'] + intervention['parent_actual_rejects']
        if arm.endswith('CONTROL'):
            gate = json.loads((WORK/arm/'control_reproduction_gate.json').read_text(encoding='utf-8'))
            assert gate['passed']
            assert intervention['target_status_counts'].get('overridden', 0) == 0
        else:
            parent_arm = 'BASE-CONTROL' if arm == 'BASE-CF' else 'HIGH-CONTROL'
            gate = json.loads((WORK/(arm+'_preintervention_gate.json')).read_text(encoding='utf-8'))
            assert gate['passed']
            parent_arrays = arrays_by_arm.get(parent_arm)
            if parent_arrays is not None:
                for key in invariant_keys:
                    err = float(np.max(np.abs(arrays[key]-parent_arrays[key])))
                    assert err <= (ST_TOL if key.startswith('service_time') else 0), (arm, key, err)
        statuses_path = WORK/arm/'intervention_audit.json'
        audits[arm] = {
            'stats_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'run_wall_seconds': status['arms'][arm]['wall_seconds'],
            'request_cost_max_error_s': max((abs(x['total_get_service_time_s']-x['demand_service_time_s']-x['extra_prefetch_service_time_s'])
                                             for x in intervention['request_costs']), default=0.),
            'target_status_counts': dict(target_statuses),
            'parent_actual_accepts': intervention['parent_actual_accepts'],
            'parent_actual_rejects': intervention['parent_actual_rejects'],
            'effective_actual_accepts': intervention['effective_actual_accepts'],
            'effective_actual_rejects': intervention['effective_actual_rejects'],
            'check_only_items_not_counted_as_actual': intervention['check_only_items_not_counted_as_actual'],
            'max_cache_entries_observed': intervention['max_cache_entries_observed_at_request_boundaries'],
            'pre_first_target_state_sha256': intervention['pre_first_target_state_sha256'],
            'pre_first_target_component_hashes': intervention['pre_first_target_state_hashes'],
            'request_cost_record_count': len(intervention['request_costs']),
        }
        summaries[arm] = {
            'arm': arm, 'parent_policy': intervention['parent'],
            'counterfactual': intervention['counterfactual'], 'evaluation_write_chunks': eval_writes,
            'evaluation_write_bytes_sample': eval_writes*CHUNK,
            'resource_eligible': eval_writes <= WRITE_CAP and intervention['max_cache_entries_observed_at_request_boundaries'] <= CAPACITY,
            'write_cap_chunks': WRITE_CAP, 'evaluation_peak_dt_pct': float(total_dt[peak_i]),
            'peak_window': peak_i, 'evaluation_average_dt_pct_time_weighted': avg_dt,
            'window_578_dt_pct': float(total_dt[578]), 'window_578_get_dt_pct': float(gets[578]/(36*.001*duration[578])*100),
            'evaluation_prefetch_read_chunks': int(pf_reads[EVAL_START_INDEX:].sum()),
            'evaluation_prefetch_read_bytes_sample': int(pf_reads[EVAL_START_INDEX:].sum())*CHUNK,
            'evaluation_prefetch_read_gib_sample': int(pf_reads[EVAL_START_INDEX:].sum())*CHUNK/2**30,
            'evaluation_prefetch_write_chunks': int(pf_writes[EVAL_START_INDEX:].sum()),
            'evaluation_demand_read_chunks': int(demand_reads[EVAL_START_INDEX:].sum()),
            'evaluation_windows': 864, 'run_seconds': status['arms'][arm]['wall_seconds'],
            'max_cache_entries_observed': intervention['max_cache_entries_observed_at_request_boundaries'],
            'parent_actual_accepts': intervention['parent_actual_accepts'],
            'parent_actual_rejects': intervention['parent_actual_rejects'],
            'effective_actual_accepts': intervention['effective_actual_accepts'],
            'effective_actual_rejects': intervention['effective_actual_rejects'],
            'actual_rejection_change': intervention['effective_actual_rejects']-intervention['parent_actual_rejects'],
            'actual_acceptance_change': intervention['effective_actual_accepts']-intervention['parent_actual_accepts'],
            'intervention_status_counts_json': json.dumps(dict(target_statuses), ensure_ascii=False, sort_keys=True),
        }
        for i in range(1008):
            windows.append({'arm': arm, 'window_index': i,
                            'phase': 'evaluation' if i >= EVAL_START_INDEX else 'warmup',
                            'window_duration_s': float(duration[i]),
                            'get_dt_including_prefetch_pct': float(gets[i]/(36*.001*duration[i])*100),
                            'put_dt_pct': float(puts[i]/(36*.001*duration[i])*100),
                            'total_dt_pct': float(total_dt[i]),
                            'demand_service_time_s': float(demand[i]),
                            'prefetch_extra_service_time_s': float(prefetch_st[i]),
                            'put_service_time_s': float(puts[i]),
                            'write_chunks': int(writes[i]), 'prefetch_write_chunks': int(pf_writes[i]),
                            'prefetch_read_chunks': int(pf_reads[i]), 'demand_read_chunks': int(demand_reads[i])})
        if baseline is None:
            baseline = total_dt
        if arm in ('BASE-CF', 'HIGH-CF'):
            summaries[arm]['peak_gain_vs_BASE'] = float((summaries['BASE-CONTROL']['evaluation_peak_dt_pct']-summaries[arm]['evaluation_peak_dt_pct']) /
                                                        summaries['BASE-CONTROL']['evaluation_peak_dt_pct'])
        if arm.endswith('-CF'):
            par = 'BASE-CONTROL' if arm.startswith('BASE') else 'HIGH-CONTROL'
            summaries[arm]['peak_gain_vs_parent'] = float((summaries[par]['evaluation_peak_dt_pct']-summaries[arm]['evaluation_peak_dt_pct']) /
                                                          summaries[par]['evaluation_peak_dt_pct'])
            summaries[arm]['writes_change_vs_parent_chunks'] = eval_writes-summaries[par]['evaluation_write_chunks']
            summaries[arm]['average_dt_change_vs_parent_pct'] = avg_dt-summaries[par]['evaluation_average_dt_pct_time_weighted']
        if arm == 'BASE-CONTROL':
            for key in invariant_keys:
                baseline_array = arrays[key]
                assert len(baseline_array) == 1008
        if arm != 'BASE-CONTROL':
            base_arr = arrays_by_arm['BASE-CONTROL']
            for key in invariant_keys:
                err = float(np.max(np.abs(arrays[key]-base_arr[key])))
                assert err <= (ST_TOL if key.startswith('service_time') else 0), (arm, key, err)

    assert summaries['BASE-CONTROL']['evaluation_write_chunks'] == WRITE_CAP
    assert abs(summaries['BASE-CONTROL']['evaluation_peak_dt_pct']-39.757485419) < 1e-6
    for arm in ARMS:
        summaries[arm]['peak_change_vs_BASE_pct'] = 100*(summaries[arm]['evaluation_peak_dt_pct'] /
                                                        summaries['BASE-CONTROL']['evaluation_peak_dt_pct']-1)
        summaries[arm]['average_change_vs_BASE_pct'] = 100*(summaries[arm]['evaluation_average_dt_pct_time_weighted'] /
                                                            summaries['BASE-CONTROL']['evaluation_average_dt_pct_time_weighted']-1)

    peak_request_rows = []
    intervention_request_rows = []
    event_rows = []
    for arm in ARMS:
        audit = json.loads((WORK/arm/'intervention_audit.json').read_text(encoding='utf-8'))
        for row in audit['request_costs']:
            row = dict(row)
            if row['role'] == 'intervention_request':
                intervention_request_rows.append(row)
            else:
                peak_request_rows.append(row)
        for target, state in zip(targets, audit['target_events']):
            assert target['event_id'] == state['event_id']
            row = {k: v for k, v in target.items() if k != 'full_block_key'}
            row.update({'arm': arm, 'full_block_key_json': json.dumps(target['full_block_key'], ensure_ascii=False),
                        'key_json': state['key'], 'status': state['status'],
                        'cached_before_rejection_request': state['cached_before_request'],
                        'pending_before_rejection_request': state['pending_before_request'],
                        'candidate_inserted_at_rejection_request': state['candidate_inserted_at_request'],
                        'parent_actual_decision': state['parent_decision'],
                        'intervention_changed_reject_to_accept': state['intervention_changed'],
                        'write_was_forced_by_intervention': state.get('write_was_forced', False),
                        'write_success': state['write_success'],
                        'write_request_sequence': state['write_request_sequence'],
                        'evicted_after_write': state['evicted_after_write'],
                        'eviction_request_sequence': state['eviction_request_sequence'],
                        'evicted_before_corresponding_peak_request': state['evicted_before_peak_request'],
                        'corresponding_peak_chunk_missing': state['peak_chunk_missing'],
                        'corresponding_peak_request_demand_service_time_s': state.get('peak_request_demand_service_time_s'),
                        'corresponding_peak_request_prefetch_service_time_s': state.get('peak_request_prefetch_service_time_s')})
            event_rows.append(row)
    dump_csv(OUT/'BC05_four_arm_summary.csv', [summaries[a] for a in ARMS])
    dump_csv(OUT/'BC05_all_windows.csv', windows)
    dump_csv(OUT/'BC05_target_events.csv', event_rows)
    dump_csv(OUT/'BC05_intervention_request_costs.csv', intervention_request_rows)
    dump_csv(OUT/'BC05_peak_request_costs.csv', peak_request_rows)

    target_coverage = {arm: audits[arm]['target_status_counts'] for arm in ARMS}
    cf_pass = {arm: summaries[arm]['resource_eligible'] and summaries[arm]['peak_gain_vs_BASE'] >= .05
               for arm in ['BASE-CF', 'HIGH-CF']}
    if any(cf_pass.values()):
        verdict = '至少一条反事实臂在容量与写入预算内，且完整评价段峰值相对 BASE-CONTROL 下降至少 5%；这是使用未来峰值信息的可实现见证，尚非在线可部署收益。'
    elif any(summaries[a]['peak_gain_vs_BASE'] >= .05 for a in ['BASE-CF', 'HIGH-CF']):
        verdict = '至少一条反事实臂降峰达到 5%，但资源资格不合格；本轮没有预算内见证。'
    else:
        verdict = '两条冻结反事实臂均未达到相对 BASE-CONTROL 的 5% 全段降峰门槛；关闭这份冻结干预方案，不外推到全部对象级准入方法。'
    report = [
        '# BC05：峰前历史拒绝反事实回放', '',
        '## 结果', '',
        '| 方法 | 评价段写入 chunks / 抽样逻辑 GiB | 写入合格 | 全段峰值 DT | 峰值窗口 | 平均 DT | 评价段预取读取 chunks / GiB | 窗口 578 DT | 回放秒数 |',
        '| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        s = summaries[arm]
        report.append(f"| {arm} | {s['evaluation_write_chunks']:,} / {s['evaluation_write_bytes_sample']/2**30:.6f} | {'是' if s['resource_eligible'] else '否'} | {s['evaluation_peak_dt_pct']:.6f}% | {s['peak_window']} | {s['evaluation_average_dt_pct_time_weighted']:.6f}% | {s['evaluation_prefetch_read_chunks']:,} / {s['evaluation_prefetch_read_gib_sample']:.6f} | {s['window_578_dt_pct']:.6f}% | {s['run_seconds']:.2f} |")
    report += ['', '主比较以完整评价段峰值为准：', '']
    report.append('| 反事实臂 | 相对自身控制峰值变化 | 相对 BASE-CONTROL 峰值变化 | 预算内 | 5% 主门槛 | 写入变化 vs 自身控制 | 平均 DT 变化 vs 自身控制 |')
    report.append('| --- | ---: | ---: | :---: | :---: | ---: | ---: |')
    for arm in ['BASE-CF', 'HIGH-CF']:
        s = summaries[arm]
        report.append(f"| {arm} | {100*s['peak_gain_vs_parent']:+.4f}% | {100*s['peak_gain_vs_BASE']:+.4f}% | {'是' if s['resource_eligible'] else '否'} | {'是' if cf_pass[arm] else '否'} | {s['writes_change_vs_parent_chunks']:+,} | {s['average_dt_change_vs_parent_pct']:+.6f} pp |")
    report += ['', f"冻结清单 SHA-256：`{manifest['targets_sha256']}`，共 {manifest['targets_count']:,} 项。BC04 峰值窗口的 1,612 个去重拒绝事件按规定分区为：峰前 {manifest['source_partition_unique_events']['before_peak']:,}、窗口内 {manifest['source_partition_unique_events']['peak']:,}、预热 {manifest['source_partition_unique_events']['warmup']:,}；只干预峰前 1,489 项。", '',
               '窗口 578 的 DT 与完整评价段峰值都报告；5% 判据只使用完整评价段的最大值。写入预算是事后资格筛选，没有引入限流器。BASE-CONTROL 对账 BC03 STATIC-BASE，HIGH-CONTROL 对账 BC03 STATIC-HIGH；反事实臂第一次计划干预请求之前的完整行为状态分别与父控制相同。', '',
               '## 触发覆盖与对象后续状态', '',
               '| 方法 | 实际改变拒绝 | 改变后实际写入 | 改变但未写入 | 父策略接受 | 父策略仍拒绝 | 已在缓存 | 请求前待处理但未决策 | 候选延后 | 无实际候选 | 写入后、峰值请求前被淘汰 | 对应峰值请求仍缺失 / 命中 |',
               '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        c = Counter(target_coverage[arm])
        ev = audits[arm]
        hit = sum(not x['corresponding_peak_chunk_missing'] for x in event_rows if x['arm'] == arm)
        total = sum(x['arm'] == arm for x in event_rows)
        report.append(f"| {arm} | {c['overridden']+c['overridden_no_write']:,} | {sum(1 for x in event_rows if x['arm']==arm and x['write_was_forced_by_intervention'] and x['write_success']):,} | {c['overridden_no_write']:,} | {c['parent_accept_unchanged']:,} | {c['parent_reject_unchanged']:,} | {c['already_present']:,} | {c['already_pending_no_decision']:,} | {c['candidate_deferred']:,} | {c['no_actual_candidate']:,} | {sum(1 for x in event_rows if x['arm']==arm and x['evicted_before_corresponding_peak_request']):,} | {total-hit:,} / {hit:,} |")
    report += ['', '按请求而非 chunk 计列需求读取与额外预取成本，见 `BC05_intervention_request_costs.csv` 和 `BC05_peak_request_costs.csv`。同一请求涉及多个目标 chunk 时，整次 seek 成本不会拆分到 chunk；目标事件 CSV 保留 chunk 的历史和后续缓存事实。目标请求成本不能跨目标相加为净收益，峰值、写入、PUT 与预取变化均以全程统计为准。', '',
               '## 审计与裁决', '',
               f"- 四次回放总耗时 {status['wall_seconds']:.2f} 秒，预算 {status['budget_seconds']} 秒；只执行冻结的四臂。", 
               '- 容量在每次请求边界及实际写入后检查，始终不超过 3,002 chunks；写入合格以评价段累计写入不超过 148,731 chunks 判断。',
               '- 两个控制臂逐窗口复现 BC03；计入用户规定的完整统计字段。反事实臂在首次目标请求前与各自控制臂的缓存、待处理队列、模型历史、预取状态、模拟器时间和累计计数摘要逐组件相同。',
               '- 决策计数同时保存干预前父策略决定与干预后实际决定；`check_only` 资格查询未被翻转，也不并入实际接受／拒绝计数。',
               '- 源码、轨迹与作者模型通过冻结哈希检查；本轮没有拟合新模型。',
               '- 已从导出的窗口 CSV 重新计算峰值、加权平均、写入和预取读取总数；逐请求服务成本相加与总读服务成本一致。' if verification and verification['passed'] else '- 独立导出复核尚未完成。',
               f"- 运行管理器曾因读取写入中的 progress.json 退出；最后一臂子进程随后完成，没有重跑。四个子进程从首臂开始到末臂完成共 {status['wall_seconds']:.2f} 秒，仍在 3,600 秒预算内；输出整理约在末臂完成 {status.get('post_completion_finalize_delay_seconds',0):.2f} 秒后完成，未追加回放。",
               f"- **裁决：** {verdict}", '',
               '## 文件', '',
               '- `BC05_four_arm_summary.csv`：主汇总表。',
               '- `BC05_all_windows.csv`：四臂 1,008 个窗口的完整 DT、GET、PUT、预取和写入序列。',
               '- `BC05_target_events.csv`：每项干预在四臂的触发、父决策、写入、淘汰和峰值命中状态。',
               '- `BC05_intervention_request_costs.csv`、`BC05_peak_request_costs.csv`：按请求计费，不拆 seek。',
               '- `BC05_audit.json`：输入哈希、复现误差、计量误差、资源条件和裁决。']
    (OUT/'BC05_report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    audit = {'protocol_passed': True, 'run_status': status, 'preflight': preflight,
             'frozen_manifest': manifest, 'summaries': summaries, 'intervention_status': target_coverage,
             'audits': audits, 'counterfactual_budget_pass': cf_pass, 'verdict': verdict,
             'all_windows': {'rows': len(windows), 'windows_per_arm': 1008},
             'target_event_rows': len(event_rows), 'intervention_request_cost_rows': len(intervention_request_rows),
             'peak_request_cost_rows': len(peak_request_rows),
             'independent_deliverable_verification': verification,
             'service_time_tolerance_s': ST_TOL, 'dt_tolerance_percentage_points': DT_TOL,
             'implementation_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in WORK.glob('*.py')},
             'deliverable_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in OUT.glob('BC05_*') if p.is_file() and p.name != 'BC05_audit.json'}}
    (OUT/'BC05_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    audit['deliverable_sha256']['BC05_audit.json'] = hashlib.sha256((OUT/'BC05_audit.json').read_bytes()).hexdigest()
    print(json.dumps({'passed': True, 'verdict': verdict, 'summaries': summaries,
                      'counterfactual_budget_pass': cf_pass}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
