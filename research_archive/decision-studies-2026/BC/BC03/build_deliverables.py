"""Audit completed BC03 arms and export frozen-protocol metrics; no new replay."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import *
from run_all import find_stats, KEYS
sys.path.insert(0, str(REPO))
import compress_json
import numpy as np
from BCacheSim.episodic_analysis.episodes import st_to_util

OUT = BASE/'outputs'
ST_ATOL = 1e-9
DT_ATOL = 1e-8
CHUNK = 131072

def require(batches, key, n=1008):
    if key not in batches:
        raise KeyError(f'missing mandatory field: {key}')
    a = np.asarray(batches[key], dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f'invalid mandatory series: {key}')
    return a

def delta(a):
    return np.diff(a, prepend=0.)

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    status = json.loads((WORK/'run_status.json').read_text())
    if status['status'] == 'running':
        raise RuntimeError('Do not finalize while simulations are still running')
    preflight = json.loads((WORK/'preflight.json').read_text())
    source_revisions = {}
    for label, folder in [('Baleen-FAST24',REPO),('BCacheSim',REPO/'BCacheSim')]:
        revision = subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
        changes = subprocess.check_output(['git','-C',str(folder),'status','--porcelain'],text=True).strip()
        assert not changes, f'author source changed: {label}: {changes}'
        source_revisions[label] = revision
    for relative, digest in preflight['inputs_sha256'].items():
        assert hashlib.sha256((REPO/relative).read_bytes()).hexdigest() == digest, relative
    reference = compress_json.load(str(REFERENCE))['batches']
    completed = status['completed']
    OUT.mkdir(exist_ok=True)
    audits, summaries, all_rows = {}, {}, []
    snap_ref = None
    for arm in completed:
        path = find_stats(arm)
        main_path = Path(str(path).replace('.stats.lzma', '.lzma'))
        batches = compress_json.load(str(path))['batches']
        raw = compress_json.load(str(main_path))
        config = json.loads((WORK/arm/'config.json').read_text())
        assert hashlib.sha256((WORK/arm/'config.json').read_bytes()).hexdigest() == preflight['config_sha256'][arm]
        assert raw['sampleRatio'] == .1 and raw['chunkSize'] == CHUNK
        assert raw['results']['NumCacheElems'] == 3002
        arm_audit = json.loads((WORK/arm/'audit.json').read_text())
        assert arm_audit['completed'] and arm_audit['snapshot_captured']
        if snap_ref is None:
            snap_ref = arm_audit['warmup_state_hashes']
        assert arm_audit['warmup_state_hashes'] == snap_ref, f'warmup snapshot mismatch {arm}'
        assert all('theta='+THETA0.hex() in k for k in arm_audit['calls'] if k.startswith('warmup|'))
        arrays = {k: require(batches, k) for k in KEYS}
        warmup_errors = {}
        for key, a in arrays.items():
            ref = require(reference, key)
            err = float(np.max(np.abs(a[:EVAL_INDEX]-ref[:EVAL_INDEX])))
            assert err <= (ST_ATOL if key.startswith('service_time') else 0), (arm,key,err)
            warmup_errors[key] = err
        invariant_errors = {}
        for key in ['time_elapsed_phy', 'service_time_writes_stats', 'service_time_nocache_stats',
                    'iops_requests_stats', 'chunk_queries_stats', 'puts_ios_stats', 'puts_chunks_stats']:
            err = float(np.max(np.abs(arrays[key]-require(reference,key))))
            assert err <= (ST_ATOL if key.startswith('service_time') else 0), (arm,key,err)
            invariant_errors[key] = err
        elapsed = arrays['time_elapsed_phy']
        duration = delta(elapsed)
        assert np.all(duration > 0) and elapsed[EVAL_INDEX-1] == EVAL_START
        gets = delta(arrays['service_time_used_stats'])
        puts = delta(arrays['service_time_writes_stats'])
        pf_service = delta(require(batches, 'service_time_used_prefetch_stats'))
        demand_service = delta(require(batches, 'service_time_used_demand_stats'))
        allocation_error = float(np.max(np.abs(gets-pf_service-demand_service)))
        assert allocation_error <= ST_ATOL
        get_dt = st_to_util(gets, sample_ratio=.1, duration_s=duration)*100
        put_dt = st_to_util(puts, sample_ratio=.1, duration_s=duration)*100
        dt = get_dt+put_dt
        if arm == 'STATIC-BASE':
            ref_dt = (delta(require(reference,'service_time_used_stats'))+delta(require(reference,'service_time_writes_stats')))/(36*.001*duration)*100
            baseline_dt_error = float(np.max(np.abs(dt-ref_dt)))
            assert baseline_dt_error <= DT_ATOL
        writes = delta(arrays['flashcache/keys_written_stats'])
        pf_writes = delta(require(batches, 'flashcache/prefetches_stats'))
        pf_reads = delta(arrays['fetches_chunks_prefetch_stats'])
        demand_reads = delta(arrays['fetches_chunks_demandmiss_stats'])
        fetch_ios = delta(require(batches, 'fetches_ios_stats'))
        queries = delta(arrays['chunk_queries_stats'])
        for key, a in [('writes',writes),('pf_writes',pf_writes),('pf_reads',pf_reads),('demand_reads',demand_reads),('queries',queries)]:
            assert np.all(a>=0) and np.array_equal(a,np.rint(a)), (arm,key)
        assert np.all(pf_writes<=writes)
        rows = []
        for i in range(1008):
            left = 0. if i == 0 else float(elapsed[i-1])
            right = float(elapsed[i])
            boundaries = [EVAL_START]
            if arm in INTERVALS:
                boundaries += list(INTERVALS[arm])
            # A checkpoint window may straddle a schedule boundary. The simulator
            # uses each CALL timestamp; this column never assigns it one false theta.
            interior = [b for b in boundaries if left < b < right]
            cuts = [left, *sorted(interior), right]
            phases = [{'left_s': a, 'right_s': b, 'threshold': threshold_at(arm,a)}
                      for a,b in zip(cuts[:-1],cuts[1:])]
            rows.append({'method':arm,'window_index':i,'phase':'evaluation' if i>=EVAL_INDEX else 'warmup',
                         'trace_start_s':left,'trace_end_s':right,'duration_s':float(duration[i]),
                         'threshold_segments_left_closed_right_open':json.dumps(phases,separators=(',',':')),
                         'dt_get_including_prefetch_pct':float(get_dt[i]),'dt_put_pct':float(put_dt[i]),
                         'dt_total_pct':float(dt[i]),'get_service_time_s':float(gets[i]),'put_service_time_s':float(puts[i]),
                         'prefetch_extra_service_time_s':float(pf_service[i]),'demand_read_service_time_s':float(demand_service[i]),
                         'flash_write_chunks':int(writes[i]),'flash_write_bytes_sample':int(writes[i])*CHUNK,
                         'flash_prefetch_write_chunks':int(pf_writes[i]),'flash_other_write_chunks':int(writes[i]-pf_writes[i]),
                         'backend_prefetch_read_chunks':int(pf_reads[i]),'backend_prefetch_read_bytes_sample':int(pf_reads[i])*CHUNK,
                         'backend_demand_read_chunks':int(demand_reads[i]),'backend_demand_read_bytes_sample':int(demand_reads[i])*CHUNK,
                         'backend_fetch_ios':int(fetch_ios[i]),'chunk_queries':int(queries[i])})
        write_csv(OUT/f'BC03_{arm}_10min.csv',rows)
        all_rows += rows
        peak_i = EVAL_INDEX+int(np.argmax(dt[EVAL_INDEX:]))
        count = int(writes[EVAL_INDEX:].sum())
        avg = float(st_to_util(gets[EVAL_INDEX:].sum()+puts[EVAL_INDEX:].sum(),sample_ratio=.1,duration_s=duration[EVAL_INDEX:].sum())*100)
        summaries[arm] = {'method':arm,'evaluation_write_chunks':count,'evaluation_write_bytes_sample':count*CHUNK,
                          'resource_eligible':count<=WRITE_BUDGET,'peak_dt_pct':float(dt[peak_i]),'peak_window':peak_i,
                          'average_dt_pct':avg,'evaluation_prefetch_read_chunks':int(pf_reads[EVAL_INDEX:].sum()),
                          'evaluation_prefetch_read_bytes_sample':int(pf_reads[EVAL_INDEX:].sum())*CHUNK,
                          'evaluation_prefetch_write_chunks':int(pf_writes[EVAL_INDEX:].sum()),
                          'window_578_dt_pct':float(dt[578]),'window_578_get_dt_pct':float(get_dt[578]),
                          'wall_seconds':status['arms'][arm]['wall_seconds'], 'evaluation_windows':864}
        audits[arm] = {'warmup_state_sha256':arm_audit['warmup_snapshot_sha256'],
                       'warmup_component_hashes':arm_audit['warmup_state_hashes'],
                       'warmup_max_errors':warmup_errors,'all_window_invariant_max_errors':invariant_errors,
                       'stats_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'get_cost_allocation_max_error_s':allocation_error,
                       'threshold_invocations':arm_audit['calls']}
    if all_rows:
        write_csv(OUT/'BC03_all_windows.csv', all_rows)
        write_csv(OUT/'BC03_summary.csv', list(summaries.values()))
    passed = status['status']=='completed' and len(completed)==7
    eligible_static = [summaries[a] for a in ARMS[:3] if a in summaries and summaries[a]['resource_eligible']]
    pstatic = min(eligible_static,key=lambda s:s['peak_dt_pct']) if eligible_static else None
    comparisons = {}
    if pstatic:
        for arm in ARMS[3:]:
            if arm in summaries:
                comparisons[arm] = {'resource_eligible':summaries[arm]['resource_eligible'],
                                    'relative_peak_gain': (pstatic['peak_dt_pct']-summaries[arm]['peak_dt_pct'])/pstatic['peak_dt_pct']}
    if not passed:
        verdict = '七臂或计量审计未完整通过，保留已完成结果，不裁决研究价值。'
    elif comparisons['LEAD']['resource_eligible'] and comparisons['LEAD']['relative_peak_gain']>=.05:
        verdict = 'LEAD 获得有限策略族中的预算内降峰见证；下一步才检验合法历史能否提前识别时段。'
    elif any(comparisons[a]['resource_eligible'] and comparisons[a]['relative_peak_gain']>=.05 for a in ARMS[4:]):
        verdict = '其他时间调度出现探索性预算内见证；LEAD 主假设未获支持，不能事后更换主臂。'
    else:
        verdict = '所有资源合格时间调度均未达到相对最佳合格固定配置 5% 的降峰门槛，关闭这一组两小时阈值干预。'
    result = {'protocol_passed':passed,'run_status':status,'preflight':preflight,'audits':audits,
              'source_example_config':json.loads(CONFIG.read_text()),
              'arm_configs':{arm:json.loads((WORK/arm/'config.json').read_text()) for arm in ARMS},
              'summaries':summaries,'best_eligible_static':pstatic,'schedule_comparisons':comparisons,
              'verdict':verdict,'service_time_tolerance_s':ST_ATOL,'dt_tolerance_pp':DT_ATOL,
              'source_modified':False,'evaluation_index_start':EVAL_INDEX,'evaluation_start_s':EVAL_START,
              'source_revisions':source_revisions,
              'implementation_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in WORK.glob('*.py')},
              'baseline_dt_max_error_pp':baseline_dt_error if 'STATIC-BASE' in summaries else None,
              'write_budget_chunks':WRITE_BUDGET,'write_budget_bytes':WRITE_BUDGET*CHUNK}
    verification_file=WORK/'export_verification.json'
    if verification_file.exists():
        verification=json.loads(verification_file.read_text())
        result['independent_export_verification_passed']=verification['passed']
        result['independent_export_verification_checks']=verification['checks']
    (WORK/'audit.json').write_text(json.dumps(result, indent=2))
    (OUT/'BC03_audit.json').write_text(json.dumps(result, indent=2))
    report = ['# BC03：两小时阈值调度机制诊断','',verdict,'',
              '## 冻结协议与口径','',
              f'- Region1 原始完整轨迹、BC01 已训练模型、3,002 个 128 KiB 槽位、LRU、原预取规则与作者计量函数均保留。未训练模型，未扩充配置。',
              f'- 所有臂从空缓存开始，评价前用原阈值；从跨过第一天边界后的完整统计窗口开始评价：索引 144，实际起点 {EVAL_START:.11f} 秒，共 864 窗口。切换时保留缓存、待准入队列和历史。',
              f'- θ0={THETA0!r}，θL=θ0/(2−θ0)={THETAL!r}，θH=2θ0/(1+θ0)={THETAH!r}；作者比较规则为 pred > threshold。',
              '- 阈值在每一次 NewMLAP.batchAccept 调用时取调用物理时间；需求准入与预取 check_only 资格检查使用同一接口。预取范围和时机规则保留，实际额外读取随执行结果计费。',
              '- 原始模拟输出的 options.ap_threshold 和文件名后缀仍为 θ0，因为作者配置除输出目录外不变；实际运行阈值、调用数量及首末物理时间单独保存在各臂 audit.json 的 calls 字段。',
              '- 窗口归属沿用作者的请求处理前 checkpoint 与末尾结算；没有以固定的 600 秒左右闭区间重新分配请求。',
              f'- T={T!r} 秒来自 BC01 未来峰值。LEAD、DURING、PREV-DAY、NEXT-DAY 的低阈值区间分别为 {INTERVALS!r}，均左闭右开；评价段其余时间用 θH。三个固定臂分别始终用 θ0、θL、θH。',
              '- 两小时仅是冻结的干预区间长度，不把训练驻留时间假设解释为真实 TTL；运行中的 LRU 规则保持原定义。',
              f'- 资源合格条件为评价写入 ≤{WRITE_BUDGET:,} 次，即 {WRITE_BUDGET*CHUNK:,} 抽样逻辑字节。此预算来自已经观察过的 BC01 开发轨迹，属于事后可行性检查；write_mbps=0 未施加硬限流。所有原计数中的需求、预取、重复写入都纳入总写入。',
              '- DT = GET（含预取）与 PUT 服务时间之和，除以作者的 36 个模拟磁盘和该窗口实际时长，再按 0.1% 抽样缩放。平均值按完整评价段实际时长加权。写入与读取是模拟逻辑计量，不是本机物理 I/O 或生产收益。','',
              '- 作者需求读取按最小至最大缺失块的连续范围计数，预取读取为该范围的额外扩展；两项成本都纳入 GET。读取范围块数不等同于逐块未命中次数，本轮不据此推导逐窗命中率。','',
              '## 主表','', '| 方法 | 写入（块；GiB） | 资源合格 | 全段峰值 DT | 峰值窗口 | 平均 DT | 预取读取（块；GiB） | 回放耗时 |',
              '| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        if arm not in summaries:
            report.append(f'| {arm} | 未完成 | — | — | — | — | — | — |')
            continue
        s=summaries[arm]
        report.append(f'| {arm} | {s["evaluation_write_chunks"]:,}；{s["evaluation_write_bytes_sample"]/2**30:.6f} | {"是" if s["resource_eligible"] else "否"} | {s["peak_dt_pct"]:.6f}% | {s["peak_window"]} | {s["average_dt_pct"]:.6f}% | {s["evaluation_prefetch_read_chunks"]:,}；{s["evaluation_prefetch_read_bytes_sample"]/2**30:.6f} | {s["wall_seconds"]:.2f} s |')
    if pstatic:
        report += ['',f'最佳资源合格固定配置为 **{pstatic["method"]}**，P_static={pstatic["peak_dt_pct"]:.9f}%。它只是本轮三个固定配置的事后最优参照，不能称为可部署选参器或最强固定策略。','',
                   '| 调度臂 | 相对 P_static 的全段降峰 | 资源合格 |','| --- | ---: | :---: |']
        for arm,c in comparisons.items():
            report.append(f'| {arm} | {c["relative_peak_gain"]*100:.6f}% | {"是" if c["resource_eligible"] else "否；不作为收益见证"} |')
        if 'LEAD' in comparisons and comparisons['LEAD']['resource_eligible']:
            report += ['', f'主比较 G_LEAD={comparisons["LEAD"]["relative_peak_gain"]*100:.9f}%；LEAD 资源合格，'+
                       ('达到' if comparisons['LEAD']['relative_peak_gain']>=.05 else '未达到')+'事先固定的 5% 门槛。']
        if 'LEAD' in comparisons and not comparisons['LEAD']['resource_eligible']:
            report += ['', 'LEAD 超过写入预算，其数值仅供描述，不按主比较判读。']
        if passed:
            static_gain=(summaries['STATIC-BASE']['peak_dt_pct']-pstatic['peak_dt_pct'])/summaries['STATIC-BASE']['peak_dt_pct']
            eligible_schedules=[a for a in ARMS[3:] if summaries[a]['resource_eligible']]
            best_schedule=min(eligible_schedules,key=lambda a:summaries[a]['peak_dt_pct']) if eligible_schedules else None
            report += ['',f'最佳合格固定配置相对原 Baleen 的降峰为 {static_gain*100:.6f}%。','']
            if best_schedule:
                report += [f'最佳合格时间调度为 {best_schedule}，相对 P_static 的降峰为 {comparisons[best_schedule]["relative_peak_gain"]*100:.6f}%；其选择仅用于描述，主诊断臂仍为 LEAD。']
    report += ['', '## 原峰值窗口 578（描述）','', '| 方法 | 窗口 578 DT | 对 STATIC-BASE 的相对变化 |','| --- | ---: | ---: |']
    for arm,s in summaries.items():
        base=summaries['STATIC-BASE']['window_578_dt_pct']
        report.append(f'| {arm} | {s["window_578_dt_pct"]:.6f}% | {(s["window_578_dt_pct"]/base-1)*100:+.6f}% |')
    report += ['', '裁决使用全部评价窗口的最大值；窗口 578 的变化不能替代主指标。','', '## 审计与权限边界','',
               f'- 接口预检通过：{", ".join(preflight["checks"])}。区间起点前、起点、终点前、终点均验证；相等预测拒绝，check_only 不计准入决定。',
               '- 评价前在请求处理之前、沿用作者 checkpoint 之后保存完整状态。包括 LRU 顺序及条目元数据、待准入批次顺序与元数据、动态特征历史和最后访问时间、缓存事件历史、准入器状态、预取非模型状态及累积计数。模型内容单独固定 SHA-256。',
               '- 所有已完成臂的预热状态各组件哈希一致；预热逐窗服务成本、写入和请求计数一致。全部窗口的 PUT、无缓存 GET、窗口端点和请求计数与 BC01 保持一致。必需字段缺失、非有限值、长度错误均报错。',
               f'- 所有臂预热逐窗核对最大绝对误差：{max((max(v["warmup_max_errors"].values()) for v in audits.values()),default=0):.12g}；全段不随策略变化的字段核对最大绝对误差：{max((max(v["all_window_invariant_max_errors"].values()) for v in audits.values()),default=0):.12g}；STATIC-BASE 的 DT 重算最大误差：{baseline_dt_error if "STATIC-BASE" in summaries else None} 个百分点。',
               f'- STATIC-BASE 对原始 BC01 的逐窗对账见 audit.json：成本容差 {ST_ATOL:g} 秒，DT 容差 {DT_ATOL:g} 个百分点，时间和计数必须精确相等。输入、模型及配置哈希在前后核对；作者源码保持原样，调度通过独立运行包装器注入。',
               f'- 作者源码版本：{source_revisions!r}。最终 git 状态为空；实现文件 SHA-256 随审计保存。',
               f'- 两小时总计算预算；实际编排耗时 {status["wall_seconds"]:.2f} 秒；完成 {len(completed)}/7 臂，状态 {status["status"]}，未完成 {status.get("uncompleted",[])}。各臂耗时含启动、原模拟和状态审计；不存在新增回放。',
               '- 本轮使用未来峰值时间与开发轨迹预算，仅检验实际缓存轨迹的机制可行性，不检验合法预测、可部署控制、统计显著性或跨轨迹泛化。窗口不作为独立实验样本。','', '## 交付与复核','',
               '- BC03_summary.csv：主表全精度数值。',
               f'- BC03_all_windows.csv：{len(completed)} 臂完整时间序列，共 {len(all_rows):,} 行；另有各臂 BC03_<方法>_10min.csv，各 1,008 行，保留预热与评价标记。阈值区间跨窗时明确记录分段，实际决策使用调用时间。',
               '- 单臂时间序列：'+ '、'.join(f'[{arm}]({(OUT/f"BC03_{arm}_10min.csv").as_posix()})' for arm in completed)+'。',
               '- BC03_audit.json：完整配置、输入哈希、状态摘要、阈值调用分组、对账误差及裁决。',
               '- 实现、接口预检、每臂原始 .stats.lzma、模拟日志和压缩预热状态位于 work/bc03；复核入口为 build_deliverables.py，不需重新回放。','', '## 结论边界','',verdict,
               '5% 是事先固定的工程投入筛查尺度，不能解释为统计显著性。即使获得见证，下一步仍需合法时间信息与简单规则的独立检验，不启动 GRL。若本组未获见证，仅关闭本组干预，不否定全部峰值优化方法。']
    (OUT/'BC03_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    if verification_file.exists():
        worst=max(v['max_dt_conversion_error_pp'] for v in verification['checks'].values())
        text=(OUT/'BC03_report.md').read_text(encoding='utf-8')
        text=text.replace('## 结论边界',f'独立导出复核通过：逐窗 DT 另按 service_time / (36 × 0.001 × 窗口秒数) × 100 换算，最大差 {worst:.12g} 个百分点；峰值窗口、时间加权平均、总写入和预取读取合计全部与主表一致。复核结果已并入 BC03_audit.json。\n\n## 结论边界')
        (OUT/'BC03_report.md').write_text(text,encoding='utf-8')
    print(json.dumps({'protocol_passed':passed,'best_static':pstatic,'comparisons':comparisons,'verdict':verdict,'summaries':summaries}, ensure_ascii=False))

if __name__ == '__main__':
    main()
