"""BC04 offline reconciliation and history provenance diagnosis. No replay."""
import collections
import csv
import gzip
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
sys.path.insert(0,str(REPO))
import compress_json
import numpy as np

ST_TOL=1e-9
DT_TOL=1e-8
KEYS=['time_elapsed_phy','time_log','service_time_used_stats','service_time_writes_stats',
      'service_time_used_demand_stats','service_time_used_prefetch_stats','service_time_nocache_stats',
      'flashcache/keys_written_stats','flashcache/prefetches_stats','fetches_chunks_prefetch_stats',
      'fetches_chunks_demandmiss_stats','fetches_ios_stats','iops_requests_stats','chunk_queries_stats',
      'puts_ios_stats','puts_chunks_stats']

def required(batches,key):
    if key not in batches:raise KeyError('missing mandatory field: '+key)
    a=np.asarray(batches[key],dtype=float)
    assert a.shape==(1008,) and np.isfinite(a).all(),key
    return a

def delta(a):return np.diff(a,prepend=0.)

def jsonlines(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def token(value):return json.dumps(value,sort_keys=True,separators=(',',':'))

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    status=json.loads((WORK/'run_status.json').read_text())
    assert status['status']=='completed' and status['attempts']==1,'No inference from incomplete replay'
    replay=json.loads((WORK/'replay_audit.json').read_text())
    preflight=json.loads((WORK/'preflight.json').read_text())
    assert replay['completed'] and preflight['passed']
    for path,digest in preflight['inputs_sha256'].items():assert sha(Path(path))==digest,path
    assert sha(WORK/'config.json')==preflight['config_sha256']
    paths=list((WORK/'raw').rglob('*.stats.lzma'))
    assert len(paths)==1
    b=compress_json.load(str(paths[0]))['batches']
    reference=compress_json.load(str(REFERENCE))['batches']
    errors={}
    for key in KEYS:
        a,c=required(b,key),required(reference,key)
        err=float(np.max(np.abs(a-c)))
        assert err<=(ST_TOL if key.startswith('service_time') else 0),(key,err)
        errors[key]=err
    elapsed=required(b,'time_elapsed_phy')
    duration=delta(elapsed)
    assert np.all(duration>0) and elapsed[143]==EVAL_START
    starts=np.r_[0.,elapsed[:-1]]
    get_st=delta(required(b,'service_time_used_stats'))
    demand_st=delta(required(b,'service_time_used_demand_stats'))
    pf_st=delta(required(b,'service_time_used_prefetch_stats'))
    put_st=delta(required(b,'service_time_writes_stats'))
    actual_dt=(get_st+put_st)/(36*.001*duration)*100
    peak_i=144+int(np.argmax(actual_dt[144:]))
    assert peak_i==578
    rows=[]
    mixed=collections.defaultdict(lambda:{'requests':0,'demand_s':0.,'prefetch_s':0.})
    other=collections.defaultdict(collections.Counter)
    perwindow_demand=[[] for _ in range(1008)]
    perwindow_pf=[[] for _ in range(1008)]
    perwindow_miss=np.zeros(1008,dtype=np.int64)
    perwindow_reads=np.zeros(1008,dtype=np.int64)
    perwindow_pf_reads=np.zeros(1008,dtype=np.int64)
    # Independent history rebuilt from observed actions, rather than using the
    # logger's History class. check_only records are separate and never mutate
    # real candidate/write/rejection counts.
    independent={}
    seen=set()
    requests=0
    chunks=0
    event_counts=collections.Counter()
    last_request=0
    history_checks=0
    qualifiers=iter(jsonlines(WORK/'prefetch_qualification_checks.jsonl.gz'))
    nextqual=next(qualifiers,None)
    scopes={'peak_578':[578], 'evaluation':list(range(144,1008))}
    aggregation={scope:{cat:{'misses':0,'hits':0,'demand':[],'prefetch':[]} for cat in CATEGORIES+['HIT']} for scope in scopes}
    peak_rows=[]
    peak_chunks=[]
    OUT.mkdir(exist_ok=True)
    peak_log=gzip.open(OUT/'BC04_peak_578_requests.jsonl.gz','wt',encoding='utf-8',compresslevel=1)
    for record in jsonlines(WORK/'all_get_requests.jsonl.gz'):
        requests+=1
        if requests%20000==0:print(f'offline history reconstruction: {requests}/127305 GETs',flush=True)
        seq=record['request_sequence']
        assert seq>last_request and record['get_sequence']==requests
        last_request=seq
        block=record['full_block_key']
        block_token=token(block)
        first=block_token not in seen
        assert record['block_had_prior_get']==(not first)
        w=record['window_index']
        assert 0<=w<1008
        time=record['trace_elapsed_s']
        assert int(np.searchsorted(elapsed,time,side='right'))==w or (w==1007 and time==elapsed[-1])
        facts=record['actual_missing_chunk_history']
        need=record['need_fetch']
        assert [f['chunk'] for f in facts]==need and len(need)==len(set(need))
        assert set(need)<=set(record['requested_chunks'])
        expected_categories=[]
        for fact in facts:
            key=token({'tuple':[block,fact['chunk']]})
            prior=independent.get(key,{})
            for field in ['candidate_count','written_count','rejection_count','eviction_count',
                          'qualification_accept_count','qualification_reject_count']:
                assert fact[field]==prior.get(field,0),(seq,key,field,fact[field],prior.get(field,0))
            for stem in ['candidate','write','rejection','eviction']:
                field='last_'+stem+'_request'
                assert fact[field]==prior.get(field), (seq,key,field)
                assert fact[field] is None or fact[field]<seq,'current event backdated'
            category=('FIRST' if first else 'ADMITTED-BEFORE' if prior.get('written_count',0) else
                      'REJECTED-BEFORE' if prior.get('rejection_count',0) else 'OTHER')
            assert fact['category']==category
            expected_categories.append(category)
            history_checks+=1
        combo=[c for c in CHUNK_CATEGORIES if c in expected_categories]
        cat='HIT' if not need else combo[0] if len(combo)==1 else 'MIXED'
        assert cat==record['request_category'] and combo==record['category_combination']
        demand=float(record['demand_service_time_s'])
        pf=float(record['prefetch_extra_service_time_s'])
        assert demand>=0 and pf>=0 and (bool(need) or (demand==pf==0))
        assert abs(record['demand_dt_pct_of_window']-demand/(36*.001*duration[w])*100)<=DT_TOL
        assert abs(record['prefetch_extra_dt_pct_of_window']-pf/(36*.001*duration[w])*100)<=DT_TOL
        perwindow_demand[w].append(demand)
        perwindow_pf[w].append(pf)
        perwindow_miss[w]+=bool(need)
        perwindow_reads[w]+=record['demand_read_span_chunks']
        perwindow_pf_reads[w]+=record['prefetch_extra_read_span_chunks']
        chunks+=len(facts)
        for scope,windows in scopes.items():
            in_scope=(w==578) if scope=='peak_578' else w>=144
            if not in_scope:continue
            a=aggregation[scope][cat]
            a['misses']+=bool(need)
            a['hits']+=not bool(need)
            a['demand'].append(demand)
            a['prefetch'].append(pf)
            if cat=='MIXED':
                m=mixed[(scope,'+'.join(combo))]
                m['requests']+=1
                m['demand_s']+=demand
                m['prefetch_s']+=pf
            for fact in facts:
                if fact['category']=='OTHER':
                    counts=other[scope]
                    counts['actual_missing_chunks']+=1
                    counts['prior_actual_candidate']+=fact['candidate_count']>0
                    counts['pending_before_request']+=fact['pending_before_request']
                    counts['prior_qualification_only']+=(fact['qualification_accept_count']+fact['qualification_reject_count']>0 and fact['candidate_count']==0)
                    counts['no_candidate_no_qualification']+=(fact['candidate_count']==0 and fact['qualification_accept_count']+fact['qualification_reject_count']==0)
        if w==578:
            peak_log.write(json.dumps(record,separators=(',',':'))+'\n')
            for fact in facts:
                peak_chunks.append({'request_sequence':seq,'window_index':w,'request_trace_elapsed_s':time,
                                    'full_block_key_json':json.dumps(block,separators=(',',':')),
                                    'request_category':cat,**fact})
            peak_rows.append({'request_sequence':seq,'get_sequence':requests,'physical_time_s':record['physical_time_s'],
                             'trace_elapsed_s':time,'window_index':w,'full_block_key_json':json.dumps(block,separators=(',',':')),
                             'requested_chunks_json':json.dumps(record['requested_chunks']),
                             'need_fetch_json':json.dumps(need),'need_prefetch_json':json.dumps(record['need_prefetch']),
                             'block_had_prior_get':record['block_had_prior_get'],'request_category':cat,
                             'category_combination':'+'.join(combo),
                             'missing_chunk_history_json':json.dumps(facts,separators=(',',':')),
                             'demand_service_time_s':demand,'prefetch_extra_service_time_s':pf,
                             'demand_dt_pct_of_window':record['demand_dt_pct_of_window'],
                             'prefetch_extra_dt_pct_of_window':record['prefetch_extra_dt_pct_of_window']})
        # Commit ONLY after checking this request's detached historical facts.
        for kind,events in record['actions'].items():
            field={'candidate':'candidate_count','write':'written_count','rejection':'rejection_count','eviction':'eviction_count'}[kind]
            for native_key,event_s in events:
                key=token(native_key)
                state=independent.setdefault(key,{})
                state[field]=state.get(field,0)+1
                state['last_'+kind+'_request']=seq
                event_counts[kind]+=1
        while nextqual is not None and nextqual['request_sequence']==seq:
            assert nextqual['check_only'] is True and nextqual['window_index']==w
            for native_key,accepted in nextqual['decisions']:
                state=independent.setdefault(token(native_key),{})
                field='qualification_accept_count' if accepted else 'qualification_reject_count'
                state[field]=state.get(field,0)+1
            event_counts['qualification_calls']+=1
            event_counts['qualification_items']+=len(nextqual['decisions'])
            nextqual=next(qualifiers,None)
        assert nextqual is None or nextqual['request_sequence']>seq
        seen.add(block_token)
    peak_log.close()
    assert nextqual is None
    assert requests==127305 and chunks==replay['totals']['missing_chunks']
    for kind,count in event_counts.items():assert count==replay['totals'][kind],kind
    measured_demand=np.asarray([math.fsum(v) for v in perwindow_demand])
    measured_pf=np.asarray([math.fsum(v) for v in perwindow_pf])
    cost_errors={'demand_service_time_s':float(np.max(np.abs(measured_demand-demand_st))),
                 'prefetch_service_time_s':float(np.max(np.abs(measured_pf-pf_st))),
                 'total_get_service_time_s':float(np.max(np.abs(measured_demand+measured_pf-get_st))),
                 'total_get_dt_pp':float(np.max(np.abs((measured_demand+measured_pf-get_st)/(36*.001*duration)*100)))}
    assert all(v<=(DT_TOL if k.endswith('_pp') else ST_TOL) for k,v in cost_errors.items()),cost_errors
    assert np.array_equal(perwindow_reads,delta(required(b,'fetches_chunks_demandmiss_stats')))
    assert np.array_equal(perwindow_pf_reads,delta(required(b,'fetches_chunks_prefetch_stats')))
    assert sum(perwindow_miss)==sum(replay['totals'].get('category/'+c,0) for c in CATEGORIES)
    summaries={}
    for scope,windows in scopes.items():
        seconds=float(duration[windows].sum())
        totals=aggregation[scope]
        total_demand=math.fsum(math.fsum(a['demand']) for a in totals.values())
        totals_out={}
        for cat,a in totals.items():
            ds,ps=math.fsum(a['demand']),math.fsum(a['prefetch'])
            row={'scope':scope,'request_category':cat,'miss_requests':a['misses'],'hit_requests':a['hits'],
                 'demand_service_time_s_sample':ds,'demand_dt_pct':ds/(36*.001*seconds)*100,
                 'share_of_scope_demand_dt_pct':ds/total_demand*100 if total_demand else 0,
                 'prefetch_extra_service_time_s_sample':ps,'prefetch_extra_dt_pct':ps/(36*.001*seconds)*100,
                 'scope_duration_s':seconds}
            rows.append(row)
            totals_out[cat]=row
        summaries[scope]=totals_out
    mixed_rows=[{'scope':s,'category_combination':c,**v} for (s,c),v in mixed.items()]
    write_csv(OUT/'BC04_summary.csv',rows)
    write_csv(OUT/'BC04_peak_578_requests.csv',peak_rows)
    write_csv(OUT/'BC04_peak_578_chunks.csv',peak_chunks)
    if mixed_rows:write_csv(OUT/'BC04_mixed_combinations.csv',mixed_rows)
    shutil.copyfile(WORK/'all_get_requests.jsonl.gz',OUT/'BC04_all_get_requests.jsonl.gz')
    shutil.copyfile(WORK/'prefetch_qualification_checks.jsonl.gz',OUT/'BC04_prefetch_qualification_checks.jsonl.gz')
    window_rows=[{'window_index':i,'phase':'evaluation' if i>=144 else 'warmup',
                  'trace_start_s':float(starts[i]),'trace_end_s':float(elapsed[i]),'duration_s':float(duration[i]),
                  'request_demand_service_time_s':float(measured_demand[i]),'request_prefetch_service_time_s':float(measured_pf[i]),
                  'raw_get_service_time_s':float(get_st[i]),'raw_put_service_time_s':float(put_st[i]),
                  'request_total_get_dt_pct':float((measured_demand[i]+measured_pf[i])/(36*.001*duration[i])*100),
                  'raw_total_dt_pct':float(actual_dt[i]),'miss_requests':int(perwindow_miss[i]),
                  'demand_read_span_chunks':int(perwindow_reads[i]),'prefetch_extra_read_span_chunks':int(perwindow_pf_reads[i]),
                  'flash_write_chunks':int(delta(required(b,'flashcache/keys_written_stats'))[i])} for i in range(1008)]
    write_csv(OUT/'BC04_window_reconciliation.csv',window_rows)
    source_revisions={}
    for label,folder in [('Baleen-FAST24',REPO),('BCacheSim',REPO/'BCacheSim')]:
        assert not subprocess.check_output(['git','-C',str(folder),'status','--porcelain'],text=True).strip()
        source_revisions[label]=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
    audit={'protocol_passed':True,'run_status':status,'preflight':preflight,'replay_audit':replay,
           'baseline_max_errors':errors,'request_cost_max_errors':cost_errors,
           'history_independently_reconstructed':True,'missing_chunk_history_checks':history_checks,
           'missing_chunks_exactly_once':chunks,'requests':requests,'unique_get_blocks':len(seen),
           'summaries':summaries,'other_chunk_details':{k:dict(v) for k,v in other.items()},
           'mixed_combinations':mixed_rows,'source_revisions':source_revisions,
           'implementation_sha256':{p.name:sha(p) for p in WORK.glob('*.py')},
           'request_log_sha256':sha(WORK/'all_get_requests.jsonl.gz'),
           'qualification_log_sha256':sha(WORK/'prefetch_qualification_checks.jsonl.gz'),
           'stats_sha256':sha(paths[0]),'service_time_tolerance_s':ST_TOL,'dt_tolerance_pp':DT_TOL,
           'peak_window':578,'evaluation_start_s':EVAL_START,'evaluation_index_start':144,
           'request_dt_units':'Within window: additive utilization percentage points; whole evaluation: time-weighted average utilization percentage points.'}
    (WORK/'audit.json').write_text(json.dumps(audit,indent=2))
    (OUT/'BC04_audit.json').write_text(json.dumps(audit,indent=2))
    report=['# BC04：原峰值未命中的历史来源','',
            '一次原 STATIC-BASE 回放和只读日志审计通过。分类是历史事实，不是拒绝错误或可盈利保留的认证；本轮没有新策略、参数搜索、训练或反事实回放。','',
            '## 范围与计费','',
            '- 沿用 Region1 完整轨迹、原 Baleen 模型、原阈值 0.798545、3,002 个配置槽位、LRU、原预取与原计量。作者配置仅更改输出目录。write_mbps=0 保持原定义。',
            f'- 从轨迹起点连续维护 GET 与 chunk 事件历史，包括预热；评价窗口为 144–1007，实际起点 {EVAL_START:.11f} 秒。',
            '- 每个 GET 在 run_get 入口冻结此前是否有块 GET、chunk 实际写入／拒绝／候选／驱逐／预取资格检查记录，以及待处理队列状态。当前请求的所有行为只供后续请求使用。',
            '- 实际写入在 QueueCache.admit 成功且写入计数增一后记录；实际拒绝在 process_admit_buffer 返回后，核对拒绝计数增量才记录。check_only 资格检查单独落盘，仅更新资格检查历史，不形成真实写入、拒绝或候选。',
            '- FIRST 优先于 ADMITTED-BEFORE，再到 REJECTED-BEFORE，最后 OTHER。FIRST 指整条轨迹第一次块 GET，不能按 chunk 首次读取或评价开始重新定义。',
            '- 作者将待处理准入队列视为命中。本轮使用实际 need_fetch 做缺失分类；连续需求读取范围内的桥接 chunk 不额外造出缺失分类。',
            '- 一次需求读取的 service_time(1, max(need_fetch)−min(need_fetch)+1) 全部归给请求类别。跨类别缺失归 MIXED，保留组合，完整 seek 不做 chunk 分摊。额外预取 service_time(0, 扩展读取范围块数) 分列。',
            '- DT 单位为模拟磁盘负载百分点：窗口 578 的逐请求贡献相加为该窗口负载；完整评价段按评价实际总时长归一化为平均负载。占比的分母仅为该范围需求读取成本，不含 PUT 或额外预取。数值是模拟计量，不是本机磁盘 I/O 或生产收益。','']
    for scope,label in [('peak_578','窗口 578'),('evaluation','完整评价段')]:
        report += ['## '+label,'',
                   '| 请求类别 | 未命中请求数 | 需求读取 DT | 占该范围需求读取 DT | 额外预取 DT |',
                   '| --- | ---: | ---: | ---: | ---: |']
        for cat in CATEGORIES:
            s=summaries[scope][cat]
            report.append(f'| {cat} | {s["miss_requests"]:,} | {s["demand_dt_pct"]:.6f}% | {s["share_of_scope_demand_dt_pct"]:.4f}% | {s["prefetch_extra_dt_pct"]:.6f}% |')
        total_get=math.fsum(v['demand_dt_pct']+v['prefetch_extra_dt_pct'] for v in summaries[scope].values())
        total_demand=math.fsum(v['demand_dt_pct'] for v in summaries[scope].values())
        total_pf=math.fsum(v['prefetch_extra_dt_pct'] for v in summaries[scope].values())
        report += ['',f'完全命中：{summaries[scope]["HIT"]["hit_requests"]:,} 次，需求读取和额外预取成本均为零。需求总 DT={total_demand:.9f}%，额外预取 DT={total_pf:.9f}%，合计 GET DT={total_get:.9f}%。','']
    report += ['## MIXED 与 OTHER 的补充事实','',
               '| 范围 | MIXED 类别组合 | 请求数 | 抽样需求服务秒 | 抽样预取服务秒 |',
               '| --- | --- | ---: | ---: | ---: |']
    for row in mixed_rows:
        report.append(f'| {row["scope"]} | {row["category_combination"]} | {row["requests"]:,} | {row["demand_s"]:.9f} | {row["prefetch_s"]:.9f} |')
    report += ['', '| 范围 | OTHER 实际缺失 chunk | 曾为实际候选 | 请求前仍在队列 | 仅有先前资格检查 | 无候选且无资格检查 |',
               '| --- | ---: | ---: | ---: | ---: | ---: |']
    for scope in scopes:
        c=other[scope]
        report.append(f'| {scope} | {c["actual_missing_chunks"]:,} | {c["prior_actual_candidate"]:,} | {c["pending_before_request"]:,} | {c["prior_qualification_only"]:,} | {c["no_candidate_no_qualification"]:,} |')
    report += ['', '上述补充项按实际缺失 chunk 计数，不分摊请求 seek，不用于计算可恢复收益。ADMITTED-BEFORE 仅说明曾实际写入现在缺失；REJECTED-BEFORE 仅说明实际拒绝发生过；OTHER 仅说明未有实际写入或拒绝记录。','',
               '## 计量与历史审计','',
               f'- 恰好一次完整回放，耗时 {status["wall_seconds"]:.2f} 秒，低于 1,800 秒预算；GET {requests:,} 次、原请求总数 {replay["requests"]:,}。',
               f'- 逐窗 GET、PUT、需求与预取成本、写入、预取读取、需求读取、请求计数和作者窗口端点与 BC03 STATIC-BASE 对账，最大误差 {max(errors.values()):.12g}。',
               f'- 请求成本逐窗对账最大误差：{cost_errors!r}。容差为服务时间 {ST_TOL:g} 秒、DT {DT_TOL:g} 个百分点；整数计数与时间端点精确相等。',
               f'- {chunks:,} 个实际缺失 chunk 均恰好归入一个类别。离线从每个请求的实际候选、实际写入、真实拒绝、驱逐记录及单独资格日志重新构造历史，逐一核对冻结事实；不复用日志器的 History 对象或分类函数。',
               f'- 手工序列通过：{", ".join(preflight["checks"])}。包括当前拒绝回填、资格检查污染真实准入历史、类别优先级、跨主机键、MIXED 与完全命中。',
               '- 源码保持原样，输入、已训练模型和配置 SHA-256 前后相同；完整审计见 BC04_audit.json。','',
               '## 交付','',
               f'- [主表 CSV]({(OUT/"BC04_summary.csv").as_posix()})：两个范围的全精度主表，另有 HIT 行。',
               f'- [峰值窗口逐请求 CSV]({(OUT/"BC04_peak_578_requests.csv").as_posix()})：窗口 578 的所有 GET，包括完全命中，保留实际缺失 chunk 的完整历史。',
               f'- [峰值窗口缺失 chunk 历史 CSV]({(OUT/"BC04_peak_578_chunks.csv").as_posix()})：按请求序号关联到逐请求明细，不给 chunk 分摊 seek 或归因收益。',
               f'- [峰值窗口完整事件 JSONL.gz]({(OUT/"BC04_peak_578_requests.jsonl.gz").as_posix()})：请求事实和本请求真实候选／写入／拒绝／驱逐事件。',
               f'- [全轨迹 GET JSONL.gz]({(OUT/"BC04_all_get_requests.jsonl.gz").as_posix()})：全部 GET 明细，供历史来源回查。',
               f'- [资格检查 JSONL.gz]({(OUT/"BC04_prefetch_qualification_checks.jsonl.gz").as_posix()})：单独的 check_only=True 调用与决定。',
               f'- [逐窗对账 CSV]({(OUT/"BC04_window_reconciliation.csv").as_posix()})、[审计 JSON]({(OUT/"BC04_audit.json").as_posix()})。','',
               '## 判读边界','',
               '本轮没有学习启动占比门槛。历史机会须进一步落实到具体时点、缓存容量与写入预算内的反事实回放，才能判断收益；分类占比不等于可恢复的峰值空间。仅有拒绝或曾缓存记录，不能认证当时决策错误。']
    (OUT/'BC04_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({'protocol_passed':True,'baseline_errors':errors,'cost_errors':cost_errors,
                      'summaries':summaries,'other':{k:dict(v) for k,v in other.items()},'wall_seconds':status['wall_seconds']}))

if __name__=='__main__':main()
