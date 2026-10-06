"""Descriptive examples from the finished peak log; no strategy or model run."""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def main():
    requests=list(csv.DictReader((OUT/'BC04_peak_578_requests.csv').open(encoding='utf-8-sig',newline='')))
    chunks=list(csv.DictReader((OUT/'BC04_peak_578_chunks.csv').open(encoding='utf-8-sig',newline='')))
    selected=[]
    # Fixed descriptive ordering, with request ID as the deterministic tie break.
    # These examples do not add independent evidence or certify counterfactual gain.
    for cat,count in [('REJECTED-BEFORE',3),('ADMITTED-BEFORE',2)]:
        selected.extend(sorted([r for r in requests if r['request_category']==cat],
                               key=lambda r:(-float(r['demand_service_time_s']),int(r['request_sequence'])))[:count])
    rows=[]
    for request in selected:
        facts=[r for r in chunks if r['request_sequence']==request['request_sequence']]
        assert facts
        stem='rejection' if request['request_category']=='REJECTED-BEFORE' else 'eviction'
        times=[float(f['last_'+stem+'_s']) for f in facts]
        event_requests=sorted({int(f['last_'+stem+'_request']) for f in facts})
        counts=[int(f['rejection_count'] if stem=='rejection' else f['eviction_count']) for f in facts]
        rows.append({'request_sequence':int(request['request_sequence']),
                     'full_block_key_json':request['full_block_key_json'], 'request_category':request['request_category'],
                     'actual_missing_chunks':len(facts),'demand_dt_pct_of_window':float(request['demand_dt_pct_of_window']),
                     'last_event_type':stem,'last_event_request_sequences_json':json.dumps(event_requests),
                     'last_event_trace_elapsed_s_min':min(times),'last_event_trace_elapsed_s_max':max(times),
                     'last_event_lag_s_min':float(request['trace_elapsed_s'])-max(times),
                     'last_event_lag_s_max':float(request['trace_elapsed_s'])-min(times),
                     'prior_event_count_min':min(counts),'prior_event_count_max':max(counts)})
    with (OUT/'BC04_descriptive_examples.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (WORK/'descriptive_examples.json').write_text(json.dumps({'selection':'top 3 pure REJECTED-BEFORE and top 2 pure ADMITTED-BEFORE by whole request demand cost; ties by ascending request sequence',
                                                           'examples':rows,'causal_or_profit_claim':False},indent=2))
    audit=json.loads((OUT/'BC04_audit.json').read_text())
    audit['descriptive_examples']=rows
    audit['implementation_sha256']={p.name:sha(p) for p in WORK.glob('*.py')}
    (WORK/'audit.json').write_text(json.dumps(audit,indent=2))
    (OUT/'BC04_audit.json').write_text(json.dumps(audit,indent=2))
    report=(OUT/'BC04_report.md').read_text(encoding='utf-8')
    text=['## 本轮具体认识','',
          '- 窗口 578 最大的单一需求成本类别是纯 REJECTED-BEFORE：77 个请求，占需求读取成本 40.1398%；完整评价段为 37.5835%。这些请求确有此前实际拒绝记录，值得先回查具体决策时点。',
          '- 峰值窗口纯 ADMITTED-BEFORE 占 18.7307%，说明存在曾缓存而当前缺失的数据；其先前驱逐可能远早于峰值，保留至峰值的容量和时间成本尚未计入。',
          '- 峰值窗口全部 17 个 MIXED 请求都是 REJECTED-BEFORE+OTHER；其 8.3131% 需求成本保留在 MIXED，没有计入任何单一历史类别的可恢复成本。',
          '- 峰值窗口 747 个 OTHER 缺失 chunk（包含 MIXED 中的 OTHER）此前无实际候选、无待处理状态，也无资格检查记录。完整评价段另有 1,333 个 OTHER chunk 仅有此前资格检查记录，未当成实际拒绝。','',
          '以下仅为描述性样例：按整请求需求成本排序，分别保留纯 REJECTED-BEFORE 前三个和纯 ADMITTED-BEFORE 前两个；并列按请求序号排序，不把样例作为独立实验或收益见证。','',
          '| 峰值请求 | 块键 | 历史类别 | 缺失 chunk | 该请求需求 DT | 最近历史事件 | 距当前请求（小时） |',
          '| --- | --- | --- | ---: | ---: | --- | ---: |']
    for r in rows:
        lag=f'{r["last_event_lag_s_min"]/3600:.3f}–{r["last_event_lag_s_max"]/3600:.3f}'
        event='实际拒绝' if r['last_event_type']=='rejection' else '实际驱逐'
        text.append(f'| {r["request_sequence"]} | {r["full_block_key_json"]} | {r["request_category"]} | {r["actual_missing_chunks"]} | {r["demand_dt_pct_of_window"]:.6f}% | {event} | {lag} |')
    text+=['',f'[样例全精度 CSV]({(OUT/"BC04_descriptive_examples.csv").as_posix()}) 保留历史事件请求序号、时间和此前事件次数。','',
           '例如请求 82821 的 64 个实际缺失 chunk 此前均未写入，最近实际拒绝请求为 81555，距峰值请求约 1.4146 小时。该事实给出了具体可回查的准入决策时点；是否应该改变该次决定，仍需在容量、写入和预取计费内另做反事实回放。现阶段没有训练新模型的依据。','']
    marker='## 判读边界'
    if '## 本轮具体认识' not in report:
        report=report.replace(marker,'\n'.join(text)+'\n'+marker)
    (OUT/'BC04_report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'descriptive_examples':rows}))

if __name__=='__main__':main()
