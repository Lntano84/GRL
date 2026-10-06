"""Final arithmetic/serialization audit of BC04 delivered tables and peak log."""
import csv
import gzip
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def read_csv(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    audit=json.loads((OUT/'BC04_audit.json').read_text())
    rows=read_csv('BC04_summary.csv')
    assert len(rows)==12
    for row in rows:
        expected=audit['summaries'][row['scope']][row['request_category']]
        for key in ['miss_requests','hit_requests']:
            assert int(row[key])==expected[key]
        for key in ['demand_dt_pct','share_of_scope_demand_dt_pct','prefetch_extra_dt_pct','scope_duration_s']:
            assert abs(float(row[key])-expected[key])<=1e-9
    for scope in ['peak_578','evaluation']:
        group=[r for r in rows if r['scope']==scope]
        assert abs(math.fsum(float(r['share_of_scope_demand_dt_pct']) for r in group)-100)<=1e-9
    requests=read_csv('BC04_peak_578_requests.csv')
    chunkrows=read_csv('BC04_peak_578_chunks.csv')
    with gzip.open(OUT/'BC04_peak_578_requests.jsonl.gz','rt',encoding='utf-8') as f:
        peak=[json.loads(line) for line in f]
    assert len(requests)==len(peak)==407
    assert len({r['request_sequence'] for r in requests})==len(requests)
    assert len(chunkrows)==sum(len(r['need_fetch']) for r in peak)
    for row,raw in zip(requests,peak):
        assert int(row['request_sequence'])==raw['request_sequence']
        assert json.loads(row['need_fetch_json'])==raw['need_fetch']
        assert json.loads(row['missing_chunk_history_json'])==raw['actual_missing_chunk_history']
        assert row['request_category']==raw['request_category']
    for cat in CATEGORIES+['HIT']:
        group=[r for r in peak if r['request_category']==cat]
        expected=audit['summaries']['peak_578'][cat]
        assert sum(bool(r['need_fetch']) for r in group)==expected['miss_requests']
        assert sum(not bool(r['need_fetch']) for r in group)==expected['hit_requests']
        assert abs(math.fsum(r['demand_dt_pct_of_window'] for r in group)-expected['demand_dt_pct'])<=1e-9
        assert abs(math.fsum(r['prefetch_extra_dt_pct_of_window'] for r in group)-expected['prefetch_extra_dt_pct'])<=1e-9
    window=read_csv('BC04_window_reconciliation.csv')
    assert len(window)==1008
    assert sum(r['phase']=='evaluation' for r in window)==864
    check={'passed':True,'summary_rows':12,'peak_get_requests':len(peak),
           'peak_missing_chunks':len(chunkrows),'window_rows':len(window),'evaluation_window_rows':864}
    audit['export_verification']=check
    audit['implementation_sha256']={p.name:sha(p) for p in WORK.glob('*.py')}
    (WORK/'audit.json').write_text(json.dumps(audit,indent=2))
    (OUT/'BC04_audit.json').write_text(json.dumps(audit,indent=2))
    report=(OUT/'BC04_report.md').read_text(encoding='utf-8')
    text=f'- 导出复核通过：主表 12 行（包括 HIT）、峰值窗口 {len(peak)} 个 GET、{len(chunkrows):,} 个实际缺失 chunk；CSV 与压缩请求 JSONL 的历史明细一致，逐类别成本合计与主表一致。\n'
    if text not in report:report=report.replace('## 交付',text+'\n## 交付')
    (OUT/'BC04_report.md').write_text(report,encoding='utf-8')
    manifest={p.name:sha(p) for p in OUT.glob('BC04*') if p.is_file()}
    (WORK/'export_manifest.json').write_text(json.dumps({'checks':check,'deliverables_sha256':manifest},indent=2))
    print(json.dumps(check))

if __name__=='__main__':main()
