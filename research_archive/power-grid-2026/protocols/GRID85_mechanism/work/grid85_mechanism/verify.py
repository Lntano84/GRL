"""Independent reconstruction with struct float32, no imported policy code."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,struct
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid85_mechanism'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
def key(r):return f32(r['reward']),-int(r['pool_id'])
def best(rs):return max((r for r in rs if r['strict_admissible']),key=key,default=None)
def pid(r):return None if r is None else r['pool_id']
def independent(c,kind,subs):
    order=c['order'];truth={r['pool_id']:r for r in c['outcomes']};rs=[];inc=None
    def add(ids):
        nonlocal inc
        for i in ids:
            r=truth[i];rs.append(r)
            if r['strict_admissible'] and (inc is None or key(r)>key(inc)):inc=r
    if kind=='FULL':add(order);return rs
    if kind.startswith('FIXED_'):add(order[:int(kind.split('_')[1])]);return rs
    add(order[:128]);start=pid(inc);safe=inc is not None and inc['rho']<.9;stale=0
    if kind!='OLD_FRAME' or not safe:
        for lo in range(min(128,len(order)),len(order),64):
            before=None if inc is None else key(inc);add(order[lo:lo+64]);after=None if inc is None else key(inc)
            changed=after is not None and (before is None or after>before);stale=0 if changed else stale+1
            if not safe and inc is not None and inc['rho']<.9:break
            if inc is not None and len(rs)>=256 and stale>=2 and len(rs)<len(order):break
    if kind=='NO_LOCAL_GUARD':return rs
    if inc is not None and len(rs)<len(order) and (not safe or pid(inc)!=start):
        known={r['pool_id'] for r in rs};sub=subs[pid(inc)];ali={int(k):int(v) for k,v in c['aliases'].items()}
        targets={v for a,v in ali.items() if subs[a]==sub};ids=[i for i in order if i not in known and i in targets]
        if kind=='BUDGET_GLOBAL_GUARD':ids=[i for i in order if i not in known][:len(ids)]
        add(ids)
    return rs
def main():
    receipt=read(OUT/'source_receipt.json')
    for r in receipt['files']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
    rows=list(csv.DictReader((OUT/'offline_full_calls.csv').open(encoding='utf-8-sig')));subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
    checks=0;cache={};rq={}
    for r in rows:
        if r['source'] not in cache:cache[r['source']]=read(ROOT/r['source'])
        c=cache[r['source']];out=independent(c,r['method'],subs);b=best(out);bf=best(c['outcomes'])
        assert len(out)==int(r['queries']) and len(out)==len({v['pool_id'] for v in out})
        assert (str(pid(b)) if b is not None else '')==r['chosen']
        if b is not None and bf is not None:assert abs((f32(bf['reward'])-f32(b['reward']))-float(r['reward_gap_full']))<1e-12
        for v in out:assert v['strict_admissible']==bool(0<v['rho']<c['rho_limit'] and not v['done'] and not v['exceptions'] and not v['illegal'] and not v['ambiguous'])
        rq[r['source'],r['method']]=len(out);checks+=1
    for s in cache:assert rq[s,'FRAME_GUARD']==rq[s,'BUDGET_GLOBAL_GUARD']
    actual=list(csv.DictReader((OUT/'actual_calls.csv').open(encoding='utf-8-sig')));cohorts=read(OUT/'actual_summary.json')
    observed_queries=0;actual_checks=0
    for r in actual:
        c=read(ROOT/r['source']);out=independent(c,'FRAME_GUARD',subs)
        assert [v['pool_id'] for v in out]==[v['pool_id'] for v in c['outcomes']]
        assert pid(best(out))==c['chosen_pool_id'];observed_queries+=len(out);actual_checks+=1
    assert observed_queries==sum(s['queries'] for s in cohorts)
    for s in cohorts:
        cc=[read(ROOT/r['source']) for r in actual if r['cohort']==s['cohort']]
        phase=Counter(v['phase'] for c in cc for v in c['outcomes'])
        assert phase['shortlist']==s['prefix_queries'] and phase['local_completion']==s['local_queries']
        assert sum(v for k,v in phase.items() if k not in ['shortlist','local_completion'])==s['global_extra_queries']
    result=dict(passed=True,independent_of_policy_imports=True,float32_via_struct=True,
                independent_full_state_method_reconstructions=checks,actual_call_reconstructions=actual_checks,
                equal_budget_full_states=len(cache),actual_queries_checked=observed_queries,
                source_hashes_checked=len(receipt['files']),new_fits=0,new_simulation=0,
                limitations='Independent saved-data arithmetic/control reconstruction; not an independent AC solve or proof of long-term causal effects.')
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
