"""Saved-feedback engineering qualification only, never new data generalization."""
from pathlib import Path
import json,hashlib,copy,struct,collections
from labels import collect,input_at,key,FIELDS
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid100_data_foundation'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def independent(call,subs,n):
    visible=call['order'][:n];truth={r['pool_id']:r for r in call['outcomes']}
    valid=[i for i in visible if truth[i]['strict_admissible']]
    k=lambda i:(struct.unpack('f',struct.pack('f',truth[i]['reward']))[0],-i)
    if not valid:return None
    before=sorted(valid,key=k)[-1];station=subs[before];pending=[i for i in call['order'][n:] if any(int(rep)==i and subs[int(alias)]==station for alias,rep in call['aliases'].items())]
    if not pending:return None
    after=sorted([before]+[i for i in pending if truth[i]['strict_admissible']],key=k)[-1]
    return before,after,k(after)[0]-k(before)[0],pending
def main():
    for p,h in read(OUT/'freeze.json')['sha256'].items():assert sha(ROOT/p)==h,p
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];refs=read(ROOT/'outputs/grid91_mechanism/design.json')['references'];sources={};counts=collections.Counter();by_cohort=collections.Counter();examples=[]
    for ref in refs:
        for p in sorted((ROOT/ref['full']).glob('call[0-9][0-9][0-9].json')):
            c=read(p);identity=p.relative_to(ROOT).as_posix();sources[identity]=sha(p);rows,excluded=collect(c,subs,identity);counts['full_teacher_calls']+=1;counts['excluded_prefixes']+=len(excluded)
            for row in rows:
                n=row['visible_count'];expected=independent(c,subs,n);assert expected is not None
                before,after,gain,pending=expected;assert row['label']['before_id']==before and row['label']['after_id']==after and row['label']['local_reward_gain']==gain and row['input']['remaining_same_station_ids']==pending;counts['independent_gain_checks']+=1
                public=dict(order=c['order'],aliases=c['aliases'],rho_limit=c['rho_limit'],visible=[{k:r[k] for k in FIELDS} for r in c['outcomes'][:n]])
                assert input_at(public,subs,n)==row['input']
                poisoned=copy.deepcopy(c)
                for r in poisoned['outcomes'][n:]:r.update(reward=9999.,rho=.001,strict_admissible=True)
                px=dict(order=poisoned['order'],aliases=poisoned['aliases'],rho_limit=poisoned['rho_limit'],visible=[{k:r[k] for k in FIELDS} for r in poisoned['outcomes'][:n]])
                assert input_at(px,subs,n)==row['input'];counts['hidden_suffix_perturbations']+=1
                assert not set(row['input'])&{'after_id','local_reward_gain','choice_changes','target','label'}
                by_cohort[ref['cohort']]+=1;counts['positive_local_gain']+=gain>0;counts['zero_local_gain']+=gain==0
                if len(examples)<6:examples.append(dict(identity=identity,partition='historical_debug',calendar_group=ref['week'].split('_')[0],sample=row))
    # Four explicit boundaries: reward improvement, reward tie, no prefix incumbent, no local remainder.
    order=list(range(256));aliases={str(i):i for i in order}
    outcomes=[dict(pool_id=i,rho=.8,reward=1.,done=False,illegal=False,ambiguous=False,exceptions=[],strict_admissible=True,forecast_s=0.) for i in order]
    c=dict(order=order,aliases=aliases,rho_limit=1.,outcomes=outcomes);c['outcomes'][200]['reward']=1.25
    rr,_=collect(c,[0]*256,'toy_gain');assert rr[0]['label']['local_reward_gain']==.25
    tie=copy.deepcopy(c);tie['outcomes'][200]['reward']=1.+1e-9;rr,_=collect(tie,[0]*256,'toy_tie');assert rr[0]['label']['local_reward_gain']==0 and not rr[0]['label']['choice_changes']
    missing=copy.deepcopy(c)
    for r in missing['outcomes']:r['strict_admissible']=False
    rr,xx=collect(missing,[0]*256,'toy_missing');assert not rr and len(xx)==2
    rr,xx=collect(c,list(range(256)),'toy_no_station');assert not rr and len(xx)==2
    counts['boundary_tests']=4
    assert counts['independent_gain_checks']>0 and counts['hidden_suffix_perturbations']==counts['independent_gain_checks']
    report=dict(passed=True,scope='Historical exposed FULL teacher fixtures only; not new training/validation/test data',counts=dict(counts),by_cohort=dict(by_cohort),sources=sources,source_manifest_sha256=sha(OUT/'freeze.json'),new_physical_forecasts=0,new_replays=0,new_fits=0)
    (OUT/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (OUT/'historical_debug_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(counts=counts,by_cohort=by_cohort),ensure_ascii=False))
if __name__=='__main__':main()
