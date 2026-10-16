"""Mechanism analysis of paid logs only; no environment or learner imports."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,gzip,hashlib,importlib.util,json,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid85_mechanism'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('grid85_frozen_guard',ROOT/'work/grid70/policies.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
SUBS=read(ROOT/'outputs/grid28/offline.json')['public_subids']
INPUTS={}
def source(p):
    INPUTS[p.relative_to(ROOT).as_posix()]=sha(p);return p
def load(p):return read(source(p))
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvsave(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def mean(x):
    a=[v for v in x if v is not None];return statistics.mean(a) if a else None
def best(rows):return max((r for r in rows if r['strict_admissible']),key=P.key,default=None)
def pid(r):return None if r is None else r['pool_id']
def keygain(a,b):return b is not None and (a is None or P.key(b)>P.key(a))
def rewardgain(a,b):return None if a is None or b is None else float(np.float32(b['reward']))-float(np.float32(a['reward']))
def signature(p):
    with np.load(source(p.with_name(p.stem+'_state.npz'))) as z:
        obs=np.ascontiguousarray(z['observation']);base=np.ascontiguousarray(z['base_action'])
        return tuple((str(x.dtype),str(x.shape),hashlib.sha256(x.tobytes()).hexdigest()) for x in (obs,base))
def files(folder):return sorted(folder.glob('call[0-9][0-9][0-9].json'))
def reconstruct(c,policy='FRAME_GUARD',aliases=None):
    truth={r['pool_id']:r for r in c['outcomes']};seen=[]
    def reveal(i,phase):
        assert i in truth and i not in seen;seen.append(i);return dict(truth[i],phase=phase)
    got=P.run(c['order'],reveal,policy,SUBS,{int(k):int(v) for k,v in c['aliases'].items()} if aliases is None else aliases,limit=c['rho_limit'])
    assert seen==[r['pool_id'] for r in got['outcomes']] and len(seen)==got['public_queries']
    return got
def fullrefs():
    result={}
    for stage in [65,68,81]:
        for folder in sorted((ROOT/f'outputs/grid{stage}/runs').glob('r0__*__FULL')):
            date=folder.name.split('__')[1]
            assert date not in result;result[date]=folder
    return result
def callrow(c,path,cohort):
    rows=c['outcomes'];prefix=[r for r in rows if r['phase']=='shortlist'];local=[r for r in rows if r['phase']=='local_completion'];glob=[r for r in rows if r['phase']!='local_completion']
    assert len(prefix)==min(128,len(c['order'])) and len(rows)==len({r['pool_id'] for r in rows})==c['public_queries']
    b0,bg,bf=best(prefix),best(glob),best(rows)
    assert pid(b0)==c['shortlist_choice'] and pid(bg)==c['pre_extra_choice'] and pid(bf)==c['chosen_pool_id']
    assert len(local)==c['local_queries'] and (b0 is None or bf is not None and P.key(bf)>=P.key(b0))
    rec=reconstruct(c)
    for k in ['chosen_pool_id','public_queries','pre_extra_choice','local_queries','stop_reason']:assert rec[k]==c[k],(str(path),k)
    assert [r['phase'] for r in rec['outcomes']]==[r['phase'] for r in rows]
    first_change=next((i+1 for i in range(len(prefix),len(rows)) if pid(best(rows[:i+1]))!=pid(b0)),None)
    last=0;old=None
    for j,r in enumerate(rows):
        if r['strict_admissible'] and (old is None or P.key(r)>P.key(old)):last=j+1;old=r
    return dict(cohort=cohort,date=path.parent.name.split('__')[1],call=path.stem,step=c['step'],pool=len(c['order']),queries=len(rows),
                initial_safe=bool(b0 is not None and b0['rho']<.9),prefix_queries=len(prefix),global_extra_queries=len(glob)-len(prefix),local_queries=len(local),
                prefix_choice=pid(b0),global_choice=pid(bg),final_choice=pid(bf),global_changed=pid(bg)!=pid(b0),local_changed=pid(bf)!=pid(bg),
                final_changed=pid(bf)!=pid(b0),global_key_improved=keygain(b0,bg),local_key_improved=keygain(bg,bf),
                global_reward_gain=rewardgain(b0,bg),local_reward_gain=rewardgain(bg,bf),
                first_prefix_choice_change_query=first_change,last_incumbent_change_query=last,
                retrospective_queries_after_last_change=len(rows)-last,stop_reason=c['stop_reason'],source=path.relative_to(ROOT).as_posix())
def inventory_actual():
    rows=[];paths=[]
    for stage,cohort in [(70,'old13'),(81,'new6')]:
        for folder in sorted((ROOT/f'outputs/grid{stage}/runs').glob('r0__*__FRAME_GUARD')):
            for p in files(folder):rows.append(callrow(load(p),p,cohort));paths.append(p)
    assert len(rows)==131
    bydate=[]
    dates=read(ROOT/'outputs/grid70/design.json')['weeks']+read(ROOT/'outputs/grid81/design.json')['weeks']
    for d in dates:
        rr=[r for r in rows if r['date']==d];bydate.append(dict(date=d,cohort='old13' if d.startswith('2035') else 'new6',calls=len(rr),
          queries=sum(r['queries'] for r in rr),prefix=sum(r['prefix_queries'] for r in rr),global_extra=sum(r['global_extra_queries'] for r in rr),local=sum(r['local_queries'] for r in rr),
          safe_calls=sum(r['initial_safe'] for r in rr),global_changed=sum(r['global_changed'] for r in rr),local_changed=sum(r['local_changed'] for r in rr),
          final_changed=sum(r['final_changed'] for r in rr),local_triggered=sum(r['local_queries']>0 for r in rr)))
    summary=[]
    for cohort in ['old13','new6']:
        rr=[r for r in rows if r['cohort']==cohort];safes=[r for r in rr if r['initial_safe']]
        summary.append(dict(cohort=cohort,dates=sum(x['cohort']==cohort for x in bydate),calls=len(rr),
             queries=sum(r['queries'] for r in rr),prefix_queries=sum(r['prefix_queries'] for r in rr),global_extra_queries=sum(r['global_extra_queries'] for r in rr),local_queries=sum(r['local_queries'] for r in rr),
             initial_safe_calls=len(safes),safe_prefix_changed=sum(r['final_changed'] for r in safes),
             global_changed_calls=sum(r['global_changed'] for r in rr),local_triggered_calls=sum(r['local_queries']>0 for r in rr),local_changed_calls=sum(r['local_changed'] for r in rr),
             final_changed_calls=sum(r['final_changed'] for r in rr),no_final_change_calls=sum(not r['final_changed'] for r in rr),
             queries_on_no_final_change_calls=sum(r['queries']-r['prefix_queries'] for r in rr if not r['final_changed']),
             stop_reasons=dict(Counter(r['stop_reason'] for r in rr))))
    return rows,paths,bydate,summary
def offline_full(refs):
    output=[];paired=[];stops=[]
    for date,folder in sorted(refs.items()):
      cohort='old13' if date.startswith('2035') else 'new6'
      for p in files(folder):
        c=load(p);rows=c['outcomes'];truth={r['pool_id']:r for r in rows};order=c['order'];assert len(rows)==len(order)==len(truth) and [r['pool_id'] for r in rows]==order
        bf=best(rows);fg=reconstruct(c);ng=reconstruct(c,aliases={});old=reconstruct(c,policy='FRAME')
        # Equal-budget global extension: budget is determined before local feedback.
        pre=ng['chosen_pool_id'];seen={r['pool_id'] for r in ng['outcomes']};aliases={int(k):int(v) for k,v in c['aliases'].items()}
        local_ids=[]
        if pre is not None and len(seen)<len(order):
            initial=best(rows[:min(128,len(rows))]);safe=initial is not None and initial['rho']<.9
            if not safe or pre!=pid(initial):
                sub=SUBS[pre];targets={v for a,v in aliases.items() if SUBS[a]==sub};local_ids=[i for i in order if i not in seen and i in targets]
        assert local_ids==fg['completion_candidate_ids']
        extra=[i for i in order if i not in seen][:len(local_ids)]
        bmrows=ng['outcomes']+[dict(truth[i],phase='global_budget_extension') for i in extra]
        assert len(bmrows)==fg['public_queries'] and len({r['pool_id'] for r in bmrows})==len(bmrows)
        methods={'FULL':rows,'FRAME_GUARD':fg['outcomes'],'NO_LOCAL_GUARD':ng['outcomes'],'BUDGET_GLOBAL_GUARD':bmrows,'OLD_FRAME':old['outcomes']}
        for k in [128,256,512]:methods[f'FIXED_{k}']=rows[:min(k,len(rows))]
        for name,rs in methods.items():
            b=best(rs);gap=rewardgain(b,bf)
            assert gap is None or gap>=-1e-6
            output.append(dict(cohort=cohort,date=date,call=p.stem,step=c['step'],method=name,pool=len(rows),queries=len(rs),
                  full_has_incumbent=bf is not None,has_incumbent=b is not None,full_choice=pid(bf),chosen=pid(b),match_full_choice=pid(b)==pid(bf),
                  has_safe_choice=b is not None and b['rho']<.9,reward_gap_full=gap,source=p.relative_to(ROOT).as_posix()))
        loc=best(fg['outcomes']);bg=best(bmrows);nob=best(ng['outcomes'])
        paired.append(dict(cohort=cohort,date=date,call=p.stem,step=c['step'],global_queries=len(ng['outcomes']),extra_queries=len(local_ids),
                           local_choice=pid(loc),global_choice=pid(bg),no_local_choice=pid(nob),local_reward_gain=rewardgain(nob,loc),global_reward_gain=rewardgain(nob,bg),
                           local_better=loc is not None and (bg is None or P.key(loc)>P.key(bg)),global_better=bg is not None and (loc is None or P.key(bg)>P.key(loc)),
                           local_greater_reward=loc is not None and bg is not None and float(np.float32(loc['reward']))>float(np.float32(bg['reward']))+1e-6,
                           global_greater_reward=loc is not None and bg is not None and float(np.float32(bg['reward']))>float(np.float32(loc['reward']))+1e-6))
        seen_fg={r['pool_id'] for r in fg['outcomes']};remaining=[r for r in rows if r['pool_id'] not in seen_fg];rembest=best(remaining)
        stops.append(dict(cohort=cohort,date=date,call=p.stem,step=c['step'],pool=len(rows),queries=fg['public_queries'],
                          remaining=len(remaining),stop_reason=fg['stop_reason'],prefix_safe=fg['prefix_safe_at_entry'],
                          remaining_better_key=rembest is not None and (loc is None or P.key(rembest)>P.key(loc)),
                          missed_reward_gain=rewardgain(loc,bf)))
    return output,paired,stops
def matched_actual(paths,refs):
    maps={};result=[];feedback=0
    for date,folder in refs.items():
        m=defaultdict(list)
        for p in files(folder):m[signature(p)].append(p)
        maps[date]=m
    for p in paths:
        c=load(p);date=p.parent.name.split('__')[1];match=maps[date].get(signature(p),[])
        same=[]
        for f in match:
            fc=load(f)
            if c['order']==fc['order'] and c['aliases']==fc['aliases'] and c['rho_limit']==fc['rho_limit']:same.append(f)
        assert len(same)<=1
        gap=None;matches=None;fullchoice=None
        if same:
            fc=load(same[0]);truth={r['pool_id']:r for r in fc['outcomes']}
            for r in c['outcomes']:
                assert all(r[k]==truth[r['pool_id']][k] for k in ['rho','reward','done','exceptions','illegal','ambiguous','strict_admissible']);feedback+=1
            bf=best(fc['outcomes']);bc=best(c['outcomes']);gap=rewardgain(bc,bf);fullchoice=pid(bf);matches=pid(bc)==pid(bf)
        result.append(dict(cohort='old13' if date.startswith('2035') else 'new6',date=date,call=p.stem,step=c['step'],matched_full=len(same)==1,
                           reward_gap_full=gap,match_full_choice=matches,full_choice=fullchoice,guard_choice=c['chosen_pool_id'],
                           full_source=same[0].relative_to(ROOT).as_posix() if same else None))
    return result,feedback
def steps(folder):
    with gzip.open(source(folder/'steps.jsonl.gz'),'rt',encoding='utf-8') as f:return [json.loads(l) for l in f]
def first_divergences(refs):
    output=[]
    old_dates=read(ROOT/'outputs/grid70/design.json')['weeks']
    reviews={r['date']:r for r in load(ROOT/'outputs/grid70/review.json')['dates']}
    for date in old_dates:
        base=refs[date].parent/f'r0__{date}__FRAME';guard=ROOT/'outputs/grid70/runs'/f'r0__{date}__FRAME_GUARD'
        a=steps(base);b=steps(guard);n=min(len(a),len(b));diff=next((i for i in range(n) if a[i]['action_hash']!=b[i]['action_hash']),None)
        if diff is not None:assert all(a[i]['after_hash']==b[i]['after_hash'] for i in range(diff))
        callpaths={load(p)['step']:p for p in files(guard)};step=b[diff]['step'] if diff is not None else None;p=callpaths.get(step);c=load(p) if p is not None else None
        oldpaths={load(p)['step']:p for p in files(base)};op=oldpaths.get(step);oc=load(op) if op is not None else None
        same=bool(p and op and signature(p)==signature(op))
        rev=reviews[date]
        output.append(dict(date=date,first_action_divergence_step=step,old_steps=len(a),guard_steps=len(b),
                           divergence_has_guard_search=c is not None,divergence_same_observation_base=same,
                           prefix_safe_at_divergence=c.get('prefix_safe_at_entry') if c else None,
                           guard_prefix_choice=c['shortlist_choice'] if c else None,old_final_choice=oc['chosen_pool_id'] if oc else None,
                           guard_final_choice=c['chosen_pool_id'] if c else None,guard_queries=c['public_queries'] if c else None,
                           old_queries=oc['public_queries'] if oc else None,guard_cost_delta_full=rev['cost_delta_full'],old_cost_delta_full=rev['original_cost_delta_full'],
                           guard_source=p.relative_to(ROOT).as_posix() if p else None))
    return output
def summarize_offline(rows):
    result=[]
    for cohort in ['old13','new6']:
      for method in ['FULL','FIXED_128','FIXED_256','FIXED_512','OLD_FRAME','NO_LOCAL_GUARD','BUDGET_GLOBAL_GUARD','FRAME_GUARD']:
        rr=[r for r in rows if r['cohort']==cohort and r['method']==method];valid=[r for r in rr if r['full_has_incumbent']]
        ds=sorted({r['date'] for r in rr})
        date_q=[mean(r['queries'] for r in rr if r['date']==d) for d in ds]
        date_match=[mean(int(r['match_full_choice']) for r in valid if r['date']==d) for d in ds]
        date_gap=[mean(r['reward_gap_full'] for r in valid if r['date']==d) for d in ds]
        result.append(dict(cohort=cohort,method=method,calls=len(rr),dates_with_search=len(ds),full_admissible_calls=len(valid),
                           queries=sum(r['queries'] for r in rr),mean_queries=mean(r['queries'] for r in rr),date_macro_mean_queries=mean(date_q),
                           missed_admissible_calls=sum(not r['has_incumbent'] for r in valid),
                           match_full_calls=sum(r['match_full_choice'] for r in valid),match_full_fraction=mean(int(r['match_full_choice']) for r in valid),
                           date_macro_match_full_fraction=mean(date_match),mean_reward_gap_on_common_admissible=mean(r['reward_gap_full'] for r in valid),
                           date_macro_mean_reward_gap_on_common_admissible=mean(date_gap),max_reward_gap=max((r['reward_gap_full'] for r in valid if r['reward_gap_full'] is not None),default=None),
                           safe_choice_calls=sum(r['has_safe_choice'] for r in valid)))
    return result
def main():
    assert not (OUT/'analysis_finished.json').exists()
    for n in ['outputs/grid70/review.json','outputs/grid81/review.json','outputs/grid84/review.json']:
        assert load(ROOT/n)['passed_saved_data']
    plan_sha=sha(OUT/'analysis_plan.md')
    actual,paths,dates,summary=inventory_actual();refs=fullrefs();assert len(refs)==19
    offline,paired,stops=offline_full(refs);matched,feedback=matched_actual(paths,refs);first=first_divergences(refs)
    for name,rows in [('actual_calls.csv',actual),('actual_dates.csv',dates),('offline_full_calls.csv',offline),('offline_method_summary.csv',summarize_offline(offline)),('equal_budget_pairs.csv',paired),('stopping_suffix_audit.csv',stops),('matched_actual_full.csv',matched),('first_action_divergences.csv',first)]:csvsave(name,rows)
    save('actual_summary.json',summary);save('offline_summary.json',summarize_offline(offline))
    save('mechanism_results.json',dict(actual=summary,offline=summarize_offline(offline),
        paired=[dict(cohort=g,calls=sum(r['cohort']==g for r in paired),active_calls=sum(r['cohort']==g and r['extra_queries']>0 for r in paired),
                     local_better=sum(r['cohort']==g and r['local_better'] for r in paired),global_better=sum(r['cohort']==g and r['global_better'] for r in paired),
                     local_greater_reward=sum(r['cohort']==g and r['local_greater_reward'] for r in paired),global_greater_reward=sum(r['cohort']==g and r['global_greater_reward'] for r in paired)) for g in ['old13','new6']],
        matched_actual=dict(calls=len(matched),matched=sum(r['matched_full'] for r in matched),verified_feedback=feedback),
        first_divergences=first,stopping=[dict(cohort=g,early_stops=sum(r['cohort']==g and r['remaining']>0 for r in stops),
          early_stops_with_better_suffix=sum(r['cohort']==g and r['remaining']>0 and r['remaining_better_key'] for r in stops)) for g in ['old13','new6']]))
    for p in [Path(__file__).resolve(),ROOT/'work/grid70/policies.py',ROOT/'work/grid67/policies.py',ROOT/'work/grid63/policies.py',ROOT/'work/grid60/policies.py',ROOT/'outputs/grid28/offline.json',OUT/'analysis_plan.md']:source(p)
    save('source_receipt.json',dict(plan_sha256=plan_sha,files=[dict(path=p,sha256=h) for p,h in sorted(INPUTS.items())],new_fits=0,new_simulation=0,scope='Post-hoc saved-data mechanism study. Same-state offline reward does not identify closed-loop value.'))
    save('analysis_finished.json',dict(passed=True,actual_calls=len(actual),dates=len(dates),full_calls=len(stops),offline_rows=len(offline),new_fits=0,new_simulation=0))
    print(json.dumps(read(OUT/'analysis_finished.json'),ensure_ascii=False));print(json.dumps(read(OUT/'mechanism_results.json'),ensure_ascii=False))
if __name__=='__main__':main()
