"""Saved-data qualification and horizon-specific single-intervention value labels."""
from pathlib import Path
import csv,gzip,hashlib,importlib.util,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid95_labels'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def steps(folder):
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:return [json.loads(x) for x in f]


def loadmod(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def folder(item):return OUT/'runs'/f"r0__{item['week']}__{item['method']}"


def window_stats(rows,calls,start,end):
    rr=[r for r in rows if start<=r['step']<=end];cc=[c for c in calls if start<=c['step']<=end]
    return dict(steps=len(rr),reaches_horizon=len(rows)>=end and not any(r['done'] and not r['complete'] for r in rows[:end]),
        cost=math.fsum(r['raw_cost'] for r in rr),all_queries=sum(r['simulations'] for r in rr),
        candidate_queries=sum(c['public_queries'] for c in cc),
        overload_steps=sum((r['label_audit']['overloaded_lines'] or 0)>0 for r in rr),
        overload_excess_sum=math.fsum(r['label_audit']['summed_overload_excess'] or 0. for r in rr),
        max_rho=max((r['label_audit']['max_rho'] for r in rr if r['label_audit']['max_rho'] is not None),default=None),
        exceptions=sum(bool(r['exceptions']) for r in rr),illegal=sum(r['illegal'] for r in rr))


def main():
    D=read(OUT/'design.json');done=read(OUT/'finished.json');assert done['passed_engineering'] and done['runs']==6
    assert read(OUT/'qualification.json')['passed']
    for n in ['code_freeze.json','data_freeze.json']:
        for rel,h in read(OUT/n).items():assert sha(ROOT/rel)==h,rel
    I=loadmod('g95_independent',ROOT/'work/grid89_review/reference_logic.py')
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
    checks=dict(runs=0,steps=0,vectors=0,cost_ledger_steps=0,independent_calls=0,feedback=0,prefix_steps=0,memory_pairs=0)
    pack={}
    for item in D['execution_order']:
        p=folder(item);s=read(p/'summary.json');rr=steps(p)
        with np.load(p/'vectors.npz') as z:av=z['action'];ov=z['observation']
        assert len(rr)==len(av)==len(ov)==s['steps']
        assert abs(math.fsum(r['raw_cost'] for r in rr)-s['cost'])<1e-6
        assert s['illegal']==s['ambiguous']==0
        assert sum(r['simulations'] for r in rr)==s['public_forecasts']
        cc=[]
        for j,r in enumerate(rr):
            assert digest(av[j])==r['action_hash'] and digest(ov[j])==r['after_hash'];checks['vectors']+=2
            if j:assert r['before_hash']==rr[j-1]['after_hash']
            if r['ledger']:
                v=r['ledger'];cost=v['marginal_cost']*sum(v[k] for k in ['losses_mwh','redispatch_mwh','curtailment_delta_mwh','storage_throughput_mwh'])
                assert abs(cost-r['raw_cost'])<=r['ledger_tolerance']+1e-9;checks['cost_ledger_steps']+=1
        interventions=0
        for path in sorted(p.glob('call[0-9][0-9][0-9].json')):
            c=read(path);cc.append(c);truth={r['pool_id']:r for r in c['outcomes']};aliases={int(k):v for k,v in c['aliases'].items()}
            omit=bool(c.get('label_intervention',{}).get('omitted_once'))
            if omit:
                assert c['label_intervention']['case_id']==item['omit_case'];interventions+=1
            pp,choice,budget=I.replay(c['order'],truth,subs,aliases,'NO_LOCAL' if omit else 'FRAME_GUARD',dict(p=128,h=64,patience=2,safe=.9,qmin=0))
            assert pp==[r['pool_id'] for r in c['outcomes']] and choice==c['chosen_pool_id'] and budget==c['completion_budget']
            assert c['local_queries']==(0 if omit else budget)
            assert len(truth)==c['public_queries']
            for r in c['outcomes']:
                assert r['strict_admissible']==bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['illegal'] and not r['ambiguous'] and not r['exceptions'])
            with np.load(path.with_name(path.stem+'_state.npz')) as z:assert digest(z['observation'])==c['state_hash']
            checks['independent_calls']+=1;checks['feedback']+=len(truth)
        assert interventions==int(item['omit_case'] is not None)
        assert len(cc)==s['n1_calls'] and sum(c['public_queries'] for c in cc)==s['n1_forecasts']
        pack[item['method']+'|'+item['week']]=dict(item=item,folder=p,summary=s,rows=rr,calls=cc)
        checks['runs']+=1;checks['steps']+=len(rr)
    labels=[];case_results=[]
    for case in D['cases']:
        base=pack['FRAME_LABEL_QUAL|'+case['week']];branch=pack['SKIP_'+case['id']+'|'+case['week']]
        a,b=base['rows'],branch['rows'];t=case['step']
        keys=['before_hash','action_hash','after_hash','raw_cost','done','complete','simulations']
        assert len(b)>=t and all(all(x[k]==y[k] for k in keys) for x,y in zip(a[:t-1],b[:t-1]))
        assert a[t-1]['before_hash']==b[t-1]['before_hash']==case['state_hash'];checks['prefix_steps']+=t-1
        wa=read(base['folder']/f"{case['id']}_witness.json");wb=read(branch['folder']/f"{case['id']}_witness.json")
        assert wa['state_hash']==wb['state_hash'] and wa['base_action_hash']==wb['base_action_hash']
        assert wa['before_search']['sha256']==wb['before_search']['sha256']
        assert wa['after_global']['sha256']==wb['after_global']['sha256']
        assert wa['global_feedback']==wb['global_feedback'];checks['memory_pairs']+=1
        assert wa['public_queries']-wb['public_queries']==case['completion_budget']
        keys2=['before_hash','action_hash','after_hash','raw_cost','done','complete']
        identical=len(a)==len(b) and all(all(x[k]==y[k] for k in keys2) for x,y in zip(a,b))
        if case['id']=='C0_SAME_ACTION':assert identical,'Same-choice omission changes physical path: labels not qualified'
        changed=next((x['step'] for x,y in zip(a,b) if x['action_hash']!=y['action_hash']),None)
        case_results.append(dict(**case,exact_preintervention_prefix=True,reviewed_memory_match=True,
            complete_steps=len(a),skip_steps=len(b),complete_finished=base['summary']['complete'],skip_finished=branch['summary']['complete'],
            physical_path_unchanged=identical,first_delivered_action_difference=changed,
            complete_full_cost=base['summary']['cost'],skip_full_cost=branch['summary']['cost']))
        for name,h in [('one_step',1),('one_hour',12),('one_day',288),('remaining_week',len(a)-t+1)]:
            end=min(t+h-1,len(a));aa=window_stats(a,base['calls'],t,end);bb=window_stats(b,branch['calls'],t,end)
            common_end=min(end,len(a),len(b))
            extdiff=[j+1 for j in range(t-1,common_end) if a[j]['label_audit']['opponent_realization']!=b[j]['label_audit']['opponent_realization']]
            both=aa['reaches_horizon'] and bb['reaches_horizon'];dc=bb['cost']-aa['cost'] if both else None
            dq=aa['all_queries']-bb['all_queries'] if both else None
            tol=max(.01,32*np.finfo(np.float64).eps*(abs(aa['cost'])+abs(bb['cost'])))
            if aa['reaches_horizon'] and not bb['reaches_horizon']:label='COMPLETE_SURVIVAL_ADVANTAGE'
            elif bb['reaches_horizon'] and not aa['reaches_horizon']:label='SKIP_SURVIVAL_ADVANTAGE'
            elif not both:label='BOTH_INCOMPLETE_COST_INCOMPARABLE'
            elif abs(dc)<=tol:label='SKIP_SAME_COST_LESS_COMPUTE' if dq>0 else 'COMPLETE_SAME_COST_LESS_COMPUTE' if dq<0 else 'EQUIVALENT_REPORTED_COST_AND_COMPUTE'
            elif dc>0:label='COST_COMPUTE_TRADEOFF_COMPLETION_CHEAPER' if dq>0 else 'COMPLETE_COST_COMPUTE_DOMINATES'
            else:label='SKIP_COST_COMPUTE_DOMINATES' if dq>=0 else 'COST_COMPUTE_TRADEOFF_SKIP_CHEAPER'
            if extdiff:label='UNRESOLVED_EXTERNAL_DIFFERENCE'
            labels.append(dict(case_id=case['id'],week=case['week'],intervention_step=t,horizon=name,
                requested_steps=h,actual_horizon_steps=end-t+1,end_step=end,complete_branch=aa,skip_branch=bb,
                delta_cost_skip_minus_complete=dc,delta_all_queries_complete_minus_skip=dq,
                delta_candidate_queries_complete_minus_skip=aa['candidate_queries']-bb['candidate_queries'] if both else None,
                equality_tolerance=tol,label=label,external_attack_differences=extdiff,
                scope='Reported cost/compute relation conditional on survival and original FRAME continuation; overload indicators remain separate. Not a universal good/bad label.'))
    save('labels.json',labels);save('case_results.json',case_results)
    with (OUT/'labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['case_id','week','intervention_step','horizon','actual_horizon_steps','label','delta_cost_skip_minus_complete','delta_all_queries_complete_minus_skip','delta_candidate_queries_complete_minus_skip','complete_steps','skip_steps','complete_overload_steps','skip_overload_steps']
        wr=csv.DictWriter(f,fields);wr.writeheader()
        for l in labels:wr.writerow({**{k:l[k] for k in fields if k in l},'complete_steps':l['complete_branch']['steps'],'skip_steps':l['skip_branch']['steps'],
            'complete_overload_steps':l['complete_branch']['overload_steps'],'skip_overload_steps':l['skip_branch']['overload_steps']})
    save('review.json',dict(passed_saved_data=True,checks=checks,same_choice_negative_control_passed=True,
        factual_trajectories_reproduced=2,reviewed_memory_matches=4,independent_ac_resolve=False,new_fits=0,
        limits='Four selected exposed states on two dates; numerical/RNG witnesses do not serialize every engine object. Full prefix reconstruction and factual qualification verified.'))
    save('decision.json',dict(stage=D['stage'],label_collection_mechanics_qualified=True,universal_binary_label_established=False,
        state_cases=4,primary_horizon='remaining_week',new_fits=0,
        results=[dict(case_id=l['case_id'],label=l['label'],delta_cost=l['delta_cost_skip_minus_complete'],delta_queries=l['delta_all_queries_complete_minus_skip']) for l in labels if l['horizon']=='remaining_week'],
        no_threshold_or_weight_selection=True))
    print(json.dumps(dict(checks=checks,decision=read(OUT/'decision.json')),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
