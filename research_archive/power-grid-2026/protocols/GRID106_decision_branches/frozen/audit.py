"""Saved vectors, policies, matched prefixes and multihorizon causal-label audit."""
import collections,gzip,json,math
import numpy as np
from common106 import ROOT,OUT,read,save,verify,digest,loadmod
def rows(p):
    with gzip.open(p/'steps.jsonl.gz','rt',encoding='utf8') as f:return [json.loads(x) for x in f]
def window(rs,cs,t,end):
    subset=rs[t-1:end];valid=[r['label_audit']['max_rho'] for r in subset if r['label_audit']['max_rho'] is not None]
    return dict(reaches_horizon=len(rs)>=end,steps=len(subset),cost=math.fsum(r['raw_cost'] for r in subset),
        all_queries=sum(r['simulations'] for r in subset),candidate_queries=sum(c['public_queries'] for c in cs if t<=c['step']<=end),
        local_queries=sum(c['local_queries'] for c in cs if t<=c['step']<=end),
        overload_steps=sum(r['label_audit']['overloaded_lines']>0 for r in subset if r['label_audit']['overloaded_lines'] is not None),
        max_rho=max(valid) if valid else None)
def main():
    assert not (OUT/'review.json').exists(),'Never repeat completed audit'
    verify();D=read(OUT/'design.json');assert read(OUT/'finished.json')['passed_engineering'] and read(OUT/'qualification.json')['passed']
    G=loadmod('grid106_audit_policy','work/grid91_mechanism/policies.py')
    H=loadmod('grid106_audit_hooks','work/grid106_decision_branches/hooks106.py')
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];checks=collections.Counter();pack={}
    for item in D['execution_order']:
        p=OUT/'runs'/f"r0__{item['week']}__{item['method']}";rs=rows(p);s=read(p/'summary.json')
        assert len(rs)==s['steps'] and abs(math.fsum(r['raw_cost'] for r in rs)-s['cost'])<.01
        assert sum(r['simulations'] for r in rs)==s['public_forecasts']
        assert sum(r['illegal'] for r in rs)==sum(r['ambiguous'] for r in rs)==sum(bool(r['exceptions']) for r in rs)==0
        with np.load(p/'vectors.npz') as z:actions=z['action'];observations=z['observation']
        assert len(actions)==len(observations)==len(rs)
        for j,r in enumerate(rs):
            assert digest(actions[j])==r['action_hash'] and digest(observations[j])==r['after_hash']
            if j:assert rs[j-1]['after_hash']==r['before_hash']
            if r['ledger']:
                ledger=r['ledger'];recomputed=ledger['marginal_cost']*sum(ledger[k] for k in ['losses_mwh','redispatch_mwh','curtailment_delta_mwh','storage_throughput_mwh'])
                assert abs(recomputed-r['raw_cost'])<=r['ledger_tolerance'];checks['stored_cost_ledgers']+=1
        cs=[]
        for f in sorted(p.glob('call[0-9][0-9][0-9].json')):
            c=read(f);cs.append(c);out=c['outcomes'];truth={r['pool_id']:r for r in out};assert len(truth)==len(out)==c['public_queries']
            for r in out:
                valid=bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['illegal'] and not r['ambiguous'] and not r['exceptions'])
                assert valid==r['strict_admissible'];checks['feedback']+=1
            with np.load(f.with_name(f.stem+'_state.npz')) as z:
                assert digest(z['observation'])==c['state_hash']==rs[c['step']-1]['before_hash']
            aliases={int(k):int(v) for k,v in c['aliases'].items()};seen=[]
            def reveal(i,phase):seen.append(i);return dict(truth[i],phase=phase)
            omitted=bool(c.get('single_intervention',{}).get('omitted_once'))
            ans=G.run(c['order'],reveal,'NO_LOCAL' if omitted else 'FRAME_GUARD',subs,aliases,limit=c['rho_limit'])
            for k in ['chosen_pool_id','public_queries','local_queries','completion_budget','completion_candidate_ids','extra_selected_ids']:
                assert ans[k]==c[k],(item['method'],f.name,k)
            assert seen==[r['pool_id'] for r in out];checks['policy_replays']+=1
        assert sum(c['public_queries'] for c in cs)==s['n1_forecasts'] and len(cs)==s['n1_calls']
        assert sum(c.get('single_intervention',{}).get('omitted_once',False) for c in cs)==int(item['omit_case'] is not None)
        pack[item['method']]=dict(folder=p,rows=rs,calls=cs,summary=s);checks['steps']+=len(rs);checks['vectors']+=2*len(rs)
    base=pack['FRAME_QUAL'];labels=[];cases=[]
    for c in D['cases']:
        branch=pack[f"SKIP_{c['step']}"];a,b=base['rows'],branch['rows'];t=c['step']
        keys=['step','before_hash','action_hash','after_hash','raw_cost','done','complete','simulations']
        assert len(b)>=t and all(all(x[k]==y[k] for k in keys) for x,y in zip(a[:t-1],b[:t-1]))
        assert a[t-1]['before_hash']==b[t-1]['before_hash']==c['state_hash']
        wa=read(base['folder']/f"{c['id']}_witness.json");wb=read(branch['folder']/f"{c['id']}_witness.json")
        assert H.equal_memory(wa['before_search'],wb['before_search']) and H.equal_memory(wa['after_global'],wb['after_global'])
        assert wa['global_feedback']==wb['global_feedback'] and wa['base_action_hash']==wb['base_action_hash']
        assert wa['public_queries']-wb['public_queries']==c['completion_budget']
        checks['qualified_prefix_steps']+=t-1;checks['memory_pairs']+=2
        # Include all earlier paid search calls, not only delivered physical vectors.
        ac=[x for x in base['calls'] if x['step']<t];bc=[x for x in branch['calls'] if x['step']<t];assert len(ac)==len(bc)
        fields=['pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible','phase','combined_hash']
        for x,y in zip(ac,bc):
            assert all(x[k]==y[k] for k in ['step','state_hash','order','aliases','chosen_pool_id','public_queries'])
            assert [{k:r[k] for k in fields} for r in x['outcomes']]==[{k:r[k] for k in fields} for r in y['outcomes']]
        cases.append(dict(case_id=c['id'],step=t,frame_steps=len(a),skip_steps=len(b),frame_complete=base['summary']['complete'],
            skip_complete=branch['summary']['complete'],frame_week_cost=base['summary']['cost'],skip_week_cost=branch['summary']['cost'],
            first_action_difference=next((x['step'] for x,y in zip(a,b) if x['action_hash']!=y['action_hash']),None),
            full_prefix_and_memory_verified=True))
        for name,h in [('one_step',1),('one_hour',12),('one_day',288),('remaining_week',len(a)-t+1)]:
            end=min(t+h-1,len(a));aa=window(a,base['calls'],t,end);bb=window(b,branch['calls'],t,end)
            ext=[j+1 for j in range(t-1,min(end,len(a),len(b))) if a[j]['label_audit']['opponent_realization']!=b[j]['label_audit']['opponent_realization']]
            both=aa['reaches_horizon'] and bb['reaches_horizon'];dc=bb['cost']-aa['cost'] if both else None;dq=aa['all_queries']-bb['all_queries'] if both else None
            if not both:label='FRAME_SURVIVAL_ADVANTAGE' if aa['reaches_horizon'] else 'INCOMPLETE_COST_INCOMPARABLE'
            elif abs(dc)<=.01:label='SAME_REPORTED_COST'
            elif dc>0:label='COMPLETION_CHEAPER_COMPUTE_TRADEOFF' if dq>0 else 'COMPLETION_COST_AND_QUERY_ADVANTAGE'
            else:label='SKIP_COST_AND_QUERY_ADVANTAGE' if dq>=0 else 'SKIP_CHEAPER_COMPUTE_TRADEOFF'
            if ext:label='UNRESOLVED_EXTERNAL_DIFFERENCE'
            labels.append(dict(case_id=c['id'],intervention_step=t,horizon=name,end_step=end,horizon_steps=end-t+1,frame=aa,skip=bb,
                delta_cost_skip_minus_frame=dc,delta_all_queries_frame_minus_skip=dq,
                delta_candidate_queries_frame_minus_skip=aa['candidate_queries']-bb['candidate_queries'] if both else None,
                external_opponent_difference_steps=ext,label=label))
    save(OUT/'labels.json',labels);save(OUT/'case_results.json',cases)
    save(OUT/'review.json',dict(passed=True,checks=dict(checks),factual_trajectory_reproduced=True,one_omission_per_branch=True,
        original_global_search_and_continuation_preserved=True,new_fits=0,audit_forecasts=0,independent_ac_resolve=False,
        memory='GRID95 numeric/used seeded RNG witnesses; same five proven-unused inherited RNG exclusions, no opaque engine clone.'))
    save(OUT/'decision.json',dict(diagnostic_completed=True,primary_horizon='remaining_week',new_fits=0,training_must_notify_user_first=True,
        ready_to_scale=False,independent_training_or_test_data=False,
        results=[x for x in labels if x['horizon']=='remaining_week']))
    print(json.dumps(dict(checks=dict(checks),cases=cases,primary=read(OUT/'decision.json')['results']),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
