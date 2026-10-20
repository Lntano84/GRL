"""Independent saved-vector, prefix, memory, query-plan and label checks."""
from pathlib import Path
from collections import Counter
import copy,csv,gzip,hashlib,importlib.util,json,math,re
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid96_value_learning'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def loadmod(name,p):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def rows(p):
    with gzip.open(p/'steps.jsonl.gz','rt',encoding='utf-8') as f:return [json.loads(x) for x in f]
def norm(w):
    v=copy.deepcopy(w['inventory'])
    for name in ['base_optimizer','restore_optimizer','reconnect','recover_topology','topology_search']:v[name]['values'].pop('space_prng',None)
    return v
def comparison(a,b,identical,external):
    if external:return 'UNRESOLVED','different_realized_external_attack'
    if identical and b['all_queries']<=a['all_queries']:return 'SKIP','equivalent_entire_physical_path'
    if a['reaches_horizon']!=b['reaches_horizon']:return ('KEEP' if a['reaches_horizon'] else 'SKIP'),'survival_priority'
    if not a['reaches_horizon']:return 'UNRESOLVED','different_both_incomplete_paths'
    tolerances=[max(.01,32*np.finfo(float).eps*(abs(a['cost'])+abs(b['cost']))),0,0,1e-6,1e-6]
    keys=['cost','all_queries','overload_steps','max_rho','overload_excess_sum']
    av=[a[k] for k in keys];bv=[b[k] for k in keys]
    if all(y<=x+t for x,y,t in zip(av,bv,tolerances)) and any(y<x-t for x,y,t in zip(av,bv,tolerances)):return 'SKIP','pareto_skip'
    if all(x<=y+t for x,y,t in zip(av,bv,tolerances)) and any(x<y-t for x,y,t in zip(av,bv,tolerances)):return 'KEEP','pareto_completion'
    return 'MIXED','mixed_or_numerically_equivalent_objectives'
def main():
    D=read(OUT/'design.json');assert read(OUT/'finished.json')['runs']==D['new_replays'];assert read(OUT/'qualification.json')['passed']
    for n in ['code_freeze.json','data_freeze.json']:
        for rel,h in read(OUT/n).items():assert sha(ROOT/rel)==h,rel
    for rel in ['work/grid02/ljn_nn/modules/base_module.py','work/grid02/ljn_nn/modules/topology_heuristic.py','work/grid02/ljn_nn/modules/convex_optim.py']:
        assert not re.search(r'space_prng|np\.random|random\.|\.sample\(|\.shuffle\(|\.permutation\(',(ROOT/rel).read_text(encoding='utf-8'))
    I=loadmod('g96_independent',ROOT/'work/grid89_review/reference_logic.py');F=loadmod('g96_features',ROOT/'work/grid94_predictability/data.py');H=loadmod('g96_horizons',ROOT/'work/grid95_labels/audit.py')
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];checks=Counter();pack={}
    def inspect(p,omit):
        key=p.as_posix()
        if key in pack:return pack[key]
        s=read(p/'summary.json');rr=rows(p)
        with np.load(p/'vectors.npz') as z:av=z['action'];ov=z['observation']
        assert len(rr)==len(av)==len(ov)==s['steps'];assert abs(math.fsum(r['raw_cost'] for r in rr)-s['cost'])<1e-6
        assert s['illegal']==s['ambiguous']==0;assert sum(r['simulations'] for r in rr)==s['public_forecasts']
        for j,r in enumerate(rr):
            assert digest(av[j])==r['action_hash'] and digest(ov[j])==r['after_hash'];checks['vector_hashes']+=2
            if j:assert r['before_hash']==rr[j-1]['after_hash']
            if r['ledger']:
                v=r['ledger'];cost=v['marginal_cost']*sum(v[k] for k in ['losses_mwh','redispatch_mwh','curtailment_delta_mwh','storage_throughput_mwh'])
                assert abs(cost-r['raw_cost'])<=r['ledger_tolerance']+1e-9;checks['cost_steps']+=1
        cc=[];interventions=0
        for path in sorted(p.glob('call[0-9][0-9][0-9].json')):
            c=read(path);cc.append(c);truth={r['pool_id']:r for r in c['outcomes']};aliases={int(k):v for k,v in c['aliases'].items()}
            skip=bool(c.get('label_intervention',{}).get('omitted_once'))
            if skip:assert c['label_intervention']['case_id']==omit;interventions+=1
            pp,ch,budget=I.replay(c['order'],truth,subs,aliases,'NO_LOCAL' if skip else 'FRAME_GUARD',dict(p=128,h=64,patience=2,safe=.9,qmin=0))
            assert pp==[r['pool_id'] for r in c['outcomes']] and ch==c['chosen_pool_id'] and budget==c['completion_budget']
            assert c['local_queries']==(0 if skip else budget) and len(truth)==c['public_queries']
            for r in c['outcomes']:assert r['strict_admissible']==bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['illegal'] and not r['ambiguous'] and not r['exceptions'])
            with np.load(path.with_name(path.stem+'_state.npz')) as z:assert digest(z['observation'])==c['state_hash']
            checks['independent_calls']+=1;checks['feedback_rows']+=len(truth)
        assert interventions==int(omit is not None);assert len(cc)==s['n1_calls'] and sum(c['public_queries'] for c in cc)==s['n1_forecasts']
        result=dict(folder=p,summary=s,rows=rr,calls=cc);pack[key]=result;checks['runs']+=1;checks['physical_steps']+=len(rr);return result
    for item in D['execution_order']:inspect(OUT/'runs'/f"r0__{item['week']}__{item['method']}",item['omit_case'])
    labels=[]
    for c in D['cases']:
        base=inspect(OUT/'runs'/f"r0__{c['week']}__FRAME_VALUE_QUAL",None)
        bp=ROOT/c['reused_branch'] if c['pilot_known'] else OUT/'runs'/f"r0__{c['week']}__SKIP_{c['id']}"
        bid=c['reuse_case_id'] if c['pilot_known'] else c['id'];branch=inspect(bp,bid)
        a,b=base['rows'],branch['rows'];t=c['step'];assert len(a)>=t and len(b)>=t
        keys=['before_hash','action_hash','after_hash','raw_cost','done','complete','simulations']
        assert all(all(x[k]==y[k] for k in keys) for x,y in zip(a[:t-1],b[:t-1]))
        assert a[t-1]['before_hash']==b[t-1]['before_hash']==c['state_hash'];checks['prefix_steps']+=t-1
        wa=read(base['folder']/f"{c['id']}_witness.json");wb=read(bp/f'{bid}_witness.json')
        assert wa['state_hash']==wb['state_hash']==c['state_hash'] and wa['base_action_hash']==wb['base_action_hash']==c['base_action_hash']
        for k in ['before_search','after_global']:assert norm(wa[k])==norm(wb[k]);checks['memory_pairs']+=1
        assert wa['global_feedback']==wb['global_feedback'] and wa['public_queries']-wb['public_queries']==c['completion_budget']
        keys2=['before_hash','action_hash','after_hash','raw_cost','done','complete']
        identical=len(a)==len(b) and all(all(x[k]==y[k] for k in keys2) for x,y in zip(a,b))
        if c['before_choice']==c['after_choice']:assert identical;checks['same_choice_negative_controls']+=1
        ext=[j+1 for j in range(t-1,min(len(a),len(b))) if a[j]['label_audit']['opponent_realization']!=b[j]['label_audit']['opponent_realization']]
        target=next(x for x in base['calls'] if x['step']==t);prior=[r for r in target['outcomes'] if r['phase']!='local_completion'];aliases={int(k):v for k,v in target['aliases'].items()}
        x,no=F.features(target['order'],prior,aliases,subs,target['rho_limit']);assert x==c['features']
        hidden=[dict(r) if r['phase']!='local_completion' else dict(r,rho=999.,reward=-999.,strict_admissible=False) for r in target['outcomes']]
        xx,_=F.features(target['order'],hidden,aliases,subs,target['rho_limit']);assert xx==x;checks['feature_hidden_suffix_checks']+=1
        horizons={}
        for name,h in [('primary',D['nominal_episode_steps']-t+1),('one_step',1),('one_hour',12),('one_day',288)]:
            end=min(t+h-1,D['nominal_episode_steps']);aa=H.window_stats(a,base['calls'],t,end);bb=H.window_stats(b,branch['calls'],t,end)
            horizons[name]=dict(end_step=end,completion=aa,skip=bb)
        aa=horizons['primary']['completion'];bb=horizons['primary']['skip'];lab,reason=comparison(aa,bb,identical,ext)
        labels.append(dict(case_id=c['id'],week=c['week'],calendar_date=c['calendar_date'],cohort=c['cohort'],step=t,pilot_known=c['pilot_known'],label=lab,reason=reason,
            features=x,before_choice=c['before_choice'],after_choice=c['after_choice'],local_queries=c['completion_budget'],physical_path_identical=identical,
            first_action_difference=next((x['step'] for x,y in zip(a,b) if x['action_hash']!=y['action_hash']),None),
            completion_steps=len(a),skip_steps=len(b),completion_complete=base['summary']['complete'],skip_complete=branch['summary']['complete'],
            external_differences=ext,horizons=horizons,
            cost_skip_minus_completion=bb['cost']-aa['cost'] if aa['reaches_horizon'] and bb['reaches_horizon'] else 0. if identical else None,
            queries_saved=aa['all_queries']-bb['all_queries'],candidate_queries_saved=aa['candidate_queries']-bb['candidate_queries']))
    assert len(labels)==56 and checks['runs']==77
    save('labels.json',labels)
    fields=['case_id','week','calendar_date','cohort','step','pilot_known','label','reason','before_choice','after_choice','local_queries','physical_path_identical','completion_steps','skip_steps','completion_complete','skip_complete','cost_skip_minus_completion','queries_saved','candidate_queries_saved']
    with (OUT/'labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fields);wr.writeheader();wr.writerows({k:r[k] for k in fields} for r in labels)
    save('review.json',dict(passed_saved_data=True,checks=dict(checks),counts=dict(Counter(r['label'] for r in labels)),headline_counts=dict(Counter(r['label'] for r in labels if not r['pilot_known'])),independent_ac_resolve=False,
        limits='Single omissions under original continuation; all exposed same-grid trajectories; full-prefix reconstruction, not opaque engine serialization. Known pilots excluded headline evaluation.'))
    print(json.dumps(read(OUT/'review.json'),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
