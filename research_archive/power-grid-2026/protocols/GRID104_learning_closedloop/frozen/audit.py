"""Saved-vector/feedback/online-policy/time checks after all benchmark processes end."""
import collections,gzip,hashlib,importlib.util,json,math,sys
from registry import ROOT,W,OUT,read,save,verify

def steps(folder):
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf8') as f:return [json.loads(s) for s in f]
def signature(rs):return hashlib.sha256(json.dumps([(r['before_hash'],r['action_hash'],r['after_hash'],r['raw_cost'],r['done'],r['complete']) for r in rs],separators=(',',':')).encode()).hexdigest()

def main():
    verify();D=read(OUT/'design.json');summaries=read(OUT/'finished.json')['summaries'];assert len(summaries)==48
    sys.path[:0]=[str(ROOT/'work/grid30'),str(ROOT/'work/grid29'),str(ROOT/'work/grid08'),str(ROOT/'work/grid28')]
    from common import np,grid2op,LightSimBackend,MaxRhoReward,operational_other_rewards,ResidualControl,OptimModule,action_vector,graph_features
    from engine import Search
    module=sys.modules[OptimModule.__module__]
    if not hasattr(OptimModule,'logger'):OptimModule.logger=module.logger
    spec=importlib.util.spec_from_file_location('grid104_audit_policy',W/'policies.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
    policies={m['method']:P.CompletionPolicy(m['method'],m['model_seed']) for m in D['methods']}
    env=grid2op.make(str(ROOT/D['environment']),test=True,backend=LightSimBackend(),reward_class=MaxRhoReward,other_rewards=operational_other_rewards())
    env.seed(0);env.set_id(D['selected'][0]['scenario']);template=env.reset();ctl=ResidualControl(env,template);search=Search(env,ctl,'STATIC_ID',128,None)
    before_forecasts=int(env.nb_highres_called);checks=collections.Counter();runrows=[];stepcache={};callcache={}
    digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    try:
        for s in summaries:
            key=(s['repeat'],s['week'],s['method']);folder=OUT/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
            assert read(folder/'summary.json')==s
            rs=steps(folder);stepcache[key]=rs;assert len(rs)==s['steps']
            assert abs(math.fsum(r['raw_cost'] for r in rs)-s['cost'])<1e-6
            assert abs(math.fsum(r['decision_s'] for r in rs)-s['controller_s'])<1e-6
            assert sum(r['simulations'] for r in rs)==s['public_forecasts']
            assert s['illegal']==s['ambiguous']==0
            with np.load(folder/'vectors.npz') as z:
                assert len(z['action'])==len(z['observation'])==len(rs)
                for j,r in enumerate(rs):
                    assert digest(z['action'][j])==r['action_hash'] and digest(z['observation'][j])==r['after_hash']
                    if j:assert r['before_hash']==rs[j-1]['after_hash']
                    if r['ledger'] is not None:assert abs(r['ledger']['recomputed_raw_cost']-r['raw_cost'])<=r['ledger_tolerance']
                    if r['exceptions']:assert r['done']
                    checks['vectors']+=2
            calls=[]
            for path in sorted(folder.glob('call[0-9][0-9][0-9].json')):
                c=read(path);calls.append(c)
                with np.load(path.with_name(path.stem+'_state.npz')) as z:
                    obs=template.copy();obs.from_vect(z['observation']);base=env.action_space();base.from_vect(z['base_action'])
                assert digest(obs.to_vect())==c['state_hash']
                eligible,reps,aliases,actions,changed=search.prepare(obs,base)
                assert sorted(reps)==c['order'] and reps==c['representatives']
                assert aliases=={int(k):int(v) for k,v in c['aliases'].items()}
                truth={r['pool_id']:r for r in c['outcomes']};assert len(truth)==len(c['outcomes'])==c['public_queries']
                for row in c['outcomes']:
                    assert row['strict_admissible']==bool(0<row['rho']<c['rho_limit'] and not row['done'] and not row['exceptions'] and not row['illegal'] and not row['ambiguous'])
                    assert row['combined_hash']==digest(action_vector(actions[row['pool_id']]+base));checks['legal_forecasts']+=1
                with np.load(path.with_name(path.stem+'_inputs.npz')) as z:
                    xx,ee,gg=graph_features(obs,env,dict(offered=False,predictions=[None,None]))
                    assert np.array_equal(xx,z['x']) and np.array_equal(ee,z['edges']) and np.array_equal(gg,z['glob']);checks['public_context']+=1
                seen=[]
                def reveal(i,phase):
                    assert i in truth and i not in seen and truth[i]['phase']==phase
                    seen.append(i);return dict(truth[i])
                result=policies[s['method']].run(c['order'],reveal,search.subids,aliases,c['rho_limit'])
                assert seen==[r['pool_id'] for r in c['outcomes']]
                for field in ['chosen_pool_id','public_queries','local_queries','completion_budget','pre_extra_choice','extra_selected_ids','completion_candidate_ids','stop_reason']:
                    assert result[field]==c[field],(path,field)
                assert result['learned_decisions']==c['learned_decisions'],path
                valid=[r for r in c['outcomes'] if r['strict_admissible']]
                choice=max(valid,key=P.key)['pool_id'] if valid else None
                assert choice==c['chosen_pool_id']
                if choice is not None:assert digest(action_vector(actions[choice]))==c['returned_action_hash']
                if s['method']=='LOCAL_BATCH1':assert c['local_queries']==min(64,c['completion_budget'])
                # Independent replay of just the paid global prefix, with the unchanged historical rule.
                global_ids=[]
                def global_reveal(i,phase):
                    assert truth[i]['phase']==phase;global_ids.append(i);return dict(truth[i])
                no=P.G.run(c['order'],global_reveal,'NO_LOCAL',search.subids,aliases,limit=c['rho_limit'])
                actual_global=[r['pool_id'] for r in c['outcomes'] if r['phase']!='local_completion']
                assert global_ids==actual_global and no['pre_extra_choice']==c['pre_extra_choice']
                assert c['extra_selected_ids']==no['completion_candidate_ids'][:c['local_queries']]
                checks['online_policy_reexecution']+=1;checks['calls']+=1;checks['online_decisions']+=len(c['learned_decisions'])
            assert len(calls)==s['n1_calls'] and sum(c['public_queries'] for c in calls)==s['n1_forecasts']
            callcache[key]=calls
            components=read(folder/'components.json');opt=math.fsum(r['wall_s'] for r in components['calls'] if r['in_control'])
            overhead=s['n1_source_s']-s['n1_forecast_s'];other=s['controller_s']-opt-s['all_forecast_s']-overhead
            feature=math.fsum(c['local_feature_s'] for c in calls);inference=math.fsum(c['local_inference_s'] for c in calls)
            assert min(opt,overhead,other,overhead-feature-inference)>=-1e-6
            active={c['step'] for c in calls};lat=[r['decision_s'] for r in rs];searchlat=[r['decision_s'] for r in rs if r['step'] in active]
            row=dict(repeat=s['repeat'],week=s['week'],date=s['date'],method=s['method'],steps=s['steps'],complete=s['complete'],cost=s['cost'],
                controller_s=s['controller_s'],candidate_queries=s['n1_forecasts'],all_queries=s['public_forecasts'],
                local_queries=sum(c['local_queries'] for c in calls),completion_calls=sum(c['completion_budget']>0 for c in calls),
                local_feature_s=feature,local_inference_s=inference,optimizer_s=opt,forecast_s=s['all_forecast_s'],search_overhead_s=overhead,other_s=other,
                all_p95=float(np.percentile(lat,95)),all_p99=float(np.percentile(lat,99)),
                search_p95=float(np.percentile(searchlat,95)) if searchlat else None,search_p99=float(np.percentile(searchlat,99)) if searchlat else None,
                clipped_decisions=sum(d['clipped_features']>0 for c in calls for d in c['learned_decisions']),
                signature=signature(rs))
            runrows.append(row);checks['steps']+=len(rs)
        repeat_pairs=0
        for selected in D['selected']:
            for m in D['methods']:
                a=next(r for r in runrows if r['repeat']==0 and r['week']==selected['scenario'] and r['method']==m['method'])
                b=next(r for r in runrows if r['repeat']==1 and r['week']==selected['scenario'] and r['method']==m['method'])
                assert a['signature']==b['signature'] and a['candidate_queries']==b['candidate_queries'];repeat_pairs+=1
        assert int(env.nb_highres_called)==before_forecasts,'Saved-data audit must not issue forecasts'
    finally:env.close()
    comparisons=[]
    for selected in D['selected']:
        week=selected['scenario'];full=read(ROOT/D['full_references'][week])
        for candidate in [m['method'] for m in D['methods'] if m['method'] not in ['FRAME_GUARD','LOCAL_BATCH1']]:
            for reference in ['FRAME_GUARD','LOCAL_BATCH1']:
                per_repeat=[]
                for repeat in [0,1]:
                    a=next(r for r in runrows if r['repeat']==repeat and r['week']==week and r['method']==candidate)
                    b=next(r for r in runrows if r['repeat']==repeat and r['week']==week and r['method']==reference)
                    complete_cost=a['cost']/b['cost']-1 if a['complete'] and b['complete'] else None
                    quality=bool(a['complete'] and complete_cost<=D['quality_threshold']) if b['complete'] else a['steps']>=b['steps']
                    ca=callcache[repeat,week,candidate];cb=callcache[repeat,week,reference];common=min(a['steps'],b['steps'])
                    active={c['step'] for c in ca+cb if c['step']<=common}
                    qa=sum(c['public_queries'] for c in ca if c['step'] in active);qb=sum(c['public_queries'] for c in cb if c['step'] in active)
                    timea=sum(r['decision_s'] for r in stepcache[repeat,week,candidate][:common]);timeb=sum(r['decision_s'] for r in stepcache[repeat,week,reference][:common])
                    per_repeat.append(dict(repeat=repeat,candidate_steps=a['steps'],reference_steps=b['steps'],candidate_complete=a['complete'],reference_complete=b['complete'],
                        quality=bool(quality),whole_week_cost_delta=complete_cost,same_horizon=a['steps']==b['steps'] and a['complete']==b['complete'],
                        controller_reduction=1-a['controller_s']/b['controller_s'],matched_horizon_controller_reduction=1-timea/timeb,
                        active_query_reduction=1-qa/qb if qb else None,matched_candidate_queries=qa,matched_reference_queries=qb,
                        full_teacher_steps=full['steps'],full_teacher_complete=full['complete'],
                        cost_delta_full=a['cost']/full['cost']-1 if a['complete'] and full['complete'] else None))
                comparisons.append(dict(week=week,date=selected['date'],candidate=candidate,reference=reference,per_repeat=per_repeat))
    gates={}
    for candidate in [m['method'] for m in D['methods'] if m['method'] not in ['FRAME_GUARD','LOCAL_BATCH1']]:
        gates[candidate]={}
        for reference in ['FRAME_GUARD','LOCAL_BATCH1']:
            cs=[c for c in comparisons if c['candidate']==candidate and c['reference']==reference]
            quality=all(p['quality'] for c in cs for p in c['per_repeat']);same=all(p['same_horizon'] for c in cs for p in c['per_repeat'])
            times=[float(np.mean([c['per_repeat'][r]['controller_reduction'] for c in cs])) for r in [0,1]]
            qs=[p['active_query_reduction'] for c in cs for p in c['per_repeat'] if p['active_query_reduction'] is not None]
            faster=sum(np.mean([p['controller_reduction'] for p in c['per_repeat']])>0 for c in cs)
            compute=bool(same and np.mean(times)>=.05 and qs and np.mean(qs)>=.05 and faster>=3 and min(times)>0)
            gates[candidate][reference]=dict(quality_all4=quality,same_horizon_all4=same,macro_controller_reduction=float(np.mean(times)),
                repeat_macro_time_reductions=times,macro_active_query_reduction=float(np.mean(qs)) if qs else None,faster_dates=faster,
                compute_pass=compute,pass_gate=quality and compute)
    rf_ready=all(g['pass_gate'] for g in gates['LOCAL_RF'].values())
    dqn_quality=all(g['quality_all4'] and g['same_horizon_all4'] for m in ['LOCAL_DQN0_0','LOCAL_DQN0_1','LOCAL_DQN0_2'] for g in gates[m].values())
    dqn_compute=all(sum(gates[m][ref]['compute_pass'] for m in ['LOCAL_DQN0_0','LOCAL_DQN0_1','LOCAL_DQN0_2'])>=2 and
        np.mean([gates[m][ref]['macro_controller_reduction'] for m in ['LOCAL_DQN0_0','LOCAL_DQN0_1','LOCAL_DQN0_2']])>=.05 and
        np.mean([gates[m][ref]['macro_active_query_reduction'] or 0. for m in ['LOCAL_DQN0_0','LOCAL_DQN0_1','LOCAL_DQN0_2']])>=.05 and
        all(np.mean([gates[m][ref]['repeat_macro_time_reductions'][r] for m in ['LOCAL_DQN0_0','LOCAL_DQN0_1','LOCAL_DQN0_2']])>0 for r in [0,1]) for ref in ['FRAME_GUARD','LOCAL_BATCH1'])
    decision=dict(ready_to_scale=bool(rf_ready or dqn_quality and dqn_compute),rf_ready=rf_ready,dqn_family_ready=bool(dqn_quality and dqn_compute),
        gates=gates,comparisons=comparisons,no_selected_best_seed=True,
        note='Development readiness, no independent-test/generalization/deployment claim; early termination retained and never sold as speedup.')
    save(OUT/'decision.json',decision)
    save(OUT/'review.json',dict(passed_saved_data=True,runs=48,checks=dict(checks),deterministic_repeat_pairs=repeat_pairs,
        descriptive_runs=runrows,independent_ac_resolve=False,audit_forecasts=0,new_model_fits=0,final_test_materialized=False))
    print(json.dumps(dict(passed_saved_data=True,checks=dict(checks),ready_to_scale=decision['ready_to_scale']),indent=2),flush=True)
if __name__=='__main__':main()
