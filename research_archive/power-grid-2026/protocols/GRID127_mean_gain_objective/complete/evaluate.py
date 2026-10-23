"""Complete saved-search evaluation after all30 endpoints. No model selection."""
from common127 import *

def stats(q,y,g,c,p,arm):
    p=p/p.sum();predg=numpy_gain(q[:,1],c,arm);predv=predg-c;go=actions(q,c,arm);err=predv-y;hub=np.where(abs(err)<1,.5*err**2,abs(err)-.5)
    return dict(states=len(y),main_huber=float(p@hub),sign_regret=float(p@(np.maximum(y,0)-go*y)),positive_misses=int(np.sum((y>0)&~go)),unprofitable_continues=int(np.sum((y<=0)&go)),gain_negative_mass=float(p@(predg < -1e-6)),mean_predicted_gain=float(p@predg),mean_gain_error=float(p@(predg-g)),continue_mass=float(p@go))

def main():
    assert not (O/'evaluation_receipt.json').exists();verify();src.configure();F=read(O/'fits.json');assert F['passed'] and len(F['records'])==30;D=read(O/'design.json');meta={p:read(S/f'{p}_metadata.json') for p in ['train','validation']};lookup={p:{(m['sample_id'],m['position']):i for i,m in enumerate(mm)} for p,mm in meta.items()};cases=src.pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cm={c['sample_id']:c for c in cases};Q={};old=np.load(S/'predictions.npz');metric={};fit_side={};full={p:[] for p in meta};oof=[]
    for key in old.files:
        if '__NET_ZERO_' in key:Q[key]=old[key]
    previous=np.load(BASE/'predictions.npz')
    for key in previous.files:
        if '__COST_POSITIVE_' in key:Q[key.replace('__COST_POSITIVE_','__POS_HUBER_')]=previous[key]
    for rec in F['records']:
        name=rec['name'];net=build(rec['seed']);cp=torch.load(O/'models'/f'{name}.pt',map_location='cpu',weights_only=False);net.load_state_dict(cp['state_dict']);assert digest(net)==rec['final_hash']
        for part in (['train','validation'] if rec['split']=='full' else ['train']):
            x,*_=getdata(rec['split'],part);q=src.forward(net,x);assert np.isfinite(q).all();Q[name+'__'+part]=q
    for split in ['full','f0','f1','f2','f3']:
        for arm in ALL:
            for seed in range(3):
                name=f'{split}__{arm}_seed{seed}';metric[name]={}
                for part in (['train','validation'] if split=='full' else ['train']):
                    x,y,g,c,ids,p=getdata(split,part);q=Q[name+'__'+part]
                    if split=='full':
                        if part=='validation':
                            cc=collections.Counter(v['date'] for v in cases if v['partition']==part);p=np.array([1/len(cc)/cc[m['date']]/len(cm[m['sample_id']]['states']) for m in meta[part]])
                        metric[name][part]=stats(q,y,g,c,p,arm)
                    else:
                        held=np.setdiff1d(np.arange(len(y)),ids);pp=np.load(S/'train_data.npz')['probability'][held];metric[name]['fit']=stats(q[ids],y[ids],g[ids],c[ids],p,arm);metric[name]['held']=stats(q[held],y[held],g[held],c[held],pp,arm)
                    rows={'fit':[],'held':[]}
                    for case in cases:
                        if case['partition']!=part:continue
                        ii=[lookup[part][case['sample_id'],j] for j in range(len(case['states']))];go=actions(q[ii],c[ii],arm);t=src.trajectory(case,f'{arm}_seed{seed}',go,D['cost_per_query'],D['gain_scale']);t['model_name']=name
                        if split=='full':full[part].append(t)
                        else:
                            isheld=case['date'] in D['folds'][int(split[1:])]['held_dates'];rows['held' if isheld else 'fit'].append(t)
                            if isheld:oof.append(t)
                    if split!='full':fit_side[name]={k:src.summarize(v) for k,v in rows.items()}
    results={'full_train':src.summaries(full['train']),'validation':src.summaries(full['validation']),'oof':src.summaries(oof)};families={};comparisons={};specific={}
    for dom in ['validation','oof']:
        v=results[dom];u=lambda arm,s:v[f'{arm}_seed{s}']['all']['macro']['mean_net_utility'];dates=sorted(v['NET_ZERO_seed0']['all']['per_date']);families[dom]={arm:float(np.mean([u(arm,s) for s in range(3)])) for arm in ALL};comparisons[dom]={}
        for arm,ref in [(a,ref) for a in ARMS for ref in ['NET_ZERO','POS_HUBER']]+[('POS_KL','POS_MSE')]:
            pairs=[u(arm,s)-u(ref,s) for s in range(3)];dd={date:float(np.mean([v[f'{arm}_seed{s}']['all']['per_date'][date]['mean_net_utility']-v[f'{ref}_seed{s}']['all']['per_date'][date]['mean_net_utility'] for s in range(3)])) for date in dates};signal=bool(np.mean(pairs)>0 and all(x>0 for x in pairs) and sum(x>0 for x in dd.values())>=(3 if dom=='validation' else 12));comparisons[dom][f'{arm}_minus_{ref}']=dict(paired_seed_differences=pairs,date_differences=dd,positive_dates=sum(x>0 for x in dd.values()),signal=signal)
    legacy=read(S/'decision.json')['legacy_validation'];support={arm:all(comparisons[dom][f'{arm}_minus_NET_ZERO']['signal'] for dom in comparisons) for arm in ARMS};huber_support={arm:all(comparisons[dom][f'{arm}_minus_POS_HUBER']['signal'] for dom in comparisons) for arm in ARMS};specific=all(comparisons[dom]['POS_KL_minus_POS_MSE']['signal'] for dom in comparisons);anchors={arm:all(families['validation'][arm]>z for z in legacy.values()) for arm in ARMS}
    decision=dict(family_utility=families,comparisons=comparisons,incremental_support=support,vs_positive_huber_support=huber_support,kl_specific_support=specific,legacy_validation=legacy,legacy_anchors_exceeded=anchors,useful_candidate={arm:bool(support[arm] and huber_support[arm] and anchors[arm]) for arm in ARMS},ready_to_scale=False,scope='Exposed-date FULL-teacher local-search development;conditional OOF fixedhistoricaltask;not physical whole-week quality/latency or new independent test.')
    np.savez_compressed(O/'predictions.npz',**Q)
    for part,rr in full.items():save(O/f'full_{part}_trajectories.json',rr)
    save(O/'oof_trajectories.json',oof)
    for dom,rr in results.items():save(O/f'{dom}_results.json',rr)
    save(O/'state_metrics.json',metric);save(O/'fit_side_results.json',fit_side);save(O/'decision.json',decision)
    # Exhaustive first-common-prefix accounting of complete local-search utility differences.
    decomposition={}
    for dom,rr in [('validation',full['validation']),('oof',oof)]:
        look={(v['method'],v['sample_id']):v for v in rr};caseids=[v['sample_id'] for v in rr if v['method']=='NET_ZERO_seed0'];cc=collections.Counter(cm[i]['date'] for i in caseids);decomposition[dom]={}
        for arm in ARMS:
            groups={}
            for seed in range(3):
                for sid in caseids:
                    a=look[f'{arm}_seed{seed}',sid];b=look[f'NET_ZERO_seed{seed}',sid];case=cm[sid];j=next((j for j,(av,bv) in enumerate(zip(a['actions'],b['actions'])) if av!=bv),None)
                    if j is None:assert a['actions']==b['actions'];key='same_actions'
                    else:
                        y=(case['gains'][j]-D['cost_per_query']*case['queries'][j])/D['gain_scale'];key=('baseCONT_newSTOP' if b['actions'][j] else 'baseSTOP_newCONT')+('__nextPositive' if y>0 else '__nextNonpositive')
                    if key not in groups:groups[key]=dict(records=0,utility_contribution=0.,physical_calls=set())
                    group=groups[key];group['records']+=1;group['physical_calls'].add(case['physical_state_group']);group['utility_contribution']+=(a['net_utility']-b['net_utility'])/3/len(cc)/cc[case['date']]
            for group in groups.values():group['distinct_physical_calls']=len(group.pop('physical_calls'))
            assert abs(sum(v['utility_contribution'] for v in groups.values())-(families[dom][arm]-families[dom]['NET_ZERO']))<1e-12;decomposition[dom][arm]=groups
    save(O/'divergence_summary.json',decomposition);save(O/'evaluation_receipt.json',dict(passed=True,all30_endpoints_before_new_model_held_validation_inference=True,reused_baselines=30,new_physical_calls=0,final_test_materialized=False));save(O/'state.json',dict(stage='evaluated_waiting_audit',fits=30));print('GRID127_EVALUATED',families,flush=True)

if __name__=='__main__':main()
