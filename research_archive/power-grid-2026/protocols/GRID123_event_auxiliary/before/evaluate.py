"""All fitted endpoints before any evaluation. Auxiliary heads never gate actions."""
import gzip
from common123 import *

def statistics(q,y,prob,event,arm):
    go=q[:,1]>0;regret=np.maximum(y,0)-go*y;error=q[:,1]-y
    loss=np.where(abs(error)<1,.5*error**2,abs(error)-.5)
    out=dict(states=len(y),positive_misses=int(((y>0)&~go).sum()),unprofitable_continues=int(((y<=0)&go).sum()),
        weighted_sign_regret=float(prob@regret),weighted_main_huber=float(prob@loss),weighted_net_prediction=float(prob@q[:,1]))
    if arm in ARMS:
        t=event if arm=='AUX_GAIN' else (y>0);logit=q[:,0].astype(float)
        out['aux_bce_own_target']=float(prob@(np.logaddexp(0.,logit)-t*logit))
    return out

def main():
    assert not (O/'evaluation_receipt.json').exists();verify();configure();F=read(O/'fits.json');assert F['passed'] and len(F['records'])==30
    D=read(O/'design.json');cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());bycase={c['sample_id']:c for c in cases}
    metas={p:read(O/f'{p}_metadata.json') for p in ['train','validation']};lookup={p:{(m['sample_id'],m['position']):i for i,m in enumerate(mm)} for p,mm in metas.items()}
    Q={};metrics={};fit_side={};full={'train':[],'validation':[]};oof=[];baseline_full={}
    for s in range(3):
        for p in metas:baseline_full[s,p]=np.full((len(metas[p]),2),np.nan,'f')
    with gzip.open(OLD/'state_predictions.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['method'].startswith('DIRECT'):
                s=int(r['method'][-1]);p=r['partition'];baseline_full[s,p][lookup[p][r['sample_id'],r['position']]]=[r['q0'],r['q1']]
    for (s,p),q in baseline_full.items():
        assert np.isfinite(q).all();Q[f'full__NET_ZERO_seed{s}__{p}']=q
    oldfold=np.load(CV/'predictions.npz')
    for fold in range(4):
        for s in range(3):Q[f'f{fold}__NET_ZERO_seed{s}__train']=oldfold[f'fold{fold}__HUBER_seed{s}']
    for rec in F['records']:
        split,seed,arm=rec['split'],rec['seed'],rec['arm'];name=f'{split}__{arm}_seed{seed}';net=build(seed);cp=torch.load(O/'models'/f'{name}.pt',weights_only=False,map_location='cpu');net.load_state_dict(cp['state_dict']);assert digest(net)==rec['final_hash']
        for part in (['train','validation'] if split=='full' else ['train']):
            x,y,event,ids,prob=getdata(split,part);q=forward(net,x);assert np.isfinite(q).all();Q[name+'__'+part]=q
    # Build full/OOF strategies using the same exact-zero stop readout for all new and matched controls.
    for split in ['full','f0','f1','f2','f3']:
        for arm in ['NET_ZERO']+ARMS:
            for seed in range(3):
                name=f'{split}__{arm}_seed{seed}';metrics[name]={}
                for part in (['train','validation'] if split=='full' else ['train']):
                    x,y,event,ids,prob=getdata(split,part);q=Q[name+'__'+part]
                    if split=='full':
                        if part=='validation':
                            count=collections.Counter(c['date'] for c in cases if c['partition']==part)
                            prob=np.array([1/len(count)/count[m['date']]/len(bycase[m['sample_id']]['states']) for m in metas[part]])
                        metrics[name][part]=statistics(q,y,prob,event,arm)
                    else:metrics[name]['fit']=statistics(q[ids],y[ids],prob,event[ids],arm)
                    rows={'fit':[],'held':[]}
                    for c in cases:
                        if c['partition']!=part:continue
                        ix=[lookup[part][c['sample_id'],j] for j in range(len(c['states']))]
                        t=trajectory(c,f'{arm}_seed{seed}',q[ix,1]>0,D['cost_per_query'],D['gain_scale']);t['model_name']=name
                        if split=='full':full[part].append(t)
                        else:
                            held=c['date'] in D['folds'][int(split[1:])]['held_dates'];rows['held' if held else 'fit'].append(t)
                            if held:oof.append(t)
                    if split!='full':fit_side[name]={k:summarize(v) for k,v in rows.items()}
    historical=read(OLD/'validation_trajectories.json')
    for r in historical:
        c=bycase[r['sample_id']];t=trajectory(c,r['method'],r['actions']+[0]*len(c['states']),D['cost_per_query'],D['gain_scale'])
        assert t['actions']==r['actions'] and abs(t['net_utility']-r['net_utility'])<1e-12;full['validation'].append(t)
    results={'full_train':summaries(full['train']),'validation':summaries(full['validation']),'oof':summaries(oof)}
    comparisons={};families={};readout={}
    for domain in ['validation','oof']:
        v=results[domain];u=lambda n:v[n]['all']['macro']['mean_net_utility']
        families[domain]={a:float(np.mean([u(f'{a}_seed{s}') for s in range(3)])) for a in ['NET_ZERO']+ARMS}
        dates=sorted(v['AUX_GAIN_seed0']['all']['per_date']);comparisons[domain]={}
        for ref in ['NET_ZERO','AUX_NETSIGN']:
            pairs=[u(f'AUX_GAIN_seed{s}')-u(f'{ref}_seed{s}') for s in range(3)]
            dd={d:float(np.mean([v[f'AUX_GAIN_seed{s}']['all']['per_date'][d]['mean_net_utility']-v[f'{ref}_seed{s}']['all']['per_date'][d]['mean_net_utility'] for s in range(3)])) for d in dates}
            comparisons[domain][ref]=dict(paired_seed_differences=pairs,date_differences=dd,positive_dates=sum(x>0 for x in dd.values()),signal=np.mean(pairs)>0 and all(x>0 for x in pairs) and sum(x>0 for x in dd.values())>=(3 if domain=='validation' else 12))
    val=results['validation'];vu=lambda n:val[n]['all']['macro']['mean_net_utility'];legacy_huber=float(np.mean([vu(f'DIRECT{s}') for s in range(3)]));legacy_dqn=float(np.mean([vu(f'DQN0_{s}') for s in range(3)]));batch=vu('BATCH1')
    source_oof=read(CV/'oof_results.json');old_oof_huber=float(np.mean([source_oof[f'HUBER_seed{s}']['all']['macro']['mean_net_utility'] for s in range(3)]))
    for domain,legacy in [('validation',legacy_huber),('oof',old_oof_huber)]:readout[domain]=dict(original_q1_vs_q0=legacy,exact_zero_readout=families[domain]['NET_ZERO'],difference=families[domain]['NET_ZERO']-legacy)
    incremental=all(comparisons[d]['NET_ZERO']['signal'] for d in comparisons);specific=all(comparisons[d]['AUX_NETSIGN']['signal'] for d in comparisons)
    anchors=all(families['validation']['AUX_GAIN']>x for x in [legacy_huber,legacy_dqn,batch])
    decision=dict(family_utility=families,comparisons=comparisons,incremental_auxiliary_support=incremental,event_specific_support=specific,legacy_anchors_exceeded=anchors,
        useful_candidate=incremental and specific and anchors,legacy_validation=dict(HUBER=legacy_huber,DQN=legacy_dqn,BATCH1=batch),readout_change=readout,ready_to_scale=False,
        scope='Repeated exposed-date teacher-path development andconditional OOF,notwholeweek quality/latency or independentconfirmation.')
    np.savez_compressed(O/'predictions.npz',**Q)
    for part in full:save(O/f'full_{part}_trajectories.json',full[part])
    save(O/'oof_trajectories.json',oof)
    for p,v in results.items():save(O/f'{p}_results.json',v)
    save(O/'fit_side_results.json',fit_side);save(O/'state_metrics.json',metrics);save(O/'decision.json',decision)
    save(O/'evaluation_receipt.json',dict(passed=True,all30_endpoints_before_evaluation=True,reused_backbones=15,new_physical_calls=0,final_test_materialized=False));save(O/'state.json',dict(stage='evaluated_waiting_audit',fits=30));print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
