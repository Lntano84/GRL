"""Join sealed cancellation predictions and replay saved one-batch prefixes."""
from pathlib import Path
import collections,json,pickle,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,R,BASE=C.O,C.R,C.BASE

def summarize(rows):
    per=[]
    for date in sorted({r['date'] for r in rows}):
        rr=[r for r in rows if r['date']==date];fg=sum(r['full_gain'] for r in rr);fq=sum(r['full_queries'] for r in rr);groups=collections.defaultdict(list)
        for r in rr:groups[r['physical_state_group']].append(r['net_utility'])
        per.append(dict(date=date,cases=len(rr),mean_net_utility=float(np.mean([r['net_utility'] for r in rr])),mean_gain=float(np.mean([r['gain'] for r in rr])),mean_queries=float(np.mean([r['queries'] for r in rr])),query_fraction=sum(r['queries'] for r in rr)/fq,gain_capture=sum(r['gain'] for r in rr)/fg if fg>0 else None,physical_group_mean_utility=float(np.mean([np.mean(v) for v in groups.values()])),action_match_rate=float(np.mean([r['action_match'] for r in rr])),eligible_gate_fraction=float(np.mean([r['eligible'] for r in rr])),append_fraction=float(np.mean([r['append'] for r in rr])),added_queries=float(np.mean([r['queries']-r['baseline_queries'] for r in rr])),added_gain=float(np.mean([r['gain']-r['baseline_gain'] for r in rr]))))
    macro={k:float(np.mean([d[k] for d in per if d[k] is not None])) for k in per[0] if k not in ['date','cases']};return dict(cases=len(rows),per_date=per,macro=macro)

def main():
    C.scored();assert C.read(O/'pre_join_publication.json')['passed'] and not (O/'join_started.json').exists();C.save('join_started.json',dict(all24predictionfiles_published=True));meta=C.read(C.META);lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};oldmap={(r['method'],r['sample_id']):r for r in C.read(BASE/'oof_trajectories.json')};oldpred={(r['seed'],r['sample_id']):r for r in C.read(BASE/'gate_predictions.json')};cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];scores={}
    for f in range(4):
        for seed in range(3):
            for arm in C.ARMS:
                z=np.load(O/'held_predictions'/f'f{f}_s{seed}_{arm}.npz')
                for i,logit,p,keep,base in zip(z['ids'],z['logit'],z['weighted_KEEP_score'],z['keep'],z['base_CONT']):scores[f,seed,arm,int(i)]=(float(logit),float(p),bool(keep),bool(base))
    assert len(scores)==9362;tr=[];pred=[];removed=[];changed=[];controls=0
    for seed in range(3):
        for c in cases:
            sid=c['sample_id']
            if (f'GATE_seed{seed}',sid) not in oldmap:continue
            base=oldmap[f'GATE_seed{seed}',sid];netbase=oldmap[f'NET_seed{seed}',sid];j=base['baseline_stop'];eligible=j<len(c['states']);f=base['fold'];tr.append(base);values={}
            for control,stop in [(base,j+int(base['append'])),(netbase,j)]:
                assert control['gain']==float(c['gains'][:stop].sum()) and control['queries']==int(c['queries'][:stop].sum()) and control['chosen_pool_id']==c['chosen'][stop];assert control['actions']==[1]*stop+([0] if stop<len(c['states']) else []);controls+=1
            for arm in C.ARMS:
                logit,p,keep,publicbase=scores[f,seed,arm,lookup[sid,j]] if eligible else (1.,1.,True,False);assert publicbase==bool(base['append']);append=bool(base['append'] and keep);end=j+int(append);v=dict(base);v.update(method=f'{arm}_seed{seed}',actions=[1]*end+([0] if end<len(c['states']) else []),batches=end,gain=float(c['gains'][:end].sum()),queries=int(c['queries'][:end].sum()),chosen_pool_id=c['chosen'][end],append=append);v['net_utility']=(v['gain']-fee*v['queries'])/scale;v['action_match']=v['chosen_pool_id']==c['chosen'][-1];assert v['queries']<=base['queries'] and base['queries']-v['queries']<=64 and v['gain']<=base['gain'];tr.append(v);values[arm]=v
                if eligible:
                    true_net=float((c['gains'][j]-fee*c['queries'][j])/scale);assert abs(true_net-oldpred[seed,sid]['actual_next_NET'])<1e-12;row=dict(fold=f,seed=seed,arm=arm,sample_id=sid,date=c['date'],physical_state_group=c['physical_state_group'],position=j,logit=logit,weighted_KEEP_score=p,keep=keep,base_continue=base['append'],new_continue=append,actual_next_NET=true_net,actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j]),NET_delta=v['net_utility']-base['net_utility'],action_match_delta=int(v['action_match'])-int(base['action_match']));pred.append(row)
                    if base['append'] and not keep:assert abs(row['NET_delta']+true_net)<1e-12;removed.append(row)
            if values['DIRECT']['append']!=values['ANCHORED']['append']:
                a,b=values['DIRECT'],values['ANCHORED'];changed.append(dict(seed=seed,fold=f,sample_id=sid,date=c['date'],physical_state_group=c['physical_state_group'],position=j,DIRECT_append=a['append'],ANCHORED_append=b['append'],actual_next_NET=float((c['gains'][j]-fee*c['queries'][j])/scale),query_delta=b['queries']-a['queries'],gain_delta=b['gain']-a['gain'],NET_delta=b['net_utility']-a['net_utility']))
    assert controls==14268 and len(pred)==9362 and len(tr)==21402;oldmetrics=C.read(BASE/'search_metrics.json');metrics={}
    for arm in ['GATE',*C.ARMS]:
        for seed in range(3):
            name=f'{arm}_seed{seed}';rr=[r for r in tr if r['method']==name];assert len(rr)==2378;metrics[name]=summarize(rr)
            if arm=='GATE':
                for k,v in metrics[name]['macro'].items():assert abs(v-oldmetrics[name]['macro'][k])<1e-12
    family={m:{k:float(np.mean([metrics[f'{m}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{m}_seed0']['macro']} for m in ['GATE',*C.ARMS]};paired=[]
    for a,b in [('GATE','DIRECT'),('GATE','ANCHORED'),('DIRECT','ANCHORED')]:
        for s in range(3):
            am,bm=metrics[f'{a}_seed{s}'],metrics[f'{b}_seed{s}'];paired.append(dict(base=a,new=b,seed=s,macro_delta={k:bm['macro'][k]-am['macro'][k] for k in am['macro']},per_date=[dict(date=x['date'],NET_delta=y['mean_net_utility']-x['mean_net_utility'],query_delta=y['mean_queries']-x['mean_queries'],gain_delta=y['mean_gain']-x['mean_gain'],action_delta_pp=100*(y['action_match_rate']-x['action_match_rate'])) for x,y in zip(am['per_date'],bm['per_date'])]))
    counts={}
    for arm in C.ARMS:
        rr=[r for r in pred if r['arm']==arm];cut=[r for r in removed if r['arm']==arm];counts[arm]=dict(eligible=4681,base_appends=sum(r['base_continue'] for r in rr),new_appends=sum(r['new_continue'] for r in rr),removed=len(cut),removed_profitable=sum(r['actual_next_NET']>0 for r in cut),removed_unprofitable=sum(r['actual_next_NET']<=0 for r in cut),removed_zero_gain=sum(r['actual_next_gain']==0 for r in cut),retained_profitable=sum(r['new_continue'] and r['actual_next_NET']>0 for r in rr),removed_profitable_calls=len({r['physical_state_group'] for r in cut if r['actual_next_NET']>0}))
    for name,v in [('oof_trajectories.json',tr),('gate_predictions.json',pred),('removed_batches.json',removed),('changed_between_arms.json',changed),('search_metrics.json',metrics),('family_summary.json',family),('paired_summary.json',paired),('gate_counts.json',counts)]:C.save(name,v)
    C.scored();C.save('review.json',dict(passed=True,all14268_savedNET208_controls_reconstructed=True,all21402_three_arm_trajectories_checked=True,all9362_sealed_prediction_ids_joined=True,no_new_CONT_actions=True,no_new_physics=True,all16_exposed_dates_all3seeds_retained=True,final8_absent=True));C.archive('evaluated');print('GRID218_JOINED',json.dumps(dict(family=family,counts=counts,changed=len(changed),paired=[dict(base=r['base'],new=r['new'],seed=r['seed'],NET_delta=r['macro_delta']['mean_net_utility']) for r in paired]),ensure_ascii=False))
if __name__=='__main__':main()
