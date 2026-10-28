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
    C.scored();assert C.read(O/'pre_join_publication.json')['passed'] and not (O/'join_started.json').exists();C.save('join_started.json',dict(allweights_and_predictions_published=True));meta=C.read(C.META);lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};old=C.read(BASE/'oof_trajectories.json');oldmap={(r['method'],r['sample_id']):r for r in old};oldpred={(r['seed'],r['sample_id']):r for r in C.read(BASE/'gate_predictions.json')};cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];scoremap={}
    for key in C.read(O/'configs.json'):
        f,s=[int(v[1:]) for v in key.split('_')];z=np.load(O/'held_predictions'/f'{key}.npz')
        for i,p,k in zip(z['ids'],z['keep_score'],z['keep']):scoremap[f,s,int(i)]=(float(p),bool(k))
    assert len(scoremap)==4681;trajectories=[];pred=[];removed=[];controls=0
    for seed in range(3):
        for c in cases:
            sid=c['sample_id'];key=f'GATE_seed{seed}',sid
            if key not in oldmap:continue
            base=oldmap[key];netbase=oldmap[f'NET_seed{seed}',sid];f=base['fold'];j=base['baseline_stop'];eligible=j<len(c['states']);p,keep=scoremap[f,seed,lookup[sid,j]] if eligible else (1.,True);append=bool(base['append'] and keep);end=j+int(append);baseend=j+int(base['append'])
            for control,controlend in [(base,baseend),(netbase,j)]:
                assert control['gain']==float(c['gains'][:controlend].sum()) and control['queries']==int(c['queries'][:controlend].sum()) and control['chosen_pool_id']==c['chosen'][controlend];assert control['actions']==[1]*controlend+([0] if controlend<len(c['states']) else []);assert abs(control['net_utility']-(control['gain']-fee*control['queries'])/scale)<1e-12;controls+=1
            v=dict(base);v.update(method=f'CANCEL_seed{seed}',actions=[1]*end+([0] if end<len(c['states']) else []),batches=end,gain=float(c['gains'][:end].sum()),queries=int(c['queries'][:end].sum()),chosen_pool_id=c['chosen'][end],append=append);v['net_utility']=(v['gain']-fee*v['queries'])/scale;v['action_match']=v['chosen_pool_id']==c['chosen'][-1];assert v['queries']<=base['queries'] and base['queries']-v['queries']<=64 and v['gain']<=base['gain'];assert not append or base['append'];trajectories.extend([base,v])
            if eligible:
                oldp=oldpred[seed,sid];true_net=float((c['gains'][j]-fee*c['queries'][j])/scale);assert abs(true_net-oldp['actual_next_NET'])<1e-12;assert oldp['gate_continue']==base['append'];rec=dict(fold=f,seed=seed,sample_id=sid,date=c['date'],physical_state_group=c['physical_state_group'],position=j,router_keep_score=p,router_keep=keep,base_continue=base['append'],new_continue=append,actual_next_NET=true_net,actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j]),action_match_delta=int(v['action_match'])-int(base['action_match']),NET_delta=v['net_utility']-base['net_utility']);pred.append(rec)
                if base['append'] and not keep:
                    assert v['gain']-base['gain']==-float(c['gains'][j]) or abs(v['gain']-base['gain']+float(c['gains'][j]))<1e-12;assert v['queries']-base['queries']==-int(c['queries'][j]);assert abs(rec['NET_delta']+true_net)<1e-12;removed.append(rec)
    assert len(pred)==4681 and len(trajectories)==14268 and controls==14268;metrics={};oldmetrics=C.read(BASE/'search_metrics.json')
    for method in ['GATE','CANCEL']:
        for seed in range(3):
            name=f'{method}_seed{seed}';rr=[r for r in trajectories if r['method']==name];assert len(rr)==2378;metrics[name]=summarize(rr)
            if method=='GATE':
                for k,v in metrics[name]['macro'].items():assert abs(v-oldmetrics[name]['macro'][k])<1e-12,(name,k,v,oldmetrics[name]['macro'][k])
                for a,b in zip(metrics[name]['per_date'],oldmetrics[name]['per_date']):
                    assert a['date']==b['date']
                    for k in a:
                        if k not in ['date','cases']:assert abs(a[k]-b[k])<1e-12,(name,a['date'],k)
    family={m:{k:float(np.mean([metrics[f'{m}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{m}_seed0']['macro']} for m in ['GATE','CANCEL']};paired=[]
    for s in range(3):
        a,b=metrics[f'GATE_seed{s}'],metrics[f'CANCEL_seed{s}'];paired.append(dict(seed=s,macro_delta={k:b['macro'][k]-a['macro'][k] for k in a['macro']},per_date=[dict(date=x['date'],NET_delta=y['mean_net_utility']-x['mean_net_utility'],query_delta=y['mean_queries']-x['mean_queries'],gain_delta=y['mean_gain']-x['mean_gain'],action_delta_pp=100*(y['action_match_rate']-x['action_match_rate'])) for x,y in zip(a['per_date'],b['per_date'])]))
    stats=dict(eligible_states=len(pred),baseline_appends=sum(r['base_continue'] for r in pred),retained_appends=sum(r['new_continue'] for r in pred),removed=len(removed),removed_profitable=sum(r['actual_next_NET']>0 for r in removed),removed_unprofitable=sum(r['actual_next_NET']<=0 for r in removed),removed_zero_gain=sum(r['actual_next_gain']==0 for r in removed),removed_profitable_unique_calls=len({r['physical_state_group'] for r in removed if r['actual_next_NET']>0}),retained_profitable=sum(r['new_continue'] and r['actual_next_NET']>0 for r in pred),baseline_profitable=sum(r['base_continue'] and r['actual_next_NET']>0 for r in pred),no_new_CONT_actions=True)
    for n,v in [('oof_trajectories.json',trajectories),('gate_predictions.json',pred),('removed_batches.json',removed),('search_metrics.json',metrics),('family_summary.json',family),('paired_summary.json',paired),('gate_counts.json',stats)]:C.save(n,v)
    C.scored();C.save('review.json',dict(passed=True,all14268_savedNET_and208_controls_reconstructed=True,all14268_control_and_cancel_trajectories_checked=True,all4681_sealed_prediction_ids_joined=True,unchanged_public_features_models_and_fees=True,no_new_continue_actions=True,all_seeds_and16dates_retained=True,new_physics=0,final8_absent=True));C.archive('evaluated');print('GRID216_JOINED',json.dumps(dict(family=family,counts=stats,paired=[dict(seed=r['seed'],NET=r['macro_delta']['mean_net_utility'],queries=r['macro_delta']['mean_queries'],action_pp=100*r['macro_delta']['action_match_rate']) for r in paired]),ensure_ascii=False))
if __name__=='__main__':main()
