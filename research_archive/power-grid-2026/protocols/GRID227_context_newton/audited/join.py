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
    C.scored();assert C.read(O/'pre_join_publication.json')['passed'] and not (O/'join_started.json').exists();assert not (C.P.O/'score_started.json').exists();C.save('join_started.json',dict(all12_mixed_predictions_published=True,source225_226_unscored=True));meta=C.read(C.META);lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};bases={(r['method'],r['sample_id']):r for r in C.read(BASE/'oof_trajectories.json')};stops={(r['method'],r['sample_id']):r for r in C.read(C.CONTROL/'oof_trajectories.json')};oldpred={(r['seed'],r['sample_id']):r for r in C.read(BASE/'gate_predictions.json')};cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];scores={}
    for file in (O/'held_predictions').glob('*.npz'):
        f,s=[int(v[1:]) for v in file.stem.split('_')];z=np.load(file)
        for i,logit,keep,base in zip(z['ids'],z['logit'],z['keep'],z['base_CONT']):scores[f,s,int(i)]=(float(logit),bool(keep),bool(base))
    assert len(scores)==4681;tr=[];pred=[];changes=[];controls=0
    for seed in range(3):
        for c in cases:
            sid=c['sample_id']
            if (f'GATE_seed{seed}',sid) not in bases:continue
            base=bases[f'GATE_seed{seed}',sid];netbase=bases[f'NET_seed{seed}',sid];stop=stops[f'ANCHORED_seed{seed}',sid];j=base['baseline_stop'];f=base['fold'];eligible=j<len(c['states']);logit,keep,publicbase=scores[f,seed,lookup[sid,j]] if eligible else (1.,True,False);assert publicbase==bool(base['append']);append=bool(base['append'] and keep);end=j+int(append)
            for control,stopend in [(base,j+int(base['append'])),(netbase,j),(stop,j+int(stop['append']))]:
                assert control['gain']==float(c['gains'][:stopend].sum()) and control['queries']==int(c['queries'][:stopend].sum()) and control['chosen_pool_id']==c['chosen'][stopend];assert control['actions']==[1]*stopend+([0] if stopend<len(c['states']) else []);assert abs(control['net_utility']-(control['gain']-fee*control['queries'])/scale)<1e-12;controls+=1
            stopcopy=dict(stop);stopcopy['method']=f'STOP_ONLY_seed{seed}';v=dict(base);v.update(method=f'MIXED_seed{seed}',actions=[1]*end+([0] if end<len(c['states']) else []),batches=end,gain=float(c['gains'][:end].sum()),queries=int(c['queries'][:end].sum()),chosen_pool_id=c['chosen'][end],append=append);v['net_utility']=(v['gain']-fee*v['queries'])/scale;v['action_match']=v['chosen_pool_id']==c['chosen'][-1];assert v['queries']<=base['queries'] and v['gain']<=base['gain'] and base['queries']-v['queries']<=64;tr.extend([base,stopcopy,v])
            if eligible:
                actual=float((c['gains'][j]-fee*c['queries'][j])/scale);assert abs(actual-oldpred[seed,sid]['actual_next_NET'])<1e-12;pred.append(dict(fold=f,seed=seed,sample_id=sid,date=c['date'],physical_state_group=c['physical_state_group'],position=j,logit=logit,keep=keep,base_continue=base['append'],stop_only_continue=stop['append'],mixed_continue=append,actual_next_NET=actual,actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j]),NET_delta_vs208=v['net_utility']-base['net_utility'],NET_delta_vsSTOP=v['net_utility']-stop['net_utility']))
                if append!=stop['append']:changes.append(dict(seed=seed,fold=f,sample_id=sid,date=c['date'],physical_state_group=c['physical_state_group'],position=j,STOP_append=stop['append'],MIXED_append=append,actual_next_NET=actual,query_delta=v['queries']-stop['queries'],gain_delta=v['gain']-stop['gain'],NET_delta=v['net_utility']-stop['net_utility'],action_match_delta=int(v['action_match'])-int(stop['action_match'])))
    assert len(pred)==4681 and len(tr)==21402 and controls==21402;metrics={};baseline_metrics=C.read(BASE/'search_metrics.json');oldmetrics=C.read(C.CONTROL/'search_metrics.json')
    for arm in ['GATE','STOP_ONLY','MIXED']:
        for seed in range(3):
            name=f'{arm}_seed{seed}';rr=[r for r in tr if r['method']==name];assert len(rr)==2378;metrics[name]=summarize(rr)
            if arm!='MIXED':
                reference=baseline_metrics[name] if arm=='GATE' else oldmetrics[f'ANCHORED_seed{seed}']
                for k,v in metrics[name]['macro'].items():assert abs(v-reference['macro'][k])<1e-12
    family={m:{k:float(np.mean([metrics[f'{m}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{m}_seed0']['macro']} for m in ['GATE','STOP_ONLY','MIXED']};paired=[]
    for a,b in [('STOP_ONLY','MIXED'),('GATE','MIXED')]:
        for seed in range(3):
            old,new=metrics[f'{a}_seed{seed}'],metrics[f'{b}_seed{seed}'];paired.append(dict(base=a,new=b,seed=seed,macro_delta={k:new['macro'][k]-old['macro'][k] for k in old['macro']},per_date=[dict(date=x['date'],NET_delta=y['mean_net_utility']-x['mean_net_utility'],query_delta=y['mean_queries']-x['mean_queries'],gain_delta=y['mean_gain']-x['mean_gain'],action_delta_pp=100*(y['action_match_rate']-x['action_match_rate'])) for x,y in zip(old['per_date'],new['per_date'])]))
    counts={}
    for arm,key in [('STOP_ONLY','stop_only_continue'),('MIXED','mixed_continue')]:
        selected=[r for r in pred if r[key]];removed=[r for r in pred if r['base_continue'] and not r[key]];counts[arm]=dict(original208_appends=sum(r['base_continue'] for r in pred),appends=len(selected),retained_profitable=sum(r['actual_next_NET']>0 for r in selected),removed_profitable=sum(r['actual_next_NET']>0 for r in removed),removed_unprofitable=sum(r['actual_next_NET']<=0 for r in removed),removed_zero_gain=sum(r['actual_next_gain']==0 for r in removed),removed_profitable_calls=len({r['physical_state_group'] for r in removed if r['actual_next_NET']>0}))
    for name,data in [('oof_trajectories.json',tr),('gate_predictions.json',pred),('changed_choices.json',changes),('search_metrics.json',metrics),('family_summary.json',family),('paired_summary.json',paired),('gate_counts.json',counts)]:C.save(name,data)
    C.scored();C.save('review.json',dict(passed=True,all21402_three_arm_trajectories_checked=True,all21402_NET208STOP_controls_reconstructed=True,all4681_sealed_predictions_joined=True,all3seeds16dates_retained=True,original225_226_unscored=True,source222_and208_controls_reused=True,new_physics=0,final8_absent=True));C.archive('evaluated');print('GRID227_MIXED_CONTEXT_EVALUATED',json.dumps(dict(family=family,counts=counts,changes=len(changes),paired=[dict(base=r['base'],seed=r['seed'],NET=r['macro_delta']['mean_net_utility'],queries=r['macro_delta']['mean_queries'],gain=r['macro_delta']['mean_gain']) for r in paired]),ensure_ascii=False))
if __name__=='__main__':main()
