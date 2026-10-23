"""Post-hoc conserved decomposition of all saved local-search trajectory differences."""
import diagnose as d
import gzip
np=d.np

def main():
    assert not (d.O/'trajectory_accounting.json').exists();d.verify()
    d.save(d.O/'posthoc_trajectory_plan.json',dict(posthoc=True,script_sha256=d.sha(d.W/'divergence.py'),
      purpose='Relate all complete saved local-search utility differences to the first actually shared-prefix action divergence; exhaustive directions and next-net sign,not selected cases.',
      limits='Saved FULL-teacher local-search path accounting,not physical closed-loop cost or a claim that next-step sign explains the entire subsequent tail.',new_fits=0,new_predictions=0))
    cases=d.src.pickle.loads((d.R/'outputs/grid102_search_rl/cases.pkl').read_bytes());by={c['sample_id']:c for c in cases};design=d.read(d.S/'design.json');rows=[];summary={}
    for dom,filename in [('validation','full_validation_trajectories.json'),('oof','oof_trajectories.json')]:
        traj=d.read(d.S/filename);look={(r['method'],r['sample_id']):r for r in traj};ids=[r['sample_id'] for r in traj if r['method']=='NET_ZERO_seed0'];counts=d.collections.Counter(by[i]['date'] for i in ids);summary[dom]={}
        for arm in ['AUX_NETSIGN','AUX_GAIN']:
            groups={}
            for seed in range(3):
                for sid in ids:
                    b=look[f'NET_ZERO_seed{seed}',sid];a=look[f'{arm}_seed{seed}',sid];c=by[sid];weight=1/3/len(counts)/counts[c['date']];p=0
                    while p<min(len(a['actions']),len(b['actions'])) and a['actions'][p]==b['actions'][p]:p+=1
                    delta=a['net_utility']-b['net_utility'];dg=(a['gain']-b['gain'])/design['gain_scale'];dc=-design['cost_per_query']*(a['queries']-b['queries'])/design['gain_scale'];assert abs(delta-dg-dc)<1e-12
                    if p==min(len(a['actions']),len(b['actions'])):
                        assert a['actions']==b['actions'] and abs(delta)<1e-12;group='same_actions';nextnet=None
                    else:
                        nextnet=float((c['gains'][p]-design['cost_per_query']*c['queries'][p])/design['gain_scale']);group=('baseCONT_auxSTOP' if b['actions'][p] else 'baseSTOP_auxCONT')+('__nextPositive' if nextnet>0 else '__nextNonpositive')
                        end=b['batches'] if b['actions'][p] else a['batches'];tail=float((c['gains'][p:end].sum()-design['cost_per_query']*c['queries'][p:end].sum())/design['gain_scale']);assert abs(delta-(-tail if b['actions'][p] else tail))<1e-12
                    if group not in groups:groups[group]=dict(records=0,mass=0.,utility_contribution=0.,gain_contribution=0.,query_charge_contribution=0.,physical_calls=set())
                    g=groups[group];g['records']+=1;g['mass']+=weight;g['utility_contribution']+=weight*delta;g['gain_contribution']+=weight*dg;g['query_charge_contribution']+=weight*dc;g['physical_calls'].add(c['physical_state_group'])
                    rows.append(dict(domain=dom,arm=arm,seed=seed,sample_id=sid,physical_state_group=c['physical_state_group'],date=c['date'],first_divergence_position=p if nextnet is not None else None,group=group,next_net=nextnet,trajectory_delta=delta,date_case_seed_weight=weight))
            for g in groups.values():g['unique_physical_calls']=len(g.pop('physical_calls'))
            expected=d.read(d.S/'decision.json')['family_utility'][dom][arm]-d.read(d.S/'decision.json')['family_utility'][dom]['NET_ZERO'];actual=sum(v['utility_contribution'] for v in groups.values());assert abs(expected-actual)<1e-12 and abs(sum(g['mass'] for g in groups.values())-1)<1e-12
            summary[dom][arm]=dict(groups=groups,total_utility_difference=actual)
    with gzip.open(d.O/'first_divergences.jsonl.gz','wt',encoding='utf8') as f:
        for row in rows:f.write(d.json.dumps(row,ensure_ascii=False)+'\n')
    d.save(d.O/'trajectory_accounting.json',dict(summary=summary,records=len(rows),all_differences_conserved=True,scope='Local-search teacher-path tails,not whole-week physical trajectories. Next sign is a descriptor,not explanation of all later rewards.'))
    print(d.json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
