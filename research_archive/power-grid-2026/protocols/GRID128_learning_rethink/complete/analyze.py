"""Literature-guided target/horizon audit of existing cases; no learning or simulator."""
from pathlib import Path
import collections
import hashlib
import itertools
import json
import math
import pickle
import sys
import numpy as np

R=Path('D:/Codex/GRL研究')
O=R/'outputs/grid128_learning_rethink'
sys.path.insert(0,str(R/'work/grid107_direct_value'))
from common107 import summarize,sealed

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def main():
    assert not O.exists();O.mkdir(parents=True)
    sealed()
    files=[Path(__file__),R/'work/grid107_direct_value/common107.py',
        R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',
        R/'outputs/grid103_learning_mechanism/validation_results.json',
        R/'outputs/grid127_mean_gain_objective/nonlearning_comparison/comparison.json',
        R/'outputs/grid127_mean_gain_objective/nonlearning_comparison/differences.json',
        R/'outputs/grid127_mean_gain_objective/posthoc_transfer/risk_parts.json',
        R/'outputs/grid125_value_underestimate/target_functionals.json']
    hashes={p.relative_to(R).as_posix():sha(p) for p in files}
    cases=pickle.loads(files[2].read_bytes());D=read(files[3]);fee=D['cost_per_query'];scale=D['gain_scale']
    result={};checks=0
    for partition in ['train','validation']:
        cs=[c for c in cases if c['partition']==partition];oracle_rows=collections.defaultdict(list);barriers=[];percase=[]
        net_positive_calls=set();positive_states=0
        for c in cs:
            rewards=(c['gains']-fee*c['queries'])/scale
            assert len(rewards)>0 and (c['gains']>=0).all() and (c['queries']>0).all()
            positive_states+=int((rewards>0).sum())
            if (rewards>0).any():net_positive_calls.add(c['physical_state_group'])
            # Every possible policy sequence reduces to the number of initial CONTINUEs before first STOP.
            candidates=np.r_[0.,np.cumsum(rewards)]
            brute=[]
            for acts in itertools.product([0,1],repeat=len(rewards)):
                p=next((i for i,a in enumerate(acts) if not a),len(acts))
                brute.append(math.fsum(float(x) for x in rewards[:p]))
            assert abs(max(brute)-float(candidates.max()))<1e-12;checks+=1
            greedy=next((i for i,v in enumerate(rewards) if v<=0),len(rewards))
            optimal=int(candidates.argmax())
            for j in range(len(rewards)):
                tail=np.cumsum(rewards[j:])
                if rewards[j]<=0 and float(tail.max())>0:
                    barriers.append(dict(sample_id=c['sample_id'],date=c['date'],call=c['physical_state_group'],position=j,
                        immediate_net=float(rewards[j]),best_future_net=float(tail.max()),batch_nets=rewards[j:].tolist()))
            for method,p in [('MYOPIC_ORACLE',greedy),('SEQUENCE_ORACLE',optimal)]:
                gain=float(c['gains'][:p].sum());q=int(c['queries'][:p].sum());value=(gain-fee*q)/scale
                assert abs(value-float(candidates[p]))<1e-12;checks+=1
                oracle_rows[method].append(dict(sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],
                    gain=gain,queries=q,full_gain=float(c['gains'].sum()),full_queries=int(c['queries'].sum()),net_utility=value))
            percase.append(dict(sample_id=c['sample_id'],date=c['date'],call=c['physical_state_group'],
                myopic_batches=greedy,optimal_batches=optimal,utility_gap=float(candidates[optimal]-candidates[greedy])))
        sums={name:summarize(rows) for name,rows in oracle_rows.items()}
        # Independent arithmetic of date-balanced oracle gap.
        date_gaps={d:math.fsum(row['utility_gap'] for row in percase if row['date']==d)/sum(row['date']==d for row in percase) for d in sorted({c['date'] for c in cs})}
        gap=math.fsum(date_gaps.values())/len(date_gaps)
        assert abs(gap-(sums['SEQUENCE_ORACLE']['macro']['mean_net_utility']-sums['MYOPIC_ORACLE']['macro']['mean_net_utility']))<1e-12;checks+=1
        result[partition]=dict(cases=len(cs),reachable_states=sum(len(c['states']) for c in cs),dates=len(date_gaps),
            physical_calls=len({c['physical_state_group'] for c in cs}),horizons=dict(collections.Counter(len(c['states']) for c in cs)),
            positive_net_states=positive_states,positive_net_calls=len(net_positive_calls),
            barrier_states=len(barriers),barrier_calls=len({b['call'] for b in barriers}),barrier_dates=sorted({b['date'] for b in barriers}),
            oracle=sums,date_macro_sequence_advantage=gap,date_gaps=date_gaps)
        save(partition+'_barriers.json',barriers);save(partition+'_all_prefix_gaps.json',percase)
    # Reproduce the existing four-validation oracle report; do not extend its zero-barrier claim to training.
    old=read(files[4])
    for name in ['ORACLE_MYOPIC','ORACLE_OPTIMAL']:
        src='MYOPIC_ORACLE' if name=='ORACLE_MYOPIC' else 'SEQUENCE_ORACLE'
        oldentry=old[name]
        oldmacro=oldentry['all']['macro'] if 'all' in oldentry else oldentry['macro']
        for k in ['mean_net_utility','gain_capture','query_fraction']:
            assert abs(oldmacro[k]-result['validation']['oracle'][src]['macro'][k])<1e-12;checks+=1
    assert hashes=={p.relative_to(R).as_posix():sha(p) for p in files};sealed()
    save('source_hashes.json',hashes);save('horizon_audit.json',result)
    save('review.json',dict(passed=True,checks=checks,all_original_cases_retained=True,independent_binary_policy_enumeration=True,
        source_hashes_unchanged=True,legacy_validation_oracle_reproduced=True,new_fits=0,new_model_forwards=0,
        physical_or_prediction_calls=0,new_data=0,final_tests_absent=True,
        interpretation='Hindsight oracle opportunity,not information available before query or achievable learning guarantee. Training16date summary is not learned-model OOF performance.'))
    print(json.dumps({p:{k:v for k,v in x.items() if k!='oracle'} for p,x in result.items()},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
