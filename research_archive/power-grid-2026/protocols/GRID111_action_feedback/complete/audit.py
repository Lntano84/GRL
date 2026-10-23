"""Independent loops, exact permutation enumeration, and all-state bound checks."""
from common111 import R,O,read,save,sha
import collections,gzip,itertools,json,pickle
import numpy as np

def legal(r):return not any([r['done'],r['illegal'],r['ambiguous'],r['exceptions']]) and np.isfinite(r['rho']) and r['rho']>0 and np.isfinite(r['reward'])
def bits(p):return [int(p[i]>0 and p[j]>0 and p[i]==p[j]) for i in range(len(p)) for j in range(i+1,len(p))]
def main():
    assert not (O/'review.json').exists();assert read(O/'analysis_receipt.json')['passed']
    results=read(O/'state_results.json');idx={(r['sample_id'],r['position']):r for r in results}
    flips=read(R/'outputs/grid109_input_coverage/positive_flip_details.json');selected={(r['sample_id'],r['position']) for r in flips}
    dates={}
    for r in results:dates.setdefault(r['date'],(r['sample_id'],r['position']))
    selected.update(dates.values());checks=collections.Counter();details=[];held=[]
    D=read(R/'outputs/grid102_search_rl/design.json');future={}
    with gzip.open(O/'future_targets.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line);future[r['sample_id'],r['position']]=r['future_outcomes']
    def loop(c,fs=None):
        use=[i for i,r in enumerate(c['seen_feedback']) if legal(r)];seen=[bits(c['seen_plans'][i]) for i in use];y=[float(np.float32(c['seen_feedback'][i]['reward'])) for i in use]
        assert max(y)<=c['incumbent_reward']+1e-12
        target=[bits(p) for p in c['next_plans']];n=len(y);mu=sum(y)/n;v=sum((a-mu)**2 for a in y)/n;pred=[];variances=[]
        for b in target:
            dist=[sum(x!=z for x,z in zip(a,b)) for a in seen];lo=min(dist);rewards=[y[j] for j,d in enumerate(dist) if d==lo];t=len(rewards)
            pred.append(sum(rewards)/t);variances.append(v*(n-t)/(t*(n-1)) if n>1 else 0)
        inc=bits(c['incumbent_plan']);contrast=[];covered=[]
        for k in range(len(inc)):
            zeros=[y[j] for j,b in enumerate(seen) if b[k]==0];ones=[y[j] for j,b in enumerate(seen) if b[k]==1]
            covered.append(bool(zeros and ones));contrast.append(sum(ones)/len(ones)-sum(zeros)/len(zeros) if zeros and ones else 0)
        direction=[]
        for b in target:
            relevant=[k for k in range(len(b)) if b[k]!=inc[k] and covered[k]]
            direction.append(sum((b[k]-inc[k])*contrast[k] for k in relevant)/len(relevant) if relevant else 0)
        out=dict(pred=pred,contrast_coverage=sum(covered)/len(covered),direction=direction)
        assert max(pred)<=c['incumbent_reward']+1e-12
        if fs is not None:
            legalnext=[i for i,r in enumerate(fs) if legal(r)]
            if legalnext:
                vals=[float(np.float32(fs[i]['reward'])) for i in legalnext]
                out.update(nearest_mse=sum((pred[i]-a)**2 for i,a in zip(legalnext,vals))/len(vals),
                    mean_mse=sum((mu-a)**2 for a in vals)/len(vals),
                    exact_unpaired_mse=sum((mu-a)**2+variances[i] for i,a in zip(legalnext,vals))/len(vals))
            else:out.update(nearest_mse=None,mean_mse=None,exact_unpaired_mse=None)
            gain=max([c['incumbent_reward']]+[float(np.float32(r['reward'])) for r in fs if r['strict_admissible']])-c['incumbent_reward']
            out['target_net']=gain-D['cost_per_query']*len(fs)
        return out
    with gzip.open(O/'public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line);key=(r['sample_id'],r['position']);c=r['public_input'];record=idx[key]
            # All-state bound independent of vectorised association code.
            seen=[float(np.float32(a['reward'])) for a in c['seen_feedback'] if legal(a)];assert max(seen)<=c['incumbent_reward']+1e-12
            gain=max([c['incumbent_reward']]+[float(np.float32(a['reward'])) for a in future[key] if a['strict_admissible']])-c['incumbent_reward']
            assert gain-D['cost_per_query']*len(c['next_ids'])==record['target_net'];assert record['positive_interpolation_gain_possible']==False
            checks['all_state_bounds_and_targets']+=1
            if key in selected:
                v=loop(c,future[key])
                for k in ['nearest_mse','mean_mse','exact_unpaired_mse','contrast_coverage','target_net']:
                    if v[k] is None:assert record[k] is None
                    else:assert abs(v[k]-record[k])<1e-12,(key,k,v[k],record[k])
                details.append(dict(sample_id=key[0],position=key[1],independent_values=v));checks['loop_contexts']+=1
    for p in read(O/'critical_details.json'):
        v=loop(p['public_input']);d=p['association'];assert np.allclose(v['pred'],d['interpolated_reward'],rtol=0,atol=1e-12)
        assert np.allclose(v['direction'],d['next_contrast_direction'],rtol=0,atol=1e-12)
        checks['live_probe_loops']+=1
    # Exhaust all assignments for small examples: confirms exact null, no random draws.
    for values in [[0.,1.,2.],[-1.,.2,.2],[2.,2.,2.]]:
        n=len(values);mu=np.mean(values);var=np.mean((np.array(values)-mu)**2)
        for t in [1,2,3]:
            for truth in [-.4,1.3]:
                empirical=np.mean([(np.mean(p[:t])-truth)**2 for p in itertools.permutations(values)])
                analytic=(mu-truth)**2+var*(n-t)/(t*(n-1));assert abs(empirical-analytic)<1e-12;checks['exact_permutation_null_examples']+=1
    assert checks['all_state_bounds_and_targets']==4438
    # Match all batch targets directly to old frozen transitions.
    old={(c['sample_id'],j):float(c['gains'][j]-D['cost_per_query']*c['queries'][j]) for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) for j in range(len(c['states']))}
    assert all(old[k]==v['target_net'] for k,v in idx.items());checks['old_target_reproductions']=len(old)
    for p,h in read(O/'input_manifest.json').items():assert sha(R/p)==h,p
    base=read(R/'outputs/grid100_data_foundation/design.json');test=[r for r in base['scenario_plan'] if r['partition']=='test'];assert len(test)==8
    assert all(not (R/base['environment']/'chronics'/r['scenario']).exists() and not (R/'outputs/grid100_data_foundation/datasets'/r['scenario']).exists() for r in test)
    save('independent_audit_details.json',details)
    save('review.json',dict(passed=True,checks=dict(checks),sources_unchanged=True,final8_test_ungenerated=True,new_model_fits=0,learned_model_inferences=0,new_physical_calls=0,
        scope='Fixed arithmetic on saved observations. Nearest-neighbour values and observational pair contrasts are diagnostics, not new trained/deployed policies.'))
    print(read(O/'review.json'),flush=True)
if __name__=='__main__':main()
