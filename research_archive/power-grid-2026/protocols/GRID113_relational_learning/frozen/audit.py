"""Independent NumPy forward, targets/draws/decisions audit. No fitting or forecast."""
import math,inspect
import numpy as np,torch
from common import *
from model import RelationalValue
def numpy_forward(w,row,arm):
    def linear(x,key):
        y=x@w[key+'.weight'].T
        return y+w[key+'.bias'] if key+'.bias' in w else y
    def enc(x,key):return np.maximum(linear(np.maximum(linear(x,key+'.0'),0),key+'.2'),0)
    a=enc(row['context_action'],'A');f=enc(row['context_feedback'],'F');q=enc(row['next_action'],'A');anchor=enc(row['anchor_action'],'A')
    if arm=='FACTOR':f=np.broadcast_to(f.mean(0),a.shape)
    v=enc(np.concatenate([a,f],-1),'V');logit=linear(q,'Q')@linear(a,'K').T/math.sqrt(8)
    exp=np.exp(logit-logit.max(1,keepdims=True));att=exp/exp.sum(1,keepdims=True);r=att@v
    g=enc(np.concatenate([row['global41'],anchor,row['known_query_cost']]),'G')
    hh=enc(np.concatenate([q,r,np.broadcast_to(g,(len(q),16))],-1),'H')
    z=np.concatenate([hh.mean(0),hh.max(0),g]);return linear(np.maximum(linear(z,'head.0'),0),'head.2')
def main():
    assert not (O/'review.json').exists();verify();configure();D=read(O/'design.json');P=read(O/'preflight.json');F=read(O/'fits.json');assert F['passed'] and len(F['records'])==6
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());source={(c['sample_id'],j):(c,j) for c in cases for j in range(len(c['states']))}
    scale=D['target']['gain_scale'];fee=D['target']['cost_per_query'];checks={};public={};metas={};targets={};pred=np.load(O/'state_predictions.npz')
    for part in ['train','validation']:
        public[part]=pickle.loads((O/f'{part}_public.pkl').read_bytes());metas[part]=read(O/f'{part}_metadata.json');targets[part]=np.load(O/f'{part}_targets.npz')['target']
        for i,r in enumerate(metas[part]):
            c,j=source[r['sample_id'],r['position']];assert c['partition']==part
            y=np.float32((c['gains'][j]-fee*c['queries'][j])/scale);assert y==targets[part][i]
            assert all(np.isfinite(v).all() for v in public[part][i].values())
            assert len(public[part][i]['next_action'])==c['queries'][j]
        checks[part+'_targets']=len(public[part])
    prob=np.load(O/'train_targets.npz')['sampling_probability'];maxerror=0.;compared=0
    # Fixed first/last/every151st plus ALL positive states; chosen by targets, not errors.
    for rec in F['records']:
        name=f"{rec['arm']}_seed{rec['seed']}";cp=torch.load(O/'models'/f'{name}.pt',weights_only=False,map_location='cpu')
        assert sha(O/'models'/f'{name}.pt')==rec['checkpoint_sha256'];assert cp['updates']==7250
        assert all(int(v['step'])==7250 for v in cp['optimizer']['state'].values())
        assert len(cp['optimizer']['state'])==len(cp['state_dict'])
        w={k:v.numpy().astype(np.float64) for k,v in cp['state_dict'].items()};assert sum(v.size for v in w.values())==5938
        for part in ['train','validation']:
            ids=sorted(set([0,len(public[part])-1]+list(range(0,len(public[part]),151))+list(np.flatnonzero(targets[part]>0))))
            for i in ids:
                q=numpy_forward(w,public[part][i],rec['arm']);saved=pred[name+'__'+part][i]
                err=float(np.max(abs(q-saved)));maxerror=max(maxerror,err);assert np.allclose(q,saved,atol=3e-5,rtol=3e-5),(name,part,i,q,saved)
                assert bool(q[1]>q[0])==bool(saved[1]>saved[0]),(name,part,i,'sign disagreement');compared+=1
        counts=np.load(O/f'exposure_{name}.npz')['state_draws'];assert counts.sum()==232000
        assert np.array_equal(counts,np.load(R/f"outputs/grid107_direct_value/training_exposure_seed{rec['seed']}.npz")['state_draws'])
        assert rec['initial_hash']==P['initial_hashes'][str(rec['seed'])]
    # Independently replay decisions over all complete saved search trajectories.
    trajectories=read(O/'validation_trajectories.json');valmap={(r['sample_id'],r['position']):i for i,r in enumerate(metas['validation'])};bycase={c['sample_id']:c for c in cases}
    for r in trajectories:
        c=bycase[r['sample_id']];p=0
        if r['method'].startswith(('JOINT_','FACTOR_')):
            seq=[]
            while p<len(c['states']):
                q=pred[r['method']+'__validation'][valmap[c['sample_id'],p]];go=bool(q[1]>q[0]);seq.append(int(go))
                if not go:break
                p+=1
            assert seq==r['actions'] and p==r['batches']
        else:p=r['batches']
        assert float(c['gains'][:p].sum())==r['gain'];assert int(c['queries'][:p].sum())==r['queries']
        assert abs((r['gain']-fee*r['queries'])/scale-r['net_utility'])<1e-12
    result=read(O/'validation_results.json');decision=read(O/'decision.json')
    def independent_utility(name):
        rr=[r for r in trajectories if r['method']==name];ds=sorted({r['date'] for r in rr})
        return float(np.mean([np.mean([r['net_utility'] for r in rr if r['date']==d]) for d in ds]))
    for name in result:assert abs(independent_utility(name)-result[name]['all']['macro']['mean_net_utility'])<1e-12
    pairs=[independent_utility(f'JOINT_seed{s}')-independent_utility(f'FACTOR_seed{s}') for s in range(3)]
    dates=decision['date_deltas'];signal=np.mean(pairs)>0 and all(x>0 for x in pairs) and sum(v>0 for v in dates.values())>=3
    assert signal==decision['mechanism_signal']
    # All critical contexts always included; independent forward and input packing.
    probes=read(O/'critical_probes.json');pc={r['context']:r for r in read(R/'outputs/grid111_action_feedback/critical_details.json')}
    for r in probes:
        c=pc[r['context']];call=read(R/c['source']);inp=pack(c['public_input'],np.asarray(call['learned_decisions'][0]['normalized_features'],np.float32),fee,scale)
        for rec in F['records']:
            name=f"{rec['arm']}_seed{rec['seed']}";cp=torch.load(O/'models'/f'{name}.pt',weights_only=False,map_location='cpu');w={k:v.numpy().astype(np.float64) for k,v in cp['state_dict'].items()}
            q=numpy_forward(w,inp,rec['arm']);assert np.allclose(q,[r['scores'][name]['q0'],r['scores'][name]['q1']],atol=3e-5,rtol=3e-5)
    sealed();assert F['wall_s']<1801
    save(O/'review.json',dict(passed=True,original_targets_checked=checks,independent_numpy_forwards=compared,maximum_abs_error=maxerror,
        all_saved_trajectories_checked=len(trajectories),independent_critical_forwards=18,all_six_optimizer_states_step7250=True,
        all_parameter_tensors_active=True,paired_initialization_and_old_sampling_counts=True,six_fits_no_validation_selection=True,
        code_data_hashes_unchanged=True,final_test_unmaterialized=True,new_physical_steps=0,
        scope='Offline saved teacher-path direct regression audit; no independent AC solution, physical quality or latency claim.'))
    save(O/'state.json',dict(stage='audited_awaiting_publication',fits=6));print('AUDIT_PASSED',compared,maxerror,flush=True)
if __name__=='__main__':main()
