"""Independent NumPy total forward and complete saved-sequence accounting; no fits."""
import importlib.util,torch,numpy as np
from common116 import *
spec=importlib.util.spec_from_file_location('audit113_numpy',R/'work/grid113_relational_learning/audit.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def numpy_total(w,row,arm):
    x=row['global41'].astype(np.float64)
    for i in [0,2,4]:
        x=x@w[f'base.q_net.{i}.weight'].T+w[f'base.q_net.{i}.bias']
        if i!=4:x=np.maximum(x,0)
    delta={k.removeprefix('delta.'):v for k,v in w.items() if k.startswith('delta.')}
    return x+old.numpy_forward(delta,row,arm.removeprefix('BASE_'))
def main():
    assert not (O/'review.json').exists();verify();configure();F=read(O/'fits.json');assert F['passed'] and len(F['records'])==6
    D=read(O/'design.json');fee=D['unchanged']['cost_per_query'];scale=D['unchanged']['gain_scale'];P=read(O/'preflight.json');Q=np.load(O/'state_predictions.npz')
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cm={(c['sample_id'],j):(c,j) for c in cases for j in range(len(c['states']))};meta={};row={};checked=0;num=0;errmax=0.
    for part in ['train','validation']:
        meta[part]=read(S/f'{part}_metadata.json');row[part]=pickle.loads((S/f'{part}_public.pkl').read_bytes());target=np.load(S/f'{part}_targets.npz')['target']
        for i,r in enumerate(meta[part]):
            c,j=cm[r['sample_id'],r['position']];assert c['partition']==part
            assert np.float32((c['gains'][j]-fee*c['queries'][j])/scale)==target[i];checked+=1
    for rec in F['records']:
        name=f"{rec['arm']}_seed{rec['seed']}";path=O/'models'/f'{name}.pt';assert sha(path)==rec['checkpoint_sha256'];cp=torch.load(path,weights_only=False,map_location='cpu')
        assert cp['updates']==7250 and all(int(s['step'])==7250 for s in cp['optimizer']['state'].values())
        assert len(cp['optimizer']['state'])==len(cp['state_dict']);assert sum(v.numel() for v in cp['state_dict'].values())==12916
        assert rec['initial_hash']==P['initial_hashes'][name];w={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()}
        for part in ['train','validation']:
            y=np.load(S/f'{part}_targets.npz')['target'];ids=sorted(set([0,len(y)-1]+list(range(0,len(y),151))+list(np.flatnonzero(y>0))))
            for i in ids:
                q=numpy_total(w,row[part][i],rec['arm']);actual=Q[name+'__'+part][i];e=float(np.max(abs(q-actual)));errmax=max(errmax,e)
                assert np.allclose(q,actual,atol=3e-5,rtol=3e-5),(name,part,i,e);assert bool(q[1]>q[0])==bool(actual[1]>actual[0]);num+=1
        exposure=np.load(O/f'exposure_{name}.npz');assert len(exposure['minibatch_losses'])==7250 and exposure['state_draws'].sum()==232000
        assert np.array_equal(exposure['state_draws'],np.load(R/f"outputs/grid107_direct_value/training_exposure_seed{rec['seed']}.npz")['state_draws'])
        assert rec['index_sha256']==P['sampling_streams'][str(rec['seed'])]['index_sha256']
        flow=read(O/f'first_steps_{name}.json');assert [r['update'] for r in flow]==[1,2]
        assert all(v==0 for k,v in flow[0]['gradient_norms_before_clip'].items() if k not in ['base','delta.head'])
        assert all(v>0 for k,v in flow[1]['gradient_norms_before_clip'].items() if k not in ['base','delta.head'])
    # All trajectories checked against original gain/query vectors and saved model decisions.
    bycase={c['sample_id']:c for c in cases};count=0
    for part in ['train','validation']:
        lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta[part])};trajectories=read(O/f'{part}_trajectories.json');result=read(O/f'{part}_results.json')
        for r in trajectories:
            c=bycase[r['sample_id']];p=0
            if r['method'].startswith(('BASE_JOINT_','BASE_FACTOR_')):
                seq=[]
                while p<len(c['states']):
                    q=Q[r['method']+'__'+part][lookup[c['sample_id'],p]];go=int(q[1]>q[0]);seq.append(go)
                    if not go:break
                    p+=1
                assert seq==r['actions'] and p==r['batches']
            else:p=r['batches']
            assert float(c['gains'][:p].sum())==r['gain'];assert int(c['queries'][:p].sum())==r['queries'];assert c['chosen'][p]==r['chosen_pool_id']
            assert abs((r['gain']-fee*r['queries'])/scale-r['net_utility'])<1e-12;count+=1
        for name,res in result.items():
            rr=[r for r in trajectories if r['method']==name];dates=sorted({r['date'] for r in rr});u=np.mean([np.mean([r['net_utility'] for r in rr if r['date']==d]) for d in dates])
            assert abs(u-res['all']['macro']['mean_net_utility'])<1e-12
    decision=read(O/'decision.json');pairs=decision['paired_seeds'];dates=decision['date_joint_minus_factor'];f=decision['family_mean_utility']
    signal=f['JOINT']>f['FACTOR'] and all(r['joint_minus_factor']>0 for r in pairs) and sum(v>0 for v in dates.values())>=3
    assert signal==decision['mechanism_signal'];assert (signal and decision['all_anchors_exceeded'])==decision['competitive_candidate']
    vr=read(O/'validation_results.json');util=lambda n:vr[n]['all']['macro']['mean_net_utility']
    for s in range(3):
        assert abs(pairs[s]['joint_minus_factor']-(util(f'BASE_JOINT_seed{s}')-util(f'BASE_FACTOR_seed{s}')))<1e-12
        assert abs(pairs[s]['joint_minus_base']-(util(f'BASE_JOINT_seed{s}')-util(f'DIRECT{s}')))<1e-12
    for d,v in dates.items():assert abs(v-np.mean([vr[f'BASE_JOINT_seed{s}']['all']['per_date'][d]['mean_net_utility']-vr[f'BASE_FACTOR_seed{s}']['all']['per_date'][d]['mean_net_utility'] for s in range(3)]))<1e-12
    anchors=all(f['JOINT']>x for x in [f['BASE'],f['DQN'],util('BATCH1')]);assert anchors==decision['all_anchors_exceeded']
    # Recompute fixed critical contexts independently, all seeds always retained.
    probes=read(O/'critical_probes.json');pc={r['context']:r for r in read(R/'outputs/grid111_action_feedback/critical_details.json')}
    for r in probes:
        public=pc[r['context']];call=read(R/public['source']);inp=pack(public['public_input'],np.asarray(call['learned_decisions'][0]['normalized_features'],np.float32),fee,scale)
        for rec in F['records']:
            name=f"{rec['arm']}_seed{rec['seed']}";cp=torch.load(O/'models'/f'{name}.pt',weights_only=False,map_location='cpu');w={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()};q=numpy_total(w,inp,rec['arm'])
            assert np.allclose(q,[r['scores'][name]['q0'],r['scores'][name]['q1']],atol=3e-5,rtol=3e-5)
    assert F['wall_s']<1801;verify();sealed()
    save(O/'review.json',dict(passed=True,targets_checked=checked,independent_numpy_total_forwards=num,maximum_abs_error=errmax,independent_critical_forwards=18,
        complete_train_and_validation_trajectories_checked=count,all_six_adam_steps7250=True,first_two_steps_in_budget_and_upstream_flow_checked=True,
        paired_initializations_original_sampling_streams_and_counts=True,all_frozen_hashes_unchanged=True,validation_only_after_six_fits=True,
        new_physical_calls=0,final_test_materialized=False,scope='Offline teacher-path audit, not independent AC solution or physical quality/latency claim.'))
    save(O/'state.json',dict(stage='audited_waiting_publication',fits=6));print('AUDIT_PASSED',num,count,errmax,flush=True)
if __name__=='__main__':main()
