"""Freeze authorized six fits after synthetic backward; zero optimizer updates."""
import copy,datetime
import torch,numpy as np
from common116 import *
def main():
    assert not O.exists(),'Keep previous attempts';O.mkdir(parents=True);configure();verify113();sealed()
    assert read(P115/'matching_check.json')['passed']
    for n,h in read(P115/'input_manifest.json').items():assert sha(R/n)==h,n
    plan=read(P115/'plan.json');D=copy.deepcopy(plan);D.update(id='GRID116_PRESERVED_BACKBONE_MODULE_TRAINING',status='authorized_six_fixed_fits',
        authorization='User: 开始训练一次吧. Execute the matched GRID115 six-fit experiment.',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        fit_order=[(r['arm'],r['seed']) for r in plan['future_fits']['order']],
        training=dict(updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_grad_norm=10.,huber_beta=1.,trace_every=500,wall_cap_s=1800),
        runtime=dict(torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1),
        stage_execution='Synthetic forward/backward only before frozen push; no optimizer update or train/validation model inference.')
    save(O/'design.json',D)
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cm={(c['sample_id'],j):(c,j) for c in cases for j in range(len(c['states']))};norm=np.load(R/'outputs/grid102_search_rl/normalization.npz')
    old=np.load(R/'outputs/grid107_direct_value/train.npz');checked={}
    for part in ['train','validation']:
        meta=read(S/f'{part}_metadata.json');rows=pickle.loads((S/f'{part}_public.pkl').read_bytes());targets=np.load(S/f'{part}_targets.npz')['target']
        for i,(r,inp) in enumerate(zip(meta,rows)):
            c,j=cm[r['sample_id'],r['position']];assert c['partition']==part
            y=np.float32((c['gains'][j]-D['unchanged']['cost_per_query']*c['queries'][j])/D['unchanged']['gain_scale']);assert y==targets[i]
            g=np.clip((c['states'][j]-norm['mean'])/norm['scale'],-20,20).astype(np.float32);assert np.array_equal(g,inp['global41'])
            assert inp['next_action'].shape[0]==c['queries'][j]
            assert set(inp)=={'context_action','context_feedback','next_action','anchor_action','global41','known_query_cost'}
            if part=='train':assert np.array_equal(g,old['X'][i]) and targets[i]==old['target'][i]
        checked[part]=len(rows)
    # Synthetic only, finite gradients; no optimizer object or step.
    rng=np.random.default_rng(116);rr=[]
    for n,m in [(8,6),(3,2)]:rr.append(dict(context_action=rng.normal(size=(n,68)).astype('f'),context_feedback=rng.normal(size=(n,8)).astype('f'),
        next_action=rng.normal(size=(m,68)).astype('f'),anchor_action=rng.normal(size=68).astype('f'),global41=rng.normal(size=41).astype('f'),known_query_cost=np.array([.03],'f')))
    b=collate(rr);hashes={};checks={};match=read(P115/'matching_check.json')
    for seed in range(3):
        for arm in ['BASE_JOINT','BASE_FACTOR']:
            m=build(seed,arm);before=digest(m);assert before==match['init'][str(seed)]['matched_full_initial_hash']
            assert sum(p.numel() for p in m.parameters())==12916
            q=m(**b);assert torch.equal(q,m.base(b['global41']))
            with torch.no_grad():
                x={k:v.clone() for k,v in b.items()}
                for key,mask in [('context_action','context_mask'),('context_feedback','context_mask'),('next_action','next_mask')]:x[key][~x[mask]]=float('nan')
                assert torch.equal(m(**x),q)
            loss=torch.nn.functional.smooth_l1_loss(q,torch.tensor([[0.,.4],[0.,-.3]]));loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            assert sum(float(p.grad.abs().sum()) for p in m.base.parameters())>0
            assert float(m.delta.head[-1].weight.grad.abs().sum())>0
            assert all(torch.count_nonzero(p.grad)==0 for name,p in m.delta.named_parameters() if not name.startswith('head.2.'))
            assert before==digest(m);m.zero_grad(set_to_none=True);hashes[f'{arm}_seed{seed}']=before
            checks[f'{arm}_seed{seed}']=dict(initial_output_equals_original=True,all_gradients_finite=True,base_and_final_correction_gradients_nonzero=True,upstream_zero_as_expected=True)
    save(O/'preflight.json',dict(passed=True,states_checked=checked,actual_parameters=12916,initial_hashes=hashes,
        original_base_hashes=match['init'],sampling_streams=match['original_sampling_streams'],synthetic_checks=checks,
        zero_residual_initial_upstream_gradient_expected=True,first_two_actual_steps_must_be_in_budget=True,
        no_optimizer_instances=True,optimizer_steps=0,new_fits=0,labels_used_only_for_formula_checks=True,final_test_materialized=False))
    code=list(W.glob('*.py'))+[R/'work/grid115_matched_module_plan/matched_model.py',R/'work/grid113_relational_learning/model.py',R/'work/grid113_relational_learning/common.py',R/'work/grid113_relational_learning/audit.py',R/'work/grid107_direct_value/common107.py',R/'work/grid112_relational_plan/schema.py']
    save(O/'code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in code})
    data=[O/'design.json',O/'preflight.json',P115/'plan.json',P115/'matching_check.json',P115/'input_manifest.json']
    data += [S/n for n in ['train_public.pkl','train_targets.npz','train_metadata.json','validation_public.pkl','validation_targets.npz','validation_metadata.json']]
    data += [R/f'outputs/grid107_direct_value/{n}' for n in ['train.npz','design.json','preflight.json','state_predictions.jsonl.gz','state_metrics.json','validation_results.json','validation_trajectories.json']]
    data += [R/f'outputs/grid107_direct_value/training_exposure_seed{s}.npz' for s in range(3)]
    data += [R/'outputs/grid111_action_feedback/critical_details.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/normalization.npz',R/'outputs/grid100_data_foundation/design.json']
    data += [R/r['source'] for r in read(R/'outputs/grid111_action_feedback/critical_details.json')]
    save(O/'data_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in data});save(O/'state.json',dict(stage='qualified_waiting_frozen_push',new_fits=0))
    print('QUALIFIED_12916_PARAMETERS_ZERO_UPDATES',flush=True)
if __name__=='__main__':main()
