"""Qualify implementation, then freeze before any optimizer is instantiated."""
import copy,datetime
from common119 import *

def main():
    assert not O.exists(),'Preserve all attempts';O.mkdir(parents=True);configure();verify113();sealed()
    plan=read(PLAN/'plan.json');match=read(PLAN/'matching_check.json');assert match['passed']
    for n,h in read(PLAN/'input_manifest.json').items():assert sha(R/n)==h,n
    d=copy.deepcopy(plan);d.update(id='GRID119_FIXED_BASE_LEARNING',status='authorized_nine_fits',
        authorization='User: 来训练看看效果. Execute the reviewed GRID118 fixed-base plan.',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fit_order=[(r['arm'],r['seed']) for r in plan['future_fits']['order']],
        training=dict(updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_grad_norm=10.,huber_beta=1.,wall_cap_s=1800),
        runtime=dict(torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1))
    save(O/'design.json',d)
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cm={(c['sample_id'],j):(c,j) for c in cases for j in range(len(c['states']))};norm=np.load(R/'outputs/grid102_search_rl/normalization.npz')
    checked={}
    for part in ['train','validation']:
        meta=read(S/f'{part}_metadata.json');rows=pickle.loads((S/f'{part}_public.pkl').read_bytes());ys=np.load(S/f'{part}_targets.npz')['target']
        assert len(meta)==len(rows)==len(ys)
        for i,(m,x) in enumerate(zip(meta,rows)):
            c,j=cm[m['sample_id'],m['position']];assert c['partition']==part
            assert np.float32((c['gains'][j]-d['unchanged']['cost_per_query']*c['queries'][j])/d['unchanged']['gain_scale'])==ys[i]
            assert np.array_equal(x['global41'],np.clip((c['states'][j]-norm['mean'])/norm['scale'],-20,20).astype('f'))
            assert set(x)=={'context_action','context_feedback','next_action','anchor_action','global41','known_query_cost'}
            assert len(x['next_action'])==c['queries'][j] and all(np.isfinite(v).all() for v in x.values())
        checked[part]=len(rows)
    rng=np.random.default_rng(119);rr=[]
    for n,m in [(8,6),(3,2)]:rr.append(dict(context_action=rng.normal(size=(n,68)).astype('f'),context_feedback=rng.normal(size=(n,8)).astype('f'),next_action=rng.normal(size=(m,68)).astype('f'),anchor_action=rng.normal(size=68).astype('f'),global41=rng.normal(size=41).astype('f'),known_query_cost=np.array([.03],'f')))
    b=collate(rr);initial={};synthetic={}
    for seed in range(3):
        for arm in ['BASE_EXTRA','FIXED_JOINT','FIXED_FACTOR']:
            name=f'{arm}_seed{seed}';m=build(seed,arm);before=digest(m);m.train()
            assert digest(m.base)==match['seeds'][str(seed)]['base_final_hash']
            assert sum(p.numel() for p in trainable(m))==(6978 if arm=='BASE_EXTRA' else 5938)
            q=m(**b);assert torch.equal(q,m.base(b['global41']))
            loss=torch.nn.functional.smooth_l1_loss(q,torch.tensor([[0.,.4],[0.,-.3]]));loss.backward();flow=gradient_flow(m)
            if arm=='BASE_EXTRA':assert flow['base']>0
            else:
                assert not m.base.training and all(p.grad is None and not p.requires_grad for p in m.base.parameters())
                assert float(m.delta.head[-1].weight.grad.abs().sum())>0
                assert all(torch.count_nonzero(p.grad)==0 for k,p in m.delta.named_parameters() if not k.startswith('head.2.'))
                # Nonzero synthetic head on a discarded copy avoids trivial zero-output invariance checks.
                probe=copy.deepcopy(m);torch.manual_seed(119000+seed)
                with torch.no_grad():probe.delta.head[-1].weight.normal_();probe.delta.head[-1].bias.normal_()
                with torch.no_grad():
                    orig=probe(**b);x={k:v.clone() for k,v in b.items()}
                    for k,mask in [('context_action','context_mask'),('context_feedback','context_mask'),('next_action','next_mask')]:x[k][~x[mask]]=float('nan')
                    assert torch.equal(probe(**x),orig)
                    x={k:v.clone() for k,v in b.items()}
                    for k in ['context_action','context_feedback','context_mask']:x[k]=x[k].flip(1)
                    assert torch.allclose(probe(**x),orig,atol=1e-5,rtol=1e-5)
                    x={k:v.clone() for k,v in b.items()}
                    for k in ['next_action','next_mask']:x[k]=x[k].flip(1)
                    assert torch.allclose(probe(**x),orig,atol=1e-5,rtol=1e-5)
            assert before==digest(m);m.zero_grad(set_to_none=True);initial[name]=before
            synthetic[name]=dict(gradient_norms=flow,initial_output_exact=True,weight_updates=0)
    save(O/'preflight.json',dict(passed=True,states_checked=checked,initial_hashes=initial,synthetic_checks=synthetic,
        frozen_base_hashes={s:r['base_final_hash'] for s,r in match['seeds'].items()},sampling_streams=match['sampling_streams'],
        synthetic_backward_only=True,optimizer_instances=0,optimizer_steps=0,new_fits=0,validation_model_inference=0,final_test_materialized=False))
    code=list(W.glob('*.py'))+[R/f'work/{p}' for p in ['grid118_fixed_base_plan/fixed_model.py','grid115_matched_module_plan/matched_model.py','grid113_relational_learning/model.py','grid113_relational_learning/common.py','grid113_relational_learning/audit.py','grid107_direct_value/common107.py','grid112_relational_plan/schema.py']]
    save(O/'code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in code})
    data=[O/'design.json',O/'preflight.json']+[PLAN/n for n in ['plan.json','matching_check.json','input_manifest.json','training_target_and_exposure_check.npz']]
    data+=[S/f'{p}_{n}' for p in ['train','validation'] for n in ['public.pkl','metadata.json','targets.npz']]
    data+=[R/f'outputs/grid107_direct_value/{n}' for n in ['design.json','fits.json','state_predictions.jsonl.gz','validation_results.json','validation_trajectories.json']]
    data+=[R/f'outputs/grid107_direct_value/models/direct_seed{s}_final.zip' for s in range(3)]
    data+=[R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/normalization.npz',R/'outputs/grid100_data_foundation/design.json',R/'outputs/grid111_action_feedback/critical_details.json',R/'outputs/grid116_matched_module/decision.json']
    data+=[R/r['source'] for r in read(R/'outputs/grid111_action_feedback/critical_details.json')]
    save(O/'data_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in data})
    save(O/'state.json',dict(stage='qualified_waiting_push',new_fits=0));print('GRID119_PREFLIGHT_PASSED_ZERO_UPDATES',flush=True)
if __name__=='__main__':main()
