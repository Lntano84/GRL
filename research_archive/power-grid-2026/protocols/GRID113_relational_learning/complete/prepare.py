"""Prepare isolated public inputs/targets, qualify engineering before six fits."""
import collections,datetime,gzip,inspect,pickle,copy,time
import numpy as np,torch
from common import *
from model import RelationalValue
def main():
    assert not O.exists(),'Preserve any earlier attempt';O.mkdir(parents=True);configure();sealed()
    P=read(R/'outputs/grid112_relational_plan/plan.json')
    for p,h in read(R/'outputs/grid112_relational_plan/input_manifest.json').items():assert sha(R/p)==h,p
    D={**P,'id':'GRID113_PAIRED_RELATIONAL_LEARNING','status':'authorized_fixed_six_fits',
       'authorization':'User approved concrete GRID112 plan: 没问题. Engineering then exactly six serial fits.',
       'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'fit_order':[(arm,seed) for seed in [0,1,2] for arm in ['JOINT','FACTOR']],
       'training':dict(updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,max_grad_norm=10.,
          huber_beta=1.,sampler_seed_offset=20261022,trace_every=500,wall_cap_s=1800),
       'runtime':dict(device='cpu',threads=1,torch=torch.__version__,numpy=np.__version__,python=sys.version),
       'executed_this_stage':'Engineering qualification; synthetic backward without optimizer. No research updates until frozen setup pushed.'}
    save(O/'design.json',D)
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());norm=np.load(R/'outputs/grid102_search_rl/normalization.npz')
    states={(c['sample_id'],j):(c,j,s) for c in cases for j,s in enumerate(c['states'])};raw={};metadata={}
    with gzip.open(R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=__import__('json').loads(line);key=r['sample_id'],r['position'];c,j,s=states[key]
            assert c['partition']==r['partition'] and c['date']==r['date']
            g=np.clip((s-norm['mean'])/norm['scale'],-20,20).astype(np.float32)
            inp=pack(r['public_input'],g,D['target']['cost_per_query'],D['target']['gain_scale'])
            assert key not in raw;raw[key]=inp
            metadata[key]={k:r[k] for k in ['sample_id','position','date','partition','physical_state_group']}
            metadata[key]['next_queries']=len(inp['next_action'])
    assert len(raw)==4438
    trmeta=read(R/'outputs/grid107_direct_value/training_rows.json');trkeys=[(r['sample_id'],r['position']) for r in trmeta]
    vakeys=[k for k in states if metadata[k]['partition']=='validation'];assert len(vakeys)==856
    for part,keys in [('train',trkeys),('validation',vakeys)]:
        with (O/f'{part}_public.pkl').open('wb') as f:pickle.dump([raw[k] for k in keys],f,protocol=4)
        save(O/f'{part}_metadata.json',[metadata[k] for k in keys])
    z=np.load(R/'outputs/grid107_direct_value/train.npz');target=z['target'];prob=z['sampling_probability']
    assert len(target)==3582
    for i,k in enumerate(trkeys):
        c,j,s=states[k];assert np.array_equal(raw[k]['global41'],z['X'][i])
        assert np.float32((c['gains'][j]-D['target']['cost_per_query']*c['queries'][j])/D['target']['gain_scale'])==target[i]
    np.savez_compressed(O/'train_targets.npz',target=target,sampling_probability=prob)
    va=np.array([(states[k][0]['gains'][k[1]]-D['target']['cost_per_query']*states[k][0]['queries'][k[1]])/D['target']['gain_scale'] for k in vakeys],np.float32)
    np.savez_compressed(O/'validation_targets.npz',target=va)
    assert not(set(r['date'] for r in trmeta)&set(metadata[k]['date'] for k in vakeys))
    initial={};streams={}
    for seed in [0,1,2]:
        torch.manual_seed(seed);a=RelationalValue('JOINT');torch.manual_seed(seed);b=RelationalValue('FACTOR')
        assert digest(a)==digest(b);assert sum(p.numel() for p in a.parameters())==5938
        initial[str(seed)]=digest(a);rng=np.random.default_rng(20261022+seed);counts=np.zeros(3582,np.int64);h=hashlib.sha256()
        for _ in range(7250):
            ids=rng.choice(3582,size=32,replace=True,p=prob);np.add.at(counts,ids,1);h.update(ids.astype('<i8').tobytes())
        assert np.array_equal(counts,np.load(R/f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz')['state_draws'])
        streams[str(seed)]=dict(index_sha256=h.hexdigest(),count_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),draws=int(counts.sum()))
    # Synthetic invariance and gradient smoke: no optimizer instantiated or stepped.
    rng=np.random.default_rng(91)
    sample=[]
    for n,m in [(7,5),(3,2)]:
        sample.append(dict(context_action=rng.normal(size=(n,68)).astype('f'),context_feedback=rng.normal(size=(n,8)).astype('f'),
            next_action=rng.normal(size=(m,68)).astype('f'),anchor_action=rng.normal(size=68).astype('f'),global41=rng.normal(size=41).astype('f'),known_query_cost=np.ones(1,'f')))
    inp=collate(sample);checks={};effects={}
    for arm in ['JOINT','FACTOR']:
        torch.manual_seed(0);net=RelationalValue(arm);before=digest(net);q=net(**inp)
        def check(name,x):
            delta=float((net(**x)-q).abs().max());checks[arm+'_'+name]=delta;assert delta<1e-6,(arm,name,delta)
        x={k:v.clone() for k,v in inp.items()};perm=torch.tensor([6,2,5,1,0,4,3])
        for k in ['context_action','context_feedback','context_mask']:x[k]=x[k][:,perm]
        check('joint_context_permutation',x)
        x={k:v.clone() for k,v in inp.items()};p=torch.tensor([4,2,1,0,3])
        for k in ['next_action','next_mask']:x[k]=x[k][:,p]
        check('next_permutation',x)
        x={k:v.clone() for k,v in inp.items()}
        for k,mask in [('context_action','context_mask'),('context_feedback','context_mask'),('next_action','next_mask')]:x[k][~x[mask]]=float('nan')
        check('poisoned_padding',x)
        x={k:v.clone() for k,v in inp.items()};x['context_feedback'][0,:7]=x['context_feedback'][0,perm]
        delta=float((net(**x)-q).abs().max());effects[arm]=delta
        if arm=='FACTOR':assert delta<1e-6
        else:assert delta>1e-8,'Constructed binding witness must distinguish pairings'
        loss=(q-torch.tensor([[.3,-.4],[-.2,.5]])).square().mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        checks[arm+'_all_parameter_tensors_have_finite_gradient']=True
        assert before==digest(net);net.zero_grad(set_to_none=True)
    assert 'target' not in inspect.signature(RelationalValue.forward).parameters
    # Explicit pure pack does not accept label/future fields, and IDs never enter tensor cache.
    probe=next(iter(raw.values()));assert set(probe)=={'context_action','context_feedback','next_action','anchor_action','global41','known_query_cost'}
    assert read(R/'outputs/grid112_relational_plan/schema_check.json')['passed']
    save(O/'preflight.json',dict(passed=True,states=4438,train=3582,validation=856,actual_parameters=5938,initial_hashes=initial,
        sampling_streams=streams,invariance_max_abs=checks,pairing_counterexample_abs=effects,optimizer_steps=0,
        synthetic_backward_only=True,train_validation_dates_disjoint=True,labels_separate=True,
        schema_all_state_bus_and_ID_checks=read(R/'outputs/grid112_relational_plan/schema_check.json'),
        no_environment_constructed=True,final_test_materialized=False))
    # Data inputs and executable source closed before research training; no validation outcome selection.
    source=list(W.glob('*.py'))+[R/'work/grid112_relational_plan/schema.py',R/'work/grid107_direct_value/common107.py']
    save(O/'code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in source})
    data=[O/'design.json',O/'preflight.json']+list(O.glob('*_public.pkl'))+list(O.glob('*_targets.npz'))+list(O.glob('*_metadata.json'))
    data += [R/p for p in read(R/'outputs/grid112_relational_plan/input_manifest.json') if p.startswith('outputs/')]
    data += [R/'outputs/grid107_direct_value/validation_trajectories.json',R/'outputs/grid107_direct_value/validation_results.json',R/'outputs/grid111_action_feedback/critical_details.json']
    data += [R/f'outputs/grid107_direct_value/training_exposure_seed{s}.npz' for s in [0,1,2]]
    save(O/'data_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in data})
    save(O/'state.json',dict(stage='qualified_awaiting_frozen_push',fits=0,optimizer_steps=0))
    print('QUALIFIED:5938params,6paired initializations,4438publicstates,zero optimizer steps',flush=True)
if __name__=='__main__':main()
