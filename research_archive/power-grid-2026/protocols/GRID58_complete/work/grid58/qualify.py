"""Input/checkpoint checks plus bounded cached public-feedback diagnosis."""
import copy,csv,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid58';HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT/'work/grid21'),str(ROOT/'work/grid20')]
import numpy as np,torch
from compressed_residual import CompressedResidualScorer
from frequency_prior import frequency_logs
from components import fingerprint,tensors
torch.set_num_threads(1)
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
D=read(OUT/'design.json');T=read(OUT/'training_finished.json');assert len(T['fits'])==2 and not (OUT/'qualification.json').exists()
for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
models={};checks=[];rows=[]
for seed in [0,1]:
    path=OUT/'models'/f'gnn_seed{seed}';source=ROOT/f'outputs/grid21/models/gnn_seed{seed}/initial.pt'
    assert sha(source)==sha(path/'initial.pt')
    state=torch.load(path/'final.pt',map_location='cpu',weights_only=True);initial=torch.load(source,map_location='cpu',weights_only=True)
    assert all(torch.equal(state[k],initial[k]) for k in state if any(x in k for x in ['core.actor.','core.critic.','core.trunk.']))
    first=ROOT/'outputs/grid21/inputs/train'/(read(ROOT/'outputs/grid21/training_finished.json')['training_input_ids'][0]+'.npz')
    with np.load(first) as z:model=CompressedResidualScorer('gnn',len(z['x']),372,len(z['base']),z['plans'].shape[1],state['inner.subids'].tolist())
    model.load_state_dict(state);model.eval();models[seed]=model
    losses=[];prior_receipts=[]
    for identifier in read(ROOT/'outputs/grid21/training_finished.json')['training_input_ids']:
        p=ROOT/'outputs/grid21/inputs/train'/f'{identifier}.npz';meta=read(p.with_suffix('.json'))
        with np.load(p) as z:
            features={k:z[k].copy() for k in ['x','edges','edge_attr','glob','base','plans']};q=z['target'].copy();mask=torch.as_tensor(z['mask'].copy());prior=torch.as_tensor(frequency_logs(z['representatives'],len(mask)))
        assert fingerprint(features)==meta['input_hash'];prior_receipts.append({'id':identifier,'prior_sha256':hashlib.sha256(prior.numpy().tobytes()).hexdigest()})
        if not q.any():continue
        with torch.no_grad():raw=model(**tensors(features))[0];logs=torch.nn.functional.log_softmax((prior+raw).masked_fill(~mask,-1e9),dim=0);pl=torch.nn.functional.log_softmax(prior.masked_fill(~mask,-1e9),dim=0)
        assert torch.isfinite(raw).all();loss=-(torch.as_tensor(q)*logs).sum()+.1*(logs.exp()*(logs-pl)).sum();losses.append(float(loss))
    assert len(losses)==148;checks.append({'seed':seed,'fixed_final_objective':sum(losses)/148,'initial_sha256':sha(path/'initial.pt'),'final_sha256':sha(path/'final.pt'),'prior_receipts':prior_receipts})
spec=importlib.util.spec_from_file_location('grid58_cached_policy',ROOT/'work/grid28/policy.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
for week in D['development_weeks']:
    folder=ROOT/'outputs/grid26/runs'/f'r0__{week}__FULL';features_folder=folder.parent/f'r0__{week}__GNN0_128_GUARD'
    for path in sorted(folder.glob('call[0-9][0-9][0-9].json')):
        full=read(path);meta=read(features_folder/path.name);truth={r['pool_id']:r for r in full['outcomes']};reps=meta['representatives'];aliases={int(k):int(v) for k,v in meta['aliases'].items()}
        with np.load(features_folder/(path.stem+'_inputs.npz')) as z:features={k:z[k].copy() for k in z.files if k!='prior'}
        assert fingerprint(features)==meta['input_hash'];prior=frequency_logs(reps,909)
        subs=models[0].inner.subids.tolist()
        static=P.local_complete(sorted(reps),lambda i,phase:{**truth[i],'phase':phase},'PATIENCE64',subs,aliases,.9)
        for seed,model in models.items():
            with torch.no_grad():scores=model(**tensors(features))[0].numpy()+prior
            order=sorted(reps,key=lambda i:(-float(scores[i]),i));result=P.local_complete(order,lambda i,phase:{**truth[i],'phase':phase},'PATIENCE64',subs,aliases,.9)
            seen={r['pool_id'] for r in result['outcomes']};altered=copy.deepcopy(truth)
            for i in set(truth)-seen:altered[i].update(rho=.01,reward=999.,strict_admissible=True)
            assert result==P.local_complete(order,lambda i,phase:{**altered[i],'phase':phase},'PATIENCE64',subs,aliases,.9)
            choice=result['chosen_pool_id'];winner=full['chosen_pool_id']
            rows.append({'seed':seed,'week':week,'call':path.name,'step':meta['step'],'queries':result['public_queries'],'static_queries':static['public_queries'],'choice':choice,'full_choice':winner,'same_full_choice':choice==winner,'rho_regret':None if choice is None or winner is None else truth[choice]['rho']-truth[winner]['rho']})
assert len(rows)==52
summaries=[]
for seed in [0,1]:
    g=[r for r in rows if r['seed']==seed];critical=next(r for r in g if r['week']=='2035-03-05_0' and r['step']==1733)
    summaries.append({'seed':seed,'queries':sum(r['queries'] for r in g),'static_queries':sum(r['static_queries'] for r in g),'same_full_choices':sum(r['same_full_choice'] for r in g),'known1733match':critical['same_full_choice']})
Q={'passed_engineering':True,'fits':2,'training_state_checks':334,'cached_state_checks':52,'summaries':summaries,'checkpoints':checks,
   'old_development_closedloop_eligible':all(s['known1733match'] for s in summaries),'new_simulations':0,'new_teacher_queries':0,
   'warning':'Cached replay and final-fit loss do not establish full control quality or runtime benefit. Strong static prior source is public author library;author teacher dates unknown.'}
(OUT/'qualification.json').write_text(json.dumps(Q,indent=2),encoding='utf-8')
with (OUT/'per_state.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'RUN_STATE.md').write_text('BOTH FITS AND CACHED QUALIFICATION COMPLETE; bounded olddevelopment eligibility='+str(Q['old_development_closedloop_eligible'])+';no rollout started.\n',encoding='utf-8')
print(json.dumps({'passed':True,'summaries':summaries,'eligible':Q['old_development_closedloop_eligible']}))
