"""After all fits, unchanged semantic search on existing exposed FULL feedback."""
import copy,csv,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid48'
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'work/grid21'),str(ROOT/'work/grid20')]
from free_residual import FreeResidualScorer
from components import tensors,fingerprint
import numpy as np
import torch
torch.set_num_threads(1)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
assert len(read(OUT/'training_finished.json')['fits'])==2
assert not (OUT/'review.json').exists()
D=read(OUT/'design.json');spec=importlib.util.spec_from_file_location('grid38_semantic',ROOT/'work/grid28/policy.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
oldrows={(r['week'],r['call'],r['ranker']):r for r in read(ROOT/'outputs/grid33/review.json')['rows']}
models={};rows=[];hashes={};begun=time.perf_counter()
for week in D['development_weeks']:
    folder=ROOT/'outputs/grid26/runs'/f'r0__{week}__FULL';features_folder=folder.parent/f'r0__{week}__GNN0_128_GUARD'
    for path in sorted(folder.glob('call[0-9][0-9][0-9].json')):
        full=read(path);meta=read(features_folder/path.name);truth={r['pool_id']:r for r in full['outcomes']};reps=meta['representatives'];aliases={int(k):int(v) for k,v in meta['aliases'].items()}
        inputs=features_folder/(path.stem+'_inputs.npz')
        for p in [path,features_folder/path.name,inputs]:hashes[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(inputs) as z:features={k:z[k].copy() for k in z.files if k!='prior'};prior=z['prior'].copy()
        assert fingerprint(features)==meta['input_hash']
        for kind in D['kinds']:
            for seed in D['seeds']:
                name=f'{kind.upper()}{seed}';checkpoint=OUT/'models'/f'{kind}_seed{seed}'/'final.pt'
                if name not in models:
                    state=torch.load(checkpoint,map_location='cpu',weights_only=True);subs=state['inner.subids'].tolist();n_nodes=len(features['x'])
                    max_edges=(state['inner.core.flat.0.weight'].shape[1]-n_nodes*14)//6 if kind=='mlp' else 372
                    model=FreeResidualScorer(kind,n_nodes,max_edges,len(features['base']),features['plans'].shape[1],subs);model.load_state_dict(state);model.eval();models[name]=(model,subs)
                model,subs=models[name];t=time.perf_counter()
                with torch.no_grad():residual=model(**tensors(features))[0].numpy()
                assert np.isfinite(residual).all()
                values=residual+prior;order=sorted(reps,key=lambda i:(-float(values[i]),i));inference_s=time.perf_counter()-t
                result=P.local_complete(order,lambda i,phase:{**truth[i],'phase':phase},'PATIENCE64',subs,aliases,.9)
                seen={r['pool_id'] for r in result['outcomes']};altered=copy.deepcopy(truth)
                for i in set(truth)-seen:altered[i].update(rho=.01,reward=999.,strict_admissible=True)
                assert result==P.local_complete(order,lambda i,phase:{**altered[i],'phase':phase},'PATIENCE64',subs,aliases,.9)
                old=oldrows[(week,path.name,name)];choice=result['chosen_pool_id'];winner=full['chosen_pool_id']
                rows.append({'week':week,'call':path.name,'step':meta['step'],'ranker':name,'queries':result['public_queries'],'old_queries':old['queries'],
                    'choice':choice,'full_choice':winner,'same_full_choice':choice==winner,'old_same_full_choice':old['same_full_choice'],
                    'rho_regret':None if choice is None or winner is None else truth[choice]['rho']-truth[winner]['rho'],
                    'old_rho_regret':old['rho_regret'],'inference_s':inference_s,'order':order,'stop_reason':result['stop_reason']})
summaries=[]
for kind in D['kinds']:
    for seed in D['seeds']:
        name=f'{kind.upper()}{seed}';g=[r for r in rows if r['ranker']==name];assert len(g)==26
        critical=next(r for r in g if r['week']=='2035-03-05_0' and r['step']==1733)
        summaries.append({'ranker':name,'queries':sum(r['queries'] for r in g),'old_queries':sum(r['old_queries'] for r in g),
            'query_change':1-sum(r['queries'] for r in g)/sum(r['old_queries'] for r in g),'same_full_choice':sum(r['same_full_choice'] for r in g),
            'old_same_full_choice':sum(r['old_same_full_choice'] for r in g),'max_rho_regret':max(r['rho_regret'] or 0. for r in g),
            'old_max_rho_regret':max(r['old_rho_regret'] or 0. for r in g),'known_critical_choice_match':critical['same_full_choice']})
R={'passed_saved_screen':True,'rows':rows,'summaries':summaries,'input_hashes':hashes,'new_fits':2,'new_forecasts':0,'reserved_evaluations':0,'wall_s':time.perf_counter()-begun,
   'warning':'26 exposed cached FULL states, not final generalization or closed-loop quality/speed. Both fixed new models and original models preserved. Timing here is only isolated forward/ranking, not controller speed. Exact-original-choice comparison uses same historical feedback and stop.'}
(OUT/'review.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
with (OUT/'per_state.csv').open('w',encoding='utf-8',newline='') as f:keys=[k for k in rows[0] if k!='order'];w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
print(json.dumps(summaries,indent=2))
