"""Public-input re-inference; dispatch-only FULL feedback; all existing rankers retained."""
import copy,csv,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid33'
sys.path[:0]=[str(ROOT/'work/grid21'),str(ROOT/'work/grid20')]
from prior import ResidualScorer
from components import tensors,fingerprint
import numpy as np
import torch
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('grid33_semantic',ROOT/'work/grid28/policy.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
D=read(OUT/'design.json');assert not (OUT/'review.json').exists()
for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
started=time.perf_counter();results=[];hashes={};models={};historical_checks=0
def load(p):
    hashes[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();return read(p)
for week in D['weeks']:
    folder=ROOT/'outputs/grid26/runs'/f'r0__{week}__FULL'
    reference=folder.parent/f'r0__{week}__GNN0_128_GUARD'
    ppo=folder.parent/f'r0__{week}__OLDNN128_GUARD'
    gnn1=folder.parent/f'r0__{week}__GNN1_128_GUARD'
    for path in sorted(folder.glob('call[0-9][0-9][0-9].json')):
        full=load(path);own=load(reference/path.name);old=load(ppo/path.name);other=load(gnn1/path.name)
        assert full['state_hash']==own['state_hash']==old['state_hash']==other['state_hash']
        for k in ['eligible','representatives','aliases']:assert own[k]==old[k]==other[k]
        inputs=reference/(path.stem+'_inputs.npz')
        hashes[inputs.relative_to(ROOT).as_posix()]=hashlib.sha256(inputs.read_bytes()).hexdigest()
        with np.load(inputs) as z:features={k:z[k].copy() for k in z.files if k!='prior'};prior=z['prior'].copy()
        assert fingerprint(features)==own['input_hash']
        tensor=tensors(features);truth={r['pool_id']:r for r in full['outcomes']}
        reps=own['representatives'];aliases={int(k):int(v) for k,v in own['aliases'].items()}
        for ranker in D['rankers']:
            before=time.perf_counter()
            if ranker['method']=='PPO':order=old['order'];subids=public_subids
            else:
                name=ranker['method']
                if name not in models:
                    state=torch.load(ROOT/ranker['checkpoint'],map_location='cpu',weights_only=True)
                    subids=state['inner.subids'].tolist();public_subids=subids
                    # Exact original library dimensions; live adapter uses2*n_line.
                    n_nodes=len(features['x']);max_edges=(state['inner.core.flat.0.weight'].shape[1]-n_nodes*14)//6 if ranker['kind']=='mlp' else 372
                    model=ResidualScorer(ranker['kind'],n_nodes,max_edges,len(features['base']),features['plans'].shape[1],subids)
                    model.load_state_dict(state);model.eval();models[name]=(model,subids)
                model,subids=models[name]
                with torch.no_grad():residual=model(**tensor)[0].numpy()
                assert np.isfinite(residual).all() and np.max(np.abs(residual))<=2.+1e-6
                values=residual+prior;order=sorted(reps,key=lambda i:(-float(values[i]),i))
                if name in ['GNN0','GNN1']:
                    assert order==(own if name=='GNN0' else other)['order'];historical_checks+=1
            inference_s=time.perf_counter()-before
            assert set(order)==set(truth)==set(reps) and len(order)==len(reps)
            result=P.local_complete(order,lambda i,phase:{**truth[i],'phase':phase},D['policy'],subids,aliases,D['safe'])
            seen={r['pool_id'] for r in result['outcomes']};assert len(seen)==result['public_queries']
            altered=copy.deepcopy(truth)
            for i in set(truth)-seen:altered[i].update(rho=.01,reward=999.,strict_admissible=True)
            again=P.local_complete(order,lambda i,phase:{**altered[i],'phase':phase},D['policy'],subids,aliases,D['safe'])
            assert result==again
            choice=result['chosen_pool_id'];winner=full['chosen_pool_id']
            assert choice is None or truth[choice]['strict_admissible']
            results.append({'week':week,'call':path.name,'step':own['step'],'ranker':ranker['method'],
                'queries':result['public_queries'],'choice':choice,'full_choice':winner,'same_full_choice':choice==winner,
                'rho_regret':None if choice is None or winner is None else truth[choice]['rho']-truth[winner]['rho'],
                'inference_s':inference_s,'stop_reason':result['stop_reason'],'order':order})
            assert time.perf_counter()-started<D['caps']['wall_s']
assert len(results)==130 and historical_checks==52
summaries=[]
for ranker in D['rankers']:
    rows=[r for r in results if r['ranker']==ranker['method']]
    critical=[r for r in rows if r['week']=='2035-03-05_0' and r['step']==1733];assert len(critical)==1
    summaries.append({'ranker':ranker['method'],'states':len(rows),'queries':sum(r['queries'] for r in rows),
        'same_full_choice':sum(r['same_full_choice'] for r in rows),
        'max_rho_regret':max(r['rho_regret'] or 0. for r in rows),
        'known_critical_choice_match':critical[0]['same_full_choice'],
        'closed_loop_qualified':critical[0]['same_full_choice']})
report={'passed':True,'summaries':summaries,'rows':results,'historical_order_checks':historical_checks,
        'unrevealed_feedback_checks':len(results),'public_input_hashes':hashes,'wall_s':time.perf_counter()-started,
        'new_fits':0,'new_forecasts':0,'reserved_evaluations':0,
        'warning':'26exposed states; same FULL feedback replay diagnosis, not deployable oracle or closed-loop survival/cost proof. Timing includes first model loading and is not deployment latency.'}
(OUT/'review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
with (OUT/'per_state.csv').open('w',newline='',encoding='utf-8') as f:
    keys=[k for k in results[0] if k!='order'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in results)
(OUT/'RUN_STATE.md').write_text('COMPLETED:existing matched-ranker diagnostic only; closed-loop pilot not started.\n',encoding='utf-8')
print(json.dumps({'summaries':summaries,'historical_order_checks':historical_checks,'permission_checks':len(results)},indent=2))
