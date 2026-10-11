"""No neural imports: public order decisions followed by feedback-only stop replay."""
import copy,csv,hashlib,importlib.util,json,os,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid35'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
D=read(OUT/'design.json');assert not (OUT/'review.json').exists()
for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
spec=importlib.util.spec_from_file_location('grid35_stop',ROOT/'work/grid28/policy.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];rows=[];hashes={};started=time.perf_counter()
for week in D['weeks']:
    folder=ROOT/'outputs/grid26/runs'/f'r0__{week}__FULL';ownfolder=folder.parent/f'r0__{week}__GNN0_128_GUARD'
    for path in sorted(folder.glob('call[0-9][0-9][0-9].json')):
        own=read(ownfolder/path.name);reps=own['representatives'];aliases={int(i):int(r) for i,r in own['aliases'].items()}
        inputs=ownfolder/(path.stem+'_inputs.npz')
        for source in [path,ownfolder/path.name,inputs]:hashes[source.relative_to(ROOT).as_posix()]=hashlib.sha256(source.read_bytes()).hexdigest()
        with np.load(inputs) as z:x=z['x'].copy()
        n=len(x)//2;assert x.shape==(2*n,14) and max(subs)<n
        scores=np.maximum(x[:n,9],x[n:,9]);assert np.isfinite(scores).all()
        # Orders are fixed from legal inputs before loading any action feedback.
        orders={'STATIC_ID':sorted(reps),'LOCAL_STRESS':sorted(reps,key=lambda i:(-float(scores[subs[i]]),i))}
        full=read(path);assert own['state_hash']==full['state_hash'];truth={r['pool_id']:r for r in full['outcomes']}
        for rule in D['rules']:
            order=orders[rule];assert len(order)==len(set(order)) and set(order)==set(truth)
            result=P.local_complete(order,lambda i,phase:{**truth[i],'phase':phase},'PATIENCE64',subs,aliases,.9)
            chosen={r['pool_id'] for r in result['outcomes']};altered=copy.deepcopy(truth)
            for i in set(truth)-chosen:altered[i].update(rho=.001,reward=999.,strict_admissible=True)
            again=P.local_complete(order,lambda i,phase:{**altered[i],'phase':phase},'PATIENCE64',subs,aliases,.9)
            assert result==again
            winner=full['chosen_pool_id'];choice=result['chosen_pool_id']
            rows.append({'week':week,'call':path.name,'step':own['step'],'rule':rule,'order':order,
                'queries':result['public_queries'],'choice':choice,'full_choice':winner,'same_full_choice':choice==winner,
                'rho_regret':None if winner is None or choice is None else truth[choice]['rho']-truth[winner]['rho'],
                'stop_reason':result['stop_reason']})
            assert time.perf_counter()-started<D['caps']['wall_s']
assert len(rows)==52
summaries=[]
for rule in D['rules']:
    group=[r for r in rows if r['rule']==rule];critical=[r for r in group if r['week']=='2035-03-05_0' and r['step']==1733];assert len(critical)==1
    summaries.append({'rule':rule,'states':len(group),'queries':sum(r['queries'] for r in group),
        'same_full_choice':sum(r['same_full_choice'] for r in group),'max_rho_regret':max(r['rho_regret'] or 0 for r in group),
        'critical_choice_match':critical[0]['same_full_choice'],'closed_loop_eligible':critical[0]['same_full_choice']})
R={'passed_engineering':True,'summaries':summaries,'rows':rows,'input_hashes':hashes,'wall_s':time.perf_counter()-started,
    'hidden_feedback_checks':52,'existing_rankers':read(ROOT/'outputs/grid33/review.json')['summaries'],
    'new_fits':0,'new_forecasts':0,'reserved_evaluations':0,
    'warning':'26exposed historical FULL states,not independent dates and not current LOCAL-policy state distribution. Cheap orders are publicly defined before read outcomes; no closed-loop quality/speed or GNN necessity claim.'}
(OUT/'review.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
print(json.dumps({'summaries':summaries,'wall_s':R['wall_s']},indent=2))
