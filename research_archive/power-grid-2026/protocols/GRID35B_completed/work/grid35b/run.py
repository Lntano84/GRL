"""Freeze then execute one additional pre-existing, nonlearned prior component."""
import copy,hashlib,importlib.util,json,os,struct,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from physics_order import order_from_public_edges
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid35b'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
assert read(ROOT/'outputs/grid35/audit.json')['passed']
OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'design.json').exists()
D={'stage':'GRID35B_GEOMETRIC_PRIOR_COMPONENT','rule':'PHYSICS_TAIL',
   'reason':'Source inspection identifies existing public_prior tail as a stronger nonlearned control: distance from endpoints of top3 currently loaded live lines,then local maximum rho,then candidateID. Remove only original PPO prefix; preserve alias canonicalization.',
   'inputs':'Current public graph edges/line rho and known candidate-to-substation mapping; no learned weights or action forecast values in order.',
   'feedback':'Same26exposed FULL feedback snapshots dispatched only after choosing next query; unchanged PATIENCE64/semanticcompletion.',
   'caps':{'new_fits':0,'new_forecasts':0,'reserved_evaluations':0},
   'gate':'Known1733critical FULL choice required for unchanged local-policy eligibility; no closed-loop assertion from passing.'}
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[Path(__file__).resolve(),ROOT/'work/grid35b/physics_order.py',ROOT/'work/grid21/prior.py',ROOT/'work/grid28/policy.py']
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
spec=importlib.util.spec_from_file_location('grid35b_stop',ROOT/'work/grid28/policy.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];rows=[];hashes={};start=time.perf_counter()
for week in read(ROOT/'outputs/grid33/design.json')['weeks']:
    folder=ROOT/'outputs/grid26/runs'/f'r0__{week}__FULL';ownfolder=folder.parent/f'r0__{week}__GNN0_128_GUARD'
    for path in sorted(folder.glob('call[0-9][0-9][0-9].json')):
        own=read(ownfolder/path.name);aliases={int(i):int(r) for i,r in own['aliases'].items()};inputs=ownfolder/(path.stem+'_inputs.npz')
        for p in [path,ownfolder/path.name,inputs]:hashes[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(inputs) as z:edges=z['edges'].copy();attrs=z['edge_attr'].copy();n=len(z['x'])//2
        count=len(attrs)//2;rho=attrs[:count,0];order=order_from_public_edges(edges,rho,n,own['eligible'],subs,aliases)
        full=read(path);truth={x['pool_id']:x for x in full['outcomes']};assert set(order)==set(truth)
        result=P.local_complete(order,lambda i,phase:{**truth[i],'phase':phase},'PATIENCE64',subs,aliases,.9)
        seen={x['pool_id'] for x in result['outcomes']};changed=copy.deepcopy(truth)
        for i in set(truth)-seen:changed[i].update(rho=.001,reward=999.,strict_admissible=True)
        assert result==P.local_complete(order,lambda i,phase:{**changed[i],'phase':phase},'PATIENCE64',subs,aliases,.9)
        winner=full['chosen_pool_id'];choice=result['chosen_pool_id']
        rows.append({'week':week,'call':path.name,'step':own['step'],'order':order,'queries':result['public_queries'],
            'choice':choice,'full_choice':winner,'same_full_choice':choice==winner,
            'rho_regret':None if winner is None or choice is None else truth[choice]['rho']-truth[winner]['rho']})
assert len(rows)==26
critical=[r for r in rows if r['week']=='2035-03-05_0' and r['step']==1733];assert len(critical)==1
summary={'rule':'PHYSICS_TAIL','states':26,'queries':sum(r['queries'] for r in rows),'same_full_choice':sum(r['same_full_choice'] for r in rows),
    'max_rho_regret':max(r['rho_regret'] or 0 for r in rows),'critical_choice_match':critical[0]['same_full_choice']}
report={'passed_engineering':True,'rows':rows,'summary':summary,'input_hashes':hashes,'wall_s':time.perf_counter()-start,
    'new_fits':0,'new_forecasts':0,'reserved_evaluations':0,'warning':'This is an existing heuristic component,not a new algorithm. Public-edge rank diagnostic only,not livequality or speed.'}
(OUT/'review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
