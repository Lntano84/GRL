"""Paid full-state parity, hidden-tail invariance, edge cases and module checks."""
from pathlib import Path
import gzip,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid86';sys.path.insert(0,str(Path(__file__).parent));import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    D=read(OUT/'design.json');assert len(D['execution_order'])==36
    for n in ['code_freeze.json','data_freeze.json']:
        for p,h in read(OUT/n).items():assert sha(ROOT/p)==h,p
    with gzip.open(ROOT/'outputs/grid85_mechanism/mechanism_feedback.json.gz','rt',encoding='utf-8') as f:pack=json.load(f)
    s=importlib.util.spec_from_file_location('grid86_independent',ROOT/'work/grid85_mechanism/verify.py');V=importlib.util.module_from_spec(s);s.loader.exec_module(V)
    cases=checks=tails=0
    for source,q in pack['records'].items():
        if not source.endswith('.json') or not source.split('/')[-2].endswith('__FULL'):continue
        c=dict(q,outcomes=[dict(zip(pack['fields'],v)) for v in q['outcomes']])
        for x in c['outcomes']:x['exceptions']=x.pop('has_exception')
        truth={r['pool_id']:r for r in c['outcomes']};aliases={int(k):int(v) for k,v in c['aliases'].items()};cases+=1
        for name in P.POLICIES:
            seen=[]
            def reveal(i,phase):
                assert i not in seen;seen.append(i);return dict(truth[i],phase=phase)
            got=P.run(c['order'],reveal,name,pack['subids'],aliases,limit=c['rho_limit'])
            ref=V.independent(c,name,pack['subids']);expected=V.pid(V.best(ref))
            assert got['public_queries']==len(ref) and got['chosen_pool_id']==expected
            assert seen==[r['pool_id'] for r in ref]
            if name!='FRAME_GUARD':assert len(seen)==min(int(name.split('_')[1]),len(c['order'])) and got['local_queries']==0
            oldseen=list(seen);seen=[];unknown=set(truth)-set(oldseen)
            def perturbed(i,phase):
                assert i not in seen;seen.append(i);r=dict(truth[i],phase=phase)
                if i in unknown:r.update(reward=9999.,rho=.01,done=False,illegal=False,ambiguous=False,exceptions=False,strict_admissible=True)
                return r
            again=P.run(c['order'],perturbed,name,pack['subids'],aliases,limit=c['rho_limit'])
            assert seen==oldseen and got==again;tails+=1;checks+=1
    assert cases==137 and checks==411
    hand=0
    for size in (0,100,600):
        order=list(range(size))
        for cap in (256,512):
            calls=[]
            def observe(i,phase):
                calls.append(i);return dict(pool_id=i,rho=.8,reward=1.+(1e-9 if i==130 else 0),strict_admissible=True,phase=phase)
            r=P.run(order,observe,f'FIXED_{cap}',[0]*size,{},limit=1.)
            assert len(calls)==min(size,cap) and r['chosen_pool_id']==(0 if size else None);hand+=1
    for cap in (256,512):
        order=list(range(600));calls=[]
        def observe(i,phase):
            calls.append(i);return dict(pool_id=i,rho=.8,reward=1.,strict_admissible=i==400,phase=phase)
        r=P.run(order,observe,f'FIXED_{cap}',[0]*600,{},limit=1.)
        assert r['chosen_pool_id']==(400 if cap==512 else None) and len(calls)==cap;hand+=1
        r=P.run(order,lambda i,p:dict(pool_id=i,rho=2.,reward=0.,strict_admissible=False,phase=p),f'FIXED_{cap}',[0]*600,{},limit=1.)
        assert r['chosen_pool_id'] is None and r['public_queries']==cap;hand+=1
    assert sha(ROOT/'work/grid86/environment.py')==sha(ROOT/'work/grid84/environment.py')
    import environment
    assert Path(environment.__file__).resolve()==(ROOT/'work/grid86/environment.py').resolve()
    for w in D['weeks']:
        a=[x['method'] for x in D['execution_order'] if x['repeat']==0 and x['week']==w];b=[x['method'] for x in D['execution_order'] if x['repeat']==1 and x['week']==w];assert a==b[::-1] and len(set(a))==3
    result=dict(passed=True,paid_full_states=cases,independent_policy_reconstructions=checks,hidden_suffix_checks=tails,edge_cases=hand,
                environment_import_verified=True,balanced_schedule=True,new_fits=0,new_simulation=0)
    (OUT/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
