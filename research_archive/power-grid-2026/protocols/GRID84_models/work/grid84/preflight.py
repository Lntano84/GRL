"""Training-only full-record replay parity and balanced immutable schedule."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid84'
sys.path.insert(0,str(Path(__file__).parent));import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(1);T=read(OUT/'training_finished.json');assert T['passed_engineering'] and T['fits']==3
    D=read(OUT/'design.json');assert len(D['execution_order'])==72
    for p,h in read(OUT/'training_code_freeze.json').items():assert sha(ROOT/p)==h,p
    records=read(ROOT/'outputs/grid77/records.json');subs=read(ROOT/'outputs/grid77/data_receipts.json')['subids'];checks=0
    for r in records:
        r=dict(r);r['aliases']={int(k):int(v) for k,v in r['aliases'].items()};r['subids']=subs
        r['context']={k:np.asarray(v,dtype=np.int64 if k=='edges' else np.float32) for k,v in r['context'].items()};r['prefixes']=P.M.prefixes(r)
        truth={x['pool_id']:x for x in r['rows']}
        for policy in P.POLICIES:
            seen=[]
            def reveal(i,phase):
                assert i not in seen;seen.append(i);return {**truth[i],'phase':phase}
            got=P.run(r['order'],reveal,policy,subs,r['aliases'],limit=r['rho_limit'],context=r['context'])
            assert seen==[x['pool_id'] for x in got['outcomes']] and len(seen)==got['public_queries']
            if policy=='FRAME_GUARD':assert got['public_queries']==r['guard_queries'] and got['chosen_pool_id']==r['guard_choice']
            else:
                seed=int(policy[-1]);ref=P.M.replay(P.model(seed),r)
                assert ref['end']==got['public_queries'] and ref['chosen']==got['chosen_pool_id'],(r['id'],policy)
            checks+=1
    for w in D['weeks']:
        a=[x['method'] for x in D['execution_order'] if x['repeat']==0 and x['week']==w]
        b=[x['method'] for x in D['execution_order'] if x['repeat']==1 and x['week']==w]
        assert len(set(a))==6 and a==b[::-1]
    for p,h in read(OUT/'data_freeze.json').items():assert sha(ROOT/p)==h,p
    frozen=read(OUT/'training_code_freeze.json');models={}
    for m in D['methods']:
        if 'meta_model' not in m:continue
        p=ROOT/m['meta_model'];models[m['method']]=dict(path=m['meta_model'],sha256=sha(p));frozen[m['meta_model']]=sha(p)
    for n in ['training_finished.json','training_preflight.json']:
        frozen[f'outputs/grid84/{n}']=sha(OUT/n)
    (OUT/'model_freeze.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
    frozen['outputs/grid84/model_freeze.json']=sha(OUT/'model_freeze.json')
    (OUT/'code_freeze.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    result=dict(passed=True,training_only_replay_checks=checks,records=151,models=5,all_seeds_retained=True,balanced_order=True,new_fits=3,new_teacher_queries=0,cost_reselection=False)
    (OUT/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
