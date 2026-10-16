"""Qualify mixed-method wiring on existing paid training/calibration feedback."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid79';sys.path.insert(0,str(Path(__file__).parent));import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    import torch;torch.set_num_threads(1)
    records=read(ROOT/'outputs/grid77/records.json');subs=read(ROOT/'outputs/grid77/data_receipts.json')['subids'];count=0
    for r in records:
        aliases={int(k):int(v) for k,v in r['aliases'].items()};context={k:np.asarray(v,dtype=np.int64 if k=='edges' else np.float32) for k,v in r['context'].items()};truth={x['pool_id']:x for x in r['rows']}
        for policy in P.POLICIES:
            seen=[]
            def observe(i,phase):
                assert i not in seen;seen.append(i);return {**truth[i],'phase':phase}
            got=P.run(r['order'],observe,policy,subs,aliases,limit=r['rho_limit'],context=None if policy=='FULL' else context)
            ref=(P.C if policy in ('FULL','CONTEXT_GATE') else P.B).run(r['order'],lambda i,p:{**truth[i],'phase':p},policy,subs,aliases,limit=r['rho_limit'],context=context)
            assert got==ref and seen==[x['pool_id'] for x in got['outcomes']]
            count+=1
    D=read(OUT/'design.json')
    for w in D['weeks']:
        a=[x['method'] for x in D['execution_order'] if x['repeat']==0 and x['week']==w];b=[x['method'] for x in D['execution_order'] if x['repeat']==1 and x['week']==w]
        assert a==b[::-1] and len(set(a))==4
    for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    result=dict(passed=True,unchanged_source_policy_checks=count,paired_dates=7,methods=4,repeats=2,new_fits=0,new_queries=0)
    (OUT/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
