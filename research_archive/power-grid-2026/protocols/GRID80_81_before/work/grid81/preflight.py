"""Source parity and causal purchase checks on old paid training/calibration records only."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid81';sys.path.insert(0,str(Path(__file__).parent))
import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    records=read(ROOT/'outputs/grid77/records.json');subs=read(ROOT/'outputs/grid77/data_receipts.json')['subids'];checks=0
    for r in records:
        aliases={int(k):int(v) for k,v in r['aliases'].items()};context={k:np.asarray(v,dtype=np.int64 if k=='edges' else np.float32) for k,v in r['context'].items()};truth={x['pool_id']:x for x in r['rows']}
        for policy in P.POLICIES:
            seen=[]
            def observe(i,phase):
                assert i not in seen;seen.append(i);return {**truth[i],'phase':phase}
            got=P.run(r['order'],observe,policy,subs,aliases,limit=r['rho_limit'],context=None if policy=='FULL' else context)
            if policy in ('FULL','FRAME_GUARD','CONTEXT_GATE'):
                ref=P.C.run(r['order'],lambda i,p:{**truth[i],'phase':p},policy,subs,aliases,limit=r['rho_limit'],context=context)
                assert got==ref
            else:
                again=P.run(r['order'],lambda i,p:{**truth[i],'phase':p},policy,subs,aliases,limit=r['rho_limit'],context=context)
                assert got==again
                d=got['gate_decision']
                if d is not None:
                    assert d['skip_probability']==4/13
                    if d['outside_support']:assert d['defer']
            assert seen==[x['pool_id'] for x in got['outcomes']] and len(seen)==got['public_queries']
            checks+=1
    D=read(OUT/'design.json')
    for w in D['weeks']:
        a=[x['method'] for x in D['execution_order'] if x['repeat']==0 and x['week']==w];b=[x['method'] for x in D['execution_order'] if x['repeat']==1 and x['week']==w]
        assert a==b[::-1] and len(set(a))==5
    for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    result=dict(passed=True,paid_training_calibration_policy_checks=checks,new_fits=0,new_queries=0,calibration_skip_numerator=4,in_support_denominator=13,random_seeds=[0,1],new_data_qualification_separate=True)
    (OUT/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
