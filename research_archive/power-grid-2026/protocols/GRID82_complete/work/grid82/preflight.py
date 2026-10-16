"""453 old paid-feedback parity checks plus immutable input/schedule qualification."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid82';sys.path.insert(0,str(Path(__file__).parent));import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    import torch;torch.set_num_threads(1)
    records=read(ROOT/'outputs/grid77/records.json');subs=read(ROOT/'outputs/grid77/data_receipts.json')['subids'];checks=0
    for r in records:
        aliases={int(k):int(v) for k,v in r['aliases'].items()};context={k:np.asarray(v,dtype=np.int64 if k=='edges' else np.float32) for k,v in r['context'].items()};truth={x['pool_id']:x for x in r['rows']}
        for policy in P.POLICIES:
            seen=[]
            def reveal(i,p):
                assert i not in seen;seen.append(i);return {**truth[i],'phase':p}
            got=P.run(r['order'],reveal,policy,subs,aliases,limit=r['rho_limit'],context=context)
            source=P.C if policy=='FRAME_GUARD' else P.B
            ref=source.run(r['order'],lambda i,p:{**truth[i],'phase':p},policy,subs,aliases,limit=r['rho_limit'],context=context)
            assert got==ref and seen==[x['pool_id'] for x in got['outcomes']];assert len(seen)==got['public_queries'];checks+=1
    D=read(OUT/'design.json')
    assert len(D['execution_order'])==36
    for w in D['weeks']:
        a=[x['method'] for x in D['execution_order'] if x['repeat']==0 and x['week']==w];b=[x['method'] for x in D['execution_order'] if x['repeat']==1 and x['week']==w];assert a==b[::-1] and len(set(a))==3
    for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    for p,h in read(OUT/'data_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    selected=read(ROOT/'outputs/grid77/selected.json');assert selected['costindex']==2 and not selected['calibration_quality_matched']
    receipt=dict(passed=True,old_paid_feedback_source_parity=checks,queries_deduplicated=True,balanced_order=True,new_fits=0,new_queries=0,conference_calibration_quality_matched=False)
    (OUT/'preflight.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':main()
