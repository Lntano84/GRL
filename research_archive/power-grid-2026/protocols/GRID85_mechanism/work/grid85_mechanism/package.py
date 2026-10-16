"""Small, self-contained one-step feedback package, without environments."""
from pathlib import Path
import csv,gzip,hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid85_mechanism'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    records={}
    for name in ['actual_calls.csv','offline_full_calls.csv']:
        for r in csv.DictReader((OUT/name).open(encoding='utf-8-sig')):
            path=r['source']
            if path in records:continue
            c=read(ROOT/path)
            records[path]=dict(order=c['order'],aliases=c['aliases'],rho_limit=c['rho_limit'],
              outcomes=[[x['pool_id'],x['rho'],x['reward'],x['done'],x['illegal'],x['ambiguous'],bool(x['exceptions']),x['strict_admissible'],x['phase']] for x in c['outcomes']])
    pack=dict(scope='Paid one-step feedback, retrospective analysis only. Closed-loop costs/first divergence still refer to original saved trajectories.',
              fields=['pool_id','rho','reward','done','illegal','ambiguous','has_exception','strict_admissible','phase'],
              subids=read(ROOT/'outputs/grid28/offline.json')['public_subids'],records=records)
    target=OUT/'mechanism_feedback.json.gz'
    with target.open('wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(json.dumps(pack,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8'))
    with gzip.open(target,'rt',encoding='utf-8') as f:again=json.load(f)
    assert again==pack
    # Recompute all one-step table choices directly from this portable package.
    spec=importlib.util.spec_from_file_location('grid85_independent',Path(__file__).with_name('verify.py'));V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
    checks=0
    for r in csv.DictReader((OUT/'offline_full_calls.csv').open(encoding='utf-8-sig')):
        q=again['records'][r['source']]
        c=dict(q,outcomes=[dict(zip(again['fields'],v)) for v in q['outcomes']])
        for x in c['outcomes']:x['exceptions']=x.pop('has_exception')
        rs=V.independent(c,r['method'],again['subids']);b=V.best(rs)
        assert len(rs)==int(r['queries']) and (str(V.pid(b)) if b else '')==r['chosen'];checks+=1
    receipt=dict(passed=True,records=len(records),compressed_bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                 package_based_method_reconstructions=checks,exception_encoding='Boolean presence preserved, detailed exception strings omitted; originals retained by source hash.',
                 new_fits=0,new_simulation=0)
    (OUT/'package_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':main()
