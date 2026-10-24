"""GRID142 adds exactly six public batch summaries to the fixed GRID140 task."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,os,pickle,shutil,sys,time
sys.path.insert(0,'D:/Codex/GRL研究/work/grid140_global_compression')
import common as B
import numpy as np
R=B.R;W=Path(__file__).parent;O=R/'outputs/grid142_batch_representation';PREV=B.O
SOURCE=B.SOURCE;P=B.P;REPO=B.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID142_batch_representation'
read=B.read;sha=B.sha;refs=B.refs;replay=B.replay;summarize=B.summarize;transform=B.transform
export_model=B.export_model;exported_scores=B.exported_scores;probability_metrics=B.probability_metrics
ADDED=['tail64_best_reward','tail64_admissible_fraction','tail64_rho_median','tail64_rho_iqr','tail64_vs_prior64_best_reward','tail64_vs_prior64_median_rho']
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def verify():
    B.verify_trained()
    for name in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/name).items():assert sha(R/p)==h,p
def verify_trained():
    verify()
    for n,h in read(O/'trained_freeze.json').items():assert sha(O/n)==h,n
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def features(c,n):
    f=P.input_at(c,n)['features'];tail=c['outcomes'][:n][-64:];prior=c['outcomes'][:n][-128:-64]
    b=P.best(tail);a=P.best(prior);rho=[float(v['rho']) for v in tail if v['rho']>0 and np.isfinite(v['rho'])];pr=[float(v['rho']) for v in prior if v['rho']>0 and np.isfinite(v['rho'])]
    f.update(tail64_best_reward=P.key(b)[0] if b is not None else None,tail64_admissible_fraction=sum(v['strict_admissible'] for v in tail)/len(tail),
        tail64_rho_median=float(np.median(rho)) if rho else None,tail64_rho_iqr=float(np.quantile(rho,.75)-np.quantile(rho,.25)) if rho else None,
        tail64_vs_prior64_best_reward=P.key(b)[0]-P.key(a)[0] if a is not None and b is not None else None,
        tail64_vs_prior64_median_rho=float(np.median(rho)-np.median(pr)) if rho and pr else None)
    return f
def eligible_data(part):
    original=B.eligible_data(part);out=[]
    for r in original:
        c=B.call_at(r['meta']['source']);f=features(c,r['meta']['visible_count']);assert {k:f[k] for k in r['features']}==r['features']
        out.append(dict(meta=r['meta'],features=f,target=r['target']))
    return out
def public_checks(rows):
    checks=0
    for r in rows:
        c=B.call_at(r['meta']['source']);n=r['meta']['visible_count'];public=dict(order=c['order'],rho_limit=c['rho_limit'],outcomes=c['outcomes'][:n]);assert features(public,n)==r['features']
        poisoned=dict(c,outcomes=c['outcomes'][:n]+[dict(t,pool_id=-1,reward=999.,rho=-999.,strict_admissible=True) for t in c['outcomes'][n:]],chosen_pool_id=-1)
        assert features(poisoned,n)==r['features'];checks+=1
    return checks
