"""One loss-weight intervention on the original18-feature GRID140 task."""
from pathlib import Path
import collections,datetime,gzip,hashlib,importlib.util,json,os,pickle,shutil,sys,time
sys.path.insert(0,'D:/Codex/GRL研究/work/grid140_global_compression')
import common as B
import numpy as np
R=B.R;W=Path(__file__).parent;O=R/'outputs/grid144_class_mass';PREV=B.O;OTHER=R/'outputs/grid142_batch_representation';DIAG=R/'outputs/grid143_continue_risk'
REPO=B.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID144_class_mass'
read=B.read;sha=B.sha;refs=B.refs;replay=B.replay;summarize=B.summarize;eligible_data=B.eligible_data;transform=B.transform
export_model=B.export_model;exported_scores=B.exported_scores;probability_metrics=B.probability_metrics
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def verify():
    B.verify_trained()
    for filename in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/filename).items():assert sha(R/p)==h,p
def verify_trained():
    verify()
    for n,h in read(O/'trained_freeze.json').items():assert sha(O/n)==h,n
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def rowfile(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def planned_weights(fit):
    old=read(PREV/'fit_sample_weights.json');assert [r['meta']['sample_id'] for r in fit]==[r['sample_id'] for r in old]
    base=np.asarray([r['weight'] for r in old],float);y=np.asarray([r['target']['stop_label'] for r in fit],bool);total=float(base.sum());stop=float(base[y].sum());cont=float(base[~y].sum());fstop=total/(2*stop);fcont=total/(2*cont);weights=base*np.where(y,fstop,fcont)
    assert abs(weights[y].sum()-total/2)<1e-10 and abs(weights[~y].sum()-total/2)<1e-10
    return base,weights,dict(base_stop_mass=stop,base_continue_mass=cont,base_continue_mass_fraction=cont/total,stop_multiplier=fstop,continue_multiplier=fcont,new_stop_mass=float(weights[y].sum()),new_continue_mass=float(weights[~y].sum()),fit_only=True,overall_date_call_totals_change=True)
def public_checks(rows):
    for r in rows:
        c=B.call_at(r['meta']['source']);n=r['meta']['visible_count'];assert B.P.input_at(dict(order=c['order'],rho_limit=c['rho_limit'],outcomes=c['outcomes'][:n]),n)['features']==r['features']
        poisoned=dict(c,outcomes=c['outcomes'][:n]+[dict(t,pool_id=-999,reward=-999.,rho=999.,strict_admissible=False) for t in c['outcomes'][n:]])
        assert B.P.input_at(poisoned,n)['features']==r['features']
    return len(rows)
def diagnostic_api():
    spec=importlib.util.spec_from_file_location('grid144_frozen_date_diagnostic',R/'work/grid143_continue_risk/run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
