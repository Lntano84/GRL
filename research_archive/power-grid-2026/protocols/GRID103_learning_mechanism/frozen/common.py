"""GRID103: fixed mechanism probe on GRID102 saved feedback."""
from pathlib import Path
import hashlib,json,os,pickle,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
W=Path(__file__).parent
OLD=ROOT/'outputs/grid102_search_rl'
OUT=ROOT/'outputs/grid103_learning_mechanism'
REPO=ROOT/'work/GRL-results-integration'
ARCHIVE=REPO/'research_archive/power-grid-2026/protocols/GRID103_learning_mechanism'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
sys.path.insert(0,str(ROOT/'work/grid102_search_rl'))
from environment import LocalSearchEnv

def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');temp.replace(p)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def test_sealed():
    d=read(ROOT/'outputs/grid100_data_foundation/design.json')
    assert not any((ROOT/d['environment']/'chronics'/r['scenario']).exists() for r in d['scenario_plan'] if r['partition']=='test')
def verify():
    for name in ['code_freeze.json','data_freeze.json']:
        for p,h in read(OLD/name).items():assert sha(ROOT/p)==h,p
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p
    test_sealed()
def load():
    with (OLD/'cases.pkl').open('rb') as f:cases=pickle.load(f)
    with np.load(OLD/'normalization.npz') as z:mean=z['mean'];scale=z['scale']
    return cases,mean,scale,read(OLD/'design.json')
def normalized(state,mean,scale):return np.clip((state-mean)/scale,-20,20).astype(np.float32)

def summarize(rows):
    """Original date macro plus equal-physical-call-within-date sensitivity."""
    per_date={}
    for date in sorted({r['date'] for r in rows}):
        rs=[r for r in rows if r['date']==date]
        gain=sum(r['full_gain'] for r in rs);queries=sum(r['full_queries'] for r in rs)
        groups=sorted({r['physical_state_group'] for r in rs})
        per_date[date]=dict(cases=len(rs),physical_groups=len(groups),
            mean_net_utility=float(np.mean([r['net_utility'] for r in rs])),
            mean_gain=float(np.mean([r['gain'] for r in rs])),mean_queries=float(np.mean([r['queries'] for r in rs])),
            query_fraction=sum(r['queries'] for r in rs)/queries,
            gain_capture=sum(r['gain'] for r in rs)/gain if gain>0 else None,
            stop_immediately=float(np.mean([r['batches']==0 for r in rs])),
            negative_utility_fraction=float(np.mean([r['net_utility'] < -1e-12 for r in rs])),
            group_balanced_utility=float(np.mean([np.mean([r['net_utility'] for r in rs if r['physical_state_group']==g]) for g in groups])))
    macro={k:float(np.mean([v[k] for v in per_date.values() if v[k] is not None]))
           if any(v[k] is not None for v in per_date.values()) else None
           for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','stop_immediately',
                     'negative_utility_fraction','group_balanced_utility']}
    return dict(cases=len(rows),dates=len(per_date),macro=macro,per_date=per_date)
