"""Finite GRID104 development loop; never touches final-test scenarios."""
from pathlib import Path
import datetime as dt,hashlib,json,os
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent
OUT=ROOT/'outputs/grid104_learning_closedloop';REPO=ROOT/'work/GRL-results-integration'
ARCHIVE=REPO/'research_archive/power-grid-2026/protocols/GRID104_learning_closedloop'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def test_sealed():
    D=read(ROOT/'outputs/grid100_data_foundation/design.json')
    for r in D['scenario_plan']:
        if r['partition']=='test':
            for p in [ROOT/D['environment']/'chronics'/r['scenario'],ROOT/'outputs/grid100_data_foundation/datasets'/r['scenario']]:assert not p.exists(),p
def verify():
    for name in ['code_freeze.json','data_freeze.json']:
        for p,h in read(OUT/name).items():assert sha(ROOT/p)==h,p
    test_sealed()
