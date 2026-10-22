from pathlib import Path
import hashlib,json,os
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent
OUT=ROOT/'outputs/grid102_search_rl';SOURCE=ROOT/'outputs/grid100_data_foundation'
ARCHIVE=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID102_search_rl'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');temp.replace(p)
def verify():
    for name in ['code_freeze.json','data_freeze.json']:
        for p,h in read(OUT/name).items():assert sha(ROOT/p)==h,p
def test_sealed():
    d=read(SOURCE/'design.json')
    assert not any((ROOT/d['environment']/'chronics'/r['scenario']).exists() for r in d['scenario_plan'] if r['partition']=='test')
