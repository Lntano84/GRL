"""Finite two-decision diagnosis; no learning entrypoint."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent
OUT=ROOT/'outputs/grid106_decision_branches';REPO=ROOT/'work/GRL-results-integration'
ARCHIVE=REPO/'research_archive/power-grid-2026/protocols/GRID106_decision_branches'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def scalar(v):
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False,default=scalar);f.write('\n');f.flush();os.fsync(f.fileno())
    t.replace(p)
def loadmod(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sealed_test():
    d=read(ROOT/'outputs/grid100_data_foundation/design.json')
    for r in d['scenario_plan']:
        if r['partition']=='test':
            assert not (ROOT/d['environment']/'chronics'/r['scenario']).exists()
            assert not (ROOT/'outputs/grid100_data_foundation/datasets'/r['scenario']).exists()
def verify():
    for n in ['code_freeze.json','data_freeze.json']:
        for p,h in read(OUT/n).items():assert sha(ROOT/p)==h,p
    sealed_test()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
