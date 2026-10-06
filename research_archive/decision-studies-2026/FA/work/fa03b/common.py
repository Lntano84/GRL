from pathlib import Path
import hashlib, json, os, sys, pickle
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/fa00'))
from core import Selector, ObservationCache, Ledger, ReplayEnvironment, Exhausted, WallLimit, keyed_seed, PARAMS
sys.path.insert(0,str(Path(__file__).resolve().parent))
OUT=ROOT/'outputs/fa03b'
ARMS=['LEX-PC-U-PD','RANDOM-ROW']
SEEDS=[7,42,99]
BUDGET=86460.0
CAP=100.0

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def dump(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f:
        pickle.dump(obj,f,protocol=5);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def load_data():
    p=json.loads((ROOT/'outputs/fa00/FA00_protocol.json').read_text())
    d=dict(np.load(ROOT/'outputs/fa00/FA00_data.npz'))
    return p,d
def verify_design():
    design=json.loads((ROOT/'outputs/fa03b_design/FA03B_design_freeze.json').read_text())
    for path,value in design['files'].items():
        assert sha(ROOT/path)==value, path
    assert design['query_pair']==566 and design['query_row']==11
    assert design['budget_s']==BUDGET
    return design
def labels_json(s):
    return [[[int(i),*list(v)] for i,v in sorted(z.items())] for z in s.label_rows]
def signature(s):
    h=hashlib.sha256(s.cache.kind.tobytes()+s.cache.value.tobytes())
    h.update(json.dumps(labels_json(s),separators=(',',':')).encode())
    h.update(s.X[s.train_rows].tobytes())
    h.update(json.dumps(PARAMS,sort_keys=True).encode());h.update(str(s.seed).encode())
    return h.hexdigest()
