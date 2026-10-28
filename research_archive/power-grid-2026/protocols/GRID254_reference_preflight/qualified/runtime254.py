"""Shared fixed reference-NN runtime. No optimizer is created on import."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUDA_VISIBLE_DEVICES']='-1'
import json,hashlib,sys,time,pickle,importlib.util,ctypes
sys.dont_write_bytecode=True
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid254_reference_preflight';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID254_reference_preflight';TRAIN_OUT=R/'outputs/grid255_reference_stop_nn'
_ORIG=None

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.tmp_{os.getpid()}')
 with tmp.open('w',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def claim(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def before(deadline):
 if time.monotonic()>=deadline:raise TimeoutError('Fixed global deadline reached; preserve partial output, no extension.')
def orig():
 global _ORIG
 if _ORIG is None:
  sp=importlib.util.spec_from_file_location('grid254_original168',R/'work/grid168_one_sided_transfer/run.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.G.configure();_ORIG=m
 return _ORIG

def active_manifest():
 p=O/'source_freeze_recovery01.json';return p if p.exists() else O/'source_freeze.json'
def verify():
 for p,h in read(active_manifest()).items():assert sha(R/p)==h,p

def memory():
 if os.name!='nt':raise RuntimeError('Resource qualification targets the declared Windows host.')
 from ctypes import wintypes
 class MS(ctypes.Structure):_fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(n,ctypes.c_ulonglong) for n in ['totalPhys','availPhys','totalPage','availPage','totalVirtual','availVirtual','availExtended']]
 s=MS();s.length=ctypes.sizeof(s);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s));return dict(total_physical=int(s.totalPhys),available_physical=int(s.availPhys),logical_cpus=os.cpu_count())
def process_memory():
 from ctypes import wintypes
 class PM(ctypes.Structure):_fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['peakWS','workingSet','peakPaged','paged','peakNonpaged','nonpaged','pagefile','peakPagefile']]
 p=PM();p.cb=ctypes.sizeof(p);ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE;handle=ctypes.windll.kernel32.GetCurrentProcess();fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(PM),wintypes.DWORD];fn.restype=wintypes.BOOL;assert fn(handle,ctypes.byref(p),p.cb);return dict(working_set=int(p.workingSet),peak_working_set=int(p.peakWS),private_commit=int(p.pagefile))
def job(name):return next(j for j in read(O/'jobs.json') if j['node']==name)
def load_public(j):
 z=np.load(O/j['package']);ids=z['fit_ids'];normalized=z['normalized_fit'];rows=pickle.loads((R/'outputs/grid113_relational_learning/train_public.pkl').read_bytes());fitrows=[dict(rows[int(i)],global41=normalized[k]) for k,i in enumerate(ids)];assert len(fitrows)==len(ids);return z,fitrows

def validate_training(out,j):
 out=Path(out).resolve();assert out==TRAIN_OUT.resolve(),'Training output must match the preregistered stage255 directory.'
 q=read(O/'qualification.json');assert q['passed'] and q['optimizer_instances']==0 and q['execution_qualified']
 auth=read(out/'authorization.json');assert auth['passed'] and auth['qualification_sha256']==sha(O/'qualification.json') and auth['qualified_freeze_sha256']==sha(O/'qualified_freeze.json') and auth['jobs_sha256']==sha(O/'jobs.json') and auth['all_sources_inputs_published'] and auth['models']==18 and auth['fit_cap_s']==600
 
 for rel,h in read(O/'qualified_freeze.json').items():assert sha(O/rel)==h,rel
 ctx=read(out/'run_context.json');assert ctx['max_workers']==6 and ctx['deadline']-ctx['started_monotonic']==600 and j['node'] in ctx['nodes'];return ctx
