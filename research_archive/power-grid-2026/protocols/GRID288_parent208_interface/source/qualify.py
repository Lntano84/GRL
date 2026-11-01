from pathlib import Path
import sys,os
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[k]="1"
sys.dont_write_bytecode=True
import json,hashlib,pickle,time,subprocess,importlib.util,traceback
import numpy as np,sklearn
R=Path("D:/Codex/GRL研究");repo=R/"work/GRL-results-integration";W=Path(__file__).parent;O=R/"outputs/grid288_parent208_interface"
def rd(p):return json.loads(p.read_text(encoding="utf8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
assert not (O/"execution_started.json").exists()
for p,h in rd(O/"source_freeze.json").items():assert sha(R/p)==h,p
g=lambda *a:subprocess.check_output(["git","-C",str(repo),*a]);head=g("rev-parse","HEAD").decode().strip();assert g("ls-remote","lntano-src","refs/heads/main").decode().split()[0]==head
assert g("show",head+":research_archive/power-grid-2026/protocols/GRID288_parent208_interface/source/source_freeze.json")== (O/"source_freeze.json").read_bytes()
assert sklearn.__version__==rd(O/"preflight.json")["sklearn_version"]=="1.7.2"
save("execution_started.json",dict(source_published=head,predict_call_cap=24,new_fits=0))
sp=importlib.util.spec_from_file_location("g288_parent",W/"runtime.py");rt=importlib.util.module_from_spec(sp);sp.loader.exec_module(rt)
raw=np.load(R/"outputs/grid214_inner_head_generation/public_raw41.npz")["values"];records=[];calls=0;(O/"public_parent").mkdir()
try:
 for role in rd(R/"outputs/grid241_parent_prior_plan/qualified_pools.json"):
  pool=role["pool"];n=pool+".npz";h=np.load(R/f"outputs/grid190_extension_gate/held_{pool}.npz");sc=np.load(R/role["parent_scaler"]);ref=np.load(R/"outputs/grid242_parent_prior_inputs/outer_parent_public"/n);ref2=np.load(R/"outputs/grid249_context_bit_plan/outer_parent_public"/n)
  assert np.array_equal(h["ids"],ref["ids"]) and np.array_equal(h["ids"],ref2["ids"])
  event,amount=[pickle.loads((R/p).read_bytes()) for p in role["parent_heads"]]
  assert event.n_features_in_==amount.n_features_in_==124 and len(event.estimators_)==len(amount.estimators_)==64 and event.n_jobs==amount.n_jobs==1
  t=time.perf_counter();x=rt.pack_parent(h["features"],raw[h["ids"].astype(int)],sc);pack_s=time.perf_counter()-t
  t=time.perf_counter();out=rt.predict_parent(x,h["features"][:,82],event,amount);calls+=2;infer_s=time.perf_counter()-t
  mismatch=int(np.count_nonzero(out["b"]!=ref["b"]));error=float(abs(out["b"]-ref["b"]).max())
  record=dict(pool=pool,rows=len(x),b_bit_exact=mismatch==0,b_mismatched=mismatch,max_abs_b_error=error,head_calls=2,charge_exact=np.array_equal(out["charge"],h["features"][:,82]),parent_entry_exact=np.array_equal(out["base_CONT"],ref["base_CONT"]),pack_batch_observation_s=pack_s,heads_batch_observation_s=infer_s)
  records.append(record);save("parity_receipts.json",records)
  assert mismatch==0 and record["parent_entry_exact"] and np.array_equal(out["b"],ref2["b"]) and np.array_equal(out["base_CONT"],ref2["base_CONT"]),(pool,"exact parent parity failure",error)
  np.savez_compressed(O/"public_parent"/n,ids=h["ids"],**out)
 assert calls==24 and sum(r["rows"] for r in records)==4681
 for p,h in rd(O/"source_freeze.json").items():assert sha(R/p)==h,p
 save("public_freeze.json",{p.relative_to(O).as_posix():sha(p) for p in (O/"public_parent").glob("*.npz")})
 save("complete.json",dict(passed=True,pools=12,rows=4681,native_head_batch_predictions=calls,scalar_head_row_predictions=9362,all_parent_b_and_entries_bit_exact=True,models_unchanged=True,new_fits=0,NN_forward=0,teacher_labels=0,physics=0,new_strategy_returns=0,no_new261_scores=True,timing_not_benchmark=True))
 print("GRID288 all12 parent b/entries exactly reproduced;24head batch calls/0fits/NN/physics")
except BaseException as e:
 save("execution_error.json",dict(error=repr(e),traceback=traceback.format_exc(),completed_batch_head_calls=calls,completed_pools=len(records),automatic_rerun_prohibited=True));raise
