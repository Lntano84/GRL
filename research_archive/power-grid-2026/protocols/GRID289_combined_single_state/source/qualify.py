from pathlib import Path
import os,sys
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[k]="1"
sys.dont_write_bytecode=True
import json,hashlib,subprocess,importlib.util,time,traceback
import numpy as np,sklearn
R=Path("D:/Codex/GRL研究");repo=R/"work/GRL-results-integration";W=Path(__file__).parent;O=R/"outputs/grid289_combined_single_state"
def rd(p):return json.loads(p.read_text(encoding="utf8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
def load(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
assert not (O/"execution_started.json").exists()
for p,h in rd(O/"source_freeze.json").items():assert sha(R/p)==h,p
assert sklearn.__version__=="1.7.2"
g=lambda *a:subprocess.check_output(["git","-C",str(repo),*a]);head=g("rev-parse","HEAD").decode().strip();assert g("ls-remote","lntano-src","refs/heads/main").decode().split()[0]==head
assert g("show",head+":research_archive/power-grid-2026/protocols/GRID289_combined_single_state/source/source_freeze.json")== (O/"source_freeze.json").read_bytes()
save("execution_started.json",dict(source_published=head,single_states=12,head_call_cap=24,linear_call_cap=12))
p=load("g289parent",R/"work/grid288_parent208_interface/runtime.py");r=load("g289resid",R/"work/grid287_frozen261_interface/runtime.py");m=load("g289combo",W/"runtime.py");raw=np.load(R/"outputs/grid214_inner_head_generation/public_raw41.npz")["values"];records=[];heads=linear=0
try:
 for ref in rd(O/"model_references.json"):
  pool=ref["pool"];n=pool+".npz";h=np.load(R/f"outputs/grid190_extension_gate/held_{pool}.npz");sel=next(x for x in rd(O/"fixed_selection.json") if x["pool"]==pool);assert int(h["ids"][0])==sel["metadata_id"]
  t=time.perf_counter();model=m.Frozen208261(R,ref,p,r);load_s=time.perf_counter()-t
  try:model.decide(h["features"][:1],raw[h["ids"][:1].astype(int)],owner_pool="wrong-owner")
  except ValueError:pass
  else:raise AssertionError("wrong owner accepted")
  out=model.decide(h["features"][:1],raw[h["ids"][:1].astype(int)],owner_pool=pool);heads+=2;linear+=1
  old=np.load(R/"outputs/grid288_parent208_interface/public_parent"/n);saved=np.load(R/"outputs/grid261_bounded_basis/outer_predictions"/n);assert np.array_equal(old["ids"][:1],h["ids"][:1]) and np.array_equal(saved["ids"][:1],h["ids"][:1])
  rec=dict(pool=pool,metadata_id=sel["metadata_id"],KEEP=model.KEEP,head_calls=2,linear_calls=1,load_observation_s=load_s,timing_observation=out["timing"],parent_exact=all(np.array_equal(out[k],old[k][:1]) for k in ["event_p","mu","charge","b","base_CONT"]),logit_exact=np.array_equal(out["logit"],saved["logit"][:1]),decision_exact=np.array_equal(out["continue_"],saved["continue_"][:1]),b_error=float(abs(out["b"]-old["b"][:1]).max()),logit_error=float(abs(out["logit"]-saved["logit"][:1]).max()))
  records.append(rec);save("single_state_receipts.json",records)
  assert rec["parent_exact"] and rec["logit_exact"] and rec["decision_exact"],rec
  if model.KEEP:assert np.array_equal(out["continue_"],out["base_CONT"])
 assert heads==24 and linear==12 and len(records)==12
 for p,h in rd(O/"source_freeze.json").items():assert sha(R/p)==h,p
 save("complete.json",dict(passed=True,pools=12,KEEP=sum(r["KEEP"] for r in records),single_row_head_predictions=heads,linear_scores=linear,all_parent_logits_decisions_exact=True,owner_guard_rejected_before_predictions=True,new_fits=0,NN_forward=0,teacher_labels=0,physics=0,new_strategy_returns=0,full_batch_repeated=False,timing_is_observation_not_benchmark=True))
 print("GRID289 composite12 single states/24heads/12linear exact;0fitNNphysics")
except BaseException as e:
 save("execution_error.json",dict(error=repr(e),traceback=traceback.format_exc(),completed_head_calls=heads,completed_linear_calls=linear,completed_records=len(records),automatic_retry_prohibited=True));raise
