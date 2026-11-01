from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[k]="1"
import json,hashlib,subprocess,importlib.util,time,traceback
import numpy as np,torch
R=Path("D:/Codex/GRL研究");repo=R/"work/GRL-results-integration";W=Path(__file__).parent;O=R/"outputs/grid290_public_nn_chain"
def rd(p):return json.loads(p.read_text(encoding="utf8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
assert not (O/"execution_started.json").exists()
for p,h in rd(O/"source_freeze.json").items():assert sha(R/p)==h,p
g=lambda *a:subprocess.check_output(["git","-C",str(repo),*a]);head=g("rev-parse","HEAD").decode().strip();assert g("ls-remote","lntano-src","refs/heads/main").decode().split()[0]==head
assert g("show",head+":research_archive/power-grid-2026/protocols/GRID290_public_nn_chain/source/source_freeze.json")== (O/"source_freeze.json").read_bytes()
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
api=load("g290_originalapi",R/"work/grid190_extension_gate/run.py");old=api.G.old_api();parent=load("g290_parent",R/"work/grid288_parent208_interface/runtime.py");resid=load("g290_resid",R/"work/grid287_frozen261_interface/runtime.py");combo=load("g290_combo",R/"work/grid289_combined_single_state/runtime.py")
refs={r["pool"]:r for r in rd(R/"outputs/grid289_combined_single_state/model_references.json")};raw=np.load(R/"outputs/grid214_inner_head_generation/public_raw41.npz")["values"]
save("execution_started.json",dict(source_published=head,NN_forward_cap=12,head_call_cap=24,linear_call_cap=12));records=[];nncalls=headcalls=linear=0;(O/"public_chain").mkdir()
try:
 for nref in rd(O/"nn_references.json"):
  pool=nref["pool"];i=nref["metadata_id"];seed=int(pool[-1]);cp=R/nref["checkpoint"];assert sha(cp)==nref["checkpoint_sha"]
  t=time.perf_counter();net=api.S.build(seed,"FACTOR");net.load_state_dict(torch.load(cp,map_location="cpu",weights_only=False)["state_dict"]);net.eval();before=api.digest(net);assert before==nref["state_hash"];model=combo.Frozen208261(R,refs[pool],parent,resid);load_s=time.perf_counter()-t
  z=np.load(O/"prepared_public"/f"{pool}.npz");row={k:z[k].copy() for k in z.files};got={};handles=[net.base.q_net[3].register_forward_hook(lambda m,i,o:got.__setitem__("main",o.detach().numpy().copy())),net.delta.head[1].register_forward_hook(lambda m,i,o:got.__setitem__("delta",o.detach().numpy().copy()))]
  t=time.perf_counter();batch=old.collate([row]);collate_s=time.perf_counter()-t
  t=time.perf_counter()
  try:
   with torch.inference_mode():q=net(**batch).numpy().copy();nncalls+=1
  finally:
   for h in handles:h.remove()
  nn_s=time.perf_counter()-t;assert got["main"].shape==(1,64) and got["delta"].shape==(1,16)
  raw83=np.concatenate([got["main"],got["delta"],q,np.stack([row["known_query_cost"]])],1).astype(float)
  oldh=np.load(R/f"outputs/grid190_extension_gate/held_{pool}.npz")["features"][:1];qerr=float(abs(q-oldh[:,80:82]).max());rawerr=float(abs(raw83-oldh).max());qclose=np.allclose(q,oldh[:,80:82],atol=3e-5,rtol=3e-5);stop=np.array_equal(q[:,1]>q[:,0],oldh[:,81]>oldh[:,80]);assert api.digest(net)==before
  assert qclose and stop and not (q[:,1]>q[:,0]).any(),(pool,"original190 NN qualification",qerr)
  out=model.decide(raw83,raw[[i]],owner_pool=pool);headcalls+=2;linear+=1;pref=np.load(R/"outputs/grid288_parent208_interface/public_parent"/f"{pool}.npz");sref=np.load(R/"outputs/grid261_bounded_basis/outer_predictions"/f"{pool}.npz")
  assert int(pref["ids"][0])==int(sref["ids"][0])==i
  rec=dict(pool=pool,metadata_id=i,KEEP=model.KEEP,q_max_error=qerr,raw83_max_error=rawerr,q_original_tolerance_passed=bool(qclose),nativeSTOP_exact=bool(stop),parent_b_max_error=float(abs(out["b"]-pref["b"][:1]).max()),logit_max_error=float(abs(out["logit"]-sref["logit"][:1]).max()),parent_entry_exact=np.array_equal(out["base_CONT"],pref["base_CONT"][:1]),final_decision_exact=np.array_equal(out["continue_"],sref["continue_"][:1]),state_hash_unchanged=True,load_observation_s=load_s,collate_observation_s=collate_s,NN_observation_s=nn_s,component_observations=out["timing"])
  records.append(rec);save("chain_receipts.json",records)
  np.savez_compressed(O/"public_chain"/f"{pool}.npz",metadata_id=np.array([i]),raw83=raw83,q=q,parent_b=out["b"],logit=out["logit"],continue_=out["continue_"])
  assert rec["parent_entry_exact"] and rec["final_decision_exact"],(pool,"final/parent decision mismatch",rec)
 assert nncalls==12 and headcalls==24 and linear==12
 for p,h in rd(O/"source_freeze.json").items():assert sha(R/p)==h,p
 save("public_freeze.json",{p.relative_to(O).as_posix():sha(p) for p in (O/"public_chain").glob("*.npz")})
 save("complete.json",dict(passed=True,pools=12,KEEP=sum(r["KEEP"] for r in records),NN_forwards=nncalls,head_single_row_calls=headcalls,linear_calls=linear,new_fits=0,teacher_labels=0,physics=0,strategy_returns=0,NN_q_max_error=max(r["q_max_error"] for r in records),raw83_max_error=max(r["raw83_max_error"] for r in records),parent_b_max_error=max(r["parent_b_max_error"] for r in records),logit_max_error=max(r["logit_max_error"] for r in records),all_nativeSTOP_parent_final_decisions_exact=True,offline_interface_series_closed=True,timing_is_not_benchmark=True))
 print("GRID290 public-context NN+composite12 states qualified;0fit/physics",rd(O/"complete.json"))
except BaseException as e:
 save("execution_error.json",dict(error=repr(e),traceback=traceback.format_exc(),NN_forwards=nncalls,head_calls=headcalls,linear_calls=linear,completed_records=len(records),auto_retry=False));raise
