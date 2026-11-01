from pathlib import Path
import sys,json,hashlib,time,importlib.util,subprocess
sys.dont_write_bytecode=True
import numpy as np
R=Path("D:/Codex/GRL研究");O=R/"outputs/grid287_frozen261_interface";W=Path(__file__).parent;repo=R/"work/GRL-results-integration"
def rd(p):return json.loads(p.read_text(encoding="utf8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
assert not (O/"qualification_started.json").exists()
for p,h in rd(O/"source_freeze.json").items():assert sha(R/p)==h,p
g=lambda *a:subprocess.check_output(["git","-C",str(repo),*a]);head=g("rev-parse","HEAD").decode().strip();assert g("ls-remote","lntano-src","refs/heads/main").decode().split()[0]==head
assert g("show",head+":research_archive/power-grid-2026/protocols/GRID287_frozen261_interface/source/source_freeze.json")== (O/"source_freeze.json").read_bytes()
save("qualification_started.json",dict(source_published=head,one_saved_input_parity=True))
sp=importlib.util.spec_from_file_location("frozen261",W/"runtime.py");rt=importlib.util.module_from_spec(sp);sp.loader.exec_module(rt)
raw=np.load(R/"outputs/grid214_inner_head_generation/public_raw41.npz")["values"];records=[]
for role in rd(R/"outputs/grid241_parent_prior_plan/qualified_pools.json"):
 pool=role["pool"];name=pool+".npz";z=np.load(R/"outputs/grid249_context_bit_plan/outer_parent_public"/name);m=np.load(R/"outputs/grid261_bounded_basis/models"/name);old=np.load(R/"outputs/grid261_bounded_basis/outer_predictions"/name)
 scaler=np.load(R/role["residual_public41_scaler"]);t0=time.perf_counter();X=rt.pack_public(raw[z["ids"].astype(int)],scaler["mean41"],scaler["scale41"],z["b"]);pack_s=time.perf_counter()-t0
 assert np.array_equal(X,z["X"]),(pool,"raw-to-public40 mismatch")
 t0=time.perf_counter();score,cont=rt.score_public40(X,z["b"],z["base_CONT"],m["theta"],float(m["alpha"][0]));score_s=time.perf_counter()-t0
 assert np.array_equal(score,old["logit"]) and np.array_equal(cont,old["continue_"]) and np.array_equal(z["ids"],old["ids"])
 keep=bool(m["fallback_KEEP"][0])
 if keep:assert float(m["alpha"][0])==0 and (m["theta"]==0).all() and np.array_equal(cont,z["base_CONT"])
 records.append(dict(pool=pool,rows=len(score),KEEP=keep,raw_pack_bit_exact=True,score_bit_exact=True,decision_bit_exact=True,max_abs_score_error=float(np.max(abs(score-old["logit"]))),pack_batch_observation_s=pack_s,score_batch_observation_s=score_s,scaler=role["residual_public41_scaler"],parent_scaler=role["parent_scaler"],parent_heads=role["parent_heads"]))
# Equality continues only within parentCONT; malformed bit/offset must not be silently accepted.
b=np.array([.2,-.2]);x=np.zeros((2,40));x[:,38]=b;x[:,39]=1;theta=np.zeros(41);theta[-1]=-.2
v,c=rt.score_public40(x,b,b>0,theta,1.);assert v[0]==0 and c.tolist()==[True,False]
bad=x.copy();bad[0,39]=0
try:rt.score_public40(bad,b,b>0,theta,1.)
except ValueError:pass
else:raise AssertionError("invalid context accepted")
assert sum(r["rows"] for r in records)==4681 and sum(r["KEEP"] for r in records)==1
save("parity_receipts.json",records);save("complete.json",dict(passed=True,pools=12,rows=4681,KEEP=1,all_raw_pack_score_ids_decisions_bit_exact=True,new_fits=0,NN_head_forward=0,new_strategy_returns=0,real_physical_calls=0,linear_score_recomputations=4681,new_policy_predictions_exported=0,limits="cached public interface parity only; parent NN/head inference not run; single batch observations not timing benchmark; no end-to-end deployment claim"))
print("GRID287 twelve pools/4681 raw inputs, logits, decisions exactly reproduced; 0fits/NNhead/physics")
