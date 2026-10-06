"""Independent event/cost and batch-selection audit. No policy imports.

For CHEAP, refit only cost regressors from the paid prefix and independently
reconstruct its ranked candidate list. These are audit computations, not extra
acquisition trajectories. Test evaluation waits for all six trajectories.
"""
import concurrent.futures
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import time
import zlib
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OUT=PROJECT/"outputs"/"fa01"
OLD=PROJECT/"outputs"/"fa00"

def read(p):return json.loads(p.read_text(encoding="utf-8"))
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding="utf-8")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def seed_key(seed,*keys):return int.from_bytes(hashlib.sha256(repr((seed,keys)).encode()).digest()[:4],"big")

def audit_selection(dest,p,X):
    started=time.perf_counter();result=read(dest/"result.json")
    arm,seed=result["arm"],result["seed"]
    kinds=np.zeros((len(p["ids"]),len(p["algorithms"])),dtype=np.int8)
    values=np.zeros(kinds.shape,float)
    known=set();pairs=list(itertools.combinations(range(len(p["algorithms"])),2))
    train=np.asarray(p["train"]);checks=[];refits=0;regfit_s=0
    params=dict(n_estimators=100,max_features="sqrt",max_depth=2**31,
                min_samples_split=2,bootstrap=True,n_jobs=1)
    with gzip.open(dest/"actions.jsonl.gz","rt",encoding="utf-8") as f:
        for line in f:
            e=json.loads(line)
            if e["type"]=="observation":
                assert e["role"]=="train" and e["planned_cap_s"]==100
                kinds[e["i"],e["a"]]=e["kind"];values[e["i"],e["a"]]=e["value"]
            elif e["type"]=="pair_label":known.add((e["i"],e["pair"]))
            elif e["type"]=="batch":
                assert e["cap_s"]==100
                candidates=[];cost=[];can=(kinds<2)&((kinds==0)|(values<100-1e-10))
                pred=None
                if arm=="CHEAP-100":
                    startfit=time.perf_counter();pred=np.zeros((len(train),len(p["algorithms"])))
                    for a in range(len(p["algorithms"])):
                        rows=train[kinds[train,a]>0]
                        model=RandomForestRegressor(**params,random_state=seed_key(seed,"reg",e["round"]-1,a))
                        model.fit(X[rows],values[rows,a]);pred[:,a]=model.predict(X[train]);refits+=1
                    pred=np.minimum(np.maximum(np.maximum(pred,values[train]),0),100)
                    regfit_s+=time.perf_counter()-startfit
                for pos,i0 in enumerate(train):
                    i=int(i0)
                    for pi,(a,b) in enumerate(pairs):
                        if (i,pi) in known or not(can[i,a] or can[i,b]):continue
                        candidates.append((i,pi))
                        if pred is not None:cost.append(float((pred[pos,a] if can[i,a] else 0)+(pred[pos,b] if can[i,b] else 0)))
                # train and pairs are sorted: this list is canonical independently
                # of the policy's score-sorted internal candidate table.
                rng=np.random.default_rng(seed_key(seed,"acquire",e["round"]))
                indices=rng.permutation(len(candidates))
                if pred is not None:indices=indices[np.argsort(np.asarray(cost)[indices],kind="stable")]
                chosen=indices[:min(p["batch_size"],len(indices))].copy()
                np.random.default_rng(seed_key(seed,"batch",e["round"])).shuffle(chosen)
                expected=[list(candidates[j]) for j in chosen]
                assert expected==e["actions"],(arm,seed,e["round"],expected[:4],e["actions"][:4])
                checks.append(dict(round=e["round"],eligible_candidates=len(candidates),selected=len(chosen),exact_order_match=True))
    return dict(arm=arm,seed=seed,batches_checked=len(checks),selections_checked=sum(c["selected"] for c in checks),
        cost_regressor_refits_for_audit=refits,cost_regressor_recompute_s=regfit_s,
        audit_wall_s=time.perf_counter()-started,checks=checks)

def write_csv(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    start=time.perf_counter();p=read(OUT/"FA01_protocol.json");oldp=read(OLD/"FA00_protocol.json")
    newpaths=[OUT/"runs"/f"{a}_s{s}" for s in p["seeds"] for a in p["arms"]]
    assert all((d/"result.json").exists() and read(d/"result.json")["status"]=="complete" for d in newpaths),"Six completed trajectories before new test evaluation"
    batch=read(OUT/"FA01_batch.json");assert batch["completed"]==6
    assert all(z["process_exit_code"]==0 for z in batch["results"])
    recovery_checks=[]
    if (OUT/"FA01_recovery.json").exists():
        for seed in p["seeds"]:
            dest=OUT/"runs"/f"CHEAP-100_s{seed}"
            meta=read(dest/"resume_manifest.json")
            archive=dest/"actions.INTERRUPTED.jsonl.gz"
            assert sha(archive)==meta["interrupted_log_sha256"]
            original_bytes=zlib.decompressobj(31).decompress(archive.read_bytes())
            original=original_bytes.splitlines()
            assert len(original)==meta["events_preserved"] and meta["discarded_events"]==0
            assert meta["resume_completed"]
            final=read(dest/"result.json")
            setup_and_bookkeeping=max(0,meta["resume_including_setup_wall_s"]-(final["wall_s"]-meta["elapsed_active_s"]))
            with gzip.open(dest/"actions.jsonl.gz","rb") as f:
                assert f.read(len(original_bytes))==original_bytes
            recovery_checks.append(dict(seed=seed,preserved_events=len(original),exact_log_prefix_match=True,
                checkpoint_round=meta["checkpoint_round"],discarded_events=0,
                resume_including_setup_wall_s=meta["resume_including_setup_wall_s"],
                resume_setup_and_bookkeeping_s=setup_and_bookkeeping))
    frozen=read(OUT/"FA01_freeze_manifest.json")
    for section in ["files","inherited_files"]:
        for path,digest in frozen[section].items():assert sha(PROJECT/path)==digest,path
    # Reuse independently audited event reconstruction, without importing policy.
    oldaudit=load_module(ROOT.parent/"fa00"/"audit_analyse.py","fa00_independent_audit")
    helper=(ROOT.parent/"fa00"/"audit_analyse.py").read_text(encoding="utf-8").split('\ndef main():')[0]
    helper=helper.replace('"FA00_protocol.json"','"FA01_protocol.json"').replace('OUT=ROOT.parents[1]/"outputs"/"fa00"','OUT=ROOT.parents[1]/"outputs"/"fa01"')
    helperpath=ROOT/"audit_base.py";helperpath.write_text(helper,encoding="utf-8")
    newaudit=load_module(helperpath,"fa01_independent_events")
    truth=oldaudit.raw_runtime_table();data=dict(np.load(OLD/"FA00_data.npz"))
    results=[];events=[];test=[];histories=[]
    for s in p["seeds"]:
        d=OLD/"runs"/f"FIXED-100_s{s}"
        r,c,tr,hr=oldaudit.audit_one(d,oldp,truth)
        r["provenance"]="reused unchanged FA00"
        results.append(r);events.append(c);test.extend(tr);histories.extend(hr)
    for d in newpaths:
        r,c,tr,hr=newaudit.audit_one(d,p,truth)
        assert r["validation_cost_s"]==0 and r["validation_comparisons"]==0
        assert all(h["cap_s"]==100 for h in read(d/"history.json"))
        seed=r["seed"]
        ref=OLD/"runs"/f"FIXED-100_s{seed}"
        assert np.array_equal(np.load(d/"predictions_r0000.npy"),np.load(ref/"predictions_r0000.npy"))
        old_init=read(ref/"history.json")[0];new_init=read(d/"history.json")[0]
        assert abs(old_init["train_cost_s"]-new_init["train_cost_s"])<1e-6
        r["provenance"]="new FA01 trajectory"
        results.append(r);events.append(c);test.extend(tr);histories.extend(hr)
    for r in results:
        parent=OLD if r["arm"]=="FIXED-100" else OUT
        h=read(parent/"runs"/f"{r['arm']}_s{r['seed']}"/"history.json")
        r["last_fitted_train_cost_s"]=h[-1]["train_cost_s"]
        r["training_cost_after_last_fit_s"]=max(0,r["train_cost_s"]-h[-1]["train_cost_s"])
    # Independent acquisition reconstruction is separate from the strategy budget.
    selection=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(audit_selection,d,p,data["X"]) for d in newpaths]
        for fut in concurrent.futures.as_completed(futures):
            c=fut.result();selection.append(c)
            write(OUT/"FA01_selection_audit_progress.json",dict(completed=len(selection),target=6,
                  audit_wall_s=time.perf_counter()-start,results=selection))
            print("selection audit",c["arm"],c["seed"],c["batches_checked"],flush=True)
    summary=[]
    for arm in ["FIXED-100"]+p["arms"]:
        rs=sorted([r for r in results if r["arm"]==arm],key=lambda r:p["seeds"].index(r["seed"]))
        summary.append(dict(arm=arm,mean_par10_s=math.fsum(r["mean_test_par10_s"] for r in rs)/3,
            seed7_par10_s=rs[0]["mean_test_par10_s"],seed42_par10_s=rs[1]["mean_test_par10_s"],seed99_par10_s=rs[2]["mean_test_par10_s"],
            mean_train_cost_s=math.fsum(r["train_cost_s"] for r in rs)/3,
            mean_remaining_s=math.fsum(r["remaining_s"] for r in rs)/3,
            mean_local_wall_s=math.fsum(r["wall_s"] for r in rs)/3,
            mean_fit_s=math.fsum(r["fit_s"] for r in rs)/3,
            mean_predict_s=math.fsum(r["predict_s"] for r in rs)/3,
            mean_selection_s=math.fsum(r["selection_s"] for r in rs)/3,
            mean_training_cost_after_last_fit_s=math.fsum(r["training_cost_after_last_fit_s"] for r in rs)/3,
            mean_labels=math.fsum(c["pair_labels"] for c in events if c["arm"]==arm)/3,
            mean_executions=math.fsum(r["executions"] for r in rs)/3,
            mean_test_timeouts=math.fsum(r["test_timeouts"] for r in rs)/3))
    base=summary[0];comparisons=[]
    for other in summary[1:]:
        directions=[]
        for seed in p["seeds"]:
            a=next(r for r in results if r["arm"]=="FIXED-100" and r["seed"]==seed)
            b=next(r for r in results if r["arm"]==other["arm"] and r["seed"]==seed)
            directions.append(dict(seed=seed,pareto_better=a["mean_test_par10_s"]<b["mean_test_par10_s"]-1e-9,
                pareto_minus_control_s=a["mean_test_par10_s"]-b["mean_test_par10_s"]))
        advantage=1-base["mean_par10_s"]/other["mean_par10_s"]
        comparisons.append(dict(control=other["arm"],pareto_relative_advantage=advantage,
            pareto_positive_seeds=sum(z["pareto_better"] for z in directions),seed_directions=directions,
            continuation_pass=advantage>=.05 and sum(z["pareto_better"] for z in directions)>=2,
            control_relative_to_pareto=other["mean_par10_s"]/base["mean_par10_s"]-1,
            closure_pass=other["mean_par10_s"]<=1.02*base["mean_par10_s"]))
    cont=all(c["continuation_pass"] for c in comparisons);close=any(c["closure_pass"] for c in comparisons)
    verdict="ACCEPT_CHEAP_BASELINE" if close else "CONTINUE_BASELINE_RESEARCH" if cont else "UNDETERMINED"
    audit=dict(new_formal_runs=6,reused_baseline_runs=3,hard_failures=0,
        frozen_hashes_unchanged=True,raw_runtime_cells=len(truth),events=sum(c["events"] for c in events),
        observation_events=sum(c["observation_events"] for c in events),test_predictions=len(test),
        max_ledger_error_s=max(c["max_ledger_error_s"] for c in events),
        per_run_event_audits=events,independent_selection_audits=selection,
        initial_predictions_and_costs_match=True,new_validation_cost_s=0,
        all_new_dispatch_caps_leq100=True,audit_total_wall_s=time.perf_counter()-start,
        external_interruption_recovery_checks=recovery_checks,
        warning="This is a development comparison on the previously inspected test fold. Audit regressor refits are not additional strategy trajectories.")
    v=dict(verdict=verdict,continuation_gate=cont,closure_gate=close,comparisons=comparisons,summary=summary,
        warning="No independent-confirmation, statistical-significance or equivalence claim; all labels are from observed feedback")
    write(OUT/"FA01_audit.json",audit);write(OUT/"FA01_verdict.json",v)
    write_csv(OUT/"FA01_summary.csv",summary)
    # Avoid one arm's provenance determining an inconsistent field order.
    write_csv(OUT/"FA01_per_run.csv",results);write_csv(OUT/"FA01_test_predictions.csv",test)
    write_csv(OUT/"FA01_round_metrics.csv",histories)
    print(json.dumps(v,indent=2),flush=True)

if __name__=="__main__":main()
