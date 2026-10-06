"""Independent post-run audit: raw ARFF CSV -> events -> PAR10 and verdict.

Never import the policy, replay environment, or model code here. Test outcomes
are not read until all 18 formal trajectories have completed.
"""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/"outputs"/"fa00"

def read_json(path):return json.loads(path.read_text(encoding="utf-8"))
def write_json(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
def near(a,b,tol=1e-6):
    if abs(float(a)-float(b))>tol:raise AssertionError((a,b))

def raw_runtime_table():
    path=ROOT/"upstream"/"DATASETS"/"ASP-POTASSCO"/"algorithm_runs.arff"
    lines=path.read_text(encoding="utf-8").splitlines()
    start=next(i for i,l in enumerate(lines) if l.strip().lower()=="@data")+1
    table={}
    for row in csv.reader(lines[start:]):
        if not row or row[0].startswith("%") or int(float(row[1]))!=1:continue
        key=(row[0],row[2]);assert key not in table
        table[key]=(float(row[3]),row[4].strip())
    return table

def audit_one(dest,p,truth):
    r=read_json(dest/"result.json");h=read_json(dest/"history.json")
    assert r["status"]=="complete"
    assert r["protocol_sha256"]==hashlib.sha256((OUT/"FA00_protocol.json").read_bytes()).hexdigest()
    cache={"train":{},"validation":{}};tc=vc=0.0;executions=hits=completed=cancelled=0
    labels={}; observations=events=0;max_ledger_error=0.;native_score_cost_checks=0
    train=set(p["train"]);valid=set(p["small32"] if r["arm"]=="SMALL32-DYN" else p["validation"])
    by_round={z["round"]:z for z in h}
    preds={k:np.load(dest/z["predictions_file"]) for k,z in by_round.items()}
    for z in h:
        assert z["prediction_sha256"]==hashlib.sha256((dest/z["predictions_file"]).read_bytes()).hexdigest()
    events_by_round={}
    with gzip.open(dest/"actions.jsonl.gz","rt",encoding="utf-8") as f:
        for line in f:
            e=json.loads(line);assert e["event_id"]==events;events+=1
            typ=e["type"]
            if typ=="observation":
                observations+=1;role=e["role"];i,a=e["i"],e["a"]
                assert i in (train if role=="train" else valid)
                if role=="validation":
                    assert r["arm"] in ["PAID-DYN","SMALL32-DYN"]
                    new=preds[e["round"]];old_pred=preds[e["round"]-1]
                    assert new[i]!=old_pred[i] and a in [int(new[i]),int(old_pred[i])]
                old=cache[role].get((i,a),(0,0.));assert old[0]==e["before_kind"];near(old[1],e["before_value"])
                planned=e["planned_cap_s"];assert 0<planned<=600
                cached=old[0]>=2 or (old[0]==1 and old[1]>=planned-1e-10)
                assert cached==e["cached"]
                if cached:
                    hits+=1;near(e["charged_s"],0);near(e["actual_cap_s"],0)
                    new=old
                else:
                    executions+=1
                    actual=min(planned,p["budget_s"]-tc-vc);near(actual,e["actual_cap_s"])
                    value,status=truth[(p["ids"][i],p["algorithms"][a])]
                    success=status=="ok" and value<=actual
                    charged=value if success else actual;near(charged,e["charged_s"])
                    if success:new=(2,value);completed+=1
                    elif actual>=600-1e-9:new=(3,600.);cancelled+=1;native_score_cost_checks+=1
                    else:new=(1,max(old[1],actual));cancelled+=1
                    if role=="train":tc+=charged
                    else:vc+=charged
                assert new[0]==e["kind"];near(new[1],e["value"])
                cache[role][(i,a)]=new
            elif typ=="pair_label":
                i,a,b=e["i"],e["a"],e["b"]
                assert i in train and (e["pair"],i) not in labels
                ka,va=cache["train"].get((i,a),(0,0.));kb,vb=cache["train"].get((i,b),(0,0.))
                assert ka==e["ka"] and kb==e["kb"] and ka and kb and (ka==2 or kb==2)
                assert not(ka!=2 and va<vb-1e-10) and not(kb!=2 and vb<va-1e-10)
                near(va,e["va"]);near(vb,e["vb"])
                pa=va if ka==2 else 10*va;pb=vb if kb==2 else 10*vb
                near(abs(pa-pb),e["weight"])
                ra,sa=truth[(p["ids"][i],p["algorithms"][a])]
                rb,sb=truth[(p["ids"][i],p["algorithms"][b])]
                # Labels can be tied, but cannot designate a strictly slower run faster.
                if e["label"]==0:assert ra<=rb+1e-6
                else:assert rb<=ra+1e-6
                labels[(e["pair"],i)]=e["label"]
            elif typ=="round_end":
                no=e["round"];new=preds[no];old=preds[no-1]
                if e["delta"] is not None:
                    delta=0.;changed=0
                    for i in sorted(valid):
                        if new[i]==old[i]:continue
                        changed+=1
                        rv,rs=truth[(p["ids"][i],p["algorithms"][int(new[i])])]
                        ov,os=truth[(p["ids"][i],p["algorithms"][int(old[i])])]
                        delta+=(rv if rs=="ok" else 6000)-(ov if os=="ok" else 6000)
                        if r["arm"]!="FREE-DYN":
                            assert cache["validation"][(i,int(new[i]))][0]>=2
                            assert cache["validation"][(i,int(old[i]))][0]>=2
                    near(delta,e["delta"]);assert changed==e["changed"]
                    current=by_round[no]["cap_s"]
                    near(e["next_cap_s"],min(600,current+100) if delta>=-1e-9 else current)
                events_by_round[no]=e
            elif typ=="free_validation":assert r["arm"]=="FREE-DYN" and e["charged_s"]==0
            elif typ=="batch":
                assert len(e["actions"])==e["size"] and e["size"]<=p["batch_size"]
                assert all(i in train and 0<=pair<55 for i,pair in e["actions"])
            elif typ=="cutoff":assert e["cap_s"]<=600
            else:raise AssertionError(typ)
            max_ledger_error=max(max_ledger_error,abs(tc-e["train_cost_s"]),abs(vc-e["validation_cost_s"]))
            near(tc,e["train_cost_s"]);near(vc,e["validation_cost_s"])
            assert tc+vc<=p["budget_s"]+1e-6
            near(max(0,p["budget_s"]-tc-vc),e["remaining_s"])
    near(tc,r["train_cost_s"]);near(vc,r["validation_cost_s"])
    assert (executions,hits,completed,cancelled)==(r["executions"],r["cache_hits"],r["completions"],r["cancellations"])
    final=np.load(dest/"final_predictions.npy")
    assert np.array_equal(final,preds[r["last_fitted_round"]])
    assert final.shape==(len(p["ids"]),) and np.all((final>=0)&(final<len(p["algorithms"])))
    score=[];per_instance=[]
    for i in p["test"]:
        value,status=truth[(p["ids"][i],p["algorithms"][int(final[i])])]
        par=value if status=="ok" else 6000.
        score.append(par);per_instance.append(dict(arm=r["arm"],seed=r["seed"],row=i,
            instance_id=p["ids"][i],algorithm=p["algorithms"][int(final[i])],runstatus=status,
            native_runtime_s=value,par10_s=par))
    history_metrics=[]
    for no,pred in preds.items():
        ps=[]
        for i in p["test"]:
            value,status=truth[(p["ids"][i],p["algorithms"][int(pred[i])])]
            ps.append(value if status=="ok" else 6000.)
        history_metrics.append(dict(arm=r["arm"],seed=r["seed"],round=no,
            mean_test_par10_s=math.fsum(ps)/len(ps),**{k:by_round[no][k] for k in
            ["train_cost_s","validation_cost_s","cap_s","labels","wall_s"]}))
    r["mean_test_par10_s"]=math.fsum(score)/len(score)
    r["test_timeouts"]=sum(z["runstatus"]=="timeout" for z in per_instance)
    r["validation_cost_fraction"]=vc/(tc+vc)
    diagnostic=dict(arm=r["arm"],seed=r["seed"],events=events,observation_events=observations,
        pair_labels=len(labels),exact_scores_checked=len(score),max_ledger_error_s=max_ledger_error,
        actual_dispatch_checks=executions,cache_checks=hits,native_timeout_score_cost_checks=native_score_cost_checks,
        fixed_budget_violations=0,wrong_labels=0,validation_permissions_violations=0,
        final_prediction_checkpoint_mismatch=0)
    return r,diagnostic,per_instance,history_metrics

def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=read_json(OUT/"FA00_protocol.json")
    paths=[OUT/"runs"/f"{a}_s{s}" for s in p["seeds"] for a in p["arms"]]
    assert all((q/"result.json").exists() and read_json(q/"result.json")["status"]=="complete" for q in paths),"18 complete runs required before test evaluation"
    freeze=read_json(OUT/"FA00_freeze_manifest.json")
    for name,sha in freeze["files"].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    source=read_json(ROOT/"source_manifest.json")
    assert source["commit"]==p["source_commit"]
    assert hashlib.sha256((ROOT/"source_manifest.json").read_bytes()).hexdigest()==read_json(OUT/"FA00_data_audit.json")["source_manifest_sha256"]
    for entry in source["files"]:
        content=(ROOT/"upstream"/entry["path"]).read_bytes()
        assert len(content)==entry["bytes"] and hashlib.sha256(content).hexdigest()==entry["sha256"]
    truth=raw_runtime_table()
    d=np.load(OUT/"FA00_data.npz")
    # Reconstruct preprocessing independently with NumPy, using only train rows.
    prep=np.load(OUT/"FA00_preprocessing.npz")
    train=np.asarray(p["train"]);val=set(p["validation"]);test=set(p["test"])
    assert len(set(train))==len(train) and not(set(train)&val or set(train)&test or val&test)
    assert set(train)|val|test==set(range(len(p["ids"])))
    assert set(p["small32"])<=val and len(set(p["small32"]))==32
    raw=prep["raw_features"][:,prep["keep"]]
    median=np.nanmedian(raw[train],axis=0)
    imputed=np.where(np.isnan(raw),median,raw)
    mean=imputed[train].mean(axis=0);scale=imputed[train].std(axis=0)
    scale=np.where(scale==0,1,scale)
    assert np.allclose(median,prep["medians"],atol=1e-10,rtol=1e-10)
    assert np.allclose((imputed-mean)/scale,d["X"],atol=1e-9,rtol=1e-9)
    for s in p["seeds"]:
        expected=np.random.default_rng(s).choice(train,20,replace=False)
        assert np.array_equal(expected,p["initial_rows"][str(s)])
    for i,name in enumerate(p["ids"]):
        for a,algo in enumerate(p["algorithms"]):
            value,status=truth[(name,algo)];near(value,d["runtimes"][i,a]);assert (status=="ok")==d["ok"][i,a]
    results=[];checks=[];test_rows=[];histories=[]
    for dest in paths:
        r,c,tr,hr=audit_one(dest,p,truth)
        results.append(r);checks.append(c);test_rows.extend(tr);histories.extend(hr)
    initial_shared=[]
    for s in p["seeds"]:
        reference=OUT/"runs"/f"FREE-DYN_s{s}"
        pred=np.load(reference/"predictions_r0000.npy")
        initial_cost=read_json(reference/"history.json")[0]["train_cost_s"]
        for arm in p["arms"]:
            dest=OUT/"runs"/f"{arm}_s{s}"
            assert np.array_equal(pred,np.load(dest/"predictions_r0000.npy"))
            near(initial_cost,read_json(dest/"history.json")[0]["train_cost_s"])
        initial_shared.append(dict(seed=s,all_six_initial_predictions_equal=True,
                                   initial_cost_s=initial_cost))
    summary=[]
    for arm in p["arms"]:
        rs=[r for r in results if r["arm"]==arm]
        summary.append(dict(arm=arm,mean_par10_s=math.fsum(r["mean_test_par10_s"] for r in rs)/3,
            seed7_par10_s=rs[0]["mean_test_par10_s"],seed42_par10_s=rs[1]["mean_test_par10_s"],seed99_par10_s=rs[2]["mean_test_par10_s"],
            mean_train_cost_s=math.fsum(r["train_cost_s"] for r in rs)/3,
            mean_validation_cost_s=math.fsum(r["validation_cost_s"] for r in rs)/3,
            pooled_validation_fraction=sum(r["validation_cost_s"] for r in rs)/sum(r["total_acquisition_cost_s"] for r in rs),
            mean_remaining_budget_s=sum(r["remaining_s"] for r in rs)/3,
            mean_local_wall_s=sum(r["wall_s"] for r in rs)/3))
    free=next(z for z in summary if z["arm"]=="FREE-DYN")
    fixed=[];closure=[]
    for z in summary:
        if z["arm"]=="FREE-DYN":continue
        paired=sum(next(r for r in results if r["seed"]==s and r["arm"]=="FREE-DYN")["mean_test_par10_s"] <
                   next(r for r in results if r["seed"]==s and r["arm"]==z["arm"])["mean_test_par10_s"]-1e-9 for s in p["seeds"])
        gap=(z["mean_par10_s"]-free["mean_par10_s"])/free["mean_par10_s"]
        closure.append(dict(arm=z["arm"],relative_to_free=gap,within_2pct_or_better=gap<=0.02))
        if z["arm"].startswith("FIXED") or z["arm"]=="GEOMETRIC":
            advantage=1-free["mean_par10_s"]/z["mean_par10_s"]
            fixed.append(dict(arm=z["arm"],free_relative_advantage=advantage,
                              free_positive_seeds=paired,pass_gate=advantage>=0.05 and paired>=2))
    paid=next(z for z in summary if z["arm"]=="PAID-DYN")
    cont=all(z["pass_gate"] for z in fixed) and paid["pooled_validation_fraction"]>=0.10
    close=any(z["within_2pct_or_better"] for z in closure)
    verdict="CLOSE_CURRENT_HYPOTHESIS" if close else "CONTINUE_SCREEN" if cont else "UNDETERMINED"
    prefix=[];trial=OUT/"INVALID_serial_trial"/"FREE-DYN_s7"
    formal=OUT/"runs"/"FREE-DYN_s7"
    for f in sorted(trial.glob("predictions_r*.npy")):
        g=formal/f.name
        if g.exists():prefix.append(dict(round=int(f.stem.split("r")[-1]),equal=bool(np.array_equal(np.load(f),np.load(g)))))
    assert all(z["equal"] for z in prefix)
    audit=dict(formal_runs=18,hard_failures=0,raw_matrix_entries_checked=len(truth),
        test_predictions_checked=len(test_rows),events=sum(c["events"] for c in checks),
        observation_events=sum(c["observation_events"] for c in checks),
        max_ledger_error_s=max(c["max_ledger_error_s"] for c in checks),
        split_disjoint_and_complete=True,train_only_preprocessing_reconstructed=True,
        initial_sample_streams_reconstructed=True,
        initial_model_fairness=initial_shared,
        source_and_policy_hashes_unchanged=True,serial_parallel_prefix_checks=prefix,
        per_run=checks)
    verdict_data=dict(verdict=verdict,continuation_gate=cont,closure_gate=close,
        fixed_controls=fixed,closure_controls=closure,paid_validation_fraction=paid["pooled_validation_fraction"],
        summary=summary,warning="One scenario, one test split, three algorithmic seeds; no significance or equivalence claim")
    write_json(OUT/"FA00_audit.json",audit);write_json(OUT/"FA00_verdict.json",verdict_data)
    write_csv("FA00_per_run.csv",results);write_csv("FA00_summary.csv",summary)
    write_csv("FA00_test_predictions.csv",test_rows);write_csv("FA00_round_metrics.csv",histories)
    print(json.dumps(verdict_data,indent=2),flush=True)

if __name__=="__main__":main()
