import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import time
from pathlib import Path
import numpy as np
from core import (T, Ledger, ReplayEnvironment, ObservationCache, Selector,
                  Exhausted, WallLimit, paid_difference)

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/"outputs"/"fa01"

def atomic_json(path,data):
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w",encoding="utf-8") as f:
        json.dump(data,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def execute(arm,seed,p,d,deadline):
    dest=OUT/"runs"/f"{arm}_s{seed}"
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/"result.json").exists():
        old=json.loads((dest/"result.json").read_text())
        if old["status"]=="complete": return old
        raise RuntimeError("Incomplete prior trajectory retained; no silent overwrite")
    started=time.perf_counter(); round_no=0; cap=100.; history=[]
    X=d["X"]; shape=d["runtimes"].shape
    train_cache=ObservationCache(shape); val_cache=ObservationCache(shape)
    log=gzip.open(dest/"actions.jsonl.gz","wt",encoding="utf-8")
    def emit(x): log.write(json.dumps(x,separators=(",",":"),allow_nan=False)+"\n")
    ledger=Ledger(p["budget_s"],emit)
    env=ReplayEnvironment(d["runtimes"],d["ok"],ledger)
    selector=Selector(X,p["algorithms"],p["train"],train_cache,seed)
    rows=p["small32"] if arm=="SMALL32-DYN" else p["validation"]
    last_predictions=None
    changed_total=0; val_comparisons=0; val_complete=0
    def save_round(pred,kind,delta=None,changed=0):
        rec=dict(round=round_no,cap_s=cap,kind=kind,
            train_cost_s=ledger.train,validation_cost_s=ledger.validation,
            remaining_s=ledger.remaining,wall_s=time.perf_counter()-started,
            fit_s=selector.fit_s,predict_s=selector.predict_s,selection_s=selector.selection_s,
            labels=sum(len(z) for z in selector.label_rows),
            exact_train_observations=int((train_cache.kind>=2).sum()),
            validation_delta_new_minus_old=delta,changed=int(changed))
        filename=f"predictions_r{round_no:04d}.npy"
        np.save(dest/filename,pred)
        rec["predictions_file"]=filename
        rec["prediction_sha256"]=hashlib.sha256((dest/filename).read_bytes()).hexdigest()
        history.append(rec)
        atomic_json(dest/"history.json",history)
        log.flush()
        # Preserve complete model and visible state after every completed fit.
        tmp=dest/"checkpoint.tmp"
        with tmp.open("wb") as f:
            pickle.dump(dict(selector=selector,train_cache=train_cache,val_cache=val_cache,
                             round=round_no,cap=cap,ledger={"train":ledger.train,"validation":ledger.validation},
                             predictions=pred),f,protocol=5)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,dest/"checkpoint.pkl")
        atomic_json(OUT/"RUN_STATE.json",dict(status="running",arm=arm,seed=seed,
            round=round_no,cap_s=cap,train_cost_s=ledger.train,validation_cost_s=ledger.validation,
            wall_s=time.perf_counter()-started))
        print(f"{arm} seed={seed} r={round_no} cap={cap:g} costs={ledger.train:.1f}+{ledger.validation:.1f} labels={rec['labels']} wall={rec['wall_s']:.1f}s",flush=True)
    status="complete"; reason="budget"; diagnostic=None
    try:
        for i in p["initial_rows"][str(seed)]:
            for a in range(shape[1]): env.observe(train_cache,i,a,100,"train")
        selector.synchronize_labels(ledger.event)
        selector.fit(0,deadline)
        last_predictions=selector.predict();save_round(last_predictions,"initial")
        if arm=="FIXED-600":cap=600.
        while ledger.remaining>1e-8:
            if time.perf_counter()>=deadline:raise WallLimit
            candidates=selector.candidates(cap)
            if not len(candidates):
                if arm in ("FIXED-100","RANDOM-100","CHEAP-100") or cap>=T:
                    reason="no_informative_candidates_at_allowed_cap";break
                cap=min(T,cap*2 if arm=="GEOMETRIC" else cap+100)
                ledger.event(dict(type="cutoff",reason="same_selector_zero_difference_no_candidates",cap_s=cap))
                continue
            round_no+=1;ledger.round=round_no;ledger.phase="train"
            batch=selector.select_batch(candidates,p["batch_size"],round_no)
            ledger.event(dict(type="batch",cap_s=cap,size=len(batch),
                              actions=[[int(r[0]),int(r[1])] for r in batch],
                              score_sha256=hashlib.sha256(batch.tobytes()).hexdigest()))
            for row in batch:
                if time.perf_counter()>=deadline:raise WallLimit
                i,pi=int(row[0]),int(row[1]);a,b=selector.pairs[pi]
                if i in selector.label_rows[pi]:continue
                env.observe(train_cache,i,a,cap,"train")
                env.observe(train_cache,i,b,cap,"train")
            selector.synchronize_labels(ledger.event)
            selector.fit(round_no,deadline)
            new_predictions=selector.predict()
            old_predictions=last_predictions
            last_predictions=new_predictions
            # Store the fitted model before validation, so budget exhaustion in
            # validation does not discard a completely trained selector.
            save_round(last_predictions,"fitted_before_validation")
            delta=None;changed=0
            if arm in ("FREE-DYN","PAID-DYN","SMALL32-DYN"):
                ledger.phase="validation";val_comparisons+=1
                if arm=="FREE-DYN":
                    delta,changed=env.free_difference(old_predictions,new_predictions,rows)
                    ledger.event(dict(type="free_validation",delta=delta,changed=changed,
                                      role="privileged",charged_s=0.0))
                else:
                    delta,changed=paid_difference(env,val_cache,old_predictions,new_predictions,rows)
                changed_total+=changed;val_complete+=1
                if delta>=-1e-9:cap=min(T,cap+100)
            elif arm=="GEOMETRIC":cap=min(T,cap*2)
            # Complete the pending history record instead of inventing another round.
            history[-1].update(validation_delta_new_minus_old=delta,changed=changed,
                next_cap_s=cap,validation_cost_s=ledger.validation,remaining_s=ledger.remaining)
            atomic_json(dest/"history.json",history)
            ledger.event(dict(type="round_end",delta=delta,changed=changed,next_cap_s=cap))
            log.flush()
    except Exhausted:
        reason="budget_during_training_or_validation"
    except WallLimit:
        status="incomplete_wall_limit";reason="global_two_hour_wall_limit"
    except Exception as exc:
        status="error";reason=repr(exc)
        diagnostic=reason
        raise
    finally:
        log.flush();log.close()
        if last_predictions is not None:
            np.save(dest/"final_predictions.npy",last_predictions)
        result=dict(arm=arm,seed=seed,status=status,stop_reason=reason,rounds=round_no,
            last_fitted_round=history[-1]["round"] if history else None,
            train_cost_s=ledger.train,validation_cost_s=ledger.validation,
            total_acquisition_cost_s=ledger.train+ledger.validation,budget_s=ledger.budget,
            remaining_s=ledger.remaining,executions=ledger.executions,cache_hits=ledger.cache_hits,
            completions=ledger.completions,cancellations=ledger.cancellations,
            validation_comparisons=val_comparisons,completed_validation_comparisons=val_complete,
            changed_validation_choices=changed_total,wall_s=time.perf_counter()-started,
            fit_s=selector.fit_s,predict_s=selector.predict_s,selection_s=selector.selection_s,
            diagnostic=diagnostic,protocol_sha256=hashlib.sha256((OUT/"FA01_protocol.json").read_bytes()).hexdigest())
        atomic_json(dest/"result.json",result)
    return result
