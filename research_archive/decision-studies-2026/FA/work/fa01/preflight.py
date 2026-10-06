import copy
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from policy import ControlSelector, choose_indices
from core import Selector, Ledger, ReplayEnvironment, ObservationCache

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OUT=PROJECT/"outputs"/"fa01"
OLD=PROJECT/"outputs"/"fa00"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")

def main():
    assert not(OUT/"FA01_freeze_manifest.json").exists()
    p=json.loads((OUT/"FA01_protocol.json").read_text(encoding="utf-8"))
    checks={}
    fixture=np.array([[9,1,.49,30],[2,0,.1,4],[4,3,.01,2],[7,2,.3,4],[5,1,.4,20]],float)
    for mode in p["arms"]:
        picked=choose_indices(fixture,3,7,1,mode)
        assert len(picked)==len(set(picked))==3
        assert set(choose_indices(fixture,99,7,1,mode))==set(range(5))
        mutated=fixture.copy();mutated[:,2]=[.8,.9,.4,.7,.3]
        if mode=="RANDOM-100":mutated[:,3]=[500,300,0,1,2]
        assert np.array_equal(fixture[picked,:2],mutated[choose_indices(mutated,3,7,1,mode),:2])
        rev=fixture[::-1].copy()
        assert np.array_equal(fixture[picked,:2],rev[choose_indices(rev,3,7,1,mode),:2])
        checks[mode+"_score_and_row_order_invariance"]=True
        checks[mode+"_unique_full_and_partial_batches"]=True
    chosen=fixture[choose_indices(fixture,3,7,1,"CHEAP-100"),3]
    assert sorted(chosen.tolist())==[2,4,4]
    checks["cheap_selects_exact_three_lowest_costs"]=True
    tied=fixture.copy();tied[:,3]=4
    sets={tuple(sorted(choose_indices(tied,2,s,1,"CHEAP-100"))) for s in [7,42,99]}
    assert len(sets)>1
    checks["equal_cost_ties_use_seed_not_identifiers"]=True
    d=dict(np.load(OLD/"FA00_data.npz"))
    logs=[];ledger=Ledger(p["budget_s"],logs.append)
    env=ReplayEnvironment(d["runtimes"],d["ok"],ledger)
    cache=ObservationCache(d["runtimes"].shape)
    for i in p["initial_rows"]["7"]:
        for a in range(len(p["algorithms"])):env.observe(cache,i,a,100,"train")
    model=Selector(d["X"],p["algorithms"],p["train"],cache,7)
    model.synchronize_labels();model.fit(0)
    pred=model.predict()
    expected=np.load(OLD/"runs"/"FIXED-100_s7"/"predictions_r0000.npy")
    assert np.array_equal(pred,expected)
    checks["actual_initial_model_matches_inherited_baseline"]=True
    candidate=model.candidates(100)
    rows=set(map(tuple,candidate[:,:2].astype(int)))
    for mode in p["arms"]:
        control=ControlSelector(d["X"],p["algorithms"],p["train"],copy.deepcopy(cache),7,mode=mode)
        control.label_rows=copy.deepcopy(model.label_rows)
        control.pair_models=copy.deepcopy(model.pair_models)
        control.regressors=copy.deepcopy(model.regressors)
        assert np.array_equal(control.predict(),expected)
        assert np.array_equal(control.candidates(100),candidate)
        batch=control.select_batch(candidate,p["batch_size"],1)
        assert len(batch)==p["batch_size"] and len(set(map(tuple,batch[:,:2])))==len(batch)
        assert set(map(tuple,batch[:,:2].astype(int)))<=rows
        # Two actual environments differ on unrevealed outcomes only. Both
        # produce the same timeout feedback through the ordinary observe API.
        worlds=[]
        for hidden in [500.,599.]:
            times=d["runtimes"].copy();ok=d["ok"].copy()
            unknown=cache.kind==0;times[unknown]=hidden;ok[unknown]=True
            eventlog=[];led=Ledger(p["budget_s"],eventlog.append)
            led.train=ledger.train
            k=copy.deepcopy(cache)
            e=ReplayEnvironment(times,ok,led)
            i,pi=int(batch[0,0]),int(batch[0,1]);a,b=control.pairs[pi]
            e.observe(k,i,a,100,"train");e.observe(k,i,b,100,"train")
            visible=copy.deepcopy(control);visible.cache=k
            nxt=visible.select_batch(visible.candidates(100),2,2)
            worlds.append((eventlog,k.snapshot(),led.remaining,nxt[:,:2].tolist()))
        assert worlds[0]==worlds[1]
        checks[mode+"_two_actual_worlds_feedback_state_action_invariance"]=True
        checks[mode+"_actual_candidates_equal_FA00"]=True
    import runner_base
    assert runner_base.OUT==OUT
    text=(ROOT/"runner_base.py").read_text(encoding="utf-8")
    assert '"FA01_protocol.json"' in text
    assert 'if arm in ("FIXED-100","RANDOM-100","CHEAP-100") or cap>=T:' in text
    checks["runner_output_isolation_and_fixed_cap_exhaustion"]=True
    result=dict(hard_failures=0,checks=checks,initial_cost_s=ledger.train,
                test_outcomes_read=False,new_formal_trajectories_started=False)
    write(OUT/"FA01_preflight.json",result)
    inherited=json.loads((OUT/"FA01_inherited_manifest.json").read_text())
    for path,digest in inherited.items():assert sha(PROJECT/path)==digest,path
    files=list(ROOT.glob("*.py"))+[OUT/"FA01_protocol.json",OUT/"FA01_preflight.json",OUT/"FA01_inherited_manifest.json"]
    freeze=dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                files={str(f.relative_to(PROJECT)):sha(f) for f in files},
                inherited_files=inherited,comparison="development screening, not independent confirmation")
    write(OUT/"FA01_freeze_manifest.json",freeze)
    write(OUT/"RUN_STATE.json",dict(status="frozen_ready",target=6,preflight_hard_failures=0))
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
