"""Materialize only frozen static inputs, then qualify without live imports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from adapter import PilotState, select
from runtime import (A, ENV, GENERATOR, I, LEGACY, OUT, R, RECOVERY, SCOPE, STATE,
                     build, expected_authorization, generation_source, git, published,
                     read, save, sha, validate_authorization, validate_claim, verify_clock)

PLAN_SOURCE = A.parent/"GRID308_action_opportunity_plan/fixed_plan.json"
META = R/"outputs/grid301_remaining_collection"
BASE = R/"work/grid100_data_foundation/trainval_env"
OLD_ENV = R/"work/grid300_new_training_dates/train_env"
OLD_META = R/"outputs/grid300_new_training_dates"
LEGACY_META = R/"outputs/grid297_paired_collection_preparation"
RESULTS = A/"results"


def freeze():
    assert not OUT.exists() and not ENV.exists() and not (A/"source_freeze.json").exists()
    plan = read(PLAN_SOURCE)
    assert plan["status"] == "DESIGN_ONLY_NOT_IMPLEMENTED_NOT_APPROVED"
    assert plan["generation_scenarios"] == plan["baselines"] == 4
    assert plan["max_cont_replays"] == 8 and plan["max_total_replays"] == 12
    caps = plan["caps"]
    assert caps == dict(wall_s=7200,generation_s=2700,per_generation_s=1800,per_replay_s=1200,
                        new_physical_steps=24204,all_forecasts=180000)
    calendar = {"dates":plan["calendar"]}
    assert len({d["scenario"] for d in calendar["dates"]})==4
    assert all(d["model_pool"]=="f0_s0" and d["scenario_id"]==100 and d["environment_seed"]==0
               for d in calendar["dates"])

    old_code = read(META/"code_freeze.json")
    assert all(sha(R/p)==h for p,h in old_code.items())
    model = plan["model_reference"]
    assert model["pool"] == "f0_s0"
    all_refs = read(META/"model_references.json")
    assert [m for m in all_refs if m["pool"]=="f0_s0"] == [model]
    old_bindings = read(LEGACY_META/"published_model_bindings.json")
    model_blobs = {}
    head = git("rev-parse","HEAD").decode().strip()
    assert git("show",head+":"+PLAN_SOURCE.relative_to(I).as_posix())==PLAN_SOURCE.read_bytes()
    for path,h in model["hashes"].items():
        binding = old_bindings[path]
        assert binding["sha256"]==h==sha(R/path)
        data = git("show",head+":"+binding["archive_path"])
        assert hashlib.sha256(data).hexdigest()==h
        model_blobs[path]=binding

    previous = read(OLD_META/"static_freeze.json")
    originals = [p for p in sorted(BASE.iterdir()) if p.is_file() and p.suffix in (".json",".csv",".py")]
    bootstrap = [p for p in sorted((BASE/"chronics/2035-01-01_8").iterdir()) if p.is_file()]
    assert len(originals)==18 and bootstrap
    for p in originals+bootstrap:
        assert previous[p.relative_to(R).as_posix()]==sha(p)
    for p in originals:
        other = OLD_ENV/p.name
        assert previous[other.relative_to(R).as_posix()] == sha(other)
        if p.name!="scenario_params.json":
            assert sha(p)==sha(other)

    template = next(x for x in read(META/"design.json")["baseline_runtime"]["execution_order"]
                    if x["pool"]=="f0_s0")
    assert template["target"] is None and template["KEEP"] is False
    baselines = []
    for index,row in enumerate(calendar["dates"]):
        baselines.append({**template,"index":index,"repeat":0,"pair_index":index,"within_pair":0,
                          "date":row["date"],"week":row["scenario"],"method":"BASELINE261__f0_s0",
                          "partition":row["role"],"target":None})
    env_rel = ENV.relative_to(R).as_posix()
    runtime = {"environment":env_rel,
               "caps":{"episode_s":caps["per_replay_s"],"wall_s":caps["wall_s"],
                       "physical_steps":caps["new_physical_steps"],"public_forecasts":caps["all_forecasts"]},
               "timing":"Original frozen control and simulator with setup inside the total window.",
               "execution_order":baselines}
    d = {"scope_id":SCOPE,"status":"PREPARATION_ONLY_NO_PHYSICAL_APPROVAL","environment":env_rel,
         "caps":caps,"calendar":calendar,"baseline_runtime":runtime,"generation_scenarios":4,
         "max_cont_runs":8,"max_replays":12,
         "scientific_plan_sha256":sha(PLAN_SOURCE),"reference_rule":plan["reference_rule"],
         "minimum_coverage_before_CONT":plan["minimum_coverage_before_CONT"]}

    OUT.mkdir()
    (ENV/"chronics").mkdir(parents=True)
    for p in originals:
        shutil.copy2(p,ENV/p.name)
    params = read(BASE/"scenario_params.json")
    params["all_dates"] = [row["date"] for row in calendar["dates"]]
    save(ENV/"scenario_params.json",params)
    save(A/"design.json",d)
    (OUT/"design.json").write_bytes((A/"design.json").read_bytes())
    save(OUT/"calendar.json",calendar)
    save(OUT/"runtime_design.json",runtime)
    save(OUT/"model_references.json",[model])
    save(A/"published_model_bindings.json",model_blobs)
    freeze_paths = [PLAN_SOURCE,GENERATOR,RECOVERY/"source_builder.py",RECOVERY/"state.py",
                    RECOVERY.parent/"guard.py",LEGACY/"adapter.py",LEGACY/"intervention.py",
                    LEGACY/"source_builder.py",A/"design.json",A/"published_model_bindings.json",
                    OUT/"design.json",OUT/"calendar.json",OUT/"model_references.json",
                    *A.glob("*.py"),*originals,*bootstrap,*[p for p in ENV.iterdir() if p.is_file()]]
    source = {**old_code,**model["hashes"],**{p.relative_to(R).as_posix():sha(p) for p in freeze_paths}}
    save(OUT/"code_freeze.json",source)
    save(A/"source_freeze.json",{**source,(OUT/"code_freeze.json").relative_to(R).as_posix():sha(OUT/"code_freeze.json")})
    print(f"GRID309 frozen: {len(source)} dependencies, {len(model_blobs)} exact original f0_s0 blobs, 4 dates")


def reject(fn,error):
    try:
        fn()
    except error:
        return
    raise AssertionError(f"{error.__name__} expected")


def row(step,call,batch,length,score,state_hash=None):
    return {"marker":{"ordinary261_continue":False,"common":{
        "step":step,"call_index":call,"batch":batch,"state_hash":state_hash or f"h{step}",
        "next_ids":list(range(length))}},"score":list(score)}


def check_selection():
    events = [row(10,0,0,64,(64,40)),row(10,1,0,64,(64,99)),row(20,2,0,64,(64,80)),
              row(25,3,0,21,(21,1)),row(30,4,0,64,(64,80)),row(40,5,0,64,(0,0))]
    records = [{"index":i,"date":str(i),"opportunities":events} for i in range(4)]
    result, chosen = select(records)
    assert result["coverage_passed"] and len(chosen)==8
    assert all(d["primary"]==events[2] and d["reference"]==events[0] and d["eligible_states"]==4
               for d in result["dates"])
    with_cost = [{**r,"summary":{"cost":(-1)**r["index"]*2000}} for r in records]
    assert select(with_cost)==(result,chosen)
    assert select(records[:2])[1]==[]
    primary_only = [{**r,"opportunities":[events[2]]} for r in records]
    assert select(primary_only)[1]==[]
    partial_reference = records[:2]+primary_only[2:]
    report, selected = select(partial_reference)
    assert report["coverage_passed"] and len(selected)==6
    return ["eight bounded selections","exact state dedup and chronological ties","same-batch lower-score reference",
            "cost ignored","too few families stops","too few references stops","missing references stay empty"]


def check_intervention():
    public = {"anchor_action":[0]*34+[1]*34,"context_action":[[0]*34+[1]*34],
              "next_action":[[0,1]+[0]*15+[0,1]+[0]*15+[1]*34]}
    raw = [0.]*41
    native = {"native_q":[1.,0.],"NN_input":public}
    stop = {"continue_":False,"log":{}}
    continue_ = {"continue_":True,"log":{}}
    base = PilotState({"arm":"BASELINE261","target":None})
    base.begin({"step":1,"state_hash":"before"})
    base.observe([],[],{},[],0.,[],[7],1,0)
    assert base.force(raw,native,stop) is stop
    assert len(base.opportunities)==1 and base.opportunities[0]["score"][0]==1
    marker = base.opportunities[0]["marker"]
    target = PilotState({"arm":"CONT_NEXT","target":marker},[])
    target.begin({"step":1,"state_hash":"before"})
    target.offer = base.offer
    assert target.force(raw,native,stop)["continue_"] is True
    target.before(1,"before","next_action")
    assert target.force(raw,native,stop) is stop
    other = PilotState({"arm":"CONT_NEXT","target":marker},[])
    other.begin({"step":1,"state_hash":"wrong"})
    other.offer = base.offer
    reject(lambda:other.force(raw,native,stop),RuntimeError)
    reject(lambda:other.before(1,"wrong","other"),RuntimeError)
    baseline_continue = PilotState({"arm":"BASELINE261","target":None})
    baseline_continue.begin({"step":1,"state_hash":"before"})
    baseline_continue.offer=base.offer
    assert baseline_continue.force(raw,native,continue_) is continue_ and not baseline_continue.opportunities
    return ["original baseline decisions","pure scoring of STOP only","one flip then original behavior",
            "wrong full public marker refuses flip","missing marker refuses physical step"]


def qualify():
    assert not RESULTS.exists() and not (OUT/"physical_authorization.json").exists()
    assert not any((ENV/"chronics").iterdir())
    head = published([*A.glob("*.py"),A/"source_freeze.json",A/"design.json",A/"published_model_bindings.json"])
    start = time.perf_counter()
    results = {}
    results["selection"] = check_selection()
    results["intervention"] = check_intervention()
    builder = build()
    compile(builder.build(),"grid309_generated_run.py","exec")
    compile(builder.instrument_worker((R/"work/grid55/worker.py").read_text(encoding="utf8")),
            "grid309_instrumented_worker.py","exec")
    compile(generation_source(),"grid309_generated_scenario.py","exec")
    results["assembled_sources_compiled"] = 3
    expected = {"scope_id":SCOPE,"design_sha256":sha(A/"design.json"),"source_freeze_sha256":sha(A/"source_freeze.json"),
                "preparation_review_sha256":"0"*64,"generation_scenarios":4,"baseline_runs":4,
                "max_cont_runs":8,"max_total_replays":12,**read(A/"design.json")["caps"]}
    receipt = {**expected,"approved_by_user":True,"user_message_reference":"synthetic only","user_instruction":"synthetic only"}
    validate_authorization(receipt,expected)
    for key in receipt:
        broken = {**receipt}
        del broken[key]
        reject(lambda:validate_authorization(broken,expected),PermissionError)
    results["rejected_missing_authorization_fields"] = len(receipt)
    launch = {"boot_id":"boot","start_monotonic":10.,"start_utc_epoch":100.,"wall_s":7200}
    ctx = {**launch,"worker_deadline_monotonic":110.}
    verify_clock(ctx,launch,109.,199.,"boot",7200)
    reject(lambda:verify_clock(ctx,launch,110.,200.,"boot",7200),TimeoutError)
    reject(lambda:verify_clock(ctx,launch,20.,7300.,"boot",7200),TimeoutError)
    reject(lambda:verify_clock(ctx,launch,20.,110.,"changed",7200),PermissionError)
    reject(lambda:verify_clock({**ctx,"start_monotonic":11.},launch,20.,110.,"boot",7200),PermissionError)
    claim = {**ctx,"kind":"replay","index":2,"token":"test-token"}
    validate_claim(claim,claim,"test-token",True,"replay",2)
    reject(lambda:validate_claim(claim,claim,"test-token",False,"replay",2),PermissionError)
    reject(lambda:validate_claim(claim,claim,"wrong",True,"replay",2),PermissionError)
    reject(lambda:validate_claim(claim,claim,"test-token",True,"generation",2),PermissionError)
    reject(lambda:validate_claim(claim,claim,"test-token",True,"replay",3),PermissionError)
    results["clock_claim_cases"] = 10
    outcomes = []
    for args in (["run"],["worker","generation","0"],["worker","replay","0"],["worker","replay","4"]):
        p = subprocess.run([sys.executable,"-B","-X","utf8",str(A/"runtime.py"),*args],
                           cwd=R,capture_output=True,text=True,encoding="utf8",timeout=15)
        assert p.returncode==2 and "physical_authorization.json absent; no live imports" in p.stderr
        outcomes.append({"args":args,"exit_code":p.returncode,"guard_before_live_imports":True})
    results["live_entries_refused"] = outcomes
    assert all(name not in sys.modules for name in ("grid2op","torch","lightsim2grid","chronix2grid"))
    assert not any((ENV/"chronics").iterdir()) and not (OUT/"claims").exists() and not (OUT/"launch_context.json").exists()
    results.update(passed=True,source_commit=head,live_generation=0,live_model_loads=0,new_predictions=0,
                   physical_steps=0,new_cost_labels=0,new_fits=0,wall_s=time.perf_counter()-start)
    save(RESULTS/"preparation_review.json",results)
    save(OUT/"preparation_review.json",results)
    save(OUT/"authorization_required.json",expected_authorization())
    print(json.dumps({k:results[k] for k in ("passed","assembled_sources_compiled","rejected_missing_authorization_fields","clock_claim_cases","wall_s")},ensure_ascii=False))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("freeze","qualify"))
    {"freeze":freeze,"qualify":qualify}[parser.parse_args().phase]()
