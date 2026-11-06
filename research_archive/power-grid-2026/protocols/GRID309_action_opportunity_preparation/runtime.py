"""Serial GRID309 raw-source acquisition. Every live entry fails closed."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

R = Path("D:/Codex/GRL研究")
I = R/"work/GRL-results-integration"
A = Path(__file__).resolve().parent
OUT = O = R/"outputs/grid309_action_opportunity_preparation"
ENV = R/"work/grid309_action_opportunity_preparation/train_env"
LEGACY = R/"work/grid297_paired_collection_preparation"
RECOVERY = R/"work/grid301_remaining_collection/recovery03"
GENERATOR = R/"work/grid300_new_training_dates/generate_worker.py"
TOKEN = "GRL_GRID309_TOKEN"
SCOPE = "GRID309_FOUR_FAMILY_ACTION_OPPORTUNITY_PILOT"
STAGES = {"generation":{"generation_manifest.json","generated_freeze.json"},
          "baseline":{"baseline_manifest.json","baseline_raw_manifest.json","selected_markers.json","runtime_design.json"}}


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


PRIOR_GUARD = module("grid309_prior_guard",RECOVERY.parent/"guard.py")
read, save, sha, git = PRIOR_GUARD.read, PRIOR_GUARD.save, PRIOR_GUARD.sha, PRIOR_GUARD.git
STATE = module("grid309_state",RECOVERY/"state.py")
PLAN = module("grid308_opportunity_score",A.parent/"GRID308_action_opportunity_plan/plan.py")


def reference(path):
    data = path.read_bytes()
    return {"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}


def checked_bytes(path,ref):
    data = path.read_bytes()
    if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise PermissionError(f"Archived source differs: {path}")
    return data


def check_reference_manifest(name):
    for rel,expected in read(A/name).items():
        if sha(R/rel) != expected:
            raise PermissionError(f"Frozen source differs: {rel}")


def design():
    return read(A/"design.json")


def expected_authorization():
    d = design()
    return {"scope_id":SCOPE,"design_sha256":sha(A/"design.json"),
            "source_freeze_sha256":sha(A/"source_freeze.json"),
            "preparation_review_sha256":sha(A/"results/preparation_review.json"),
            "generation_scenarios":4,"baseline_runs":4,"max_cont_runs":8,"max_total_replays":12,
            **d["caps"]}


def validate_authorization(receipt,expected):
    if receipt.get("approved_by_user") is not True:
        raise PermissionError("Explicit GRID309 approval missing")
    for key in ("user_message_reference","user_instruction"):
        if not isinstance(receipt.get(key),str) or not receipt[key].strip():
            raise PermissionError("Approval message and reference required")
    for key,value in expected.items():
        if type(receipt.get(key)) is not type(value) or receipt[key] != value:
            raise PermissionError(f"Approval differs on {key}")


def published(files):
    head = git("rev-parse","HEAD").decode().strip()
    if git("status","--porcelain").strip() or git("ls-remote","lntano-src","refs/heads/main").decode().split()[0] != head:
        raise PermissionError("Published HEAD/remote/worktree barrier")
    for path in files:
        rel = path.relative_to(I).as_posix()
        if git("show",head+":"+rel) != path.read_bytes():
            raise PermissionError(f"Published bytes differ: {rel}")
    return head


def require_authorization():
    receipt = OUT/"physical_authorization.json"
    if not receipt.is_file():
        raise PermissionError("GRID309 physical_authorization.json absent; no live imports")
    validate_authorization(read(receipt),expected_authorization())
    tracked = A/"authorization/physical_authorization.json"
    if not tracked.is_file() or tracked.read_bytes() != receipt.read_bytes():
        raise PermissionError("Exact approval must be published")
    review = A/"results/preparation_review.json"
    if read(review)["passed"] is not True:
        raise PermissionError("Offline preparation not accepted")
    head = published((tracked,A/"design.json",A/"source_freeze.json",review))
    check_reference_manifest("source_freeze.json")
    return head


def verify_clock(ctx,launch,now_mono,now_utc,boot,wall_s):
    for key,value in launch.items():
        if ctx.get(key) != value:
            raise PermissionError(f"Clock/launch differs on {key}")
    if boot != launch["boot_id"]:
        raise PermissionError("Boot changed; no restart")
    if now_mono < launch["start_monotonic"] or now_utc < launch["start_utc_epoch"]-2:
        raise PermissionError("Clock moved backwards")
    if now_mono >= min(launch["start_monotonic"]+wall_s,ctx.get("worker_deadline_monotonic",float("inf"))) or now_utc >= launch["start_utc_epoch"]+wall_s:
        raise TimeoutError("GRID309 approved time cap reached")


def check(ctx):
    if (OUT/"STOP_REQUEST.json").exists():
        raise RuntimeError("User stop; preserve work")
    launch = read(OUT/"launch_context.json")
    verify_clock(ctx,launch,time.perf_counter(),time.time(),PRIOR_GUARD.boot_id(),launch["wall_s"])


def validate_claim(ctx,receipt,token,parent_ok,kind,index):
    if (ctx != receipt or ctx.get("kind") != kind or ctx.get("index") != index
        or not token or token != ctx.get("token") or not parent_ok):
        raise PermissionError("Worker is outside the exact manager claim")


def claim(kind,index):
    require_authorization()
    launch = read(OUT/"launch_context.json")
    ctx = read(OUT/"run_context.json")
    path = OUT/"claims"/f"{kind}_{index:03d}.json"
    validate_claim(ctx,read(path),os.environ.get(TOKEN),PRIOR_GUARD.ancestor(launch["manager_pid"]),kind,index)
    if launch["authorization_sha256"] != sha(OUT/"physical_authorization.json"):
        raise PermissionError("Launch approval changed")
    check(ctx)
    with path.with_suffix(".started").open("x",encoding="utf8") as file:
        file.write(str(os.getpid()))
    return ctx


def build():
    builder = module("grid309_frozen_builder",RECOVERY/"source_builder.py")
    original = builder.build
    instrument = builder.instrument_worker

    def assemble():
        text = builder.old.replace_once(original(),"OUT=ROOT/'outputs/grid301_remaining_collection'",
                                        "OUT=ROOT/'outputs/grid309_action_opportunity_preparation'")
        return text
    def instrument_with_initial_forecasts(text):
        return builder.old.replace_once(instrument(text),
            "    initial = int(env.nb_highres_called)",
            "    initial = int(env.nb_highres_called)\n"
            "    _intervention.forecasts = initial\n"
            "    if initial > _forecast_remaining: raise RuntimeError('initial forecast cap')")
    builder.build = assemble
    builder.instrument_worker = instrument_with_initial_forecasts
    return builder


def generation_source():
    builder = build()
    src = builder.old.replace_once(GENERATOR.read_text(encoding="utf8"),
                                  "ctx=claim('generation',index)","ctx=run_ctx")
    return builder.old.replace_once(src,"schema_forecasts=int(env.nb_highres_called);env.close();check(ctx)",
        "schema_forecasts=int(env.nb_highres_called);env.close();check(ctx)\n"
        " if schema_forecasts>ctx['forecast_remaining']:raise RuntimeError('schema forecast cap')")


def folder(item):
    return OUT/"runs"/f"r{item['repeat']}__{item['week']}__{item['method']}"


def verify_stage(stage):
    release = read(OUT/f"{stage}_publication.json")
    if release.get("passed") is not True or set(release["files"]) != STAGES[stage]:
        raise PermissionError(f"Unpublished {stage} source")
    for name,expected in release["files"].items():
        archived = A/f"{stage}_public"/name
        if sha(archived) != expected or sha(OUT/name) != expected:
            raise PermissionError(f"{stage} source differs")
        if hashlib.sha256(git("show",release["commit"]+":"+archived.relative_to(I).as_posix())).hexdigest()!=expected:
            raise PermissionError(f"{stage} Git blob differs")


def verify_generated(week):
    prefix = (ENV/"chronics"/week).relative_to(R).as_posix()+"/"
    catalog = read(OUT/"generated_freeze.json")
    chosen = {path:value for path,value in catalog.items() if path.startswith(prefix)}
    actual = {p.relative_to(R).as_posix() for p in (ENV/"chronics"/week).iterdir() if p.is_file()}
    if not chosen or set(chosen) != actual or any(sha(R/p)!=h for p,h in chosen.items()):
        raise PermissionError("Generated scenario bytes differ")


def worker(kind,index):
    ctx = claim(kind,index)
    d = design()
    sys.modules["guard"] = sys.modules[__name__]
    if kind == "generation":
        assert 0 <= index < 4
        assert read(OUT/"design.json") == d and read(OUT/"calendar.json")==d["calendar"]
        sys.argv = [str(GENERATOR),"--index",str(index)]
        check(ctx)
        exec(compile(generation_source(),str(GENERATOR),"exec"),
             {"__name__":"__main__","__file__":str(GENERATOR),"run_ctx":ctx})
        return
    verify_stage("generation")
    runtime = read(OUT/"runtime_design.json")
    if index < 4:
        assert runtime == d["baseline_runtime"]
    else:
        verify_stage("baseline")
        assert read(OUT/"selected_markers.json")["coverage_passed"]
    item = runtime["execution_order"][index]
    assert index == item["index"] and index < 12
    assert item["pool"] == "f0_s0" and item["partition"] == "prospective_calibration_pilot_only"
    verify_generated(item["week"])
    sys.modules["source_builder"] = build()
    sys.modules["paired_adapter"] = module("paired_adapter",A/"adapter.py")
    entry = str(R/"work/grid55/run.py")
    sys.argv = [entry,"--worker",str(index)]
    check(ctx)
    exec(compile(sys.modules["source_builder"].build(),entry,"exec"),
         {"__name__":"__main__","__file__":entry,"run_ctx":ctx,
          "check_runtime_budget":lambda:check(ctx)})


def supervised(command,context,log_path,cwd):
    env = {**os.environ,TOKEN:context["token"],"PYTHONDONTWRITEBYTECODE":"1",
           "TEMP":str(R/"tmp/grid100"),"TMP":str(R/"tmp/grid100")}
    with log_path.open("x",encoding="utf8") as log:
        child = subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        save(OUT/"active_worker.json",{"active":True,"pid":child.pid,"kind":context["kind"],"index":context["index"]})
        try:
            while child.poll() is None:
                check(context)
                time.sleep(.25)
            if child.returncode:
                raise RuntimeError(f"{context['kind']} {context['index']} exited {child.returncode}")
        except BaseException:
            if child.poll() is None:
                subprocess.run(["taskkill","/PID",str(child.pid),"/T","/F"],stdout=log,stderr=subprocess.STDOUT,check=False)
                child.wait()
            raise
        finally:
            save(OUT/"active_worker.json",{"active":False,"kind":context["kind"],"index":context["index"]})
    check(context)


def publish_stage(stage,launch,deadline):
    check(launch)
    assert not read(OUT/"active_worker.json")["active"]
    target = A/f"{stage}_public"
    target.mkdir(exist_ok=False)
    copies = []
    for name in sorted(STAGES[stage]):
        path = target/name
        data = (OUT/name).read_bytes()
        assert len(data) < 30_000_000
        path.write_bytes(data)
        copies.append(path)
    actual = set(git("-c","core.quotepath=false","status","--short","--untracked-files=all").decode("utf8").splitlines())
    expected = {"?? "+path.relative_to(I).as_posix() for path in copies}
    if actual != expected:
        raise RuntimeError("Stage publication path whitelist differs")
    context = {**launch,"kind":"publication_"+stage,"index":0,"worker_deadline_monotonic":deadline}
    save(OUT/"pipeline_state.json",{"phase":stage+"_publication"})
    supervised(["cmd.exe","/d","/c",str(I/"scripts/git_push.cmd"),"GRID309 publish "+stage+" before next stage"],
               context,OUT/f"publish_{stage}.log",I)
    head = published(copies)
    check(context)
    save(OUT/f"{stage}_publication.json",{"passed":True,"commit":head,"files":{p.name:sha(p) for p in copies}})


def manager():
    head = require_authorization()
    d = design()
    caps = d["caps"]
    if (OUT/"claims").exists() or (OUT/"runs").exists() or any((ENV/"chronics").iterdir()):
        raise RuntimeError("Previous physical work exists; no retries")
    launch = {"manager_pid":os.getpid(),"token":uuid.uuid4().hex,"boot_id":PRIOR_GUARD.boot_id(),
              "start_monotonic":time.perf_counter(),"start_utc_epoch":time.time(),"wall_s":caps["wall_s"],
              "authorization_sha256":sha(OUT/"physical_authorization.json"),"authorization_commit":head}
    with (OUT/"launch_context.json").open("x",encoding="utf8") as file:
        json.dump(launch,file)
    if (OUT/"STOP_REQUEST.json").exists():
        raise RuntimeError("User stop")
    from adapter import select
    generation,records,steps,forecasts = [],[],0,0
    def child(kind,index,deadline):
        if forecasts >= caps["all_forecasts"] or (kind=="replay" and steps>=caps["new_physical_steps"]):
            raise TimeoutError("GRID309 physical or forecast cap reached")
        ctx = {**launch,"kind":kind,"index":index,"worker_deadline_monotonic":deadline,
               "physical_remaining":caps["new_physical_steps"]-steps,"forecast_remaining":caps["all_forecasts"]-forecasts}
        check(ctx)
        path = OUT/"claims"/f"{kind}_{index:03d}.json"
        path.parent.mkdir(exist_ok=True)
        with path.open("x",encoding="utf8") as file:
            json.dump(ctx,file)
        save(OUT/"run_context.json",ctx)
        supervised([sys.executable,"-B","-X","utf8","-u",str(Path(__file__).resolve()),"worker",kind,str(index)],
                   ctx,OUT/f"{kind}_{index:03d}.log",R)

    def replay(item):
        nonlocal steps,forecasts
        index = item["index"]
        assert index == len(records) and not folder(item).exists()
        save(OUT/"pipeline_state.json",{"phase":"baseline" if index<4 else "CONT","index":index,"accepted":len(records)})
        child("replay",index,time.perf_counter()+caps["per_replay_s"])
        root = folder(item)
        summary = read(root/"summary.json")
        receipt = read(root/"intervention.json")
        assert summary == read(OUT/f"worker_result_{index:02d}.json")
        assert summary["steps"] == receipt["steps"] and summary["new_model_fits"] == receipt["new_cost_labels"] == 0
        if item["target"] is not None:
            assert receipt["prefix_matched"] and receipt["event"]["common"] == item["target"]["common"]
            assert receipt["event"]["forced_continue"] and receipt["event"]["delivered"] is not None
        else:
            assert receipt["status"] == "BASELINE261_SOURCE" and receipt["event"] is None
        steps += summary["steps"]
        forecasts += receipt["forecasts"]
        assert steps <= caps["new_physical_steps"] and forecasts <= caps["all_forecasts"]
        records.append({"index":index,"date":item["date"],"summary":summary,"event":receipt["event"],
                        "opportunities":receipt["opportunities"],"forecasts":receipt["forecasts"]})
        save(OUT/"manifest.json",{"records":records,"physical_steps":steps,"forecasts":forecasts})

    try:
        gen_deadline = launch["start_monotonic"]+caps["generation_s"]
        for i,item in enumerate(d["calendar"]["dates"]):
            save(OUT/"pipeline_state.json",{"phase":"generation","index":i,"accepted":len(generation)})
            child("generation",i,min(gen_deadline,time.perf_counter()+caps["per_generation_s"]))
            row = read(OUT/"generation_receipts"/f"{i:03d}.json")
            assert row["item"] == item and row["new_model_predictions"] == row["new_policy_steps"] == 0
            generation.append(row)
            forecasts += row["schema_forecast_calls"]
            assert forecasts <= caps["all_forecasts"]
            save(OUT/"generation_manifest.json",{"records":generation,"schema_forecasts":forecasts})
        generated = {path:h for row in generation for path,h in row["files"].items()}
        assert generated and sum(len(row["files"]) for row in generation)==len(generated)
        save(OUT/"generated_freeze.json",generated)
        publish_stage("generation",launch,gen_deadline)
        for item in d["baseline_runtime"]["execution_order"]:
            replay(item)
        review, selection = select(records)
        assert len(selection) <= 8
        runtime = d["baseline_runtime"]
        original = list(runtime["execution_order"])
        opposites = []
        for base_index,role,row in selection:
            base = original[base_index]
            opposites.append({**base,"index":4+len(opposites),"pair_index":base_index,"arm":"CONT_NEXT",
                              "method":f"CONT_NEXT_{role}__{base['pool']}","target":row["marker"],
                              "baseline_folder":folder(base).relative_to(R).as_posix(),"stratum":role})
        runtime["execution_order"] = original+opposites
        raw = {p.relative_to(R).as_posix():reference(p)
               for item in original for p in sorted(folder(item).iterdir()) if p.is_file()}
        save(OUT/"baseline_raw_manifest.json",raw)
        save(OUT/"baseline_manifest.json",{"records":records.copy(),"physical_steps":steps,"forecasts":forecasts})
        save(OUT/"selected_markers.json",{"selection":opposites,**review})
        save(OUT/"runtime_design.json",runtime)
        publish_stage("baseline",launch,launch["start_monotonic"]+caps["wall_s"])
        for item in opposites:
            replay(item)
        check(launch)
        save(OUT/"finished.json",{"generation_scenarios":len(generation),"baseline_runs":4,
             "cont_runs":len(opposites),"coverage_passed":review["coverage_passed"],"physical_steps":steps,
             "forecasts":forecasts,"new_cost_labels":0,"new_fits":0,"wall_s":time.perf_counter()-launch["start_monotonic"]})
        save(OUT/"pipeline_state.json",{"phase":"source_collection_finished","accepted":len(records)})
    except BaseException:
        save(OUT/"collection_error.json",{"traceback":traceback.format_exc(),"accepted":len(records),
             "completed_generation":len(generation),"accepted_physical_steps":steps,"accepted_forecasts":forecasts,
             "preserve_partial":True,"no_retry":True})
        save(OUT/"pipeline_state.json",{"phase":"stopped","accepted":len(records)})
        raise


if __name__=="__main__":
    sys.modules["runtime"] = sys.modules[__name__]
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("run","worker"))
    parser.add_argument("kind",nargs="?",choices=("generation","replay"))
    parser.add_argument("index",nargs="?",type=int)
    args=parser.parse_args()
    try:
        if args.mode=="run":
            if args.kind is not None:
                parser.error("run accepts no worker arguments")
            manager()
        else:
            if args.kind is None or args.index is None:
                parser.error("worker requires kind and index")
            worker(args.kind,args.index)
    except PermissionError as err:
        print(str(err),file=sys.stderr)
        raise SystemExit(2)
