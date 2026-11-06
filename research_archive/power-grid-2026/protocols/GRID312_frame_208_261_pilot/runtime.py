"""Fixed three-arm physical pilot and exact original policy assembly."""
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
from types import SimpleNamespace
import uuid

R = Path("D:/Codex/GRL研究")
I = R/"work/GRL-results-integration"
A = Path(__file__).resolve().parent
O = OUT = R/"outputs/grid312_frame_208_261_pilot"
BASE = R/"outputs/grid291_control_preparation"
OLD = R/"work/grid291_control_preparation"
ARMS = ("FRAME_GUARD","PARENT208","BOUNDED261")
CAPS = {"wall_s":3600,"episode_s":1200,"physical_steps":48408,"public_forecasts":400000}
TOKEN = "GRL_GRID312_TOKEN"


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


prior = module("grid312_existing_guard",R/"work/grid301_remaining_collection/guard.py")
read, save, sha, git = prior.read, prior.save, prior.sha, prior.git
builder = module("grid312_existing_worker_builder",R/"work/grid297_paired_collection_preparation/source_builder.py")


def assemble():
    code = builder.replace_once(builder.build(),"OUT=ROOT/'outputs/grid297_paired_collection_preparation'",
                                "OUT=ROOT/'outputs/grid312_frame_208_261_pilot'")
    return builder.replace_once(code,"payload297=models297.FrozenModels(item['pool'])",
                                "payload297=None if item['arm']=='FRAME_GUARD' else models297.FrozenModels(item['pool'])")


def instrument_worker(code):
    text = builder.instrument_worker(code)
    return builder.replace_once(text,"    initial = int(env.nb_highres_called)",
        "    initial = int(env.nb_highres_called)\n    _intervention.forecasts = initial\n"
        "    if initial > _forecast_remaining: raise RuntimeError('initial forecast cap')")


def adapter_source():
    code = (OLD/"adapter.py").read_text(encoding="utf8")
    code = builder.replace_once(code,"policy=P.CompletionPolicy(item['arm'],models,N)",
        "policy=N.CompletionPolicy('FRAME_GUARD') if item['arm']=='FRAME_GUARD' else P.CompletionPolicy(item['arm'],models,N)")
    return builder.replace_once(code,"provider=N.PublicState(obs,base,actions,self.env);",
        "provider=None if item['arm']=='FRAME_GUARD' else N.PublicState(obs,base,actions,self.env);")


def install(search,item,payload,expected=None):
    assert expected is None and (payload is None)==(item["arm"]=="FRAME_GUARD")
    ns = {}
    exec(compile(adapter_source(),str(OLD/"adapter.py"),"exec"),ns)
    ns["install"](search,item,payload)

    def accept(row):
        counter.steps += 1

    def record():
        return {"steps":counter.steps,"forecasts":counter.forecasts,"new_model_fits":0}

    counter = SimpleNamespace(steps=0,forecasts=0,accept=accept,record=record)
    return counter


def check(context):
    launch = read(O/"launch_context.json")
    if any(context.get(k)!=v for k,v in launch.items()) or prior.boot_id()!=launch["boot_id"]:
        raise PermissionError("Launch identity changed")
    if (O/"STOP_REQUEST.json").exists():
        raise RuntimeError("Stop requested")
    now = time.perf_counter()
    if now < launch["start_monotonic"] or time.time() < launch["start_utc"]-2:
        raise PermissionError("Clock moved backwards")
    if now >= min(launch["start_monotonic"]+CAPS["wall_s"],context.get("worker_deadline",float("inf"))) or time.time()>=launch["start_utc"]+CAPS["wall_s"]:
        raise TimeoutError("GRID312 cap reached")


def require_publication():
    p = A/"authorization.json"
    if not p.is_file():
        raise PermissionError("GRID312 scope receipt absent before live imports")
    receipt = read(p)
    assert receipt["user_instruction"]=="来比一次吧，看看学习有没有机会打过"
    assert receipt["design_sha256"]==sha(A/"design.json") and receipt["source_sha256"]==sha(A/"source_freeze.json")
    assert receipt["preparation_sha256"]==sha(A/"offline_review.json") and read(A/"offline_review.json")["passed"]
    design = read(A/"design.json")
    assert design["caps"]==CAPS and len(design["execution_order"])==24
    head = git("rev-parse","HEAD").decode().strip()
    assert not git("status","--porcelain").strip() and git("ls-remote","lntano-src","refs/heads/main").decode().split()[0]==head
    for path in (p,A/"design.json",A/"source_freeze.json",A/"offline_review.json",Path(__file__)):
        assert git("show",head+":"+path.relative_to(I).as_posix())==path.read_bytes()
    assert (O/"design.json").read_bytes()==(A/"design.json").read_bytes()
    return design,head


def worker(index):
    design,_ = require_publication()
    assert 0 <= index < 24 and design["execution_order"][index]["index"]==index
    context = read(O/"run_context.json")
    path = O/"claims"/f"{index:02d}.json"
    assert read(path)==context and context["index"]==index
    if not os.environ.get(TOKEN) or os.environ[TOKEN]!=context["token"] or not prior.ancestor(context["manager_pid"]):
        raise PermissionError("Not the exact manager child")
    check(context)
    with path.with_suffix(".started").open("x",encoding="utf8") as stream:
        stream.write(str(os.getpid()))
    for name in ("guard","source_builder","paired_adapter"):
        sys.modules[name]=sys.modules[__name__]
    entry = str(R/"work/grid55/run.py")
    sys.argv = [entry,"--worker",str(index)]
    exec(compile(assemble(),entry,"exec"),{"__name__":"__main__","__file__":entry,
         "run_ctx":context,"check_runtime_budget":lambda:check(context)})


def folder(item):
    return O/"runs"/f"r{item['repeat']}__{item['week']}__{item['method']}"


def launch_worker(context):
    j = context["index"]
    env = {**os.environ,TOKEN:context["token"],"PYTHONDONTWRITEBYTECODE":"1",
           "TEMP":str(R/"tmp/grid100"),"TMP":str(R/"tmp/grid100")}
    with (O/f"replay_{j:02d}.log").open("x",encoding="utf8") as log:
        child = subprocess.Popen([sys.executable,"-B","-X","utf8","-u",str(Path(__file__)),"worker",str(j)],
                                 cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        save(O/"active_worker.json",{"active":True,"pid":child.pid,"index":j})
        try:
            while child.poll() is None:
                check(context)
                time.sleep(.25)
            if child.returncode:
                raise RuntimeError(f"Replay {j} exited {child.returncode}")
        except BaseException:
            if child.poll() is None:
                subprocess.run(["taskkill","/PID",str(child.pid),"/T","/F"],stdout=log,stderr=subprocess.STDOUT,check=False)
                child.wait()
            raise
        finally:
            save(O/"active_worker.json",{"active":False,"index":j})
    check(context)


def run():
    design,head = require_publication()
    assert not (O/"claims").exists() and not (O/"runs").exists()
    launch = {"token":uuid.uuid4().hex,"manager_pid":os.getpid(),"boot_id":prior.boot_id(),
              "start_monotonic":time.perf_counter(),"start_utc":time.time(),"authorization_commit":head}
    with (O/"launch_context.json").open("x",encoding="utf8") as stream:
        json.dump(launch,stream)
    records = []
    steps = forecasts = 0
    try:
        for path,expected in read(A/"source_freeze.json").items():
            assert sha(R/path)==expected,path
        for item in design["execution_order"]:
            j = item["index"]
            check(launch)
            assert j==len(records) and not folder(item).exists()
            context = {**launch,"index":j,"worker_deadline":time.perf_counter()+CAPS["episode_s"],
                       "forecast_remaining":CAPS["public_forecasts"]-forecasts,
                       "physical_remaining":CAPS["physical_steps"]-steps}
            if min(context["physical_remaining"],context["forecast_remaining"])<=0:
                raise TimeoutError("GRID312 query or step cap reached")
            p = O/"claims"/f"{j:02d}.json"
            p.parent.mkdir(exist_ok=True)
            with p.open("x",encoding="utf8") as stream:
                json.dump(context,stream)
            save(O/"run_context.json",context)
            save(O/"pipeline_state.json",{"phase":"running","index":j,"accepted":len(records)})
            launch_worker(context)
            root = folder(item)
            summary = read(root/"summary.json")
            counted = read(root/"intervention.json")
            assert summary==read(O/f"worker_result_{j:02d}.json")
            assert all(summary[k]==v for k,v in item.items())
            assert summary["new_model_fits"]==counted["new_model_fits"]==0
            assert summary["steps"]==counted["steps"]<=context["physical_remaining"]
            assert summary["public_forecasts"]<=counted["forecasts"]<=context["forecast_remaining"]
            steps += counted["steps"]
            forecasts += counted["forecasts"]
            records.append({"index":j,"summary":summary,"counted_forecasts":counted["forecasts"]})
            save(O/"manifest.json",{"records":records,"steps":steps,"forecasts":forecasts})
        check(launch)
        save(O/"finished.json",{"accepted":len(records),"steps":steps,"forecasts":forecasts,
                                 "wall_s":time.perf_counter()-launch["start_monotonic"],"new_fits":0})
        save(O/"pipeline_state.json",{"phase":"finished","accepted":len(records)})
    except BaseException:
        save(O/"collection_error.json",{"traceback":traceback.format_exc(),"accepted":len(records),
                                         "steps_accepted":steps,"forecasts_accepted":forecasts,
                                         "wall_s":time.perf_counter()-launch["start_monotonic"],"no_retry":True})
        save(O/"pipeline_state.json",{"phase":"stopped","accepted":len(records)})
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("run","worker"))
    parser.add_argument("index",nargs="?",type=int)
    args=parser.parse_args()
    try:
        if args.mode=="run":
            assert args.index is None
            run()
        else:
            worker(args.index)
    except PermissionError as exc:
        print(str(exc),file=sys.stderr)
        raise SystemExit(2)
