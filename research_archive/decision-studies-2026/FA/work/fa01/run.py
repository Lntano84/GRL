"""Six independent trajectories; reuse the FA00 loop with fixed 100s controls."""
import argparse
import concurrent.futures
import functools
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from policy import ControlSelector
import runner_base

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OUT=PROJECT/"outputs"/"fa01"
OLD=PROJECT/"outputs"/"fa00"

def read(path):return json.loads(path.read_text(encoding="utf-8"))
def verify_freeze():
    frozen=read(OUT/"FA01_freeze_manifest.json")
    for section in ["files","inherited_files"]:
        for path,digest in frozen[section].items():
            assert hashlib.sha256((PROJECT/path).read_bytes()).hexdigest()==digest,path

def worker(arm,seed,seconds):
    verify_freeze()
    p=read(OUT/"FA01_protocol.json")
    d=dict(np.load(OLD/"FA00_data.npz"))
    runner_base.Selector=functools.partial(ControlSelector,mode=arm)
    original=runner_base.atomic_json
    dest=OUT/"runs"/f"{arm}_s{seed}"
    def isolated_state(path,data):
        if path.name=="RUN_STATE.json":path=dest/"RUN_STATE.json"
        original(path,data)
    runner_base.atomic_json=isolated_state
    runner_base.execute(arm,seed,p,d,time.perf_counter()+seconds)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--worker",action="store_true")
    ap.add_argument("--arm");ap.add_argument("--seed",type=int);ap.add_argument("--seconds",type=float)
    args=ap.parse_args()
    if args.worker:return worker(args.arm,args.seed,args.seconds)
    verify_freeze();p=read(OUT/"FA01_protocol.json")
    assert read(OUT/"FA01_preflight.json")["hard_failures"]==0
    assert not(OUT/"FA01_batch.json").exists(),"No silent rerun or overwrite"
    started=time.perf_counter();deadline=started+p["wall_limit_s"]
    order=[(a,s) for s in p["seeds"] for a in p["arms"]]
    def launch(item):
        arm,seed=item;dest=OUT/"runs"/f"{arm}_s{seed}"
        dest.mkdir(parents=True,exist_ok=True)
        assert not(dest/"result.json").exists()
        seconds=max(0,deadline-time.perf_counter())
        command=[sys.executable,str(Path(__file__).resolve()),"--worker","--arm",arm,"--seed",str(seed),"--seconds",str(seconds)]
        with (dest/"console.log").open("w",encoding="utf-8") as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:code=proc.wait(timeout=max(0,deadline-time.perf_counter())+15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait();code=-999
        r=read(dest/"result.json") if (dest/"result.json").exists() else dict(arm=arm,seed=seed,status="missing_result")
        r["process_exit_code"]=code
        return r
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=p["workers"]) as pool:
        futures=[pool.submit(launch,item) for item in order]
        for fut in concurrent.futures.as_completed(futures):
            r=fut.result();completed.append(r)
            runner_base.atomic_json(OUT/"FA01_batch.json",dict(results=completed,
                completed=sum(z.get("status")=="complete" and z.get("process_exit_code")==0 for z in completed),
                formal_target=6,parallel_wall_s=time.perf_counter()-started,workers=p["workers"]))
            print(r.get("arm"),r.get("seed"),r.get("status"),r.get("wall_s"),flush=True)
            runner_base.atomic_json(OUT/"RUN_STATE.json",dict(status="running",completed=len(completed),target=6,
                parallel_wall_s=time.perf_counter()-started))
    done=sum(z.get("status")=="complete" and z.get("process_exit_code")==0 for z in completed)
    runner_base.atomic_json(OUT/"RUN_STATE.json",dict(status="complete" if done==6 else "incomplete",
        completed=done,target=6,parallel_wall_s=time.perf_counter()-started,unfinished_process=False))

if __name__=="__main__":main()
