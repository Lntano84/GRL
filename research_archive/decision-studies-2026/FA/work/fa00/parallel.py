"""Execution-only parallel orchestration; every trajectory keeps one-thread RFs."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import run
import numpy as np

OUT=run.OUT

def worker(arm,seed,seconds):
    p=json.loads((OUT/"FA00_protocol.json").read_text())
    d=dict(np.load(OUT/"FA00_data.npz"))
    dest=OUT/"runs"/f"{arm}_s{seed}"
    original=run.atomic_json
    def isolated_state(path,data):
        if path.name=="RUN_STATE.json": path=dest/"RUN_STATE.json"
        original(path,data)
    run.atomic_json=isolated_state
    run.execute(arm,seed,p,d,time.perf_counter()+seconds)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--worker",action="store_true")
    ap.add_argument("--arm");ap.add_argument("--seed",type=int);ap.add_argument("--seconds",type=float)
    ap.add_argument("--workers",type=int,default=6);args=ap.parse_args()
    if args.worker:return worker(args.arm,args.seed,args.seconds)
    p=json.loads((OUT/"FA00_protocol.json").read_text())
    metadata=json.loads((OUT/"FA00_execution_schedule.json").read_text())
    started=time.perf_counter();deadline=started+metadata["remaining_wall_limit_s"]
    order=[(a,s) for s in p["seeds"] for a in p["arms"]]
    def launch(item):
        a,s=item;dest=OUT/"runs"/f"{a}_s{s}";dest.mkdir(parents=True,exist_ok=True)
        remaining=deadline-time.perf_counter()
        if remaining<=0:return dict(arm=a,seed=s,not_started=True)
        command=[sys.executable,str(Path(__file__).resolve()),"--worker","--arm",a,"--seed",str(s),"--seconds",str(remaining)]
        with (dest/"console.log").open("w",encoding="utf-8") as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:code=process.wait(timeout=max(0,deadline-time.perf_counter())+15)
            except subprocess.TimeoutExpired:
                process.kill();process.wait();code=-999
        path=dest/"result.json"
        result=json.loads(path.read_text()) if path.exists() else dict(arm=a,seed=s,status="missing_result")
        result["process_exit_code"]=code
        return result
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(launch,item):item for item in order}
        for future in concurrent.futures.as_completed(futures):
            r=future.result();completed.append(r)
            run.atomic_json(OUT/"FA00_batch.json",dict(results=completed,
                completed=sum(z.get("status")=="complete" for z in completed),formal_target=18,
                parallel_wall_s=time.perf_counter()-started,workers=args.workers))
            print(r.get("arm"),r.get("seed"),r.get("status"),r.get("wall_s"),flush=True)
            run.atomic_json(OUT/"RUN_STATE.json",dict(status="running",completed=len(completed),target=18,
                parallel_wall_s=time.perf_counter()-started,workers=args.workers))
    done=sum(z.get("status")=="complete" for z in completed)
    run.atomic_json(OUT/"RUN_STATE.json",dict(status="complete" if done==18 else "incomplete",
        completed=done,target=18,parallel_wall_s=time.perf_counter()-started,unfinished_process=False))

if __name__=="__main__":main()
