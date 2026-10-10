"""Serial local continuation; not a recurring automation or background monitor."""
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid20"
started = time.perf_counter()
print("Waiting for the already-running frozen GRID19 collection; no duplicate producer.",flush=True)
while not (ROOT / "outputs/grid19/finished.json").exists():
    assert time.perf_counter()-started < 7200, "Teacher completion not observed within pipeline wait cap."
    if list((ROOT / "outputs/grid19/runs").glob("*/failure.json")):
        raise RuntimeError("Teacher engineering failure recorded; do not start fitting.")
    time.sleep(5)
for name in ["grid19/audit","grid19/closeout","grid20/run","grid20/audit","grid20/closeout"]:
    log = ROOT / "outputs" / name.split('/')[0] / (name.split('/')[1]+".log")
    print(json.dumps({"starting":name}),flush=True)
    with log.open("w",encoding="utf-8") as stream:
        result = subprocess.run([sys.executable,"-u",str(ROOT / "work" / (name+".py"))],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    print(json.dumps({"finished":name,"exit_code":result.returncode}),flush=True)
    assert result.returncode == 0, f"Stage {name} failed; see {log}"
print(json.dumps({"pipeline_complete":True,"wall_s":time.perf_counter()-started}),flush=True)
