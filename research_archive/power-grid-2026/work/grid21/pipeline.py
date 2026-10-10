"""Run the frozen V1 fits, independent saved-data checks and closeout once."""
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid21"
start = time.perf_counter()
assert not (OUT / "finished.json").exists()
for name in ["run", "audit", "closeout"]:
    log = OUT / (name + ".log")
    assert not log.exists(), log
    print(json.dumps({"starting": "grid21/" + name}), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, "-u", str(ROOT / "work/grid21" / (name + ".py"))], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    print(json.dumps({"finished": "grid21/" + name, "exit_code": result.returncode}), flush=True)
    assert result.returncode == 0, str(log)
print(json.dumps({"pipeline_complete": True, "wall_s": time.perf_counter()-start}), flush=True)
