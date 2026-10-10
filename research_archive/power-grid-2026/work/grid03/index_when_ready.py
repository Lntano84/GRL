"""Wait for the validated assembly, then index; no independent retry/download."""
import json
import time
from pathlib import Path
from index_archive import main

state=Path(__file__).resolve().parents[2]/"outputs/grid03/download_parallel_state.json"
deadline=time.monotonic()+7200
while time.monotonic()<deadline:
    if state.exists() and json.loads(state.read_text(encoding="utf-8")).get("complete"):
        main()
        break
    time.sleep(10)
else:
    raise TimeoutError("Archive not complete; no index attempted")
