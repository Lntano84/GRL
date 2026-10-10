"""Parent enforces 180s after import and 180s maximum import wait."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[2]
out=root/'outputs/grid00'
ready=out/'import_ready.json'
if ready.exists():
    raise RuntimeError('Prior import marker exists; archive run before rerunning')
started=time.monotonic()
deadline=started+180
import_seen=False
with (out/'probe.log').open('w',encoding='utf-8') as log:
    process=subprocess.Popen([sys.executable,str(root/'work/grid00/probe.py')],stdout=log,stderr=subprocess.STDOUT)
    while process.poll() is None:
        if ready.exists() and not import_seen:
            import_seen=True
            deadline=time.monotonic()+180
        if time.monotonic()>deadline:
            process.kill()
            process.wait()
            (out/'timeout.json').write_text(json.dumps({'import_seen':import_seen,'wall_s':time.monotonic()-started}),encoding='utf-8')
            raise TimeoutError('GRID00 watchdog limit reached')
        time.sleep(.1)
    print('exit',process.returncode,'total_wall_s',time.monotonic()-started)
    sys.exit(process.returncode)
