"""Observe remaining-phase process wall time without changing frozen timed control."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid34'
assert json.loads((OUT/'pilot_audit.json').read_text())['all_rankers_qualified']
assert not (OUT/'remaining_phase_timing.json').exists() and not (OUT/'finished.json').exists()
stamp=time.perf_counter();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (OUT/'run_remaining.log').open('w',encoding='utf-8') as f:
    proc=subprocess.run([sys.executable,str(ROOT/'work/grid34/run.py'),'--phase','remaining'],cwd=ROOT,
        stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
result={'phase':'remaining_only','started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'process_wall_s':time.perf_counter()-stamp,'exit_code':proc.returncode,
        'scope':'Only36remainingworkers plus this driver startup, not12previouspilotworkers. Per-step controller clocks include full decision cost.'}
(OUT/'remaining_phase_timing.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result));sys.exit(proc.returncode)
