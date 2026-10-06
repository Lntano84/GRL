"""Read completed replay metrics and current progress; never run a simulator."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from protocol import *
from run_all import find_stats
import compress_json
import numpy as np

status=json.loads((WORK/'run_status.json').read_text())
rows=[]
for arm in status['completed']:
    b=compress_json.load(str(find_stats(arm)))['batches']
    d=lambda k:np.diff(np.asarray(b[k],dtype=float),prepend=0.)
    duration=d('time_elapsed_phy')
    cost=d('service_time_used_stats')+d('service_time_writes_stats')
    dt=cost/(36*.001*duration)*100
    writes=int(d('flashcache/keys_written_stats')[144:].sum())
    rows.append({'arm':arm,'peak_dt_pct':float(max(dt[144:])),
                 'peak_window':144+int(np.argmax(dt[144:])),
                 'writes':writes,'eligible':writes<=WRITE_BUDGET,
                 'average_dt_pct':float(cost[144:].sum()/(36*.001*duration[144:].sum())*100),
                 'prefetch_reads':int(d('fetches_chunks_prefetch_stats')[144:].sum()),
                 'window_578_dt_pct':float(dt[578])})
current=status.get('current_arm')
progress=WORK/current/'progress.json' if current else None
print(json.dumps({'status':status['status'],'completed_metrics':rows,
                  'progress':json.loads(progress.read_text()) if progress and progress.exists() else None}))
