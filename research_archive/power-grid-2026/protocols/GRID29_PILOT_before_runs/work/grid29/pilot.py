"""Protected GNN0 + optional generic DC-program reuse; original runner retained."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid29/pilot'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int)
args=parser.parse_args()
D=json.loads((OUT/'runtime_design.json').read_text())
for path,h in json.loads((OUT/'code_freeze.json').read_text()).items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
if args.worker is not None:
    item=D['execution_order'][args.worker]
    sys.path[:0]=[str(ROOT/'work/grid29'),str(ROOT/'work/grid08')]
    from common import OptimModule
    from cache_dc import run_dc_cached
    spec=importlib.util.spec_from_file_location('grid28_cache_pilot',ROOT/'work/grid28/run.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver);driver.OUT=OUT
    old_run=OptimModule.run_dc;old_init=OptimModule.__init__;optimizers=[]
    def capture(self,*a,**kw):
        old_init(self,*a,**kw);optimizers.append(self)
    OptimModule.__init__=capture
    if item['cache_dc']:OptimModule.run_dc=run_dc_cached
    try:
        result=driver.worker(item,D)
        result['cache_counts']=[dict(getattr(m,'_grid29_dc_counts',{})) for m in optimizers]
        result['resident_dc_programs']=[len(getattr(m,'_grid29_dc_programs',{})) for m in optimizers]
        save(OUT/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"/'summary.json',result)
        save(OUT/f"worker_result_{args.worker:02d}.json",result)
    finally:
        OptimModule.run_dc=old_run;OptimModule.__init__=old_init
else:
    assert not (OUT/'finished.json').exists()
    stamp=time.perf_counter();results=[]
    (OUT/'RUN_STATE.md').write_text('RUNNING: one serial fresh worker, original/cache balanced order.\n')
    for i,item in enumerate(D['execution_order']):
        assert time.perf_counter()-stamp<D['caps']['wall_s']
        with (OUT/f'worker_{i:02d}.log').open('w',encoding='utf-8') as f:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(i)],cwd=ROOT,
                                stdout=f,stderr=subprocess.STDOUT,timeout=D['caps']['episode_s']+60)
        if proc.returncode:
            (OUT/'RUN_STATE.md').write_text(f'FAILED: worker{i}, preserve log and partial output. Stop expansion and diagnose.\n')
            raise RuntimeError(f'worker{i} exit{proc.returncode}')
        results.append(json.loads((OUT/f'worker_result_{i:02d}.json').read_text()))
        save(OUT/'manifest.json',{'completed':results,'planned':len(D['execution_order'])})
        print(json.dumps({'completed':len(results),'planned':4,'method':item['method'],'repeat':item['repeat'],
                          'controller_s':results[-1]['controller_s'],'exact_path':results[-1]['exact_full_trajectory']}),flush=True)
    counts={'physical_steps':sum(s['steps'] for s in results),'public_forecasts':sum(s['public_forecasts'] for s in results)}
    assert counts['physical_steps']<=D['caps']['physical_steps'] and counts['public_forecasts']<=D['caps']['public_forecasts']
    save(OUT/'finished.json',{'passed_engineering':True,'serial_workers':1,'counts':counts,'summaries':results,
                           'wall_s':time.perf_counter()-stamp,'new_fits':0,'reserved_evaluations':0})
    (OUT/'RUN_STATE.md').write_text('RUNS COMPLETED: four exact protected trajectories; saved-data audit not yet completed.\n')
