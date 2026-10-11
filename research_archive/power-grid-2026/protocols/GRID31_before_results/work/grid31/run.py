"""GRID31 three measured arms; derived from retained GRID29C. Original/DC/DC+safe."""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid31'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
D=load(OUT/'design.json')
for file,h in load(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==h,file
if args.worker is not None:
    item=D['execution_order'][args.worker]
    sys.path[:0]=[str(ROOT/'work/grid30'),str(ROOT/'work/grid29'),str(ROOT/'work/grid08')]
    from common import ResidualControl,OptimModule
    from cache_dc import run_dc_cached
    from cache_safe import compute_safe_cached
    spec=importlib.util.spec_from_file_location('grid29c_worker',ROOT/'work/grid28/run.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker);worker.OUT=OUT
    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose
    old_solve=OptimModule._solve_problem
    originals={name:getattr(OptimModule,name) for name in ['run_dc','compute_optimum_safe','compute_optimum_unsafe']}
    optimizers=[];components=[];solvers=[];phase='other';step=0;in_control=False
    def init(self,*a,**kw):
        old_init(self,*a,**kw)
        self.base.optim._timer_owner='base';self.safe._timer_owner='restore'
        optimizers.extend([self.base.optim,self.safe])
    def propose(self,obs,*a,**kw):
        global step,in_control
        assert not in_control;step=int(obs.current_step)+1;in_control=True
        try:return old_propose(self,obs,*a,**kw)
        finally:in_control=False
    def solve(self,prob,*a,**kw):
        solver=kw.get('solver_type',a[0] if a else None)
        stamp=time.perf_counter()
        try:return old_solve(self,prob,*a,**kw)
        finally:
            stats=prob.solver_stats
            solvers.append({'step':step,'in_control':in_control,'phase':phase,'owner':getattr(self,'_timer_owner','setup'),
                'solver':None if solver is None else str(solver),'wall_s':time.perf_counter()-stamp,
                'status':str(prob.status),'compilation_s':prob.compilation_time,
                'backend_s':None if stats is None else stats.solve_time,
                'iterations':None if stats is None else stats.num_iters})
    def wrapper(name,fn):
        def call(self,*a,**kw):
            global phase
            previous=phase;phase=name;stamp=time.perf_counter()
            try:return fn(self,*a,**kw)
            finally:
                elapsed=time.perf_counter()-stamp
                components.append({'step':step,'in_control':in_control,'phase':name,'owner':getattr(self,'_timer_owner','setup'),'wall_s':elapsed})
                phase=previous
        return call
    ResidualControl.__init__=init;ResidualControl.propose=propose;OptimModule._solve_problem=solve
    for name,fn in originals.items():
        chosen=run_dc_cached if name=='run_dc' and item['cache_dc'] else compute_safe_cached if name=='compute_optimum_safe' and item['cache_safe'] else fn
        setattr(OptimModule,name,wrapper(name,chosen))
    try:
        summary=worker.worker(item,D);assert summary['exact_full_trajectory']
        path=OUT/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
        save(path/'components.json',{'calls':components,'solver_records':solvers,
            'cache_counts':[dict(getattr(m,'_grid29_dc_counts',{})) for m in optimizers],
            'safe_cache_counts':[dict(getattr(m,'_grid30_safe_counts',{})) for m in optimizers],
            'warning':'Both arms carry same component timers. Null stats remain null. Explicit solver records only for additive solve accounting.'})
        save(OUT/f'worker_result_{args.worker:02d}.json',summary)
    finally:
        ResidualControl.__init__=old_init;ResidualControl.propose=old_propose;OptimModule._solve_problem=old_solve
        for name,fn in originals.items():setattr(OptimModule,name,fn)
else:
    assert not (OUT/'finished.json').exists()
    stamp=time.perf_counter();summaries=[]
    (OUT/'RUN_STATE.md').write_text('RUNNING:one serial fresh worker;6balanced original/DC/DC+safe component profiles.\n',encoding='utf-8')
    for index,item in enumerate(D['execution_order']):
        assert time.perf_counter()-stamp<D['caps']['wall_s']
        with (OUT/f'worker_{index:02d}.log').open('w',encoding='utf-8') as f:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(index)],cwd=ROOT,
                stdout=f,stderr=subprocess.STDOUT,timeout=D['caps']['episode_s']+60,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if proc.returncode:
            (OUT/'RUN_STATE.md').write_text(f'FAILED worker{index}:preserve partial run and log.\n',encoding='utf-8')
            raise RuntimeError((index,proc.returncode))
        summaries.append(load(OUT/f'worker_result_{index:02d}.json'))
        save(OUT/'manifest.json',{'summaries':summaries,'planned':6})
        print(json.dumps({'completed':len(summaries),'planned':6,'method':item['method'],'repeat':item['repeat'],
                         'controller_s':summaries[-1]['controller_s'],'exact_path':summaries[-1]['exact_full_trajectory']}),flush=True)
    assert sum(s['steps'] for s in summaries)<=D['caps']['physical_steps']
    assert sum(s['public_forecasts'] for s in summaries)<=D['caps']['public_forecasts']
    save(OUT/'finished.json',{'passed_engineering':True,'summaries':summaries,'wall_s':time.perf_counter()-stamp,
         'new_fits':0,'reserved_evaluations':0})
    (OUT/'RUN_STATE.md').write_text('RUNS COMPLETED:component accounting audit pending.\n',encoding='utf-8')
