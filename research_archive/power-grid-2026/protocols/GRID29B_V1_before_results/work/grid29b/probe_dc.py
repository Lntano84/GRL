"""Actual controller untouched; same-real-memory isolated DC probes."""
import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid29b'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
sys.path[:0]=[str(ROOT/'work/grid29'),str(ROOT/'work/grid08')]
from common import np,OptimModule,ResidualControl,write_json
from cache_dc import run_dc_cached
D=json.loads((OUT/'design.json').read_text())
for path,h in json.loads((OUT/'code_freeze.json').read_text()).items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
assert not (OUT/'profile.json').exists()
old_dc=OptimModule.run_dc;old_solve=OptimModule._solve_problem;old_propose=ResidualControl.propose
old_init=ResidualControl.__init__
replicas={};pairs=[];solver_records=[];vectors=[];phase='actual_other';in_control=False


def solve(self,prob,*args,**kwargs):
    solver_type=kwargs.get('solver_type',args[0] if args else None)
    stamp=time.perf_counter()
    try:return old_solve(self,prob,*args,**kwargs)
    finally:
        stats=prob.solver_stats
        solver_records.append({'phase':phase,'in_control':in_control,'owner':getattr(self,'_probe_owner','unknown'),
            'solver':None if solver_type is None else str(solver_type),'status':str(prob.status),
            'wall_s':time.perf_counter()-stamp,'compilation_s':prob.compilation_time,
            'backend_s':None if stats is None else stats.solve_time,'iterations':None if stats is None else stats.num_iters})


def dc(self,obs):
    global phase
    before=phase;phase='actual_dc';stamp=time.perf_counter()
    try:actual=bool(old_dc(self,obs))
    finally:actual_s=time.perf_counter()-stamp;phase=before
    actual_flow=self.flow_computed.copy()
    if self not in replicas:
        replicas[self]=[OptimModule(self.env,self.action_space,config=self.config) for _ in range(2)]
        for name,mod in zip(['shadow_original','shadow_cached'],replicas[self]):mod._probe_owner=name
    flows={};times={};convergence={}
    order=[0,1] if len(pairs)%2==0 else [1,0]
    for slot in order:
        mod=replicas[self][slot];name='ORIGINAL' if slot==0 else 'CACHED'
        mod._powerlines_x.value=np.array(self._powerlines_x.value,copy=True)
        mod._storage_power_obs.value=float(self._storage_power_obs.value);mod.max_iter=self.max_iter
        phase='shadow_'+name.lower();stamp=time.perf_counter()
        try:convergence[name]=bool(old_dc(mod,obs) if slot==0 else run_dc_cached(mod,obs))
        finally:times[name]=time.perf_counter()-stamp;phase=before
        flows[name]=mod.flow_computed.copy()
    errors={}
    for name,flow in flows.items():
        assert convergence[name]==actual
        assert np.array_equal(np.isfinite(flow),np.isfinite(actual_flow))
        finite=np.isfinite(flow)
        errors[name]=float(np.max(np.abs(flow[finite]-actual_flow[finite]))) if finite.any() else 0.
        assert errors[name]<=1e-6,(name,obs.current_step,errors[name])
    row={'step':int(obs.current_step),'owner':getattr(self,'_probe_owner','setup'),
         'in_control':in_control,'actual_dc_s':actual_s,'shadow_s':times,'flow_error_MW':errors,
         'storage_power_accumulator':float(self._storage_power_obs.value),'converged':actual,'order':order}
    vectors.append(np.stack([actual_flow,flows['ORIGINAL'],flows['CACHED']]))
    pairs.append(row)
    assert np.array_equal(actual_flow,self.flow_computed,equal_nan=True)
    return actual


def tagged_init(self,*args,**kwargs):
    old_init(self,*args,**kwargs)
    self.base.optim._probe_owner='base_optimizer';self.safe._probe_owner='restoration_optimizer'


def proposed(self,*args,**kwargs):
    global in_control
    assert not in_control;in_control=True
    try:return old_propose(self,*args,**kwargs)
    finally:in_control=False


OptimModule.run_dc=dc;OptimModule._solve_problem=solve
ResidualControl.__init__=tagged_init;ResidualControl.propose=proposed
try:
    spec=importlib.util.spec_from_file_location('grid28_live_dc_probe',ROOT/'work/grid28/run.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker);worker.OUT=OUT
    settings=json.loads((ROOT/'outputs/grid28/runtime_design.json').read_text());settings['caps']=D['caps']
    item={'repeat':0,'week':D['week'],'method':'LIVE_DC_PROFILE','rule':'GNN_seed0','budget':128,
          'checkpoint':'outputs/grid21/models/gnn_seed0/final.pt','policy':'PROTECTED'}
    summary=worker.worker(item,settings);assert summary['exact_full_trajectory']
    active=[r for r in pairs if r['in_control']]
    totals={name:sum(r['shadow_s'][name] for r in active) for name in ['ORIGINAL','CACHED']}
    result={'passed':True,'summary':summary,'all_dc_calls':len(pairs),'controlled_dc_calls':len(active),
            'pairs':pairs,'solver_records':solver_records,'controlled_shadow_s':totals,
            'paired_dc_reduction':1-totals['CACHED']/totals['ORIGINAL'],
            'max_live_memory_flow_error_MW':max(max(r['flow_error_MW'].values()) for r in pairs),
            'cache_counts':[dict(mods[1]._grid29_dc_counts) for mods in replicas.values()],
            'new_fits':0,'reserved_evaluations':0,
            'warning':'Instrumented actual original trajectory; full controller timer includes shadow work and cannot be used as an acceleration result. Isolated probes use actual current storage memory and reactance, but never deploy their outputs.'}
    np.savez_compressed(OUT/'dc_vectors.npz',flows=np.stack(vectors))
    write_json(OUT/'profile.json',result)
    (OUT/'RUN_STATE.md').write_text('COMPLETED: exact protected actual trajectory; paired live-state DC probes, not total speed measurement. Saved-vector audit pending.\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['pairs','solver_records']},indent=2),flush=True)
finally:
    OptimModule.run_dc=old_dc;OptimModule._solve_problem=old_solve
    ResidualControl.__init__=old_init;ResidualControl.propose=old_propose
