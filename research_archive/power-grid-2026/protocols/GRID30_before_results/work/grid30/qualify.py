"""Manual real-Parameter cases then isolated probes under actual control memory."""
import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid30'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
sys.path[:0]=[str(ROOT/'work/grid30'),str(ROOT/'work/grid29'),str(ROOT/'work/grid08')]
from common import np,OptimModule,write_json,action_vector
import cvxpy as cp
from cache_safe import compute_safe_cached,COEFFICIENTS
D=json.loads((OUT/'design.json').read_text())
for path,h in json.loads((OUT/'code_freeze.json').read_text()).items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
assert not (OUT/'qualification.json').exists()
original_safe=OptimModule.compute_optimum_safe;original_solve=OptimModule._solve_problem


class Toy:
    nb_max_bus=2;bus_storage=None;SOLVER_TYPES=OptimModule.SOLVER_TYPES
    def __init__(self):
        for name in COEFFICIENTS:setattr(self,name,cp.Parameter(value=1.,nonneg=True))
        self._powerlines_x=cp.Parameter(1,value=[2.],pos=True)
        self.bus_or=cp.Parameter(1,value=[0.]);self.bus_ex=cp.Parameter(1,value=[1.])
        self.bus_load=cp.Parameter(1,value=[1.]);self.bus_gen=cp.Parameter(1,value=[0.])
        for name,value in {'load_per_bus':[0.,8.],'gen_per_bus':[8.,0.],
            'redisp_up':[10.,10.],'redisp_down':[10.,10.],
            'curtail_up':[0.,0.],'curtail_down':[0.,0.],'storage_up':[0.,0.],
            'storage_down':[0.,0.],'_past_dispatch':[3.,-3.],
            '_past_state_of_charge':[0.,0.],'_storage_target_bus':[0.,0.]}.items():
            setattr(self,name,cp.Parameter(2,value=value))
        self._storage_power_obs=cp.Parameter(value=0.)
        self._prev_por_error=cp.Parameter(1,value=[0.]);self._th_lim_mw=cp.Parameter(1,value=[20.])
        self.flow_computed=np.full(1,np.nan)
    def _mask_theta_zero(self):return np.array([True,False])
    def _aux_compute_kcl(self,inj,f):
        return [inj[i]+cp.sum(f[self.bus_or.value==i])-cp.sum(f[self.bus_ex.value==i]) for i in range(2)]
    def _solve_problem(self,p,solver_type=None):
        p.solve(solver=cp.OSQP,warm_start=False,max_iter=100000)
        self.last_status=p.status
        return p.status in ['optimal','optimal_inaccurate']


original,cached=Toy(),Toy();obs=SimpleNamespace(delta_time=5.)
manual=[]
def check(name,expected_red=None,expected_miss=None):
    a=original_safe(original,obs);b=compute_safe_cached(cached,obs)
    assert original.last_status==cached.last_status
    assert np.array_equal(np.isfinite(original.flow_computed),np.isfinite(cached.flow_computed))
    for x,y in zip(a,b):assert np.allclose(x,y,rtol=0,atol=1e-7)
    if expected_red is not None:assert np.allclose(b[2],expected_red,rtol=0,atol=1e-6)
    if expected_miss is not None:assert cached._grid30_safe_counts['miss']==expected_miss
    manual.append({'name':name,'passed':True,'status':original.last_status,'cache':dict(cached._grid30_safe_counts)})
check('analytic_quadratic_target',[-1.5,1.5],1)
for m in [original,cached]:m._past_dispatch.value=[1.,-1.]
check('refresh_target_without_recompile',[-.5,.5],1)
for m in [original,cached]:m.redisp_down.value=[.2,10.]
check('refresh_bounds_without_recompile',[-.2,.2],1)
for m in [original,cached]:m.redisp_down.value=[10.,10.];m._penalty_redispatching_safe.value=2.
check('coefficient_invalidation',[-1./3,1./3],2)
for m in [original,cached]:m._powerlines_x.value=[3.]
check('reactance_invalidation',[-1./3,1./3],3)
for m in [original,cached]:m.bus_or.value=[1.];m.bus_ex.value=[0.]
check('topology_invalidation',[-1./3,1./3],4)
obs.delta_time=10.;check('time_resolution_invalidation',[-1./3,1./3],5)
for m in [original,cached]:
    m.redisp_up.value=[0.,0.];m.redisp_down.value=[0.,0.];m._th_lim_mw.value=[1.]
check('same_infeasible_zero_and_nan_semantics',[0.,0.],5)
assert np.isnan(cached.flow_computed).all()
write_json(OUT/'manual.json',{'passed':True,'checks':manual})
print(json.dumps({'manual_passed':len(manual)}),flush=True)

replicas={};pairs=[];outputs=[];param_arrays={};solver_records=[]
def solve(self,p,*a,**kw):
    stamp=time.perf_counter()
    try:return original_solve(self,p,*a,**kw)
    finally:
        stats=p.solver_stats
        self._grid30_latest_stats={'status':str(p.status),'wall_s':time.perf_counter()-stamp,
            'compile_s':p.compilation_time,'backend_s':None if stats is None else stats.solve_time,
            'iterations':None if stats is None else stats.num_iters}


def qualified(mod):
    def call(prob,solver_type=None):
        ok=solve(mod,prob,solver_type=solver_type)
        return bool(ok and prob.status in ['optimal','optimal_inaccurate'])
    return call


def safe(self,obs,*args,**kwargs):
    if self not in replicas:
        replicas[self]=[OptimModule(self.env,self.action_space,config=self.config) for _ in range(2)]
        for clone in replicas[self]:clone._solve_problem=qualified(clone)
    clones=replicas[self]
    for name,param in vars(self).items():
        if isinstance(param,cp.Parameter):
            value=None if param.value is None else np.array(param.value,copy=True)
            param_arrays.setdefault(name,[]).append(value)
            for clone in clones:getattr(clone,name).value=None if value is None else value.copy()
    for clone in clones:
        clone.flow_computed[:]=self.flow_computed;clone.max_iter=self.max_iter
    stamp=time.perf_counter();actual=original_safe(self,obs,*args,**kwargs);actual_s=time.perf_counter()-stamp
    actual_values=np.concatenate([self.flow_computed.copy()]+[v.copy() for v in actual])
    actual_status=self._grid30_latest_stats['status'];actual_post=float(self._storage_power_obs.value)
    actual_action=action_vector(self.to_grid2op(obs,*[v.copy() for v in actual],base_action=self.action_space(),safe=True))
    times={};errors={};statuses={};action_equal={};values={};order=[0,1] if len(pairs)%2==0 else [1,0]
    for slot in order:
        clone=clones[slot];name='ORIGINAL' if slot==0 else 'CACHED';stamp=time.perf_counter()
        result=original_safe(clone,obs,*args,**kwargs) if slot==0 else compute_safe_cached(clone,obs,*args,**kwargs)
        times[name]=time.perf_counter()-stamp;statuses[name]=clone._grid30_latest_stats['status']
        vector=np.concatenate([clone.flow_computed.copy()]+[v.copy() for v in result]);values[name]=vector
        assert statuses[name]==actual_status
        assert float(clone._storage_power_obs.value)==actual_post
        assert np.array_equal(np.isfinite(vector),np.isfinite(actual_values))
        finite=np.isfinite(vector)
        errors[name]=float(np.max(np.abs(vector[finite]-actual_values[finite]))) if finite.any() else 0.
        assert errors[name]<=1e-6,(name,obs.current_step,errors[name])
        action=action_vector(clone.to_grid2op(obs,*[v.copy() for v in result],base_action=clone.action_space(),safe=True))
        action_equal[name]=bool(np.array_equal(action,actual_action));assert action_equal[name],(name,obs.current_step)
        solver_records.append({'pair':len(pairs),'method':name,**clone._grid30_latest_stats})
    outputs.append(np.stack([actual_values,values['ORIGINAL'],values['CACHED']]))
    pairs.append({'step':int(obs.current_step),'actual_s':actual_s,'shadow_s':times,
        'max_numerical_error_MW':errors,'same_grid2op_action':action_equal,'statuses':statuses,'order':order})
    return actual


OptimModule._solve_problem=solve;OptimModule.compute_optimum_safe=safe
try:
    spec=importlib.util.spec_from_file_location('grid30_worker',ROOT/'work/grid28/run.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker);worker.OUT=OUT
    settings=json.loads((ROOT/'outputs/grid28/runtime_design.json').read_text());settings['caps']=D['caps']
    item={'repeat':0,'week':D['week'],'method':'SAFE_QP_SHADOW','rule':'GNN_seed0','budget':128,
        'checkpoint':'outputs/grid21/models/gnn_seed0/final.pt','policy':'PROTECTED'}
    summary=worker.worker(item,settings);assert summary['exact_full_trajectory'] and len(pairs)==153
    times={name:sum(p['shadow_s'][name] for p in pairs) for name in ['ORIGINAL','CACHED']}
    result={'passed':True,'manual_checks':len(manual),'live_pairs':len(pairs),'pairs':pairs,
        'solver_records':solver_records,'summary':summary,'shadow_times_s':times,
        'max_numerical_error_MW':max(max(p['max_numerical_error_MW'].values()) for p in pairs),
        'cache_counts':[dict(mods[1]._grid30_safe_counts) for mods in replicas.values()],
        'warning':D['no_speed_claim'],'new_fits':0,'reserved_evaluations':0}
    np.savez_compressed(OUT/'safe_vectors.npz',outputs=np.stack(outputs))
    np.savez_compressed(OUT/'pre_solve_parameters.npz',**{k:np.stack(v) for k,v in param_arrays.items()})
    write_json(OUT/'qualification.json',result)
    (OUT/'RUN_STATE.md').write_text('QUALIFIED:manual and live numerical/action qualification; saved-data audit pending, no cached safe control deployed.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['pairs','solver_records','summary']},indent=2),flush=True)
finally:
    OptimModule._solve_problem=original_solve;OptimModule.compute_optimum_safe=original_safe
