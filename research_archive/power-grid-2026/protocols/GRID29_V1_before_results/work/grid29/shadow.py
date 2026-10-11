"""Same-observation numerical qualification; no AC/closed-loop replay or fitting."""
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
sys.path[:0] = [str(ROOT/'work/grid29'), str(ROOT/'work/grid08')]
from common import np, make_env, OptimModule, write_json
from cache_dc import solve_current, run_dc_cached
import cvxpy as cp
OUT = ROOT / 'outputs/grid29'
started = time.perf_counter()
D = json.loads((OUT / 'design.json').read_text())
for p,h in json.loads((OUT/'code_freeze.json').read_text()).items():
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h, p
assert not (OUT / 'shadow.json').exists()


class Toy:
    nb_max_bus=2
    bus_storage=None
    def __init__(self):
        self.bus_or=cp.Parameter(1,value=[0.]);self.bus_ex=cp.Parameter(1,value=[1.])
        self.bus_load=cp.Parameter(1,value=[1.]);self.bus_gen=cp.Parameter(1,value=[0.])
        self._powerlines_x=cp.Parameter(1,value=[2.],nonneg=True);self.flow_computed=np.full(1,np.nan)
        self.load_per_bus=cp.Parameter(2,value=[0.,7.]);self.gen_per_bus=cp.Parameter(2,value=[7.,0.])
    def _mask_theta_zero(self):return np.array([True,False])
    def _aux_compute_kcl(self,inj,f):
        a=int(self.bus_or.value[0]);b=int(self.bus_ex.value[0])
        return [inj[i]+(f[0] if i==a else 0)-(f[0] if i==b else 0) for i in range(2)]
    def _solve_problem(self,p,solver_type):
        p.solve(solver=solver_type,warm_start=False,verbose=False,max_iter=100000)
        return p.status=='optimal'

t=Toy();assert solve_current(t);assert np.allclose(t.flow_computed,[7.],rtol=0,atol=1e-7)
t.gen_per_bus.value=[11.,0.];t.load_per_bus.value=[0.,11.]
assert solve_current(t);assert np.allclose(t.flow_computed,[11.],rtol=0,atol=1e-7)
assert t._grid29_dc_counts['hit']==1
t.bus_or.value=[1.];t.bus_ex.value=[0.]
assert solve_current(t);assert np.allclose(t.flow_computed,[-11.],rtol=0,atol=1e-7)
assert t._grid29_dc_counts['miss']==2
t._powerlines_x.value=[3.];assert solve_current(t)
assert t._grid29_dc_counts['miss']==3
# A triangle also checks that changed impedance changes the actual line flow,
# not just a cache counter (two-bus current is determined by demand).
class Triangle(Toy):
    nb_max_bus=3
    def __init__(self):
        self.bus_or=cp.Parameter(3,value=[0.,1.,0.]);self.bus_ex=cp.Parameter(3,value=[1.,2.,2.])
        self.bus_load=cp.Parameter(1,value=[2.]);self.bus_gen=cp.Parameter(1,value=[0.])
        self._powerlines_x=cp.Parameter(3,value=[1.,1.,1.],nonneg=True);self.flow_computed=np.full(3,np.nan)
        self.load_per_bus=cp.Parameter(3,value=[0.,0.,9.]);self.gen_per_bus=cp.Parameter(3,value=[9.,0.,0.])
    def _mask_theta_zero(self):return np.array([True,False,False])
    def _aux_compute_kcl(self,inj,f):
        return [inj[i]+cp.sum(f[self.bus_or.value==i])-cp.sum(f[self.bus_ex.value==i]) for i in range(3)]
tri=Triangle();assert solve_current(tri);assert np.allclose(tri.flow_computed,[3.,3.,6.],rtol=0,atol=1e-7)
tri._powerlines_x.value[:]=[1.,1.,2.]
assert solve_current(tri);assert np.allclose(tri.flow_computed,[4.5,4.5,4.5],rtol=0,atol=1e-7)
assert tri._grid29_dc_counts['miss']==2
manual=6
records=[];counts=[]
env=make_env();env.seed(0);env.set_id(D['shadow_weeks'][0]);template=env.reset()
try:
    original=OptimModule(env,env.action_space);cached=OptimModule(env,env.action_space)
    for week in D['shadow_weeks']:
        folder=ROOT/'outputs/grid26/runs'/f'r0__{week}__FULL'
        with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(line) for line in f]
        vectors=np.load(folder/'vectors.npz')['observation']
        valid=[i for i,r in enumerate(steps) if not r['done']]
        selected=sorted(set(valid[::64]+[valid[-1]]))
        for i in selected:
            obs=template.copy();obs.from_vect(vectors[i])
            assert obs.current_step==steps[i]['step']
            times={};flows={};ok={}
            order=['ORIGINAL','CACHED'] if len(records)%2==0 else ['CACHED','ORIGINAL']
            for method in order:
                mod=original if method=='ORIGINAL' else cached
                mod._storage_power_obs.value=0.
                stamp=time.perf_counter()
                ok[method]=bool(mod.run_dc(obs) if method=='ORIGINAL' else run_dc_cached(mod,obs))
                times[method]=time.perf_counter()-stamp;flows[method]=mod.flow_computed.copy()
            finite={k:bool(np.isfinite(v).all()) for k,v in flows.items()}
            diff=float(np.max(np.abs(flows['ORIGINAL']-flows['CACHED']))) if all(finite.values()) else None
            agree=ok['ORIGINAL']==ok['CACHED'] and finite['ORIGINAL']==finite['CACHED'] and (diff is None or diff<=1e-6)
            records.append({'week':week,'step':steps[i]['step'],'same':agree,'convergence':ok,'finite':finite,'max_flow_error_MW':diff,'times_s':times})
            if not agree:
                write_json(OUT/'shadow_failure.json',records[-1]);raise AssertionError(records[-1])
        counts.append({'week':week,'samples':len(selected),'cache_counts':dict(cached._grid29_dc_counts),'resident_programs':len(cached._grid29_dc_programs)})
        print(json.dumps(counts[-1]),flush=True)
    totals={k:sum(r['times_s'][k] for r in records) for k in ['ORIGINAL','CACHED']}
    result={'passed':True,'manual_checks':manual,'states':len(records),'records':records,'counts':counts,'total_s':totals,
            'time_reduction':1-totals['CACHED']/totals['ORIGINAL'],
            'max_flow_error_MW':max(r['max_flow_error_MW'] for r in records if r['max_flow_error_MW'] is not None),
            'wall_s':time.perf_counter()-started,'new_fits':0,'reserved_evaluations':0,
            'warning':D['warning']+' '+D['shadow_state']}
    write_json(OUT/'shadow.json',result)
    (OUT/'RUN_STATE.md').write_text('SHADOW PASSED: stored-state numerical check only; live closed-loop quality is not yet established.\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
finally:env.close()
