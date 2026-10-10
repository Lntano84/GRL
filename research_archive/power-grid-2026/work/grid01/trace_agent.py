"""Transparent telemetry: forwards the original arguments, results and exceptions."""
import functools
import hashlib
import time
import numpy as np
from grid2op.Observation import BaseObservation
from grid2op.dtypes import dt_float

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def action_vector(action):
    # to_vect() mutates a cache that some action setters do not invalidate.
    # Read exactly the same fields without filling/reading that cache.
    fields=[action._get_array_from_attr_name(name).astype(dt_float) for name in type(action).attr_list_vect]
    return np.concatenate(fields) if fields else np.array([],dtype=dt_float)

def finite(value):
    x=float(value)
    return x if np.isfinite(x) else None

def flags(info):
    return {'illegal':bool(info.get('is_illegal',False)), 'ambiguous':bool(info.get('is_ambiguous',False)),
            'exceptions':[str(x) for x in info.get('exception',[])]}

class Telemetry:
    def __init__(self,agent):
        self.step=0
        self.stack=[]
        self.events=[]
        self.forecasts={}
        self.original_simulate=BaseObservation.simulate
        trace=self
        original=self.original_simulate
        @functools.wraps(original)
        def simulate(obs,action,*args,**kwargs):
            key=digest(action_vector(action))
            t=time.perf_counter()
            result=original(obs,action,*args,**kwargs)
            elapsed=time.perf_counter()-t
            future,reward,done,info=result
            horizon=int(kwargs.get('time_step',args[0] if args else 1))
            record={'kind':'simulate','step':trace.step,'module':trace.stack[-1] if trace.stack else 'agent',
                'horizon':horizon,'wall_s':elapsed,'action_hash':key,'rho_max':finite(future.rho.max()),
                'reward':finite(reward),'done':bool(done),**flags(info)}
            trace.events.append(record)
            # Store only current decision's already-requested results in a temporary map.
            trace.forecasts[(key,horizon)]=(np.array(future.rho,copy=True),np.array(future.p_or,copy=True),record)
            return result
        BaseObservation.simulate=simulate
        for name in ['reconnect','recover_topo','topo_12_unsafe','topo_n1_unsafe','optim']:
            module=getattr(agent,name)
            self.wrap_module(module,name)
        self.wrap_solver(agent.optim)

    def wrap_module(self,module,name):
        original=module.get_act
        @functools.wraps(original)
        def call(*args,**kwargs):
            t=time.perf_counter()
            before=sum(e['kind']=='simulate' for e in self.events)
            self.stack.append(name)
            try:
                result=original(*args,**kwargs)
                self.events.append({'kind':'module','step':self.step,'module':name,'wall_s':time.perf_counter()-t,
                    'returned_none':result is None,'returned_action_hash':digest(action_vector(result)) if result is not None else None,
                    'simulate_calls':sum(e['kind']=='simulate' for e in self.events)-before})
                return result
            except Exception as e:
                self.events.append({'kind':'module_error','step':self.step,'module':name,'wall_s':time.perf_counter()-t,
                    'error':type(e).__name__+': '+str(e)})
                raise
            finally:
                self.stack.pop()
        module.get_act=call

    def wrap_solver(self,module):
        original=module._solve_problem
        @functools.wraps(original)
        def solve(prob,solver_type=None):
            t=time.perf_counter()
            result=original(prob,solver_type=solver_type)
            self.events.append({'kind':'solve','step':self.step,'module':'optim','solver_argument':solver_type,
                'root_dispatch':solver_type is None,'wall_s':time.perf_counter()-t,
                'returned_success':bool(result),'problem_status':str(prob.status),'problem_value':finite(prob.value) if prob.value is not None else None})
            return result
        module._solve_problem=solve

    def begin(self,step):
        self.step=step
        self.events=[]
        self.forecasts={}

    def close(self):
        BaseObservation.simulate=self.original_simulate
