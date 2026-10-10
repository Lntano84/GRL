"""Zero physical steps: source-greedy semantic fixtures and AC cache comparison."""
import hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace,MethodType
started=time.perf_counter()
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid05'
sys.path[:0]=[str(ROOT/'work/grid01'),str(ROOT/'work/grid02'),str(ROOT/'work/grid03'),str(ROOT/'work/grid04')]
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from fallback import install_rule,attach_rank_trace
from trace_agent import Telemetry,action_vector,digest
from io_utils import encode
from ljn_nn.modules.base_module import GreedyModule

class Action:
    attr_list_vect=['v']
    def __init__(self,row,bias=0):self.row=row;self.bias=bias
    def _get_array_from_attr_name(self,name):return np.asarray([self.row,self.bias],dtype=np.float32)
    def __add__(self,other):return Action(self.row,self.bias+other.bias)

class Observation:
    def __init__(self,values,trace,exceptions=None,done=None):
        self.rho=np.array([1.1]);self.values=values;self.trace=trace;self.calls=[]
        self.exceptions=exceptions or {};self.done=done or {}
    def simulate(self,action):
        self.calls.append(action.row);rho=float(self.values[action.row])
        nxt=SimpleNamespace(rho=np.asarray([rho]))
        record={'rho_max':rho,'done':bool(self.done.get(action.row,False)),
            'exceptions':self.exceptions.get(action.row,[]),'reward':2-rho}
        self.trace.forecasts[(digest(action_vector(action)),1)]=(nxt.rho,np.zeros(1),record)
        return nxt,record['reward'],record['done'],{'exception':record['exceptions']}

class Module:
    def __init__(self,first,full):
        self.top_k=20;self.null_action_reward=-100.
        self.gym_env=SimpleNamespace(action_space=SimpleNamespace(from_gym=lambda i:Action(int(i))),
            observation_space=SimpleNamespace(to_gym=lambda o:o))
        self.get_top_k=lambda o,top_k:np.asarray(first if top_k==20 else full)
        self._get_tested_action=lambda o:[self.gym_env.action_space.from_gym(i) for i in self.get_top_k(o,self.top_k)]
        self.get_act=MethodType(GreedyModule.get_act,self)

def fixture(name,values,expect_expand,first=None,full=None,bias=0,exceptions=None,done=None,kwargs=None):
    first=first or list(range(20));full=full or list(range(352));kwargs=kwargs or {}
    trace=SimpleNamespace(step=1,events=[],forecasts={})
    module=Module(first,full);holder={'trace':trace}
    install_rule(module,.9,holder)
    attach_rank_trace(module,trace,'FALLBACK90')
    obs=Observation(values,trace,exceptions,done)
    result=module.get_act(obs,Action(-1,bias),0,**kwargs)
    event=[e for e in trace.events if e['kind']=='fallback'][0]
    assert event['expanded']==expect_expand,(name,event)
    assert len(obs.calls)==len(set(obs.calls))==(352 if expect_expand else 20),name
    ref=Module(first,full);ref.top_k=352 if expect_expand else 20
    ref_trace=SimpleNamespace(step=1,events=[],forecasts={})
    ref_obs=Observation(values,ref_trace,exceptions,done)
    target=ref.get_act(ref_obs,Action(-1,bias),0,**kwargs)
    assert (result is None)==(target is None),name
    if result is not None:assert digest(action_vector(result))==digest(action_vector(target)),name
    assert np.array_equal(module.resulting_rewards,ref.resulting_rewards),name
    return {'fixture':name,'expanded':expect_expand,'simulated_unique_rows':len(obs.calls),'passed':True}

fixtures=[]
for name,best,expand in [('safe',.89,False),('exact_boundary',.9,True),('unsafe',.95,True),('near_below',.899999,False)]:
    a=np.full(352,1.05);a[3]=best;a[77]=.7
    fixtures.append(fixture(name,a,expand))
a=np.full(352,1.2);a[77]=.7;fixtures.append(fixture('no_initial_improvement',a,True))
fixtures.append(fixture('no_improvement_anywhere',np.full(352,1.2),True))
a=np.full(352,1.05);a[3]=.95;a[77]=.7
fixtures.append(fixture('base_action_preserved',a,True,bias=7))
a[4]=.5
fixtures.append(fixture('exception_is_not_a_safe_candidate',a,True,exceptions={4:['invalid']}))
fixtures.append(fixture('terminal_is_not_a_safe_candidate',a,True,done={4:True}))
a=np.full(352,.95);full=[77]+[i for i in range(352) if i!=77]
fixtures.append(fixture('full_order_tie',a,True,full=full))
a=np.full(352,1.05);a[3]=.89;a[77]=.7
fixtures.append(fixture('kwargs_threshold_preserved',a,True,kwargs={'rho_threshold':.8}))

import grid2op
from lightsim2grid import LightSimBackend
from ljn_nn import make_agent_topoNN
from ljn_nn.modules.rewards import MaxRhoReward
from score_adapter import operational_other_rewards
design=json.loads((OUT/'design_freeze.json').read_text())
assert hashlib.sha256((OUT/'design_freeze.json').read_bytes()).hexdigest()==(OUT/'design_freeze.sha256').read_text()
env=grid2op.make(str(ROOT/design['data_root']),test=True,backend=LightSimBackend(),reward_class=MaxRhoReward,other_rewards=operational_other_rewards())
env.seed(0);env.set_id(design['selected_development_weeks'][0]);obs=env.reset()
agent=make_agent_topoNN(env,str(ROOT/'work/grid02/ljn_nn'));agent.seed(0);agent.reset(obs)
module=agent.topo_12_unsafe;source_get_act=module.get_act;holder={}
install_rule(module,agent.rho_safe,holder,force_expand_for_test=True)
trace=Telemetry(agent);holder['trace']=trace;attach_rank_trace(module,trace,'PREFLIGHT_FORCED')
initial=obs.to_vect().copy();before_count=int(env.nb_highres_called)
trace.begin(1)
expanded=module.get_act(obs,env.action_space(),0.)
expanded_rewards=module.resulting_rewards.copy();expanded_actions=[digest(action_vector(a)) for a in module.tested_action]
first_events=list(trace.events)
assert len([e for e in first_events if e['kind']=='simulate'])==352
module.top_k=352
reference=source_get_act(obs,env.action_space(),0.)
assert np.array_equal(expanded_rewards,module.resulting_rewards)
assert expanded_actions==[digest(action_vector(a)) for a in module.tested_action]
assert (expanded is None)==(reference is None)
assert expanded is None or digest(action_vector(expanded))==digest(action_vector(reference))
assert np.array_equal(initial,obs.to_vect(),equal_nan=True)
assert np.array_equal(initial,env.get_obs().to_vect(),equal_nan=True)
native_calls=int(env.nb_highres_called)-before_count
assert native_calls==704<=design['preflight']['maximum_native_simulates']
prior_failed=json.loads((OUT/'preflight_failed_attempts.json').read_text()) if (OUT/'preflight_failed_attempts.json').exists() else []
result={'passed':True,'physical_steps':0,'native_simulates':native_calls,'source_full_greedy_rewards_actions_and_selection_match':True,
    'fixture_count':len(fixtures),'fixtures':fixtures,'wall_s':time.perf_counter()-started,
    'prior_failed_wall_s':sum(r['wall_s'] for r in prior_failed),
    'forced_expansion_used_only_in_preflight':True,'initial_observation_hash':digest(initial)}
assert result['wall_s']+result['prior_failed_wall_s']<=design['preflight']['maximum_wall_s']
(OUT/'preflight_events.json').write_text(encode(trace.events,indent=2),encoding='utf-8')
(OUT/'preflight.json').write_text(encode(result,indent=2),encoding='utf-8')
trace.close();env.close()
print(encode(result,indent=2))
