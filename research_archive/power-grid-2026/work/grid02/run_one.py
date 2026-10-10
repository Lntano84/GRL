import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

parser=argparse.ArgumentParser()
parser.add_argument('--scenario',type=int,choices=[0,1],required=True)
parser.add_argument('--mode',choices=['full','smoke'],required=True)
parser.add_argument('--policy',choices=['FULL','NN20','NN352'],required=True)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid02'/f'{args.policy}_scenario{args.scenario}_{args.mode}'
import sys
sys.path.insert(0,str(ROOT/'work/grid01'))
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'steps.jsonl').exists():
    raise RuntimeError('Existing trajectory must not be overwritten')
import_started=time.perf_counter()
import numpy as np
import cvxpy as cp
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import grid2op
from lightsim2grid import LightSimBackend
from ljn_nn import make_agent_challenge,make_agent_topoNN
from ljn_nn.modules.rewards import MaxRhoReward
from trace_agent import Telemetry,digest,finite,flags,action_vector

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

save('import_ready.json',{'import_s':time.perf_counter()-import_started})
started=time.perf_counter()
env=None
trace=None
completed=0
try:
    data=Path(grid2op.__file__).parent/'data/l2rpn_idf_2023'
    t=time.perf_counter()
    env=grid2op.make(str(data),test=True,backend=LightSimBackend(),reward_class=MaxRhoReward)
    env.seed(0)
    env.set_id(args.scenario)
    obs=env.reset()
    make_s=time.perf_counter()-t
    t=time.perf_counter()
    factory=make_agent_challenge if args.policy=='FULL' else make_agent_topoNN
    agent=factory(env,str(ROOT/'work/grid02/ljn_nn'))
    if args.policy=='NN352':
        agent.topo_12_unsafe.top_k=int(agent.topo_12_unsafe.model.action_space.n)
        assert agent.topo_12_unsafe.top_k==352
    agent.seed(0)
    agent.reset(obs)
    agent_s=time.perf_counter()-t
    trace=Telemetry(agent)
    if args.policy in ['NN20','NN352']:
        original_topk=agent.topo_12_unsafe.get_top_k
        def traced_topk(gym_obs,top_k):
            start_topk=time.perf_counter()
            result=original_topk(gym_obs,top_k)
            trace.events.append({'kind':'nn_topk','step':trace.step,'module':'topo_12_unsafe',
                'wall_s':time.perf_counter()-start_topk,'input_hash':digest(gym_obs),'ids':result.tolist()})
            return result
        agent.topo_12_unsafe.get_top_k=traced_topk
    (OUT/'steps').mkdir(exist_ok=True)
    np.savez_compressed(OUT/'initial.npz',observation=obs.to_vect(),rho=obs.rho,p_or=obs.p_or)
    save('metadata.json',{'scenario_id':env.chronics_handler.get_id(),'seed':0,'mode':args.mode,
        'max_steps':575 if args.mode=='full' else 24,'policy':args.policy,'native_max_episode_duration':int(env.max_episode_duration()),'NN_topk':agent.topo_12_unsafe.top_k if args.policy!='FULL' else None,'make_s':make_s,'agent_construct_s':agent_s,
        'original_commit':'ca0637eab9f098be7f206ed0e46a3900cd4deec0',
        'versions':{n:importlib.metadata.version(n) for n in ['grid2op','numpy','cvxpy','osqp','scs','lightsim2grid','torch','stable-baselines3','gymnasium']},
        'reward_class':type(env._reward_helper.template_reward).__name__,'parameters':env.parameters.to_dict(),
        'rho_danger':agent.rho_danger,'rho_safe':agent.rho_safe,'optim_config':agent.optim.config,
        'author_solver_order':agent.optim.SOLVER_TYPES,'native_action_keys':sorted(env.action_space().authorized_keys),
        'simulation_quota':'Native unlimited setting; record calls and bounded process time',
        'action_vector_layout':[{ 'name':name,'size':int(env.action_space()._get_array_from_attr_name(name).size)} for name in type(env.action_space()).attr_list_vect],
        'telemetry':'Passive wrappers; fresh field vectorization bypasses action cache; timestamps include wrapper costs; no additional simulation.'})
    reward=0.
    done=False
    with (OUT/'steps.jsonl').open('w',encoding='utf-8') as step_file, (OUT/'events.jsonl').open('w',encoding='utf-8') as event_file:
        for step in range(1,576 if args.mode=='full' else 25):
            if time.perf_counter()-started>900:
                raise TimeoutError('Frozen 900s trajectory cap reached')
            trace.begin(step)
            step_start=time.perf_counter()
            before=obs.to_vect().copy()
            before_rho=obs.rho.copy()
            before_counter=int(env.nb_highres_called)
            t=time.perf_counter()
            act=agent.act(obs,reward,done)
            act_s=time.perf_counter()-t
            # Agent's search may update simulation counters, not the physical observation.
            physical_unchanged=np.array_equal(before,env.get_obs().to_vect(),equal_nan=True)
            assert physical_unchanged
            act_vect=action_vector(act)
            act_hash=digest(act_vect)
            selected_forecast=trace.forecasts.get((act_hash,1))
            t=time.perf_counter()
            obs,reward,done,info=env.step(act)
            env_step_s=time.perf_counter()-t
            native_calls=int(env.nb_highres_called)-before_counter
            traced_calls=sum(e['kind']=='simulate' for e in trace.events)
            assert native_calls==traced_calls,(native_calls,traced_calls)
            vector_file=OUT/'steps'/f'{step:04}.npz'
            np.savez_compressed(vector_file,observation=obs.to_vect(),action=act_vect,rho=obs.rho,p_or=obs.p_or,
                forecast_rho=selected_forecast[0] if selected_forecast else np.full(env.n_line,np.nan),
                forecast_p_or=selected_forecast[1] if selected_forecast else np.full(env.n_line,np.nan))
            record={'step':step,'current_step':int(obs.current_step),'timestamp':str(obs.get_time_stamp()),
                'before_observation_hash':digest(before),'observation_hash':digest(obs.to_vect()),'action_hash':act_hash,
                'before_rho_max':finite(before_rho.max()),'after_rho_max':finite(obs.rho.max()),
                'reward':finite(reward),'done':bool(done),**flags(info),
                'chronics_done':bool(env.chronics_handler.done()),
                'exception_types':[type(x).__name__ for x in info.get('exception',[])],
                'disconnected_lines_count':int(np.count_nonzero(info.get('disc_lines',[])>=0)) if isinstance(info.get('disc_lines'),np.ndarray) else None,
                'physical_observation_unchanged_during_act':physical_unchanged,'act_wall_s':act_s,'env_step_s':env_step_s,
                'simulate_calls':native_calls,'selected_action_forecast_available':selected_forecast is not None,
                'selected_forecast_rho_max':finite(selected_forecast[0].max()) if selected_forecast else None,
                'selected_forecast_done':selected_forecast[2]['done'] if selected_forecast else None,
                'selected_forecast_flags':{k:selected_forecast[2][k] for k in ['illegal','ambiguous','exceptions']} if selected_forecast else None,
                'nonzero_action':not np.array_equal(act_vect,action_vector(env.action_space())),
                'changed_topology_entries':int(np.count_nonzero(act.set_bus)),
                'redispatch_L1':float(np.abs(act.redispatch).sum()),'storage_L1':float(np.abs(act.storage_p).sum()),
                'curtailment_modified':int(np.count_nonzero(act.curtail!=-1)),
                'file':str(vector_file.relative_to(OUT)),'file_sha256':hashlib.sha256(vector_file.read_bytes()).hexdigest()}
            for event in trace.events:
                event_file.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
            event_file.flush()
            os.fsync(event_file.fileno())
            record['step_wall_before_step_log_s']=time.perf_counter()-step_start
            step_file.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            step_file.flush()
            os.fsync(step_file.fileno())
            completed=step
            if step%24==0 or done:
                print('scenario',args.scenario,args.mode,'step',step,'rho',record['after_rho_max'],'act_s',round(act_s,4),'sim',native_calls,flush=True)
            if done:
                break
    save('completion.json',{'status':'COMPLETE_BOUNDED_PREFIX','steps':completed,'done':bool(done),
        'wall_after_import_s':time.perf_counter()-started})
except Exception as e:
    save('failure.json',{'type':type(e).__name__,'message':str(e),'completed_steps':completed,
        'wall_after_import_s':time.perf_counter()-started,'traceback':traceback.format_exc()})
    if trace is not None:
        save('failure_step_events.json',trace.events)
    raise
finally:
    if trace is not None:
        trace.close()
    if env is not None:
        env.close()
