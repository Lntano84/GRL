"""GRID05 worker, derived from the frozen GRID04 worker; fixed fallback only."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

parser=argparse.ArgumentParser()
parser.add_argument('--scenario',required=True)
parser.add_argument('--policy',choices=['NN20','FALLBACK90'],required=True)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid05/runs'/f'{args.scenario}__{args.policy}'
sys.path[:0]=[str(ROOT/'work/grid01'),str(ROOT/'work/grid02'),str(ROOT/'work/grid03'),str(ROOT/'work/grid04')]
assert not (OUT/'steps.jsonl').exists(),'Never overwrite a trajectory'
OUT.mkdir(parents=True,exist_ok=True)
def save(name,value):
    (OUT/name).write_text(encode(value,indent=2),encoding='utf-8')
import_started=time.perf_counter()
import numpy as np
from io_utils import encode
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import grid2op
from lightsim2grid import LightSimBackend
from ljn_nn import make_agent_topoNN
from ljn_nn.modules.rewards import MaxRhoReward
from trace_agent import Telemetry,digest,finite,flags,action_vector
from score_adapter import operational_other_rewards,public_cost_ledger,cost_rounding_tolerance,PREFIX
from fallback import attach_rank_trace,install_rule

save('import_ready.json',{'import_s':time.perf_counter()-import_started})
design=json.loads((ROOT/'outputs/grid05/design_freeze.json').read_text(encoding='utf-8'))
assert args.scenario in design['selected_development_weeks']
started=time.perf_counter()
env=None
trace=None
completed=0
physical_calls=0
try:
    t=time.perf_counter()
    env=grid2op.make(str(ROOT/design['data_root']),test=True,backend=LightSimBackend(),
        reward_class=MaxRhoReward,other_rewards=operational_other_rewards())
    env.seed(design['environment_seed'])
    env.set_id(args.scenario)
    obs=env.reset()
    make_s=time.perf_counter()-t
    horizon=int(env.max_episode_duration())
    assert horizon<=2017
    t=time.perf_counter()
    agent=make_agent_topoNN(env,str(ROOT/'work/grid02/ljn_nn'))
    agent.seed(design['environment_seed'])
    agent.reset(obs)
    agent_s=time.perf_counter()-t
    holder={}
    if args.policy=='FALLBACK90':
        install_rule(agent.topo_12_unsafe,agent.rho_safe,holder)
    trace=Telemetry(agent)
    holder['trace']=trace
    attach_rank_trace(agent.topo_12_unsafe,trace,args.policy)
    (OUT/'steps').mkdir()
    np.savez_compressed(OUT/'initial.npz',observation=obs.to_vect(),rho=obs.rho,p_or=obs.p_or,
        gen_p=obs.gen_p,load_p=obs.load_p,actual_dispatch=obs.actual_dispatch,
        curtailment_mw=obs.curtailment_mw,storage_power=obs.storage_power)
    save('metadata.json',{'scenario_id':env.chronics_handler.get_id(),'scenario':args.scenario,'policy':args.policy,
        'seed':design['environment_seed'],'max_steps':horizon,'make_s':make_s,'agent_construct_s':agent_s,
        'reward_class':type(env._reward_helper.template_reward).__name__,'parameters':env.parameters.to_dict(),
        'gen_cost_per_MW':env.gen_cost_per_MW.tolist(),'delta_time_seconds':float(env.delta_time_seconds),
        'gen_renewable':env.gen_renewable.tolist(),'rho_danger':agent.rho_danger,'rho_safe':agent.rho_safe,
        'optim_config':agent.optim.config,'pool_size':352,'top_k':20,
        'fixed_fallback':{'enabled':args.policy=='FALLBACK90','threshold':agent.rho_safe,'expanded_pool':352,'forced_test':False},
        'versions':{n:importlib.metadata.version(n) for n in ['grid2op','numpy','cvxpy','osqp','scs','lightsim2grid','torch','stable-baselines3','gymnasium']},
        'action_vector_layout':[{'name':name,'size':int(env.action_space()._get_array_from_attr_name(name).size)} for name in type(env.action_space()).attr_list_vect],
        'competition_normalized_score_certified':False,
        'telemetry':'Existing passive GRID01 wrappers; fresh vectors, includes telemetry cost, no extra simulation.'})
    reward=0.
    done=False
    with (OUT/'steps.jsonl').open('w',encoding='utf-8') as step_file,(OUT/'events.jsonl').open('w',encoding='utf-8') as event_file:
        for step in range(1,horizon+1):
            if time.perf_counter()-started>design['caps']['per_run_after_import_s']:
                raise TimeoutError('Frozen trajectory cap reached')
            trace.begin(step)
            step_start=time.perf_counter()
            before=obs.to_vect().copy()
            before_rho=obs.rho.copy()
            previous_curtailed=float(obs.curtailment_mw.sum(dtype=np.float64))
            before_counter=int(env.nb_highres_called)
            t=time.perf_counter()
            act=agent.act(obs,reward,done)
            act_s=time.perf_counter()-t
            physical_unchanged=np.array_equal(before,env.get_obs().to_vect(),equal_nan=True)
            assert physical_unchanged
            vector=action_vector(act)
            act_hash=digest(vector)
            selected=trace.forecasts.get((act_hash,1))
            t=time.perf_counter()
            obs,reward,done,info=env.step(act)
            physical_calls+=1
            env_step_s=time.perf_counter()-t
            native_calls=int(env.nb_highres_called)-before_counter
            traced_calls=sum(e['kind']=='simulate' for e in trace.events)
            assert native_calls==traced_calls,(native_calls,traced_calls)
            native_flags=flags(info)
            raw=float(info['rewards'][f'{PREFIX}_grid_operational_cost'])
            # A blackout's error-terminal fallback is not a measured physical ledger.
            eligible=bool((obs.gen_p>0).any()) and not native_flags['exceptions']
            ledger=public_cost_ledger(obs,env,previous_curtailed) if eligible else None
            tolerance=float(cost_rounding_tolerance(obs,env)) if eligible else None
            residual=ledger['recomputed_raw_cost']-raw if eligible else None
            path=OUT/'steps'/f'{step:04}.npz'
            np.savez_compressed(path,observation=obs.to_vect(),action=vector,rho=obs.rho,p_or=obs.p_or,
                gen_p=obs.gen_p,load_p=obs.load_p,actual_dispatch=obs.actual_dispatch,
                curtailment_mw=obs.curtailment_mw,storage_power=obs.storage_power,
                gen_p_before_curtail=obs.gen_p_before_curtail,
                forecast_rho=selected[0] if selected else np.full(env.n_line,np.nan),
                forecast_p_or=selected[1] if selected else np.full(env.n_line,np.nan))
            record={'step':step,'current_step':int(obs.current_step),'timestamp':str(obs.get_time_stamp()),
                'before_observation_hash':digest(before),'observation_hash':digest(obs.to_vect()),'action_hash':act_hash,
                'before_rho_max':finite(before_rho.max()),'after_rho_max':finite(obs.rho.max()),
                'reward':finite(reward),'done':bool(done),**native_flags,'chronics_done':bool(env.chronics_handler.done()),
                'exception_types':[type(x).__name__ for x in info.get('exception',[])],
                'physical_observation_unchanged_during_act':physical_unchanged,'act_wall_s':act_s,'env_step_s':env_step_s,
                'simulate_calls':native_calls,'ledger_eligible':eligible,'ledger':ledger,'raw_operational_cost':raw,
                'cost_residual':residual,'cost_rounding_tolerance':tolerance,
                'renewable_component':float(info['rewards'][f'{PREFIX}_new_renewable_sources_usage']),
                'assistant_component':float(info['rewards'][f'{PREFIX}_assistant_confidence']),
                'selected_action_forecast_available':selected is not None,
                'selected_forecast_rho_max':finite(selected[0].max()) if selected else None,
                'nonzero_action':not np.array_equal(vector,action_vector(env.action_space())),
                'changed_topology_entries':int(np.count_nonzero(act.set_bus)),
                'file':str(path.relative_to(OUT)),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            for event in trace.events:
                event_file.write(encode(event)+'\n')
            event_file.flush()
            os.fsync(event_file.fileno())
            record['step_wall_before_step_log_s']=time.perf_counter()-step_start
            step_file.write(encode(record)+'\n')
            step_file.flush()
            os.fsync(step_file.fileno())
            completed=step
            if eligible:
                assert abs(residual)<=tolerance,('Public ledger mismatch',residual,tolerance)
            if step%96==0 or done:
                print(args.scenario,args.policy,'step',step,'rho',record['after_rho_max'],'sim',native_calls,flush=True)
            if done:
                break
    save('completion.json',{'status':'NATIVE_EPISODE_END' if done else 'NATIVE_HORIZON_REACHED',
        'steps':completed,'physical_calls':physical_calls,'done':bool(done),
        'reached_native_horizon':bool(obs.current_step>=horizon or env.chronics_handler.done()),
        'wall_after_import_s':time.perf_counter()-started})
except Exception as e:
    save('failure.json',{'type':type(e).__name__,'message':str(e),'completed_steps':completed,
        'physical_calls':physical_calls,'wall_after_import_s':time.perf_counter()-started,'traceback':traceback.format_exc()})
    if trace is not None:
        save('failure_step_events.json',trace.events)
    raise
finally:
    save('execution_count.json',{'physical_calls':physical_calls,'saved_steps':completed,'wall_after_import_s':time.perf_counter()-started})
    if trace is not None: trace.close()
    if env is not None: env.close()
