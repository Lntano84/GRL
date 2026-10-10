"""Formal-grid compatibility and passive scoring smoke, <=36 physical steps."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import grid2op
from lightsim2grid import LightSimBackend

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"work/grid02"))
sys.path.insert(0,str(ROOT/"work/grid01"))
from ljn_nn import make_agent_topoNN
from ljn_nn.modules.rewards import MaxRhoReward
from trace_agent import action_vector, digest, flags
from score_adapter import operational_other_rewards, public_cost_ledger, cost_rounding_tolerance, PREFIX

OUT=ROOT/"outputs/grid03/formal_smoke_corrected_ledger"
DATA=ROOT/"work/grid03/formal_env_initialized"
split=json.loads((ROOT/"outputs/grid03/scenario_split_frozen.json").read_text(encoding="utf-8"))
scenario=min(split["extracted_subsets"]["development"],key=lambda x:hashlib.sha256(("GRID03-smoke:"+x).encode()).hexdigest())
if OUT.exists():
    raise RuntimeError("Existing smoke output; do not overwrite")
OUT.mkdir(parents=True)
started=time.monotonic()
physical_steps=0


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if hasattr(v,"item") else str(v)),encoding="utf-8")


def make(path,scored=False):
    kwargs={"other_rewards":operational_other_rewards()} if scored else {}
    env=grid2op.make(str(path),test=True,backend=LightSimBackend(),reward_class=MaxRhoReward,**kwargs)
    env.seed(0)
    env.set_id(scenario)
    return env,env.reset()


def signature(env):
    attrs=["name_line","name_gen","name_load","name_sub","name_storage","sub_info","gen_type","gen_renewable",
           "gen_cost_per_MW","gen_pmax","gen_pmin","line_or_to_subid","line_ex_to_subid","gen_to_subid","load_to_subid",
           "storage_to_subid","line_or_pos_topo_vect","line_ex_pos_topo_vect","gen_pos_topo_vect","load_pos_topo_vect","storage_pos_topo_vect"]
    return {k:np.asarray(getattr(env,k)).tolist() for k in attrs}|{"thermal_limit":env.get_thermal_limit().tolist(),
        "parameters":env.parameters.to_dict(),"action_vect_size":env.action_space.size(),"obs_vect_size":env.observation_space.size(),
        "delta_time_seconds":env.delta_time_seconds,"rules_type":type(env._game_rules.legal_action).__name__,
        "rule_substations_by_area":{str(k):np.asarray(v).tolist() for k,v in env._game_rules.legal_action.substations_id_by_area.items()},
        "action_keys":sorted(env.action_space().authorized_keys),"action_vect_fields":list(type(env.action_space()).attr_list_vect),
        "alertable_line_ids":np.asarray(env.alertable_line_ids).tolist(),"n_busbar_per_sub":env.n_busbar_per_sub}


env=None
control=None
bundled=None
try:
    env,obs=make(DATA,True)
    control,obs_c=make(DATA,False)
    native_horizon=int(env.max_episode_duration())
    source=Path(grid2op.__file__).parent/"data/l2rpn_idf_2023"
    bundled=grid2op.make(str(source),test=True,backend=LightSimBackend(),reward_class=MaxRhoReward)
    formal_sig=signature(env)
    bundled_sig=signature(bundled)
    diffs={key:{"formal":formal_sig[key],"bundled":bundled_sig[key]} for key in formal_sig if formal_sig[key]!=bundled_sig[key]}
    save("compatibility.json",{"formal":formal_sig,"bundled":bundled_sig,"differences":diffs,"scenario":scenario,"native_horizon":native_horizon})
    assert not diffs,"Native grid/config difference; no silent overrides"
    assert np.array_equal(obs.to_vect(),obs_c.to_vect())
    bundled.close()
    bundled=None
    passive_agent=make_agent_topoNN(env,str(ROOT/"work/grid02/ljn_nn"))
    passive_agent.seed(0)
    passive_agent.reset(obs)
    passive_reward=0.0
    records=[]
    renewable=int(np.flatnonzero(env.gen_renewable)[np.argmax(obs.gen_p[env.gen_renewable])])
    for step in range(1,13):
        previous_curtailed_mw=float(obs.curtailment_mw.sum(dtype=np.float64))
        if step==3:
            limit=float(np.clip(0.98*obs.gen_p[renewable]/env.gen_pmax[renewable],0.0,0.95))
            payload={"curtail":[(renewable,limit)]}
        elif step==5:
            payload={"set_storage":[(0,1.0)]}
        elif step==7:
            payload={"curtail":[(renewable,1.0)]}
        else:
            payload={}
        # Use the existing controller so a no-op blackout cannot masquerade as an
        # interface failure. Extra hand probes are applied only in a safe state.
        action=passive_agent.act(obs,passive_reward,False)
        probe_applied=bool(payload) and float(obs.rho.max())<0.9
        if probe_applied:
            action+=env.action_space(payload)
        action_c=control.action_space.from_vect(action_vector(action))
        obs,reward,done,info=env.step(action)
        obs_c,reward_c,done_c,info_c=control.step(action_c)
        physical_steps+=2
        passive_reward=float(reward)
        same=np.array_equal(obs.to_vect(),obs_c.to_vect()) and float(reward)==float(reward_c) and bool(done)==bool(done_c) and flags(info)==flags(info_c)
        ledger=public_cost_ledger(obs,env,previous_curtailed_mw)
        reported=float(info["rewards"][f"{PREFIX}_grid_operational_cost"])
        residual=ledger["recomputed_raw_cost"]-reported
        tolerance=cost_rounding_tolerance(obs,env)
        record={"step":step,"action_hash":digest(action_vector(action)),"observation_hash":digest(obs.to_vect()),"same_as_unscored":same,
                "reported_operational_cost":reported,"ledger":ledger,"residual":residual,"rounding_tolerance":tolerance,"probe_applied":probe_applied,
                "done":bool(done),"native_flags":flags(info)}
        records.append(record)
        save("passive_steps.json",records)
        assert same,"Passive metrics changed the physical/control trajectory"
        assert abs(residual)<=tolerance,"Independent raw cost reconciliation failed"
        assert not info.get("is_illegal") and not info.get("is_ambiguous") and not info.get("exception")
        if done:
            break
    control.close()
    control=None
    env.close()
    env,obs=make(DATA,True)
    agent=make_agent_topoNN(env,str(ROOT/"work/grid02/ljn_nn"))
    agent.seed(0)
    agent.reset(obs)
    # Check every candidate against the current action vector layout, retaining original order/contents.
    checks=[]
    for i in range(agent.topo_12_unsafe.gym_env.action_space.n):
        action=agent.topo_12_unsafe.gym_env.action_space.from_gym(i)
        vec=action_vector(action)
        rebuilt=env.action_space.from_vect(vec)
        ambiguous,why=rebuilt.is_ambiguous()
        assert not ambiguous and np.array_equal(action_vector(rebuilt),vec)
        checks.append({"row":i,"hash":digest(vec),"ambiguous":bool(ambiguous)})
    assert len(checks)==352
    save("mapping_checks.json",checks)
    nn=[]
    reward=0.0
    done=False
    for step in range(1,13):
        previous_curtailed_mw=float(obs.curtailment_mw.sum(dtype=np.float64))
        t=time.monotonic()
        action=agent.act(obs,reward,done)
        act_s=time.monotonic()-t
        obs,reward,done,info=env.step(action)
        physical_steps+=1
        assert not info.get("is_illegal") and not info.get("is_ambiguous") and not info.get("exception")
        ledger=public_cost_ledger(obs,env,previous_curtailed_mw)
        reported=float(info["rewards"][f"{PREFIX}_grid_operational_cost"])
        assert abs(ledger["recomputed_raw_cost"]-reported)<=cost_rounding_tolerance(obs,env)
        nn.append({"step":step,"action_hash":digest(action_vector(action)),"observation_hash":digest(obs.to_vect()),"rho_max":float(obs.rho.max()),
                   "search_reward":float(reward),"operational_cost":reported,"ledger":ledger,
                   "act_s":act_s,"native_flags":flags(info),"done":bool(done)})
        save("nn_steps.json",nn)
        if done:
            break
    save("summary.json",{"passed":True,"scenario":scenario,"native_horizon":native_horizon,"physical_steps":physical_steps,
        "passive_pairs":len(records),"nn_steps":len(nn),"mapping_checks":len(checks),"max_cost_abs_residual":max(abs(r["residual"]) for r in records),
        "wall_s":time.monotonic()-started,"test_or_validation_agent_outcomes_used":False,"research_claim":"None; bounded interface qualification only."})
    print(json.dumps({"passed":True,"physical_steps":physical_steps,"native_horizon":native_horizon,"scenario":scenario}),flush=True)
finally:
    save("execution_count.json",{"physical_steps":physical_steps,"wall_s":time.monotonic()-started})
    for e in [env,control,bundled]:
        if e is not None:
            e.close()
