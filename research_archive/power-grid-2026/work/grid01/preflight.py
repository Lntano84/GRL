import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid01'
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

started=time.perf_counter()
try:
    import numpy as np
    import cvxpy as cp
    import grid2op
    from lightsim2grid import LightSimBackend
    from ljn_heuristic import make_agent_challenge
    from ljn_heuristic.modules.rewards import MaxRhoReward
    # Layered environment must preserve every original runtime package version.
    old=[]
    for line in (ROOT/'outputs/grid00/requirements.lock').read_text(encoding='utf-8').splitlines():
        name,version=line.split(' --hash=')[0].split('==')
        assert importlib.metadata.version(name)==version,(name,version,importlib.metadata.version(name))
        old.append(name)
    data=Path(grid2op.__file__).parent/'data/l2rpn_idf_2023'
    env=grid2op.make(str(data),test=True,backend=LightSimBackend(),reward_class=MaxRhoReward)
    try:
        env.seed(0)
        env.set_id(0)
        obs=env.reset()
        assets=[]
        for p in sorted((ROOT/'work/grid01/original_ljn/assets').glob('*.npz')):
            with np.load(p,allow_pickle=False) as z:
                acts=z['action_space']
                assert acts.ndim==2 and acts.shape[1]==env.action_space.size()
                assert np.isfinite(acts).all()
                max_error=0.
                ambiguities=0
                not_single_sub=0
                for row in acts:
                    act=env.action_space()
                    act.from_vect(row)
                    max_error=max(max_error,float(np.max(np.abs(act.to_vect()-row))))
                    ambiguities+=int(act.is_ambiguous()[0])
                    subs=act.as_dict().get('set_bus_vect',{}).get('modif_subs_id',[])
                    not_single_sub+=int(len(subs)!=1)
                assert max_error==0.
                assets.append({'name':p.name,'shape':list(acts.shape),'count':len(acts),'max_roundtrip_error':max_error,
                    'intrinsic_ambiguities':ambiguities,'not_single_substation':not_single_sub,
                    'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        t=time.perf_counter()
        agent=make_agent_challenge(env,str(ROOT/'work/grid01/ljn_heuristic'))
        construct_s=time.perf_counter()-t
        assert len(agent.topo_12_unsafe.topo_act_list)==421
        assert len(agent.topo_n1_unsafe.topo_act_list)==909
        save('preflight.json',{'status':'PASS_CONSTRUCTION_AND_ASSETS','versions_unchanged':old,'assets':assets,
            'versions':{name:importlib.metadata.version(name) for name in ['grid2op','numpy','cvxpy','osqp','scs','lightsim2grid']},
            'installed_cvxpy_solvers':cp.installed_solvers(),'author_solver_order':agent.optim.SOLVER_TYPES,
            'agent_construct_s':construct_s,'wall_s':time.perf_counter()-started,
            'scenario':env.chronics_handler.get_id(),'initial_rho_max':float(obs.rho.max()),
            'action_dimension':int(env.action_space.size()),'reward':type(env._reward_helper.template_reward).__name__,
            'native_parameters':env.parameters.to_dict(),
            'warning':'No agent action/actual step executed in preflight; contextual legality not certified.'})
        print('PASS; original assets reconstructed; non-NN author class constructed',flush=True)
    finally:
        env.close()
except Exception as e:
    save('preflight_failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
    raise
