"""Build a GRID02 harness from the already audited GRID01 harness, with explicit edits."""
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / 'work/grid01/run_one.py').read_text(encoding='utf-8')
modified = source
edits = [
    ("parser.add_argument('--mode',choices=['full','replay'],required=True)", "parser.add_argument('--mode',choices=['full','smoke'],required=True)\nparser.add_argument('--policy',choices=['FULL','NN20','NN352'],required=True)"),
    ("OUT=ROOT/'outputs/grid01'/f'scenario{args.scenario}_{args.mode}'", "OUT=ROOT/'outputs/grid02'/f'{args.policy}_scenario{args.scenario}_{args.mode}'\nimport sys\nsys.path.insert(0,str(ROOT/'work/grid01'))"),
    ("import cvxpy as cp", "import cvxpy as cp\nimport torch\ntorch.set_num_threads(1)\ntorch.set_num_interop_threads(1)"),
    ("from ljn_heuristic import make_agent_challenge", "from ljn_nn import make_agent_challenge,make_agent_topoNN"),
    ("from ljn_heuristic.modules.rewards", "from ljn_nn.modules.rewards"),
    ("agent=make_agent_challenge(env,str(ROOT/'work/grid01/ljn_heuristic'))", "factory=make_agent_challenge if args.policy=='FULL' else make_agent_topoNN\n    agent=factory(env,str(ROOT/'work/grid02/ljn_nn'))\n    if args.policy=='NN352':\n        agent.topo_12_unsafe.top_k=int(agent.topo_12_unsafe.model.action_space.n)\n        assert agent.topo_12_unsafe.top_k==352"),
    ("trace=Telemetry(agent)", "trace=Telemetry(agent)\n    if args.policy in ['NN20','NN352']:\n        original_topk=agent.topo_12_unsafe.get_top_k\n        def traced_topk(gym_obs,top_k):\n            start_topk=time.perf_counter()\n            result=original_topk(gym_obs,top_k)\n            trace.events.append({'kind':'nn_topk','step':trace.step,'module':'topo_12_unsafe',\n                'wall_s':time.perf_counter()-start_topk,'input_hash':digest(gym_obs),'ids':result.tolist()})\n            return result\n        agent.topo_12_unsafe.get_top_k=traced_topk"),
    ("'max_steps':258 if args.mode=='full' else 24", "'max_steps':575 if args.mode=='full' else 24,'policy':args.policy,'native_max_episode_duration':int(env.max_episode_duration()),'NN_topk':agent.topo_12_unsafe.top_k if args.policy!='FULL' else None"),
    ("['grid2op','numpy','cvxpy','osqp','scs','lightsim2grid']", "['grid2op','numpy','cvxpy','osqp','scs','lightsim2grid','torch','stable-baselines3','gymnasium']"),
    ("for step in range(1,259 if args.mode=='full' else 25):", "for step in range(1,576 if args.mode=='full' else 25):"),
    ("time.perf_counter()-started>600", "time.perf_counter()-started>900"),
    ("Frozen 600s trajectory cap reached", "Frozen 900s trajectory cap reached"),
    ("'reward':finite(reward),'done':bool(done),**flags(info),", "'reward':finite(reward),'done':bool(done),**flags(info),\n                'chronics_done':bool(env.chronics_handler.done()),\n                'exception_types':[type(x).__name__ for x in info.get('exception',[])],\n                'disconnected_lines_count':int(np.count_nonzero(info.get('disc_lines',[])>=0)) if isinstance(info.get('disc_lines'),np.ndarray) else None,"),
]
for old, new in edits:
    if old not in modified:
        raise RuntimeError(f'Missing harness edit: {old}')
    modified = modified.replace(old, new)
(ROOT / 'work/grid02/run_one.py').write_text(modified, encoding='utf-8')
(ROOT / 'outputs/grid02/harness_GRID01_to_GRID02.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),modified.splitlines(True),fromfile='GRID01/run_one.py',tofile='GRID02/run_one.py')),encoding='utf-8')
print('Built GRID02 run harness; source diff saved.')
