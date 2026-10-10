import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid02'
sys.path.insert(0, str(ROOT / 'work/grid01'))
t0 = time.perf_counter()
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import grid2op
from lightsim2grid import LightSimBackend
from ljn_nn.modules.rewards import MaxRhoReward
from trace_agent import action_vector, digest

env = None
result = {'status': 'RUNNING', 'actual_steps': 0}
def save():
    (OUT / 'preflight.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
try:
    result['versions'] = {n: importlib.metadata.version(n) for n in ['torch', 'stable-baselines3', 'gymnasium', 'numpy', 'scipy', 'cvxpy', 'lightsim2grid', 'grid2op']}
    result['torch_threads'] = torch.get_num_threads()
    env = grid2op.make(str(Path(grid2op.__file__).parent / 'data/l2rpn_idf_2023'), test=True, backend=LightSimBackend(), reward_class=MaxRhoReward)
    env.seed(0)
    env.set_id(0)
    obs = env.reset()
    result['native_max_episode_duration'] = int(env.max_episode_duration())
    migration=json.loads((OUT/'mapping_migration.json').read_text(encoding='utf-8'))
    assert migration['status']=='PASS_EXACT_LEGACY_STORAGE_MIGRATION'
    result['legacy_mapping_migration_checked']=True
    p = ROOT / 'work/grid02/ljn_nn/assets/nn_act_space/action_12_unsafe_nn.npz'
    with np.load(p, allow_pickle=True) as data:
        result['mapping_keys'] = data.files
        result['vector_shape'] = list(data['vect_actions'].shape)
        result['curriculum_shape'] = list(data['curriculum_id_actions'].shape)
        actions = data['g2op_id_actions']
        vectors = np.asarray(data['vect_actions'])
    result['mapping_count'] = len(actions)
    result['mapping_original_type'] = str(type(actions[0]))
    array = np.stack([action_vector(x) for x in actions])
    # Object semantics must agree with the separately supplied vector map.
    if vectors.shape == array.T.shape:
        vectors = vectors.T
    assert vectors.shape == array.shape, (vectors.shape, array.shape)
    assert np.array_equal(array, vectors, equal_nan=True), float(np.max(np.abs(array-vectors)))
    rebuilt = [env.action_space.from_vect(v) for v in array]
    assert all(np.array_equal(action_vector(a), v, equal_nan=True) for a, v in zip(rebuilt, array))
    result['ambiguous'] = sum(bool(a.is_ambiguous()[0]) for a in actions)
    assert result['ambiguous'] == 0
    result['mapping_vector_sha256'] = digest(array)
    result['mapping_vectors_equal_objects_and_rebuild'] = True
    # Source models have pickle metadata. Use only the pinned, hashed author archive.
    from ljn_nn import make_agent_topoNN
    t = time.perf_counter()
    agent = make_agent_topoNN(env, str(ROOT / 'work/grid02/ljn_nn'))
    result['construct_s'] = time.perf_counter() - t
    model = agent.topo_12_unsafe.model
    result['policy_class'] = str(type(model.policy))
    result['observation_shape'] = list(model.observation_space.shape)
    result['action_count'] = int(model.action_space.n)
    result['topk'] = agent.topo_12_unsafe.top_k
    assert result['action_count'] == len(actions)
    assert result['topk'] == 20
    gym_obs = agent.topo_12_unsafe.gym_env.observation_space.to_gym(obs)
    assert np.array_equal(gym_obs, obs.rho)
    with torch.no_grad():
        distribution = model.policy.get_distribution(torch.from_numpy(gym_obs).reshape(1, -1))
        logits = distribution.distribution.logits.cpu().numpy()[0]
        ids = agent.topo_12_unsafe.get_top_k(gym_obs, 20)
        ids2 = agent.topo_12_unsafe.get_top_k(gym_obs, 20)
    assert np.isfinite(logits).all()
    assert np.array_equal(ids, ids2)
    assert len(set(ids)) == 20 and (ids >= 0).all() and (ids < len(actions)).all()
    selected = agent.topo_12_unsafe._get_tested_action(obs)
    assert all(np.array_equal(action_vector(a), array[i]) for a, i in zip(selected, ids))
    result['initial_top20'] = ids.tolist()
    result['input_exactly_observed_rho'] = True
    # Compare published action sets; policies do not necessarily share candidate universes.
    full = np.load(ROOT / 'work/grid01/original_ljn/assets/action_12_unsafe.npz')['action_space']
    full_hashes = {digest(v) for v in full}
    result['mapping_overlap_with_full421'] = sum(digest(v) in full_hashes for v in array)
    result['unique_mapping_actions'] = len({digest(v) for v in array})
    result['model_parameters'] = sum(p.numel() for p in model.policy.parameters())
    np.savez_compressed(OUT / 'preflight_vectors.npz', mapping_vectors=array, initial_rho=gym_obs, initial_logits=logits, initial_top20=ids)
    result['status'] = 'PASS_NO_PHYSICAL_STEPS'
except Exception as error:
    result['status'] = 'FAIL_QUALIFICATION'
    result['error'] = type(error).__name__ + ': ' + str(error)
    result['traceback'] = traceback.format_exc()
    raise
finally:
    result['wall_s'] = time.perf_counter() - t0
    save()
    if env is not None:
        env.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ['traceback']}, ensure_ascii=False), flush=True)
