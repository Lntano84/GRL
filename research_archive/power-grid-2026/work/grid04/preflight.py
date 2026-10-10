"""Zero physical steps: qualification of reused assets and shortlist semantics."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'work/grid01'),str(ROOT/'work/grid02'),str(ROOT/'work/grid03')]
import grid2op
from lightsim2grid import LightSimBackend
from ljn_nn import make_agent_topoNN
from ljn_nn.modules.rewards import MaxRhoReward
from trace_agent import action_vector
from policies import random_ids,local_ids,attach_shortlist
from io_utils import encode
OUT=ROOT/'outputs/grid04'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
reused=json.loads((OUT/'reused_assets_frozen.json').read_text())
assert all(sha(ROOT/p)==v for p,v in reused.items())
design=json.loads((OUT/'design_freeze.json').read_text())
assert sha(OUT/'design_freeze.json')==(OUT/'design_freeze.sha256').read_text()
assert len(design['plan'])==20
assert local_ids(np.array([1.4,1.1,0.1]),[np.array([0]),np.array([1]),np.array([2])],3,2)==[0,1]
assert random_ids(1)==random_ids(1) and random_ids(1)!=random_ids(2)
assert len(set(random_ids(1)))==20
assert json.loads(encode({'float':np.float32(1.5),'integer':np.int64(2),'boolean':np.bool_(True)}))=={'float':1.5,'integer':2,'boolean':True}
try:
    encode({'x':np.float32(np.nan)})
    raise AssertionError('Nonfinite telemetry must fail')
except ValueError: pass
env=grid2op.make(str(ROOT/design['data_root']),test=True,backend=LightSimBackend(),reward_class=MaxRhoReward)
try:
    env.seed(0);env.set_id(design['selected_development_weeks'][0]);obs=env.reset()
    agent=make_agent_topoNN(env,str(ROOT/'work/grid02/ljn_nn'))
    class Dummy:
        step=0
        events=[]
    trace=Dummy()
    attach_shortlist(agent,'LOCAL20',env,trace)
    before=obs.to_vect().copy()
    first=agent.topo_12_unsafe._get_tested_action(obs)
    ids=trace.events[-1]['ids']
    assert len(first)==len(set(ids))==20
    assert all(0<=i<352 for i in ids)
    assert np.array_equal(before,obs.to_vect())
    hashes=[]
    for i in range(352):
        action=agent.topo_12_unsafe.gym_env.action_space.from_gym(i)
        rebuilt=env.action_space.from_vect(action_vector(action))
        assert not rebuilt.is_ambiguous()[0] and np.array_equal(action_vector(rebuilt),action_vector(action))
        hashes.append(hashlib.sha256(action_vector(action).tobytes()).hexdigest())
    incident=[x.tolist() for x in agent.topo_12_unsafe.grid04_incident]
    (OUT/'pool_metadata.json').write_text(json.dumps({'action_hashes':hashes,'incident_lines':incident},indent=2),encoding='utf-8')
    result={'passed':True,'physical_steps':0,'mapping_count':352,'local_hand_fixture':True,
        'deterministic_hash_permutation':True,'candidate_enumeration_passive':True,
        'reused_assets_unchanged':len(reused),'native_horizon':int(env.max_episode_duration()),
        'local_first_ids':ids,'pool_metadata_sha256':sha(OUT/'pool_metadata.json')}
    result['strict_numpy_scalar_json']=True
    (OUT/'preflight_v3.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
finally: env.close()
