"""Training-only paired failure diagnostic; never modifies a training run.

Public forecast branches and privileged actual-data branches are kept separate.
Neither branch supplies training labels or changes an active version's screen.
"""
import gzip, traceback
from common import *

source = OUT/'v1/train/gnn/e0_2035-10-29_15'
target = OUT/'ramp_diagnostic.json'
assert not target.exists(), 'Refuse to overwrite a completed diagnostic'
rows = [json.loads(x) for x in gzip.open(source/'steps.jsonl.gz','rt')]
vectors = np.load(source/'vectors.npz')
metadata = json.loads((source/'metadata.json').read_text())
report = {'source':str(source.relative_to(ROOT)), 'physical_steps':0,
          'forecast_steps':0, 'qualified':False,
          'scope':'One training counterexample, two-step branches. Actual branches are privileged offline diagnostics, never policy input.'}
env = make_env()

def decode(vector):
    action=env.action_space()
    action.from_vect(vector)  # Grid2Op mutates in place and returns None.
    return action

def pack(obs, done, info):
    f=flags(info)
    return {'done':bool(done),**f,'rho_max':float(obs.rho.max()),
            'gen_margin_down':obs.gen_margin_down.tolist(),
            'actual_dispatch':obs.actual_dispatch.tolist(),
            'target_dispatch':obs.target_dispatch.tolist(),
            'observation_hash':digest(obs.to_vect())}

try:
    env.seed(metadata['seed']); env.set_id(metadata['scenario']); obs=env.reset()
    for i in range(131):
        assert digest(obs.to_vect())==rows[i]['before_hash'], f'Before-prefix mismatch at {i+1}'
        act=decode(vectors['action'][i])
        assert digest(action_vector(act))==rows[i]['action_hash']
        obs,_,done,info=env.step(act); report['physical_steps']+=1
        assert digest(obs.to_vect())==rows[i]['after_hash'], f'After-prefix mismatch at {i+1}'
        assert not done
    report['prefix_verified_steps']=131
    assert digest(obs.to_vect())==rows[131]['before_hash']
    noop=env.action_space()
    assert np.array_equal(vectors['action'][132],action_vector(noop)), 'Archived continuation is not no-op'
    restore=decode(vectors['action'][131])
    assert rows[131]['proposal']['offered'] and rows[131]['choice']==1
    report['archived_one_step_screen']=rows[131]['proposal']
    report['branches']={}
    for mode in ['public_forecast','privileged_actual']:
        for choice,act in [('restore',restore),('hold',noop)]:
            branch=obs.get_forecast_env() if mode=='public_forecast' else env.copy()
            try:
                if mode=='public_forecast':branch.reset()
                outcomes=[]
                for a in [act,noop]:
                    bo,_,bd,bi=branch.step(a)
                    report['forecast_steps' if mode=='public_forecast' else 'physical_steps']+=1
                    outcomes.append(pack(bo,bd,bi))
                    if bd:break
                report['branches'][mode+'_'+choice]=outcomes
            finally:branch.close()
    report['qualified']=True
except Exception:
    report['error']=traceback.format_exc()
finally:
    env.close();write_json(target,report)
print(json.dumps(report,indent=2,default=json_default))
assert report['qualified'], 'Diagnostic prefix/interface qualification failed; do not interpret branches'
