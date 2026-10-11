import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT / "work/grid28"),str(ROOT / "work/grid08")]
from common import np,make_env,ResidualControl,action_vector,digest,write_json
from engine import Search
OUT=ROOT / "outputs/grid28"
D=json.loads((OUT / "runtime_design.json").read_text())
env=make_env();obs0=env.reset();ctl=ResidualControl(env,obs0);counter=int(env.nb_highres_called)
checks=[];fake_queries=0
saved_class=type(obs0);original=saved_class.simulate;owned="simulate" in saved_class.__dict__
active={}
def fake(self,action,**kwargs):
    global fake_queries
    if not active:return original(self,action,**kwargs)
    r=active['truth'][digest(action_vector(action))];assert r['pool_id'] not in active['seen']
    active['seen'].append(r['pool_id']);fake_queries+=1
    return SimpleNamespace(rho=np.asarray([r['rho']])),r['reward'],r['done'],{
        'exception':['saved'] if r['exceptions'] else [],'is_illegal':r['illegal'],'is_ambiguous':r['ambiguous']}
saved_class.simulate=fake
try:
    searchers={}
    for m in D['methods']:
        searchers[m['method']]=Search(env,ctl,m['rule'],128,ROOT / m['checkpoint'] if m['checkpoint'] else None)
        searchers[m['method']].policy=m['policy']
    offline=json.loads((OUT / 'offline.json').read_text())['rows']
    for week in D['weeks']:
        directory=ROOT / 'outputs/grid26/runs' / f'r0__{week}__FULL'
        for path in sorted(directory.glob('call[0-9][0-9][0-9].json')):
            full=json.loads(path.read_text())
            with np.load(path.with_name(path.stem+'_state.npz')) as f:
                obs=obs0.copy();obs.from_vect(f['observation']);base=env.action_space();base.from_vect(f['base_action'])
            assert digest(obs.to_vect())==full['state_hash']
            for m in D['methods']:
                search=searchers[m['method']]
                elig,reps,aliases,actions,changed=search.prepare(obs,base)
                assert reps==full['representatives'] and aliases=={int(k):v for k,v in full['aliases'].items()}
                assert search.subids==json.loads((OUT / 'offline.json').read_text())['public_subids']
                active.update(truth={r['combined_hash']:r for r in full['outcomes']},seen=[])
                selected,log,_,_=search.choose(obs,base,0.)
                assert active['seen']==[r['pool_id'] for r in log['outcomes']]
                if m['policy']=='PROTECTED':
                    expected=json.loads((directory.parent / f'r0__{week}__GNN0_128_GUARD' / path.name).read_text())
                    assert log['public_queries']==expected['public_queries'] and log['chosen_pool_id']==expected['chosen_pool_id']
                    assert active['seen']==[r['pool_id'] for r in expected['outcomes']]
                else:
                    ranker='GNN0_128_GUARD' if m['checkpoint'] else 'OLDNN128_GUARD'
                    expected=next(r for r in offline if r['week']==week and r['call']==path.name and r['ranker']==ranker and r['policy']==m['policy'])
                    assert log['public_queries']==expected['queries'] and log['chosen_pool_id']==expected['choice']
                assert (None if selected is None else digest(action_vector(selected)))==(None if log['chosen_pool_id'] is None else digest(action_vector(actions[log['chosen_pool_id']])))
                active.clear();checks.append({'week':week,'call':path.name,'method':m['method'],'passed':True})
    assert int(env.nb_highres_called)==counter
finally:
    if owned:saved_class.simulate=original
    else:delattr(saved_class,'simulate')
    env.close()
assert len(checks)==78
write_json(OUT / 'preflight.json',{'passed':True,'checks':checks,'fake_queries':fake_queries,'new_ac_forecasts':0,'new_fits':0,
    'scope':'Real runtime masks/aliases/rank/dispatch agree with frozen offline policy; provider returns only purchased outcomes; no new AC or physical replay.'})
paths=list((ROOT / 'work/grid28').glob('*.py'))+[ROOT / 'work/grid27/policy.py',ROOT / 'work/grid26/engine.py',ROOT / 'work/grid24/guard_adapter.py',ROOT / 'work/grid08/common.py',ROOT / 'work/grid20/components.py',ROOT / 'work/grid21/prior.py']
write_json(OUT / 'code_freeze.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
print(json.dumps({'passed':True,'checks':len(checks),'fake_queries':fake_queries}))
