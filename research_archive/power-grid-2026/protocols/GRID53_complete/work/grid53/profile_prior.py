"""Profile unchanged public prior on one saved old-development observation."""
import cProfile, hashlib, json, os, pstats, sys, time
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid53'
sys.path[:0]=[str(ROOT/'work/grid08'),str(ROOT/'work/grid21'),str(ROOT/'work/grid20')]
from common import np, make_env, ResidualControl, digest
from prior import public_prior

folder=ROOT/'outputs/grid51/runs/r0__2035-03-05_0__COMPRESSED_GNN0'
call_path=next(iter(sorted(folder.glob('call[0-9][0-9][0-9].json'))))
call=json.loads(call_path.read_text(encoding='utf-8'))
state_path=call_path.with_name(call_path.stem+'_state.npz')
env=make_env();env.seed(0);env.set_id('2035-03-05_0');initial=env.reset()
ctl=ResidualControl(env,initial)
with np.load(state_path) as z:
    obs=initial.copy();obs.from_vect(z['observation'])
assert digest(obs.to_vect())==call['state_hash']
ids=call['eligible'];aliases={int(k):v for k,v in call['aliases'].items()}
counts=int(env.nb_highres_called)
# Warm only already-legal input inference; do not simulate any candidate.
value,order=public_prior(obs,env,ctl,ids,aliases)
assert digest(value)==call['prior_hash'], 'Saved prior must match before timing.'
profiles=[];wall=[]
for repeat in range(3):
    pr=cProfile.Profile();begun=time.perf_counter();pr.enable()
    result,ordered=public_prior(obs,env,ctl,ids,aliases)
    pr.disable();elapsed=time.perf_counter()-begun
    assert np.array_equal(result,value) and ordered==order
    stats=pstats.Stats(pr)
    rows=[]
    for (file,line,name),(primitive,calls,self_s,cumulative_s,callers) in stats.stats.items():
        rows.append({'file':file,'line':line,'name':name,'primitive_calls':primitive,'calls':calls,
                     'self_s':self_s,'cumulative_s':cumulative_s})
    profiles.append({'repeat':repeat,'wall_s':elapsed,'largest_self':sorted(rows,key=lambda r:-r['self_s'])[:30],
                     'largest_cumulative':sorted(rows,key=lambda r:-r['cumulative_s'])[:30]})
    wall.append(elapsed)
assert int(env.nb_highres_called)==counts
R={'passed':True,'state_source':state_path.relative_to(ROOT).as_posix(),
   'state_sha256':hashlib.sha256(state_path.read_bytes()).hexdigest(),
   'prior_source_sha256':hashlib.sha256((ROOT/'work/grid21/prior.py').read_bytes()).hexdigest(),
   'saved_prior_matched':True,'exact_repeat_scores_and_order':True,'new_candidate_forecasts':0,
   'new_fits':0,'profiles':profiles,
   'limitations':'One saved old-development state; cProfile adds overhead. Local bottleneck diagnosis, not closed-loop speed or GNN contribution. No historical result changed.'}
(OUT/'prior_profile.json').write_text(json.dumps(R,indent=2,allow_nan=False),encoding='utf-8')
env.close()
print(json.dumps({'passed':True,'walls_s':wall,'largest_self':profiles[-1]['largest_self'][:12],
                  'largest_cumulative':profiles[-1]['largest_cumulative'][:12]},indent=2))
