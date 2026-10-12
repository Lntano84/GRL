"""Check original/cached priors and reconstructed old/new/PPO ranking exactly."""
import hashlib,json,math,os,sys,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid54'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
D=read(OUT/'design.json')
for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
assert not (OUT/'qualification.json').exists()
sys.path[:0]=[str(ROOT/'work/grid40'),str(ROOT/'work/grid28'),str(ROOT/'work/grid50'),str(ROOT/'work/grid54'),str(ROOT/'work/grid08')]
from common import np, make_env, ResidualControl, digest, action_vector
import engine
from adapter import install_search_adapter
from online_adapter import install_free_adapter
install_search_adapter(engine.Search);install_free_adapter(engine.Search)
from cache_prior import ORIGINAL_PRIOR,public_prior_cached,install_prior_cache
from components import fingerprint
patched=install_prior_cache();assert 'online_adapter' in patched
env=make_env();env.seed(0);env.set_id('2035-03-05_0');initial=env.reset();ctl=ResidualControl(env,initial)
searches={};rows=[];start_count=int(env.nb_highres_called);begun=time.perf_counter()
for n,case in enumerate(D['cases']):
    p=ROOT/case['call'];assert sha(p)==case['sha256']
    state=p.with_name(p.stem+'_state.npz');assert sha(state)==case['state_sha256']
    c=read(p);item=case['item']
    with np.load(state) as z:
        obs=initial.copy();obs.from_vect(z['observation'])
        base=env.action_space();base.from_vect(z['base_action'])
    assert digest(obs.to_vect())==c['state_hash']
    eligible=c['eligible'];reps=c['representatives'];aliases={int(k):v for k,v in c['aliases'].items()}
    outputs={};times={}
    funcs=[('original',ORIGINAL_PRIOR),('cached',public_prior_cached)]
    if n%2:funcs.reverse()
    for name,fn in funcs:
        stamp=time.perf_counter();outputs[name]=fn(obs,env,ctl,eligible,aliases);times[name]=time.perf_counter()-stamp
    a,order=outputs['original'];b,border=outputs['cached']
    assert np.array_equal(a,b) and order==border and digest(b)==c['prior_hash'],case['call']
    method=item['method']
    if method not in searches:
        searches[method]=engine.Search(env,ctl,item['rule'],item['budget'],ROOT/item['checkpoint'] if item['checkpoint'] else None)
    search=searches[method]
    actual,pvalue,features,details=search.rank(obs,base,eligible,reps,aliases)
    assert actual==c['order'] and np.array_equal(pvalue,b),(method,case['call'])
    assert (None if features is None else fingerprint(features))==c['input_hash'],case['call']
    rows.append({'source':case['call'],'method':method,'original_s':times['original'],'cached_s':times['cached'],
                 'scores_order_saved_rank_and_features_exact':True})
assert int(env.nb_highres_called)==start_count
steady_metadata_stats=ctl._grid54_prior_metadata.stats();assert steady_metadata_stats['builds']==1

# Runtime-only mutation probes. Restore originals; no artifact/library file changes.
case=D['cases'][0];c=read(ROOT/case['call'])
with np.load((ROOT/case['call']).with_name(Path(case['call']).stem+'_state.npz')) as z:
    obs=initial.copy();obs.from_vect(z['observation'])
eligible=c['eligible'];aliases={int(k):v for k,v in c['aliases'].items()}
library=ctl.base.topo_n1_unsafe.topo_act_list;original_action=library[0]
def mutation_check(name):
    count=ctl._grid54_prior_metadata.builds
    a,o=ORIGINAL_PRIOR(obs,env,ctl,eligible,aliases);b,bo=public_prior_cached(obs,env,ctl,eligible,aliases)
    assert np.array_equal(a,b) and o==bo,name
    assert ctl._grid54_prior_metadata.builds==count+1,name
    return {'name':name,'exact':True,'cache_rebuilt':True}
mutation=[]
try:
    library[0]=original_action.copy();mutation.append(mutation_check('same-valued object replacement'))
    changed=library[0].set_bus.copy();position=int(np.flatnonzero(changed)[0]);changed[position]=3-int(changed[position])
    library[0].set_bus=changed;mutation.append(mutation_check('setter mutation'))
    # Bypass the action's vector-cache invalidation on purpose.
    library[0]._set_topo_vect[position]=3-int(library[0]._set_topo_vect[position])
    mutation.append(mutation_check('in-place mutation without vector-cache reset'))
finally:
    library[0]=original_action
mutation.append(mutation_check('restored original object'))
assert int(env.nb_highres_called)==start_count
original_sum=math.fsum(r['original_s'] for r in rows);cached_sum=math.fsum(r['cached_s'] for r in rows)
R={'passed':True,'cases':len(rows),'all_scores_orders_ranks_features_exact':True,'patched_modules':patched,
   'new_candidate_forecasts':0,'new_fits':0,'metadata_stats_before_mutations':steady_metadata_stats,
   'mutation_checks':mutation,'original_prior_s':original_sum,'cached_prior_s':cached_sum,
   'local_prior_time_reduction':1-cached_sum/original_sum,'first_cached_call_s':rows[0]['cached_s'],
   'cached_steady_mean_s':math.fsum(r['cached_s'] for r in rows[1:])/(len(rows)-1),
   'wall_s':time.perf_counter()-begun,'rows':rows,
   'limitations':'130old saved cases and runtime mutation probes; no new closed-loop speed evidence. Cache is shared engineering,not GNN-exclusive. Fresh canonical-vector validation remains in paid online cost.'}
(OUT/'qualification.json').write_text(json.dumps(R,indent=2,allow_nan=False),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('QUALIFICATION PASSED:130exact scores/orders/ranks/features;mutation invalidation;no forecasts/fits.\n',encoding='utf-8')
env.close()
print(json.dumps({k:v for k,v in R.items() if k!='rows'},indent=2))
