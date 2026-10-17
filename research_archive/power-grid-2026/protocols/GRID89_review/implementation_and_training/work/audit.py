"""Saved-vector/feedback/cost audit, no independent AC re-solving."""
from pathlib import Path
from collections import defaultdict
import csv,gzip,hashlib,importlib.util,json,math,statistics,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid89_review'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=load('audit_review_policy',W/'policies.py');I=load('audit_review_independent',W/'reference_logic.py')
def digest(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def signature(rows):return hashlib.sha256(json.dumps([(r['before_hash'],r['action_hash'],r['after_hash'],r['raw_cost'],r['done'],r['complete']) for r in rows],separators=(',',':')).encode()).hexdigest()
def prior_signature(path):
 with gzip.open(path,'rt',encoding='utf-8') as f:return signature([json.loads(l) for l in f])
def mean(xs):
 a=[x for x in xs if x is not None];return statistics.mean(a) if a else None
def main():
 assert read(OUT/'benchmark_finished.json')['runs']==246
 for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 for p,h in read(OUT/'data_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 sys.path[:0]=[str(ROOT/'work/grid08'),str(W)]
 from common import grid2op,LightSimBackend,MaxRhoReward,operational_other_rewards,graph_features
 A=load('audit_review_adapter',W/'adapter.py');subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];allrows=[];sig_by_run={};counts=dict(steps=0,calls=0,paid_queries=0,independent_policy_checks=0,public_context_checks=0,repeat_trajectories=0,prior_trajectories=0)
 for phase in ['order','parameters','meta','sealed']:
  base=OUT/phase;D=read(base/'design.json');S=read(base/'finished.json')['summaries'];assert len(S)==len(D['execution_order'])
  assert {(s['week'],s['method'],s['repeat']) for s in S}=={(s['week'],s['method'],s['repeat']) for s in D['execution_order']}
  env=grid2op.make(str(ROOT/D['environment']),test=True,backend=LightSimBackend(),reward_class=MaxRhoReward,other_rewards=operational_other_rewards());obs0=env.reset();nb=int(env.nb_highres_called)
  try:
   for s in S:
    folder=base/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}";assert s==read(folder/'summary.json');assert s['illegal']==s['ambiguous']==0
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(l) for l in f]
    assert len(steps)==s['steps'];assert abs(math.fsum(x['raw_cost'] for x in steps)-s['cost'])<1e-6;assert abs(math.fsum(x['decision_s'] for x in steps)-s['controller_s'])<1e-6
    assert sum(x['simulations'] for x in steps)==s['public_forecasts'];assert sum(bool(x['exceptions']) for x in steps)==s['exception_steps']
    with np.load(folder/'vectors.npz') as z:actions=z['action'];observations=z['observation']
    assert len(actions)==len(observations)==len(steps)
    for j,x in enumerate(steps):
     assert digest(actions[j])==x['action_hash'] and digest(observations[j])==x['after_hash']
     if x['ledger'] is not None:assert abs(x['ledger']['recomputed_raw_cost']-x['raw_cost'])<=x['ledger_tolerance']
     if x['exceptions']:assert x['done']
    sig=signature(steps);rk=(phase,s['week'],s['method']);sig_by_run[(phase,s['week'],s['method'],s['repeat'])]=sig
    if s['repeat']:
     assert sig==sig_by_run[(phase,s['week'],s['method'],0)];counts['repeat_trajectories']+=1
    prior=None
    if phase=='parameters' and s['method']=='FRAME_GUARD':prior=ROOT/'outputs/grid70/runs'/f"r0__{s['week']}__FRAME_GUARD/steps.jsonl.gz"
    if phase in ('order','meta') and s['method'] in ('FRAME_GUARD','FRAME_GUARD__STATIC_ID'):prior=ROOT/'outputs/grid82/runs'/f"r0__{s['week']}__FRAME_GUARD/steps.jsonl.gz"
    if phase=='meta' and s['method'].startswith('META_COST0_'):
     seed=s['training_seed'];prior=ROOT/'outputs/grid84/runs'/f"r0__{s['week']}__META_CONTEXT{seed}/steps.jsonl.gz"
    if prior is not None:assert sig==prior_signature(prior),(phase,s['method'],s['week']);counts['prior_trajectories']+=1
    calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))];assert len(calls)==s['n1_calls'];assert sum(c['public_queries'] for c in calls)==s['n1_forecasts']
    for j,c in enumerate(calls):
     truth={r['pool_id']:r for r in c['outcomes']};assert len(truth)==c['public_queries'];aliases={int(k):int(v) for k,v in c['aliases'].items()}
     with np.load(folder/f'call{j:03d}_state.npz') as z:obs=obs0.copy();obs.from_vect(z['observation'])
     assert c['order']==A.expected_order(s,c['representatives'],subs,obs,env)
     for r in c['outcomes']:assert r['strict_admissible']==bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['exceptions'] and not r['illegal'] and not r['ambiguous'])
     context=None
     if s['policy']!='FULL':
      with np.load(folder/f'call{j:03d}_inputs.npz') as z:context={k:z[k] for k in ['x','edges','glob']}
      xx,ee,gg=graph_features(obs,env,dict(offered=False,predictions=[None,None]));assert np.array_equal(xx,context['x']) and np.array_equal(ee,context['edges']) and np.array_equal(gg,context['glob']);counts['public_context_checks']+=1
     seen=[]
     def reveal(i,phase_name):
      assert i in truth and i not in seen and truth[i]['phase']==phase_name;seen.append(i);return truth[i]
     got=P.run(c['order'],reveal,s['policy'],subs,aliases,s.get('parameters'),context=context,limit=c['rho_limit'],model_path=s.get('meta_model'))
     for k in ['chosen_pool_id','public_queries','stop_reason','local_queries','completion_budget','extra_selected_ids']:assert got[k]==c[k],(phase,s['method'],k)
     assert seen==[r['pool_id'] for r in c['outcomes']]
     if s['policy']!='META':
      cfg=dict(P.DEFAULT,**s.get('parameters',{}));plan,choice,budget=I.replay(c['order'],truth,subs,aliases,s['policy'],cfg)
      assert plan==seen and choice==c['chosen_pool_id'];assert s['policy'] in ('FULL','FIXED_512') or budget==c['completion_budget'];counts['independent_policy_checks']+=1
     else:assert got['meta_events']==c['meta_events']
     if c['chosen_pool_id'] is not None:
      independent=max((r for r in c['outcomes'] if r['strict_admissible']),key=I.k);assert independent['pool_id']==c['chosen_pool_id']
     counts['calls']+=1;counts['paid_queries']+=c['public_queries']
    cp=read(folder/'components.json');optim=math.fsum(x['wall_s'] for x in cp['calls'] if x['in_control']);fore=s['all_forecast_s'];search=s['n1_source_s']-s['n1_forecast_s'];rest=s['controller_s']-optim-fore-search
    assert min(optim,fore,search,rest)>=-1e-6
    active={c['step'] for c in calls};lat=[x['decision_s'] for x in steps];n1=[x['decision_s'] for x in steps if x['step'] in active]
    allrows.append(dict(phase=phase,date=s['week'],method=s['method'],policy=s['policy'],order_mode=s.get('order_mode','STATIC_ID'),repeat=s['repeat'],steps=s['steps'],complete=s['complete'],cost=s['cost'],queries=s['n1_forecasts'],controller_s=s['controller_s'],search_steps=len(n1),all_p50=float(np.percentile(lat,50)),all_p95=float(np.percentile(lat,95)),all_p99=float(np.percentile(lat,99)),search_p50=float(np.percentile(n1,50)) if n1 else None,search_p95=float(np.percentile(n1,95)) if n1 else None,search_p99=float(np.percentile(n1,99)) if n1 else None,optimizer_s=optim,forecast_s=fore,search_nonforecast_s=search,remainder_s=rest,trajectory_signature=sig))
    counts['steps']+=len(steps)
   assert int(env.nb_highres_called)==nb
  finally:env.close()
 # Attach quality only using the same-cohort FULL. META uses old FULL for quality, never time.
 oldfull={s['week']:s for s in read(ROOT/'outputs/grid81/finished.json')['summaries'] if s['method']=='FULL' and s['repeat']==0}
 for r in allrows:
  if r['phase']=='meta':b=oldfull[r['date']]
  else:
   fullmethod='FULL__'+r['order_mode'] if r['phase']=='order' else 'FULL';b=next(x for x in allrows if x['phase']==r['phase'] and x['date']==r['date'] and x['method']==fullmethod and x['repeat']==r['repeat'])
  r['cost_delta_full']=r['cost']/b['cost']-1 if r['complete'] and b['complete'] else None
  r['same_horizon_full']=r['steps']==b['steps'] and r['complete']==b['complete']
  r['quality_pass']=bool((r['complete'] if b['complete'] else r['steps']>=b['steps']) and (r['cost_delta_full'] is None or r['cost_delta_full']<=.005))
 groups=[]
 for phase in ['order','parameters','meta','sealed']:
  D=read(OUT/phase/'design.json')
  for m in D['methods']:
   for date in D['weeks']:
    rr=[r for r in allrows if r['phase']==phase and r['method']==m['method'] and r['date']==date];assert len(rr)==D['repeats']
    g={k:rr[0][k] for k in ['phase','date','method','policy','order_mode','steps','complete','cost','queries','cost_delta_full','same_horizon_full','quality_pass']}
    for k in ['controller_s','all_p50','all_p95','all_p99','search_p50','search_p95','search_p99','optimizer_s','forecast_s','search_nonforecast_s','remainder_s']:g[k]=mean(r[k] for r in rr)
    g['controller_min']=min(r['controller_s'] for r in rr);g['controller_max']=max(r['controller_s'] for r in rr);g['controller_sd']=statistics.stdev(r['controller_s'] for r in rr) if len(rr)>1 else None;groups.append(g)
 order_invariance=[]
 for date in read(OUT/'order/design.json')['weeks']:
  ss=[sig_by_run[('order',date,'FULL__'+mode,0)] for mode in read(OUT/'design.json')['order']['orders']];order_invariance.append(dict(date=date,full_order_invariant=len(set(ss))==1))
 csvsave(OUT/'per_run.csv',allrows);csvsave(OUT/'per_date.csv',groups)
 result=dict(passed_saved_data=True,runs=len(allrows),checks=counts,full_order_invariance=order_invariance,independent_ac_resolve=False,rows=allrows,groups=groups,scope='Saved vectors/feedback/timers and independent list-based control audit; generation scenarios are same grid with independent stochastic seed; old order/parameter/META data exposed.')
 save(OUT/'review.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','groups']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
