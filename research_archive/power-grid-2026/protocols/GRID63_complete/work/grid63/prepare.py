"""Freeze the sole requested closed-loop budget-matched comparison."""
import hashlib,json,py_compile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid63'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(HERE));import policies as P
assert not OUT.exists(), 'Preserve any previous stage'
# Hand counterexamples check both directions, a legal alias, and empty proposal.
order=list(range(400));subs=[1]*400;subs[0]=subs[399]=0;aliases={i:i for i in order}
def reveal(i,p):return {'pool_id':i,'reward':2. if i==399 else 1.,'rho':1.,'strict_admissible':True,'phase':p}
a=P.run(order,reveal,'FRAME',subs,aliases);b=P.run(order,reveal,'GLOBAL_MATCH',subs,aliases)
assert a['public_queries']==b['public_queries']==257 and a['chosen_pool_id']==399 and b['chosen_pool_id']==0
def global_good(i,p):return {**reveal(i,p),'reward':3. if i==256 else reveal(i,p)['reward']}
assert P.run(order,global_good,'GLOBAL_MATCH',subs,aliases)['chosen_pool_id']==256
assert P.run(order,global_good,'FRAME',subs,aliases)['chosen_pool_id']==399
for policy in P.POLICIES:
 c=P.run(order,lambda i,p:{**reveal(i,p),'strict_admissible':False},policy,subs,aliases)
 assert c['chosen_pool_id'] is None and c['public_queries']==400
# A later alias can make an earlier canonical representative belong to the station.
subs[398]=0;aliases[398]=350;order2=[i for i in order if i!=398]
c=P.run(order2,reveal,'FRAME',subs,aliases);assert 350 in c['completion_candidate_ids']
# Recover every old full-state FRAME exactly; compare equal budgets and hide unqueried rewards.
plan=read(ROOT/'outputs/review_budgetmatch_20261014/plan.json');real_subs=read(ROOT/plan['substation_metadata'])['public_subids'];checks=0
for source in plan['sources']:
 c=read(ROOT/source);truth={r['pool_id']:r for r in c['outcomes']};o=sorted(c['representatives']);al={int(k):int(v) for k,v in c['aliases'].items()}
 results={}
 for policy in ['FRAME','GLOBAL_MATCH']:
  seen=[]
  def observe(i,p):
   assert i not in seen;seen.append(i);return {**truth[i],'phase':p}
  r=P.run(o,observe,policy,real_subs,al)
  changed={i:dict(v) for i,v in truth.items()}
  for i in set(truth)-set(seen):changed[i].update(reward=999.,rho=.001,strict_admissible=True)
  assert P.run(o,lambda i,p:{**changed[i],'phase':p},policy,real_subs,al)==r
  results[policy]=r;checks+=1
 original=P.old.run(o,lambda i,p:{**truth[i],'phase':p},'FRAME',real_subs,al)
 for field in ['outcomes','chosen_pool_id','public_queries','stop_reason','local_queries']:
  assert results['FRAME'][field]==original[field],(source,field)
 assert results['FRAME']['public_queries']==results['GLOBAL_MATCH']['public_queries']
 assert results['FRAME']['completion_budget']==results['GLOBAL_MATCH']['completion_budget']
weeks=read(ROOT/'outputs/grid61/design.json')['weeks']+read(ROOT/'outputs/grid62/design.json')['weeks']
assert len(weeks)==7 and len(set(weeks))==7
methods=[dict(method=p,rule='STATIC_ID',budget=128,checkpoint=None,policy=p,cache_dc=True,cache_safe=True) for p in P.POLICIES]
execution=[]
for rep in range(3):
 for date_index,w in enumerate(weeks):
  shift=(date_index+rep)%3;ordered=methods[shift:]+methods[:shift]
  execution.extend({**m,'week':w,'repeat':rep,'stage_phase':'budget_matched'} for m in ordered)
D={'stage':'GRID63_BUDGET_MATCHED_CLOSED_LOOP','authorization':'User 2026-10-14 explicitly requests FRAME versus equal-budget global expansion closed-loop comparison.','weeks':weeks,'methods':methods,'execution_order':execution,'repeats':3,'environment_seed':0,'serial_workers':1,
 'scope':'Seven already-exposed dates; 3 policies x 3 balanced timing repeats = 63 new runs. Development mechanism comparison, not new independent confirmation. No new model, ranking, date, training or parameter selection.',
 'timing':'Serial fresh workers; rotate three policies within each date over three repeats so each occupies every ordinal position once. Full controller includes the common legal completion-budget computation for FRAME/GLOBAL_MATCH. No concurrent training or heavy audit.',
 'method_freeze':{'ranking':'Same original author teacher-frequency order','initial':128,'batch':64,'minimum_before_plateau':256,'stale_batches':2,'safe_rho':.9,'global_match':'After common progressive stop, compute legal same-substation unqueried L using aliases and incumbent. q=|L| before extra feedback. Query next q remaining global representatives exactly, with no early stop in this added pass. FRAME queries L. Both retain best admissible incumbent.','budget_limit':'Exact equal extra query count for identical input state/prefix. Closed-loop trajectories differ, so whole-week query counts need not be equal.','tie':'Original fp32 reward, smaller retained ID'},
 'primary':'No earlier failure and full-week cost relative same-batch FULL; paired FRAME versus GLOBAL_MATCH full-week cost expressed as (J_GLOBAL-J_FRAME)/J_FULL. Query totals, actual full-controller time and latency separate.',
 'quality_threshold':.005,'material_cost_threshold':.001,
 'decision_rule':'If FRAME quality fails, report FRAME_QUALITY_FAILED. If FRAME meets quality and GLOBAL violates it on at least one date, or FRAME improves GLOBAL cost by at least .1% of FULL in at least two comparable complete dates without a converse >=.1% loss, report STRUCTURAL_INCREMENT_SUPPORTED_IN_MATRIX. If GLOBAL meets all quality and no FRAME >=.1% FULL cost gain occurs, report GLOBAL_COMPETITIVE_NO_MATERIAL_STRUCTURE_GAIN. Otherwise MIXED_OR_SMALL. Thresholds are prospective investment screens, not significance or equivalence certification. Retain all adverse dates.',
 'statistics':'Average three timing repeats within date before date ratios; report every repetition, mean/std/range, all-step and search-active-step P95/P99/max. Dates are correlated within one grid; repetitions are not independent scenarios. No cherry-picking repeats or old timing reuse.',
 'permission':'Only current public observation, library metadata and dispatched feedback. No other trajectory, full unqueried reward or test cost enters controller.',
 'caps':{'episode_s':1200,'wall_s':14400,'physical_steps':150000,'public_forecasts':1200000,'new_fits':0},
 'continuation':'Audit complete then report all results. No automatic model changes, parameter sweep, new grids or new dates. Preserve partial errors; resume only fully persisted receipts and unchanged code hashes.',
 'baseline_note':'Same hybrid LJN control, continuous/recovery logic, DC/safeQP and common metadata caches. FULL exhaustive only over current legal representative pool. GNN/RL not loaded or trained.'}
OUT.mkdir();(OUT/'design.json').write_text(json.dumps(D,ensure_ascii=False,indent=2),encoding='utf-8')
freeze={}
for p,h in read(ROOT/'outputs/grid62/code_freeze.json').items():assert sha(ROOT/p)==h,p;freeze[p]=h
for p in list(HERE.glob('*.py'))+[OUT/'design.json']:
 if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
 freeze[p.relative_to(ROOT).as_posix()]=sha(p)
(OUT/'code_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
(OUT/'preflight.json').write_text(json.dumps({'passed':True,'old_frame_states_exact':35,'hidden_feedback_checks':checks,'equal_query_states':35,'hand_direction_and_alias_checks':7,'planned':63,'new_fits':0},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:63serial same-date FRAME/GLOBAL_MATCH/FULL runs;no training.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'runs':63,'same_state_checks':35,'hidden_feedback_checks':checks}))
