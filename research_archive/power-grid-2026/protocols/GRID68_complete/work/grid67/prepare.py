"""Freeze both finite experiments before any new-date control results."""
from pathlib import Path
import hashlib,importlib.util,json,py_compile,sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(HERE));import policies as P
def main():
 qualified=read(ROOT/'outputs/grid66/fresh_scene_qualification.json');assert qualified['passed_local_outcome_nonexposure']
 # Causal finite-policy tests on all retained exhaustive historical states.
 plan=read(ROOT/'outputs/review_budgetmatch_20261014/plan.json');subs=read(ROOT/plan['substation_metadata'])['public_subids'];checks=0
 for rel in plan['sources']:
  c=read(ROOT/rel);truth={r['pool_id']:r for r in c['outcomes']};order=sorted(c['representatives']);aliases={int(k):int(v) for k,v in c['aliases'].items()}
  for policy in P.POLICIES:
   seen=[]
   def reveal(i,phase):seen.append(i);return {**truth[i],'phase':phase}
   result=P.run(order,reveal,policy,subs,aliases,limit=c['rho_limit']);assert len(seen)==len(set(seen))==result['public_queries']
   changed={i:dict(v) for i,v in truth.items()}
   for i in set(truth)-set(seen):changed[i].update(reward=999.,rho=.001,strict_admissible=True)
   assert P.run(order,lambda i,p:{**changed[i],'phase':p},policy,subs,aliases,limit=c['rho_limit'])==result;checks+=1
   if policy=='FRAME':assert result==P.old.run(order,lambda i,p:{**truth[i],'phase':p},'FRAME',subs,aliases)
 for stage,policies,weeks,repeats in [(67,['FULL','FRAME','FRAME_P64','FRAME_P256','FRAME_S1','FRAME_S3'],read(ROOT/'outputs/grid61/design.json')['weeks'],1),(68,['FULL','FRAME','GLOBAL_MATCH'],[f['week'] for f in qualified['families']],2)]:
  out=ROOT/f'outputs/grid{stage}';assert not out.exists();out.mkdir()
  methods=[dict(method=p,rule='STATIC_ID',budget=128,checkpoint=None,policy=p,cache_dc=True,cache_safe=True) for p in policies];execution=[]
  for rep in range(repeats):
   for j,w in enumerate(weeks):
    shift=j%len(methods);seq=methods[shift:]+methods[:shift]
    if rep:seq=seq[::-1]
    execution.extend({**m,'week':w,'repeat':rep,'stage_phase':'sensitivity' if stage==67 else 'new_scenes'} for m in seq)
  D={'stage':f'GRID{stage}','weeks':weeks,'methods':methods,'execution_order':execution,'repeats':repeats,'environment_seed':0,'serial_workers':1,'quality_threshold':.005,
   'timing':'Original GRID55 worker and GRID28 fast control, shared DC/safeQP caches, fresh serial workers. No parallel training or heavy audit; cold imports/setup kept separate; all online search/control counted.',
   'method_freeze':{'initial':128,'batch':64,'minimum_plateau':256,'patience':2,'safe_rho':.9,'ranking':'original author frequency (STATIC_ID)','completion':'unchanged incumbent-substation alias completion'},
   'scope':('Four already-exposed development dates. Single-axis prefix64/128/256 and patience1/2/3; no cross-product, no tuning or default replacement. One timing repeat, quality/query sensitivity primary.' if stage==67 else 'All six locally outcome-unused date families remaining after actual exposure audit, selected without control results. Two timing repeats; one existing grid. Dates earlier extracted/schema-checked or named in unselected pools are not claimed never read. Author prior training exposure unknown; not cross-grid or universal confirmation.'),
   'decision_rule':('Report every parameter and date. Default unchanged regardless of winner. Robustness means no earlier failure, each comparable complete cost<=FULL*1.005 and mean query saving>=50%; adverse settings reported individually.' if stage==67 else 'Retain all six dates and failures. Support if FRAME no earlier failure on all6, every comparable complete cost<=FULL*1.005, at least3 common complete weeks, macro query saving>=50%, macro full-controller saving>=5%; otherwise limited/mixed. Report global comparison without requiring FRAME wins. No replacing dates or tuning.'),
   'statistics':'Per-date equal weights after averaging repeats; cost only jointly complete; time only equal horizon; no-query dates separate. P95/P99 all steps and search steps separate. Repeats not independent dates.',
   'caps':{'episode_s':1200,'wall_s':14400,'physical_steps':len(execution)*2100,'public_forecasts':len(execution)*40000,'new_fits':0},'frozen_before_any_new_date_outcomes':True}
  (out/'design.json').write_text(json.dumps(D,ensure_ascii=False,indent=2),encoding='utf-8')
  freeze=read(ROOT/'outputs/grid65/code_freeze.json')
  for p,h in freeze.items():assert sha(ROOT/p)==h,p
  extra=list(HERE.glob('*.py'))+[out/'design.json',ROOT/'outputs/grid66/fresh_scene_qualification.json',ROOT/'work/grid66/prepare.py']
  if stage==68:extra += [ROOT/row['path'] for f in qualified['families'] for row in f['files']]
  for p in extra:
   if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
   freeze[p.relative_to(ROOT).as_posix()]=sha(p)
  (out/'code_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
  (out/'preflight.json').write_text(json.dumps({'passed':True,'historical_policy_suffix_checks':checks,'old_frame_exact_states':len(plan['sources']),'new_fits':0,'runs':len(execution)},indent=2),encoding='utf-8')
  (out/'RUN_STATE.md').write_text('FROZEN BEFORE NEW OUTCOMES: finite serial experiment; no training.\n',encoding='utf-8')
  print(json.dumps({'stage':stage,'runs':len(execution),'weeks':weeks,'checks':checks}),flush=True)
if __name__=='__main__':main()
