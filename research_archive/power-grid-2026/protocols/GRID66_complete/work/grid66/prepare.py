"""Read-only historical metrics and outcome-exposure qualification, before new replay."""
from pathlib import Path
import collections,csv,gzip,hashlib,json,math,re,statistics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid66'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvsave(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def q(a,p):
 if not a:return None
 a=sorted(a);k=(len(a)-1)*p;i=int(k);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(k-i)
def main():
 assert not OUT.exists();OUT.mkdir()
 dates=collections.defaultdict(set);receipts=[]
 # Read names of every historical output directory, including early failed runs.
 for st in (ROOT/'outputs').glob('grid*'):
  if not st.is_dir() or st==OUT:continue
  for p in st.rglob('*'):
   if not p.is_dir():continue
   matches=re.findall(r'2035-\d{2}-\d{2}_\d+',p.name)
   for v in matches:dates[v[:10]].add(p.relative_to(ROOT).as_posix())
 # Include original historical exposure audit and all actually fitted data identities.
 legacy=ROOT/'outputs/grid_research_audit/split_exposure.json'
 for d in read(legacy)['exposed_dates']:dates[d].add(legacy.relative_to(ROOT).as_posix())
 for rel,key in [('outputs/grid58/design.json','training_weeks'),('outputs/grid08/design.json','evaluation_weeks')]:
  p=ROOT/rel
  for v in read(p)[key]:dates[v[:10]].add(rel+':'+key)
  receipts.append({'path':rel,'sha256':sha(p)})
 index=read(ROOT/'outputs/grid03/archive_index.json');catalogue=sorted({n[:10] for n in index['scenario_files']})
 available=sorted(set(catalogue)-set(dates))
 assert available==['2035-01-01','2035-02-19','2035-04-02','2035-07-09','2035-12-10','2035-12-17'],available
 env=ROOT/'work/grid03/formal_env_initialized';seed='GRID68_FRESH_FAMILIES_20261015_V1'
 families=[]
 for d in available:
  candidates=sorted(p.name for p in (env/'chronics').iterdir() if p.is_dir() and p.name.startswith(d+'_'))
  assert candidates
  week=min(candidates,key=lambda x:hashlib.sha256((seed+'|'+x).encode()).hexdigest())
  files=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted((env/'chronics'/week).iterdir()) if p.is_file()]
  families.append({'date':d,'week':week,'quarter':(int(d[5:7])-1)//3+1,'eligible_preextracted_variants':candidates,'files':files})
 qualification={'passed_local_outcome_nonexposure':True,'new_date_families':6,'selection_seed':seed,'families':families,'historical_exposed_families':{d:sorted(v) for d,v in sorted(dates.items())},'receipts':receipts,
 'exclusion_semantics':'Outcome/teacher/fit exposure, including failed run directories and old exposure audit; mere archive/schema extraction or membership in an unselected candidate pool is not an observed control result.',
 'correction':'GRID62 conservatively treated every preextracted family as exposed. Six preextracted families never used for control/teacher outcomes remain. No previous verdict is changed.',
 'external_exposure':'Released author action-frequency/PPO training dates remain unknown. Fresh locally, not certified unseen to external prior. Public synthetic IDF118 scenarios, not operational logs or cross-grid evaluation.',
 'environment_static_hashes':{p.name:sha(p) for p in env.iterdir() if p.is_file()},'archive_index_sha256':sha(ROOT/'outputs/grid03/archive_index.json')}
 save(OUT/'fresh_scene_qualification.json',qualification)
 F=read(ROOT/'outputs/grid65/finished.json');rows=[]
 for s in F['summaries']:
  folder=ROOT/'outputs/grid65/runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
  with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(l) for l in f]
  c=read(folder/'components.json');calls=[x for x in c['calls'] if x['in_control']]
  phase={p:math.fsum(x['wall_s'] for x in calls if x['phase']==p) for p in ['run_dc','compute_optimum_safe','compute_optimum_unsafe']}
  other=s['controller_s']-s['all_forecast_s']-sum(phase.values());assert other>=-1e-6
  total=s['all_forecast_s']+sum(phase.values())+other;assert abs(total-s['controller_s'])<1e-7
  search_steps={read(p)['step'] for p in folder.glob('call[0-9][0-9][0-9].json')}
  times=[x['decision_s'] for x in steps];active=[x['decision_s'] for x in steps if x['step'] in search_steps]
  rows.append({'date':s['week'],'method':s['method'],'repeat':s['repeat'],'steps':s['steps'],'complete':s['complete'],'cost':s['cost'],'controller_s':s['controller_s'],'all_p95_s':q(times,.95),'all_p99_s':q(times,.99),'max_s':max(times),'search_steps':len(active),'search_p95_s':q(active,.95),'search_p99_s':q(active,.99),'n1_queries':s['n1_forecasts'],'n1_forecast_s':s['n1_forecast_s'],'other_forecast_s':s['all_forecast_s']-s['n1_forecast_s'],'dc_s':phase['run_dc'],'safe_qp_s':phase['compute_optimum_safe'],'unsafe_qp_s':phase['compute_optimum_unsafe'],'other_control_s':other,'accounting_abs_error':abs(total-s['controller_s'])})
 csvsave('historical_run_metrics.csv',rows)
 family=[]
 for d in read(ROOT/'outputs/grid65/design.json')['weeks']:
  for method in ['FULL','FRAME','GLOBAL_MATCH','META_DQN0','META_DQN1']:
   a=[x for x in rows if x['date']==d and x['method']==method];assert len(a)==2
   z={k:a[0][k] for k in ['date','method','steps','complete','cost','n1_queries']}
   assert all(a[0][k]==a[1][k] for k in ['steps','complete','cost','n1_queries'])
   for key in ['controller_s','all_p95_s','all_p99_s','max_s','search_p95_s','search_p99_s','n1_forecast_s','other_forecast_s','dc_s','safe_qp_s','unsafe_qp_s','other_control_s']:
    v=[x[key] for x in a if x[key] is not None]
    z[key+'_mean']=statistics.mean(v) if v else None
    if key=='controller_s':z.update(controller_std_s=statistics.stdev(v),controller_min_s=min(v),controller_max_s=max(v))
   z['search_steps_per_repeat']=a[0]['search_steps'];family.append(z)
 csvsave('historical_date_metrics.csv',family)
 save(OUT/'metrics_audit.json',{'passed':True,'source_stage':'GRID65','runs':70,'date_method_groups':35,'max_partition_error_s':max(x['accounting_abs_error'] for x in rows),'costs_and_queries_exact_across_repeats':True,'quantile':'Linear interpolation per run, then date means; active-search and all-decision quantiles separate. Two repeats measure timing variation, not independent dates.','partition':'Forecast + disjoint outer run_dc/safe/unsafe calls + residual controller. Solver/backend compilation nested in outer calls, never added twice.','independent_ac_resolve':False})
 (OUT/'RUN_STATE.md').write_text('COMPLETE: historical metrics and six locally unused date-family qualification, no new control results.\n',encoding='utf-8')
 print(json.dumps({'qualified_families':[(x['week']) for x in families],'metrics_runs':len(rows)},ensure_ascii=False))
if __name__=='__main__':main()
