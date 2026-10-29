"""GRID272 fixed saved-score/guard-source/ownfit support diagnosis only."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,pickle,collections
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid272_union_limits';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID272_union_limits';Q=R/'outputs/grid271_union_guard';U=R/'outputs/grid270_union_guard_plan';F=R/'outputs/grid267_conditional_forest_plan';N=R/'outputs/grid268_conditional_forest';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def rd(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
 for p,h in rd(O/'source_freeze.json').items():assert sha(R/p)==h,p
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def published(phase):
 verify();h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 for rel,s in rd(A/phase/'archive_manifest.json').items():assert hashlib.sha256(git('show',h+':'+(A/phase/rel).relative_to(REPO).as_posix())).hexdigest()==s
 return h

def prepare():
 assert not O.exists() and rd(Q/'NEXT_PROTOCOL_PLAN.json')['id']=='GRID272';h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 ar=REPO/'research_archive/power-grid-2026/protocols/GRID271_union_guard'
 for n in ['NEXT_PROTOCOL_PLAN.json','matched268_changes.json','profitable_misses.json','complete.json','independent_audit.json','matched261_summary.json']:
  assert git('show',h+':'+(ar/'audited'/n).relative_to(REPO).as_posix())==(Q/n).read_bytes()
 for phase,freeze in [('calibrated','calibrated_freeze.json'),('scored','scored_freeze.json')]:
  for rel,s in rd(Q/freeze).items():assert sha(Q/rel)==s and hashlib.sha256(git('show',h+':'+(ar/phase/rel).relative_to(REPO).as_posix())).hexdigest()==s
 arU=REPO/'research_archive/power-grid-2026/protocols/GRID270_union_guard_plan/qualified';arF=REPO/'research_archive/power-grid-2026/protocols/GRID267_conditional_forest_plan/qualified'
 for rel,s in rd(U/'qualified_freeze.json').items():assert sha(U/rel)==s and hashlib.sha256(git('show',h+':'+(arU/rel).relative_to(REPO).as_posix())).hexdigest()==s
 for p in (F/'qualified_inputs').glob('*_fit.npz'):
  rel=p.relative_to(F).as_posix();assert sha(p)==rd(F/'qualified_freeze.json')[rel] and git('show',h+':'+(arF/rel).relative_to(REPO).as_posix())==p.read_bytes()
 O.mkdir();save('input_publication.json',dict(passed=True,git_sha=h,all_fixed271scores_changes_and270guards_and267fitinputs_specificblobs=True))
 save('design.json',dict(id='GRID272',scope='Readonly13publishedrestorations/10residualmisses/543guardlimitingrows andownfit support,0fit/scalar/alternativepolicy/forestNNforward.',public='Fixedall13+10focusedIDs andall543unionrows;reusepublishedforest_score/oldnewtau/calibrationscores. TraceALLlimitingties/origin_current/origin_legacy/fitrole andadded43belowoldtau. Matchall543IDs toownfitpublicX32/ids/domain_STOP,record exact40/exact39/contextonlydifference ormissing. No nearestk/thresholdscan/leafgate.',labels='Afterpublicsavedtablepublished,readonly eachownfit y/NET/weight/source_mass. Summarize STOP/CONT risk andpositive coverage for fixedguardcontext matches,ALLpoolsinclKEEP. Inspectfrozen treefeatureusage/nodes only,no predict/apply/new probability. Do not alter data/weights or exportnewtrainingcandidate yet.',interpretation='13restorations1positive12nonpositive/10misses descriptive,not independent;onlyowntraininglabels/publicsupport toqualify unique finite next hypothesis,no outerIDs rescued. Prior235/253 ancestry conclusions reused,not newbroad audit.',limits='Currentbranch106/0newfits;expanded500to543constraints notsamesample. Old6continuations8285updates/604.703actual/678.156cap separate. No final8/physics/grid/paper/FRAME/install,feeslabels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged. Queryproxy notwholeweek oractualtime.'))
 paths=[W/'run.py',O/'design.json',O/'input_publication.json',Q/'NEXT_PROTOCOL_PLAN.json',Q/'matched268_changes.json',Q/'profitable_misses.json',Q/'matched261_summary.json',Q/'search_metrics.json',Q/'model_references.json' if (Q/'model_references.json').exists() else U/'model_references.json',META,*list((Q/'cutoffs').glob('*.json')),*list((Q/'calibration_predictions').glob('*.npz')),*list((Q/'outer_predictions').glob('*.npz')),*list((U/'qualified_inputs').glob('*.npz')),*list((F/'qualified_inputs').glob('*_fit.npz')),*list((N/'models').glob('*.pkl'))]
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID272_SOURCE_FROZEN')

def public():
 verify();h=published('source');assert not (O/'public_started.json').exists();save('public_started.json',dict(source_published=h));meta=rd(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};pools=[];coverage=[];focused=[]
 focus=[dict(r,focus='restored') for r in rd(Q/'matched268_changes.json')]+[dict(r,focus='missed') for r in rd(Q/'profitable_misses.json')]
 for file in sorted((U/'qualified_inputs').glob('*.npz')):
  pool=file.stem;f,s=[int(v[1:]) for v in pool.split('_')];u=np.load(file);cal=np.load(Q/'calibration_predictions'/file.name);fit=np.load(F/'qualified_inputs'/f'{pool}_fit.npz');cut=rd(Q/'cutoffs'/f'{pool}.json');pred=np.load(Q/'outer_predictions'/file.name);assert np.array_equal(u['ids'],cal['ids']);fidx={int(v):i for i,v in enumerate(fit['ids'])};tau=cut['tau'];keep=cut['KEEP'];limits=[];added=[]
  for j,mid in enumerate(u['ids']):
   mid=int(mid);m=meta[mid];k=fidx.get(mid);equal39=k is not None and np.array_equal(u['X32'][j,:39],fit['X32'][k,:39]);equal40=k is not None and np.array_equal(u['X32'][j],fit['X32'][k]);onlybit=bool(equal39 and not equal40)
   row=dict(pool=pool,id=mid,date=m['date'],call=m['physical_state_group'],fit_role=bool(u['fit_role'][j]),origin_current=bool(u['origin_current'][j]),origin_legacy=bool(u['origin_legacy'][j]),guard_context=float(u['X32'][j,39]),ownfit_id_present=k is not None,ownfit_context=None if k is None else float(fit['X32'][k,39]),equal39=bool(equal39),equal40=bool(equal40),context_only_difference=onlybit,score=float(cal['score'][j]),tau=tau,old_tau=cut['old_tau'],is_limiting=bool(not keep and cal['score'][j]==tau),below_old_tau=bool(not keep and cal['score'][j]<cut['old_tau']))
   assert not onlybit or row['guard_context']==1 and row['ownfit_context']==0;coverage.append(row)
   if row['is_limiting']:limits.append(row)
   if not row['origin_current']:added.append(row)
  pools.append(dict(pool=pool,KEEP=keep,rows=len(u['ids']),added_rows=len(added),added_below_old=sum(r['below_old_tau'] for r in added),lowered=tau<cut['old_tau'],old_tau=cut['old_tau'],new_tau=tau,all_limiting_ties=limits,added_score_min=None if not added else min(r['score'] for r in added)))
  pi={int(v):j for j,v in enumerate(pred['ids'])}
  for r in focus:
   if r['fold']!=f or r['seed']!=s:continue
   mid=lookup[r['sample_id'],r['position']];j=pi[mid];focused.append(dict(pool=pool,id=mid,focus=r['focus'],date=r['date'],call=r['call'],sample_id=r['sample_id'],position=r['position'],forest_score=float(pred['forest_score'][j]),old_tau=cut['old_tau'],new_tau=tau,margin=float(pred['forest_score'][j]-tau),parent261_logit=float(pred['parent261_logit'][j]),current_continue=bool(pred['continue_'][j])))
 assert len(coverage)==543 and len(focused)==23 and sum(r['added_rows'] for r in pools)==43;save('public_guard_coverage.json',coverage);save('public_pool_limits.json',pools);save('public_focused_scores.json',focused);save('public_freeze.json',{n:sha(O/n) for n in ['public_guard_coverage.json','public_pool_limits.json','public_focused_scores.json']});archive('public');print('GRID272_PUBLIC_FIXED23_AND543_NO_NEW_PREDICTION')

def analyze():
 verify();h=published('public');assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(public_published=h));meta=rd(META);coverage=rd(O/'public_guard_coverage.json');out=[];matches=[]
 for file in sorted((F/'qualified_inputs').glob('*_fit.npz')):
  pool=file.name[:-8];z=np.load(file);bundle=pickle.loads((N/'models'/f'{pool}.pkl').read_bytes());keep=bundle['KEEP'];idx={int(v):i for i,v in enumerate(z['ids'])};parts=[]
  for stop in [True,False]:
   mask=z['domain_STOP']==stop
   for pos in [True,False]:
    mm=mask&((z['y']>0)==pos);parts.append(dict(domain_STOP=stop,positive=pos,rows=int(mm.sum()),risk=float(z['weight'][mm].sum()),calls=len({meta[int(v)]['physical_state_group'] for v in z['ids'][mm]}),dates=len({meta[int(v)]['date'] for v in z['ids'][mm]})))
  splits=bit_trees=nodes=leaves=0;feature=collections.Counter();treeuse=[]
  if not keep:
   for t in bundle['model'].estimators_:
    ff=t.tree_.feature;used=ff[ff>=0];feature.update(map(int,used));nodes+=int(t.tree_.node_count);leaves+=int((ff<0).sum());splits+=int((used==39).sum());bit_trees+=int((used==39).any());treeuse.append(int((used==39).sum()))
  out.append(dict(pool=pool,KEEP=keep,rows=len(z['ids']),partitions=parts,total_risk=float(z['weight'].sum()),context_bit_split_nodes=splits,context_bit_trees=bit_trees,trees=0 if keep else 64,nodes=nodes,leaves=leaves,all_feature_split_counts={str(k):v for k,v in sorted(feature.items())}))
  for r in [r for r in coverage if r['pool']==pool]:
   k=idx.get(r['id']);v=dict(**r,fit_y=None if k is None else int(z['y'][k]),fit_NET=None if k is None else float(z['actual_NET'][k]),fit_weight=None if k is None else float(z['weight'][k]),fit_source_mass=None if k is None else float(z['source_mass'][k]),fit_domain_STOP=None if k is None else bool(z['domain_STOP'][k]))
   if k is not None:assert z['y'][k]==1 and z['actual_NET'][k]>0 and v['fit_domain_STOP']==(v['ownfit_context']==1)
   matches.append(v)
 active=[r for r in matches if r['pool']!='f0_s1'];legacy_only=[r for r in active if not r['origin_current']];present=[r for r in legacy_only if r['ownfit_id_present']];context=[r for r in present if r['context_only_difference']];limits=[r for r in matches if r['is_limiting']];pools=rd(O/'public_pool_limits.json');focus=rd(O/'public_focused_scores.json')
 summary=dict(pools12=len(out),active11=sum(not r['KEEP'] for r in out),guard543=len(matches),active507=len(active),added43=len(legacy_only),added_below_old=sum(r['below_old_tau'] for r in legacy_only),cutoffs_lowered=sum(r['lowered'] for r in pools),all_limiting_rows=len(limits),limiting_fit=sum(r['fit_role'] for r in limits),limiting_added_only=sum(not r['origin_current'] for r in limits),added_ownfit_id_present=len(present),added_fit_context_only_difference=len(context),added_fit_same40=sum(r['equal40'] for r in present),added_fit_positive_weight=float(sum(r['fit_weight'] for r in present)),added_unique_fit_calls=len({r['call'] for r in present}),added_unique_fit_dates=len({r['date'] for r in present}),forest_context_bit_trees=sum(r['context_bit_trees'] for r in out),trees704=sum(r['trees'] for r in out),focus23=len(focus),residual10=sum(r['focus']=='missed' for r in focus),restored13=sum(r['focus']=='restored' for r in focus),zero_newfits=True,zero_newforward=True)
 save('ownfit_risk_and_structure.json',out);save('ownfit_guard_source_support.json',matches);save('diagnosis_summary.json',summary);verify();archive('analyzed');print('GRID272_READONLY_DIAGNOSED',json.dumps(summary))
if __name__=='__main__':globals()[sys.argv[1]]()
