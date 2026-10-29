"""GRID281 fixed bounds, roots and all40published changes; no tree traversal."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,pickle,collections,ast
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid281_partition_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID281_partition_support';Q=R/'outputs/grid280_invocation_conditioned';OLD=R/'outputs/grid277_selector_free';F=R/'outputs/grid279_conditional_fit_plan';U=R/'outputs/grid276_selector_input_plan';E=R/'outputs/grid278_invocation_support';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def rd(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
 for p,h in rd(O/'source_freeze.json').items():assert sha(R/p)==h,p

def path_constant(p,key):
 n=next(n.value for n in ast.parse(p.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets));assert isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div) and isinstance(n.left,ast.Name) and isinstance(n.right,ast.Constant);return {'R':R,'REPO':REPO}[n.left.id]/n.right.value

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
 assert not O.exists() and rd(Q/'NEXT_PROTOCOL_PLAN.json')['id']=='GRID281';h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h;arQ=path_constant(R/'work/grid280_invocation_conditioned/run.py','A');arOld=path_constant(R/'work/grid277_selector_free/run.py','A')
 for n in ['NEXT_PROTOCOL_PLAN.json','matched277_changes.json','matched277_summary.json','search_metrics.json','per_seed_preservation_and_exposure.json','complete.json']:assert git('show',h+':'+(arQ/'audited'/n).relative_to(REPO).as_posix())==(Q/n).read_bytes()
 for out,ar in [(Q,arQ),(OLD,arOld)]:
  for rel,s in rd(out/'trained_freeze.json').items():assert sha(out/rel)==s and hashlib.sha256(git('show',h+':'+(ar/'trained'/rel).relative_to(REPO).as_posix())).hexdigest()==s
 for out,ar in [(F,path_constant(R/'work/grid279_conditional_fit_plan/run.py','A')),(U,path_constant(R/'work/grid276_selector_input_plan/run.py','A'))]:
  for rel,s in rd(out/'qualified_freeze.json').items():assert sha(out/rel)==s and hashlib.sha256(git('show',h+':'+(ar/'qualified'/rel).relative_to(REPO).as_posix())).hexdigest()==s
 O.mkdir();save('input_publication.json',dict(passed=True,git_sha=h,allmodels_andowntraininginputs_and40changes_published=True))
 save('design.json',dict(id='GRID281',scope='Limitedreadonly all40published changes andowntraining39D bounds/frozen roots,0fit/scalar/newapply/predict/forestNNforward/alternativepolicy.',public='ALL12pools xALL39features float32 actualforestinput min/max/nonconstant andALLoldextrema rowindices inclremovedties;verifynew279X32=old276X32[fixed278E]. No featureorcutoff selection. Inspect eachfrozen277/280tree rootfeature/threshold/n_node_samples/weighted_n_node_samples;no traversal. FixedALL40focusIDs/oldnewdecisions exported beforeownfitlabels.',labels='Afterpublicbounds/extremamembershippublished,read onlyown276/279y/weight/NET toannotate removedextrema androotweightedmass. Published40outcomes usedsolely win/loss/call/date/crossseed recurrence andmacrocontribution,notnewsubsetpolicyreturns.',interpretation='Smallremovedrisk maystillchange rawsample minleafcounts/randomcandidate ranges,structuraldescriptionsnotpairedcausalproof. Keepseed0smallgain_querysignal ANDallseed1/2negativecases,notpickdeploymentseed/subset.',counts='Current168 actual0fits/forward;all331excluded/16copies/543guardexposure and6continuations8285updates/604.703actual/678.156cap retained.',bounds='Parentown12/internalcal4/exposedrelatedteacherprefixes,notOOFblind/fullweek/time. No final8/physics/grid/paper/FRAME/install;feeNETlabels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.'))
 paths=[W/'run.py',O/'design.json',O/'input_publication.json',Q/'NEXT_PROTOCOL_PLAN.json',Q/'matched277_changes.json',Q/'matched277_summary.json',Q/'search_metrics.json',Q/'per_seed_preservation_and_exposure.json',META,F/'qualified_freeze.json',U/'qualified_freeze.json',E/'public_freeze.json',*list((E/'public_entry').glob('*_fit.npz')),*list((F/'qualified_inputs').glob('*_fit.npz')),*list((U/'qualified_inputs').glob('*_fit.npz')),*list((Q/'models').glob('*.pkl')),*list((OLD/'models').glob('*.pkl'))]
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID281_SOURCE_FROZEN')

def public():
 verify();h=published('source');assert not (O/'public_started.json').exists();save('public_started.json',dict(source_published=h));ranges=[];roots=[];pools=[];meta=rd(META);lookup={(v['sample_id'],v['position']):i for i,v in enumerate(meta)}
 for file in sorted((F/'qualified_inputs').glob('*_fit.npz')):
  pool=file.name[:-8];old=np.load(U/'qualified_inputs'/file.name);new=np.load(file);pub=np.load(E/'public_entry'/file.name);mask=pub['eligible'];assert np.array_equal(old['ids'],pub['ids']) and np.array_equal(old['X32'][mask],new['X32']) and np.array_equal(np.flatnonzero(mask),new['original_row_index']);keep=bool(old['fallback_KEEP'][0]);ox=old['X32'];nx=new['X32'];assert ox.shape[1]==nx.shape[1]==39
  changed=0
  for j in range(39):
   omin,omax=float(ox[:,j].min()),float(ox[:,j].max());nmin,nmax=float(nx[:,j].min()),float(nx[:,j].max());mi=np.flatnonzero(ox[:,j]==omin);ma=np.flatnonzero(ox[:,j]==omax);change=omin!=nmin or omax!=nmax;changed+=change;ranges.append(dict(pool=pool,KEEP=keep,feature=j,old_min=omin,old_max=omax,new_min=nmin,new_max=nmax,old_nonconstant=omin!=omax,new_nonconstant=nmin!=nmax,range_changed=change,old_min_indices=mi.tolist(),old_max_indices=ma.tolist(),removed_min_indices=mi[~mask[mi]].tolist(),removed_max_indices=ma[~mask[ma]].tolist()))
  a=pickle.loads((OLD/'models'/f'{pool}.pkl').read_bytes());b=pickle.loads((Q/'models'/f'{pool}.pkl').read_bytes());assert a['KEEP']==b['KEEP']==keep
  oldleaf=newleaf=0
  if not keep:
   for k,(ta,tb) in enumerate(zip(a['model'].estimators_,b['model'].estimators_)):
    af,bf=int(ta.tree_.feature[0]),int(tb.tree_.feature[0]);at,bt=float(ta.tree_.threshold[0]),float(tb.tree_.threshold[0]);oldleaf+=int((ta.tree_.children_left==-1).sum());newleaf+=int((tb.tree_.children_left==-1).sum());roots.append(dict(pool=pool,tree=k,old_feature=af,new_feature=bf,old_threshold=at,new_threshold=bt,same_root_feature=af==bf,same_root_split=af==bf and at==bt,old_root_n=int(ta.tree_.n_node_samples[0]),new_root_n=int(tb.tree_.n_node_samples[0]),old_root_weight=float(ta.tree_.weighted_n_node_samples[0]),new_root_weight=float(tb.tree_.weighted_n_node_samples[0]),old_nodes=int(ta.tree_.node_count),new_nodes=int(tb.tree_.node_count),old_root_feature_range_changed=False if af<0 else ranges[-39+af]['range_changed']))
  pools.append(dict(pool=pool,KEEP=keep,old_rows=len(ox),new_rows=len(nx),changed_ranges=changed,old_leaves=oldleaf,new_leaves=newleaf))
 focus=[]
 for r in rd(Q/'matched277_changes.json'):focus.append(dict(pool=f"f{r['fold']}_s{r['seed']}",id=lookup[r['sample_id'],r['position']],seed=r['seed'],date=r['date'],call=r['call'],sample_id=r['sample_id'],position=r['position'],old_continue=r['old_continue'],new_continue=r['new_continue']))
 assert len(ranges)==468 and len(roots)==704 and len(focus)==40;save('public_all39_ranges.json',ranges);save('public_all704_roots.json',roots);save('public_pool_structure.json',pools);save('public_all40_focus.json',focus);save('public_freeze.json',{n:sha(O/n) for n in ['public_all39_ranges.json','public_all704_roots.json','public_pool_structure.json','public_all40_focus.json']});archive('public');print('GRID281_PUBLIC468RANGES_704ROOTS_40FOCUS_NO_TRAVERSAL')

def analyze():
 verify();h=published('public');assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(public_published=h));ranges=rd(O/'public_all39_ranges.json');roots=rd(O/'public_all704_roots.json');meta=rd(META);own=[];annotated=[]
 for file in sorted((F/'qualified_inputs').glob('*_fit.npz')):
  pool=file.name[:-8];old=np.load(U/'qualified_inputs'/file.name);new=np.load(file);pub=np.load(E/'public_entry'/file.name);mask=pub['eligible'];keep=bool(old['fallback_KEEP'][0]);assert np.array_equal(old['y'][mask],new['y']) and np.array_equal(old['weight'][mask],new['weight']);own.append(dict(pool=pool,KEEP=keep,old_risk=float(old['weight'].sum()),new_risk=float(new['weight'].sum()),removed_risk=float(old['weight'][~mask].sum()),removed_rows=int((~mask).sum()),removed_positive=int(((old['y']>0)&~mask).sum())))
  for r in [v for v in ranges if v['pool']==pool]:
   ii=sorted(set(r['removed_min_indices'])|set(r['removed_max_indices']));annotated.append(dict(pool=pool,feature=r['feature'],range_changed=r['range_changed'],removed_extrema_occurrences=len(ii),removed_extrema_positive=int(old['y'][ii].sum()) if ii else 0,removed_extrema_risk=float(old['weight'][ii].sum()) if ii else 0.,ids=[int(old['ids'][j]) for j in ii]))
  for root in [v for v in roots if v['pool']==pool]:assert root['old_root_n']==len(old['ids']) and root['new_root_n']==len(new['ids']) and abs(root['old_root_weight']-old['weight'].sum())<1e-8 and abs(root['new_root_weight']-new['weight'].sum())<1e-8
 changes=rd(Q/'matched277_changes.json');metrics=rd(Q/'search_metrics.json');calls=collections.defaultdict(list)
 for r in changes:calls[(r['sample_id'],r['position'])].append(r)
 states=[dict(sample_id=k[0],position=k[1],seeds=sorted({r['seed'] for r in v}),records=len(v),directions=[dict(seed=r['seed'],old=r['old_continue'],new=r['new_continue']) for r in v]) for k,v in calls.items()];cohorts=[];contributions=[]
 for seed in range(3):
  ss=[r for r in changes if r['seed']==seed];nd={r['date']:r['cases'] for r in metrics[f'ENTRY_CONDITIONED_seed{seed}']['per_date']};group=collections.defaultdict(lambda:dict(NET=0.,query=0.,records=0))
  for r in ss:
   g=group[r['call']];g['NET']+=r['NET_delta']/(16*nd[r['date']]);g['query']+=r['query_delta']/(16*nd[r['date']]);g['records']+=1
  cohorts.append(dict(seed=seed,records=len(ss),calls=len({r['call'] for r in ss}),dates=len({r['date'] for r in ss}),raw_query_delta=sum(r['query_delta'] for r in ss),profitable_losses=sum(not r['new_continue'] and r['actual_next_NET']>0 for r in ss),macro_NET_delta=sum(g['NET'] for g in group.values()),macro_query_delta=sum(g['query'] for g in group.values())))
  contributions.extend(dict(seed=seed,call=k,**v) for k,v in sorted(group.items()))
 active=[r for r in own if not r['KEEP']];ar=[r for r in ranges if not r['KEEP']];summary=dict(all_feature_pool_pairs468=len(ranges),active_feature_pool_pairs429=len(ar),active_range_changed=sum(r['range_changed'] for r in ar),pools_with_range_change=len({r['pool'] for r in ar if r['range_changed']}),nonconstant_to_constant=sum(r['old_nonconstant'] and not r['new_nonconstant'] for r in ar),roots704=len(roots),same_root_feature=sum(r['same_root_feature'] for r in roots),same_root_split=sum(r['same_root_split'] for r in roots),old_root_feature_range_changed=sum(r['old_root_feature_range_changed'] for r in roots),old_leaves=sum(r['old_leaves'] for r in rd(O/'public_pool_structure.json')),new_leaves=sum(r['new_leaves'] for r in rd(O/'public_pool_structure.json')),raw_rows_removed=sum(r['removed_rows'] for r in active),removed_risk=sum(r['removed_risk'] for r in active),old_risk=sum(r['old_risk'] for r in active),focus40=len(changes),unique_focus_states=len(states),states_cross2seeds=sum(len(r['seeds'])>=2 for r in states),states_cross3seeds=sum(len(r['seeds'])==3 for r in states),per_seed=cohorts,newfits0=True,newapply_predict0=True,newstrategy_returns0=True)
 save('ownfit_extrema_support.json',annotated);save('ownfit_risk_support.json',own);save('all_change_call_contributions.json',contributions);save('cross_seed_related_states.json',states);save('diagnosis_summary.json',summary);verify();archive('analyzed');print('GRID281_READONLY_DIAGNOSED',json.dumps(summary))
if __name__=='__main__':globals()[sys.argv[1]]()
