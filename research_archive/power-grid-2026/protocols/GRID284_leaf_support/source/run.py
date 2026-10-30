"""GRID284 fixed all-leaf support and all saved changes, read-only."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,pickle,collections,ast
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid284_leaf_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID284_leaf_support';Q=R/'outputs/grid283_fixed_partition_leaf';ROUTES=R/'outputs/grid282_fixed_partition_plan';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def rd(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def verify():
 for rel,h in rd(O/'source_freeze.json').items():assert sha(R/rel)==h,rel

def actualA(grid):
 src=(R/f'work/{grid}/run.py').read_text(encoding='utf-8-sig');n=next(x.value for x in ast.parse(src).body if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='A' for y in x.targets));assert isinstance(n,ast.BinOp) and isinstance(n.right,ast.Constant);return REPO/n.right.value

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8')
def published(phase):
 verify();h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 for rel,s in rd(A/phase/'archive_manifest.json').items():assert hashlib.sha256(git('show',h+':'+(A/phase/rel).relative_to(REPO).as_posix())).hexdigest()==s
 return h

def prepare():
 assert not O.exists() and rd(Q/'NEXT_PROTOCOL_PLAN.json')['id']=='GRID284';h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h;aq=actualA('grid283_fixed_partition_leaf');assert rd(Q/'complete.json')['passed']
 for n in ['NEXT_PROTOCOL_PLAN.json','complete.json','matched280_changes.json','matched277_changes.json','matched280_summary.json','matched277_summary.json','partition_references.json']:
  assert git('show',h+':'+(aq/'audited'/n).relative_to(REPO).as_posix())==(Q/n).read_bytes()
 for phase,name in [('trained','trained_freeze.json'),('scored','scored_freeze.json')]:
  for rel,s in rd(Q/name).items():assert sha(Q/rel)==s and hashlib.sha256(git('show',h+':'+(aq/phase/rel).relative_to(REPO).as_posix())).hexdigest()==s
 refs=rd(Q/'partition_references.json');loc=rd(ROUTES/'qualified_artifact_locations.json')
 for rel,v in loc.items():assert sha(ROUTES/rel)==v['sha256'] and hashlib.sha256(git('show',h+':'+v['git_path'])).hexdigest()==v['sha256']
 for ref in refs:
  for v in ref['inputs'].values():assert sha(R/v['relative_to_R'])==v['sha256'] and hashlib.sha256(git('show',h+':'+v['git_path'])).hexdigest()==v['sha256']
  assert sha(R/ref['partition_relative_to_R'])==ref['partition_sha256']
 O.mkdir();save('input_publication.json',dict(passed=True,git_sha=h,all283results_models_scores_and282routes279inputs_published=True));save('partition_references.json',refs)
 save('design.json',dict(id='GRID284',scope='ONEread-only all6715leaves/all43vs280+3vs277savedchanges;0fit/scalar/newprobability/treeapply/NNforward/newpolicy orhybridreturns.',public='ALL frozen277leaf IDs and282savedfit/cal membership rowindices;no traversal/noownfit target reading. ALL published43+3changes selectedwithouttargetfilter;publishedoldnewscore/tau/margins only. Membership firstpublish thenown279yweight annotation.',analysis='ALL6715leaf conditionalclassmass/risk/positive support vsoriginal277staticleafweightedstatistics/prior and283savedstatistics. Explainzero fallback andmassshifts;do notrefit tables/recompute counterfactual policy scores orcutoffs. ALLquality/query/call/date/seed changes withmacroconservation,adversecases retained.',numerics='Fixed np.mean vsnativeoriginalsum floatingorder caveat;publishedscore/margins only,notnewaccumulatorpolicy. Tolerance1e-10 forstaticmass identity only,notthresholdtuning.',stopping='User2026-10-30 requestsfinishthisanalysis thenSTOPnewexperiments;grlremainsPAUSED. Prepareall-experiment inventory andweeklyoverview/upload,notnexttraining.',limits='Branch190 unchanged;6715leafprobparams/13430stats/704oldtrees;331excluded/16copies/543guard and6oldcontinuations8285updates604.703actual678.156cap retained. Parentown12/internalcal4/exposedrelatedprefixes notindependentOOFblind/fullweek/time. No final8/physics/grid/paper/FRAME/install;originalcostsNETlabels/controller/orderaliasesstation128/64/stagnation2/.9 unchanged.'))
 paths=[*W.glob('*.py'),O/'design.json',O/'input_publication.json',O/'partition_references.json',Q/'NEXT_PROTOCOL_PLAN.json',META]
 for folder in [Q,R/'outputs/grid277_selector_free',R/'outputs/grid280_invocation_conditioned']:
  paths.extend(p for p in folder.glob('*.json') if p.name in ['matched280_changes.json','matched277_changes.json','matched280_summary.json','matched277_summary.json','search_metrics.json','gate_counts.json','per_seed_preservation_and_exposure.json']);paths.extend((folder/'models').glob('*.pkl'));paths.extend((folder/'outer_predictions').glob('*.npz'))
 for ref in refs:paths.append(R/ref['partition_relative_to_R']);paths.extend(R/v['relative_to_R'] for v in ref['inputs'].values())
 paths+=list((ROUTES/'routes').glob('*.npz'))+list((ROUTES/'original_leaf_parameters').glob('*.npz'))+list((Q/'leaf_statistics').glob('*.npz'))
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID284_SOURCE_FROZEN')

def public():
 h=published('source');assert not (O/'public_started.json').exists();save('public_started.json',dict(source_published=h));members=[];focus=[];meta=rd(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)}
 for ref in rd(O/'partition_references.json'):
  pool=ref['pool'];old=pickle.loads((R/ref['partition_relative_to_R']).read_bytes());fit=np.load(ROUTES/'routes'/f'{pool}_fit.npz');cal=np.load(ROUTES/'routes'/f'{pool}_cal.npz')
  if ref['KEEP']:assert fit['leaf_ids'].shape[1]==cal['leaf_ids'].shape[1]==0;continue
  for j,t in enumerate(old['model'].estimators_):
   for leaf in np.flatnonzero(t.tree_.children_left==-1):members.append(dict(pool=pool,tree=j,leaf=int(leaf),fit_rows=np.flatnonzero(fit['leaf_ids'][:,j]==leaf).tolist(),cal_rows=np.flatnonzero(cal['leaf_ids'][:,j]==leaf).tolist()))
 for label,folder in [('280',R/'outputs/grid280_invocation_conditioned'),('277',R/'outputs/grid277_selector_free')]:
  for r in rd(Q/f'matched{label}_changes.json'):
   pool=f"f{r['fold']}_s{r['seed']}";i=lookup[r['sample_id'],r['position']];old=np.load(folder/'outer_predictions'/f'{pool}.npz');new=np.load(Q/'outer_predictions'/f'{pool}.npz');a=np.flatnonzero(old['ids']==i);b=np.flatnonzero(new['ids']==i);assert len(a)==len(b)==1;a=int(a[0]);b=int(b[0]);ob=pickle.loads((folder/'models'/f'{pool}.pkl').read_bytes());nb=pickle.loads((Q/'models'/f'{pool}.pkl').read_bytes());scoreterm=float(new['forest_score'][b]-old['forest_score'][a]);tauterm=nb['tau']-ob['tau'];margin=float(new['gate_margin'][b]-old['gate_margin'][a]);assert abs(margin-(scoreterm-tauterm))<1e-15
   focus.append(dict(control=label,pool=pool,id=i,seed=r['seed'],date=r['date'],call=r['call'],sample_id=r['sample_id'],position=r['position'],old_continue=r['old_continue'],new_continue=r['new_continue'],old_score=float(old['forest_score'][a]),new_score=float(new['forest_score'][b]),old_tau=ob['tau'],new_tau=nb['tau'],old_margin=float(old['gate_margin'][a]),new_margin=float(new['gate_margin'][b]),score_term=scoreterm,cutoff_term=tauterm,margin_delta=margin))
 assert len(members)==6715 and len(focus)==46;save('public_all6715_leaf_members.json',members);save('public_all46_saved_change_margins.json',focus);save('public_freeze.json',{n:sha(O/n) for n in ['public_all6715_leaf_members.json','public_all46_saved_change_margins.json']});archive('public');print('GRID284_PUBLIC_MEMBERS6715_FOCUS46_NO_FORWARD')

def analyze():
 h=published('public');assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(public_published=h));leafrows=[];poolrows=[];members=rd(O/'public_all6715_leaf_members.json');meta=rd(META);refs=rd(O/'partition_references.json');focus=rd(O/'public_all46_saved_change_margins.json')
 for ref in refs:
  pool=ref['pool'];fit=np.load(R/ref['inputs']['fit']['relative_to_R']);cal=np.load(R/ref['inputs']['cal']['relative_to_R']);stats=np.load(Q/'leaf_statistics'/f'{pool}.npz');old=pickle.loads((R/ref['partition_relative_to_R']).read_bytes());new=pickle.loads((Q/'models'/f'{pool}.pkl').read_bytes());keep=ref['KEEP'];pp=[]
  if keep:poolrows.append(dict(pool=pool,KEEP=True,fit_rows=len(fit['ids']),positive_rows=int(fit['y'].sum())));continue
  for r in [v for v in members if v['pool']==pool]:
   j=r['tree'];leaf=r['leaf'];ii=r['fit_rows'];ci=r['cal_rows'];tree=old['model'].estimators_[j].tree_;om=float(tree.weighted_n_node_samples[leaf]);prior=float(tree.value[leaf,0,1]/tree.value[leaf,0,:].sum());op=prior*om;nm=float(stats[f'mass_{j}'][leaf]);np_=float(stats[f'positive_mass_{j}'][leaf]);assert len(ii)==int(stats[f'count_{j}'][leaf]);assert abs(float(fit['weight'][ii].sum())-nm)<1e-10 and abs(float(fit['weight'][np.asarray(ii,dtype=int)[fit['y'][ii]>0]].sum())-np_)<1e-10
   row=dict(pool=pool,tree=j,leaf=leaf,old_raw_count=int(tree.n_node_samples[leaf]),new_raw_count=len(ii),old_weight=om,new_weight=nm,removed_weight=om-nm,old_positive_weight=op,new_positive_weight=np_,positive_weight_delta=np_-op,old_frequency=prior,new_frequency=float(new['tables'][j][leaf]),frequency_delta=float(new['tables'][j][leaf]-prior),zero_fallback=nm==0,fit_positive_rows=int(fit['y'][ii].sum()),fit_negative_rows=len(ii)-int(fit['y'][ii].sum()),fit_calls=len({meta[int(fit['ids'][i])]['physical_state_group'] for i in ii}),fit_positive_calls=len({meta[int(fit['ids'][i])]['physical_state_group'] for i in ii if fit['y'][i]>0}),guard_rows=len(ci),guard_fit8_rows=int(cal['fit_role'][ci].sum()),guard_cal4_rows=len(ci)-int(cal['fit_role'][ci].sum()))
   leafrows.append(row);pp.append(row)
  poolrows.append(dict(pool=pool,KEEP=False,fit_rows=len(fit['ids']),positive_rows=int(fit['y'].sum()),leaves=len(pp),positive_mass_maxabschange=max(abs(v['positive_weight_delta']) for v in pp),conditional_leaves_frequency_increased=sum(v['frequency_delta']>1e-12 for v in pp),frequency_decreased_beyond_roundoff=sum(v['frequency_delta'] < -1e-12 for v in pp),removed_risk_per_tree=sum(v['removed_weight'] for v in pp)/64,raw_removed_per_tree=sum(v['old_raw_count']-v['new_raw_count'] for v in pp)/64,empty_fallback=sum(v['zero_fallback'] for v in pp)))
 summary=dict(leaves6715=len(leafrows),positive_mass_max_abs_delta=max(abs(v['positive_weight_delta']) for v in leafrows),leaves_positive_mass_changed_gt1e10=sum(abs(v['positive_weight_delta'])>1e-10 for v in leafrows),frequency_increased_gt1e12=sum(v['frequency_delta']>1e-12 for v in leafrows),frequency_decreased_gt1e12=sum(v['frequency_delta'] < -1e-12 for v in leafrows),empty_fallback=sum(v['zero_fallback'] for v in leafrows),removed_weight_per_tree_total=sum(v['removed_weight'] for v in leafrows)/64,removed_rows_per_tree_total=sum(v['old_raw_count']-v['new_raw_count'] for v in leafrows)/64,newfits0=True,newforward0=True,newstrategy0=True)
 annotated=[];cohorts=[];states=[];calls=[]
 for label in ['280','277']:
  changes=rd(Q/f'matched{label}_changes.json');matched=rd(Q/f'matched{label}_summary.json');metrics=rd(Q/'search_metrics.json');dd=collections.defaultdict(list)
  for r in changes:dd[r['sample_id'],r['position']].append(r)
  states.extend(dict(control=label,sample_id=k[0],position=k[1],seeds=sorted({r['seed'] for r in v}),records=len(v)) for k,v in dd.items())
  for seed in range(3):
   ss=[r for r in changes if r['seed']==seed];nd={r['date']:r['cases'] for r in metrics[f'FIXED_LEAF_seed{seed}']['per_date']};groups=collections.defaultdict(lambda:dict(records=0,NET=0.,query=0.,rawquery=0))
   for r in ss:
    g=groups[r['call']];g['records']+=1;g['NET']+=r['NET_delta']/(16*nd[r['date']]);g['query']+=r['query_delta']/(16*nd[r['date']]);g['rawquery']+=r['query_delta']
   d=matched['per_seed'][seed]['macro_delta'];assert abs(sum(v['NET'] for v in groups.values())-d['mean_net_utility'])<1e-12 and abs(sum(v['query'] for v in groups.values())-d['mean_queries'])<1e-12
   cohorts.append(dict(control=label,seed=seed,records=len(ss),calls=len(groups),dates=len({r['date'] for r in ss}),rawquery=sum(r['query_delta'] for r in ss),profit_restored=sum(r['new_continue'] and r['actual_next_NET']>0 for r in ss),profit_lost=sum(not r['new_continue'] and r['actual_next_NET']>0 for r in ss),NET_delta=d['mean_net_utility'],query_delta=d['mean_queries']));calls.extend(dict(control=label,seed=seed,call=k,**v) for k,v in sorted(groups.items()))
  for f in [v for v in focus if v['control']==label]:
   r=next(v for v in changes if v['sample_id']==f['sample_id'] and v['seed']==f['seed'] and v['position']==f['position']);annotated.append(dict(**f,actual_next_NET=r['actual_next_NET'],actual_next_queries=r['actual_next_queries'],NET_delta=r['NET_delta'],raw_query_delta=r['query_delta'],larger_abs_margin_component='score' if abs(f['score_term'])>=abs(f['cutoff_term']) else 'cutoff',minimum_endpoint_abs_margin=min(abs(f['old_margin']),abs(f['new_margin']))))
 summary.update(focus46=len(annotated),vs280_changes=43,vs277_changes=3,per_seed=cohorts,unique_states_vs280=sum(r['control']=='280' for r in states),cross2seed_states_vs280=sum(r['control']=='280' and len(r['seeds'])>=2 for r in states),cross3seed_states_vs280=sum(r['control']=='280' and len(r['seeds'])==3 for r in states),minimum_abs_margin_vs277=min(v['minimum_endpoint_abs_margin'] for v in annotated if v['control']=='277'),score_dominant_vs277=sum(v['control']=='277' and v['larger_abs_margin_component']=='score' for v in annotated))
 for n,v in [('all_leaf_class_support.json',leafrows),('pool_class_support.json',poolrows),('all_change_margins_and_outcomes.json',annotated),('all_call_contributions.json',calls),('all_related_state_recurrence.json',states),('diagnosis_summary.json',summary)]:save(n,v)
 verify();archive('analyzed');print('GRID284_READONLY_DIAGNOSED',json.dumps(summary))
if __name__=='__main__':globals()[sys.argv[1]]()
