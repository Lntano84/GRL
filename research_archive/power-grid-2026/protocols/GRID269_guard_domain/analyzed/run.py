"""GRID269 readonly guard-domain coverage and fixed-tree support; no cutoff fit."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,pickle,ast,collections
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid269_guard_domain';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID269_guard_domain';N=R/'outputs/grid268_conditional_forest';Q=R/'outputs/grid267_conditional_forest_plan';LEGACY=R/'outputs/grid249_context_bit_plan';RAW=LEGACY/'outer_parent_public';META=R/'outputs/grid113_relational_learning/train_metadata.json';ROLES=R/'outputs/grid241_parent_prior_plan/qualified_pools.json'
def rd(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def path_constant(file,key):
 tree=ast.parse(file.read_text(encoding='utf-8-sig'));node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets));assert isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and isinstance(node.left,ast.Name) and isinstance(node.right,ast.Constant);return {'R':R,'REPO':REPO}[node.left.id]/node.right.value

def verify():
 for p,h in rd(O/'source_freeze.json').items():assert sha(R/p)==h,p

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 if phase=='public':shutil.copytree(O/'union_inputs',d/'union_inputs')
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def published(phase):
 h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 for rel,v in rd(A/phase/'archive_manifest.json').items():assert hashlib.sha256(git('show',h+':'+(A/phase/rel).relative_to(REPO).as_posix())).hexdigest()==v
 return h

def prepare():
 assert not O.exists() and rd(N/'complete.json')['passed'];h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 s250=R/'work/grid250_context_residual/run.py';s249=R/'work/grid249_context_bit_plan/run.py';s246=R/'work/grid246_canonical_plan/run.py';assert path_constant(s250,'Q')==LEGACY and path_constant(s249,'Q')==R/'outputs/grid246_canonical_plan' and path_constant(s246,'Q')==path_constant(s249,'PUBLIC')
 legacy_ar=path_constant(s249,'A')/'qualified'
 for f in (LEGACY/'calibration_inputs').glob('*.npz'):assert git('show',h+':'+(legacy_ar/'calibration_inputs'/f.name).relative_to(REPO).as_posix())==f.read_bytes()
 metakey=META.relative_to(R).as_posix();assert rd(R/'outputs/grid250_context_residual/source_freeze.json')[metakey]==sha(META)==rd(Q/'source_freeze.json')[metakey]
 for name in ['matched261_changes.json','matched261_summary.json','per_seed_preservation_and_exposure.json','NEXT_PROTOCOL_PLAN.json','complete.json']:
  ar=REPO/'research_archive/power-grid-2026/protocols/GRID268_conditional_forest/audited'/name;assert git('show',h+':'+ar.relative_to(REPO).as_posix())==(N/name).read_bytes()
 assert rd(LEGACY/'qualification.json')['allnative_contextbits_verified'];O.mkdir();save('source_resolution.json',dict(legacy_from250_Q=str(LEGACY),canonical_from249_Q=str(path_constant(s249,'Q')),parent_from246_Q=str(path_constant(s246,'Q')),legacy_archive_from_actual249_A=str(legacy_ar),global_metadata_250_and267_hash_identical=True,original_nativeSTOP_qualification_reused=True))
 save('design.json',dict(id='GRID269',scope='Readonly268fixedforest/KEEP exposure,232cancellation leafsupport andown12guarddomaincoverage. 0model/scalar/cutoff/alternativepolicy/NNfits.',guard_selection='L=legacy249nativeNET12STOP y>0;C=current267referenceNN8STOP y>0,perpoolglobalmetadataids. ReportALLL/C/intersection/missing. ReachableL requires frozen261 score>=0 andb>0 usinglegacyX first38tanh;reportunreachableL separately,do notsilentlydrop.',union='FixedU=C union reachableL,dedupby(pool,id),currentrowsfirst. Newoldmembers selectedonlybyown12y/source/unchangedparententry,neverouterlossIDs. VerifyNET/role/featurealignment andown12roleexcludesouter4. Exportmemberdefinition/inputs BEFOREdiagnosingleaflabels. No forestprobabilityornewtau onold-onlymembers in269.',alignment='LegacyX64 first38tanh,originalb/STOPbit preserved,X32 castsame40. SharedNET/y/roles andforestX32 exact,old/newparentcontinue compare. LogfullX64maxdiff;do not invent targets/scales. If conflictno futureuniontrainingqualificationuntilresolved.',leaf_public='Fixedtree.apply only:cache ownfitX32/ids perleaf and232publicfocus leafkeys;no fit y/weight inpublicleafpairselection. Thenpublishbeforeowntraininglabelsupport. Cal y usedsolelyalreadyknownown12guardselection,notnewouterlabels.',leaf_analysis='Positive risk fraction/count/calls onvisitedleaves,occurrenceweighted across64trees descriptive/notindependent. Reproduce publishedforestscore, no leaf-basedhandpolicy or alternative threshold.',exposure='Common active dates fixed by intersection ofpublishedrealforestavailabilitybeforeoutcomeaverages,not selectedbyquality. Reportall3seeds/common12andkeepall16originalresults,neverdeploysubsetorbestseed.',ancestry='Reuse235/253known parentown12/internalcal4limitations,not broadancestryrerun. Currentbranch95plus6historicalcontinuations8285updates/604.703sactual/678.156cap.',bounds='No final8/physical/grid/paper/FRAME/install,fee/labels/baselinecontroller/order/aliases/station/128/64/stagnation2/.9 unchanged. No tau/quantile/hybridstrategy return,parameter/risk/seedscan ordateexception.'))
 paths=[W/'run.py',O/'design.json',O/'source_resolution.json',s250,s249,s246,R/'outputs/grid250_context_residual/source_freeze.json',Q/'source_freeze.json',LEGACY/'qualification.json',N/'NEXT_PROTOCOL_PLAN.json',N/'matched261_changes.json',N/'matched261_summary.json',N/'search_metrics.json',N/'per_seed_preservation_and_exposure.json',R/'outputs/grid261_bounded_basis/search_metrics.json',META,ROLES,*list((LEGACY/'calibration_inputs').glob('*.npz')),*list((Q/'qualified_inputs').glob('*.npz')),*list((N/'models').glob('*.pkl')),*list((N/'outer_predictions').glob('*.npz')),*list((N/'calibration_predictions').glob('*.npz')),*list(RAW.glob('*.npz'))]
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID269_SOURCE_FROZEN')

def public():
 verify();h=published('source');assert not (O/'public_started.json').exists();save('public_started.json',dict(source_published=h));meta=rd(META);roles={r['pool']:r for r in rd(ROLES)};coverage=[];selection=[];limits=[];(O/'union_inputs').mkdir()
 for file in sorted((N/'models').glob('*.pkl')):
  pool=file.stem;b=pickle.loads(file.read_bytes());old=np.load(LEGACY/'calibration_inputs'/f'{pool}.npz');new=np.load(Q/'qualified_inputs'/f'{pool}_cal.npz');base=np.load(Q/'qualified_inputs'/f'{pool}_base.npz');ox=old['X'].copy();ox[:,:38]=np.tanh(ox[:,:38]);ox32=ox.astype(np.float32);oldscore=old['b']+float(base['alpha'][0])*((old['b']+np.sum(ox*base['theta'][:-1],axis=1)+base['theta'][-1])-old['b']);newscore=new['b']+float(base['alpha'][0])*((new['b']+np.sum(new['X']*base['theta'][:-1],axis=1)+base['theta'][-1])-new['b']);ol={int(v):i for i,v in enumerate(old['ids']) if old['y'][i]>0};cu={int(v):i for i,v in enumerate(new['ids']) if new['y'][i]>0};assert len(np.unique(old['ids']))==len(old['ids']) and len(np.unique(new['ids']))==len(new['ids']);shared=set(ol)&set(cu);missing=set(ol)-set(cu);reach={mid for mid,i in ol.items() if old['b'][i]>0 and oldscore[i]>=0};assert all(new['b'][i]>0 and newscore[i]>=0 for i in cu.values())
  fullmax=0.;NETok=roleok=X32ok=parentok=True
  for mid in shared:
   i,j=ol[mid],cu[mid];NETok &= float(old['actual_NET'][i])==float(new['actual_NET'][j]);roleok &= bool(old['fit_role'][i])==bool(new['fit_role'][j]);X32ok &= np.array_equal(ox32[i],new['X32'][j]);parentok &= bool(old['b'][i]>0 and oldscore[i]>=0)==bool(new['b'][j]>0 and newscore[j]>=0);fullmax=max(fullmax,float(abs(ox[i]-new['X'][j]).max()))
  added=missing&reach;union=sorted(set(cu)|reach);data={k:[] for k in ['X','X32','b','ids','y','actual_NET','fit_role','origin_current','origin_legacy','parent261_score']};own=set(roles[pool]['fit_dates'])|set(roles[pool]['cal4_dates']);outer=set(roles[pool]['outer_dates'])
  for mid in union:
   isnew=mid in cu;i=cu[mid] if isnew else ol[mid];src=new if isnew else old;xx=new['X'][i] if isnew else ox[i];xx32=new['X32'][i] if isnew else ox32[i];score=newscore[i] if isnew else oldscore[i];mt=meta[mid];assert mt['date'] in own and mt['date'] not in outer and xx[39]==1 and np.isfinite(xx32).all() and src['actual_NET'][i]>0 and score>=0
   data['X'].append(xx);data['X32'].append(xx32);data['b'].append(src['b'][i]);data['ids'].append(mid);data['y'].append(1);data['actual_NET'].append(src['actual_NET'][i]);data['fit_role'].append(src['fit_role'][i]);data['origin_current'].append(isnew);data['origin_legacy'].append(mid in ol);data['parent261_score'].append(score)
  np.savez_compressed(O/'union_inputs'/f'{pool}.npz',**{k:np.asarray(v,dtype=(np.float32 if k=='X32' else np.int64 if k in ['ids','y'] else bool if k in ['fit_role','origin_current','origin_legacy'] else np.float64)) for k,v in data.items()})
  row=dict(pool=pool,KEEP=b['KEEP'],legacy_positive=len(ol),current_positive=len(cu),shared=len(shared),legacy_missing_from_current=len(missing),legacy_parent_reachable=len(reach),legacy_parent_STOP=len(set(ol)-reach),added_reachable=len(added),union_positive=len(union),shared_NET_exact=bool(NETok),shared_role_exact=bool(roleok),shared_forestX32_exact=bool(X32ok),shared_parentdecision_same=bool(parentok),shared_X64_maxdiff=fullmax,union_semantics_aligned=bool(NETok and roleok and X32ok and parentok));coverage.append(row)
  selection.append(dict(**row,legacy_ids=sorted(ol),current_ids=sorted(cu),shared_ids=sorted(shared),legacy_missing_ids=sorted(missing),added_reachable_ids=sorted(added),inherited_parent_STOP_ids=sorted(set(ol)-reach),union_ids=union))
  cp=np.load(N/'calibration_predictions'/f'{pool}.npz');assert np.array_equal(cp['ids'],new['ids']);lim=np.flatnonzero(cp['positive']&(cp['score']==b['tau']))
  if not b['KEEP']:
   for i in lim:limits.append(dict(pool=pool,id=int(new['ids'][i]),date=meta[int(new['ids'][i])]['date'],call=meta[int(new['ids'][i])]['physical_state_group'],fit_role=bool(new['fit_role'][i]),also_legacy=int(new['ids'][i]) in ol,tau=b['tau']))
 save('guard_coverage.json',dict(pools=coverage,legacy_positive=sum(r['legacy_positive'] for r in coverage),current_positive=sum(r['current_positive'] for r in coverage),shared=sum(r['shared'] for r in coverage),missing_legacy=sum(r['legacy_missing_from_current'] for r in coverage),added_reachable=sum(r['added_reachable'] for r in coverage),legacy_parent_STOP=sum(r['legacy_parent_STOP'] for r in coverage),union_positive=sum(r['union_positive'] for r in coverage),all_shared_semantics_aligned=all(r['union_semantics_aligned'] for r in coverage),eligible_new_scalar_pools=[r['pool'] for r in coverage if r['added_reachable']>0 and not r['KEEP']]))
 save('guard_selection.json',selection);save('current_limiting_guards.json',limits)
 changes=rd(N/'matched261_changes.json');lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};focus=[];members={};cache={}
 for r in changes:
  pool=f"f{r['fold']}_s{r['seed']}"
  if pool not in cache:
   fit=np.load(Q/'qualified_inputs'/f'{pool}_fit.npz');bundle=pickle.loads((N/'models'/f'{pool}.pkl').read_bytes());assert not bundle['KEEP'];fit_leaves=[t.apply(fit['X32']) for t in bundle['model'].estimators_];cache[pool]=(fit,bundle,fit_leaves,np.load(RAW/f'{pool}.npz'),np.load(N/'outer_predictions'/f'{pool}.npz'))
  fit,bundle,fit_leaves,raw,pred=cache[pool];mid=lookup[r['sample_id'],r['position']];j=int(np.flatnonzero(raw['ids']==mid)[0]);jj=int(np.flatnonzero(pred['ids']==mid)[0]);x=raw['X'][j].copy();x[:38]=np.tanh(x[:38]);x32=x.astype(np.float32)[None,:];keys=[]
  for ti,tree in enumerate(bundle['model'].estimators_):
   leaf=int(tree.apply(x32)[0]);key=f'{pool}|{ti}|{leaf}';keys.append(key)
   if key not in members:members[key]=dict(pool=pool,tree=ti,leaf=leaf,fit_indices=np.flatnonzero(fit_leaves[ti]==leaf).tolist())
  focus.append(dict(pool=pool,seed=r['seed'],fold=r['fold'],sample_id=r['sample_id'],position=r['position'],metadata_id=mid,date=r['date'],call=r['call'],forest_score=float(pred['forest_score'][jj]),tau=bundle['tau'],leaf_keys=keys))
 exposure=rd(N/'per_seed_preservation_and_exposure.json');common=sorted(set.intersection(*(set(r['active_forest_dates']) for r in exposure)))
 save('leaf_public_focus.json',focus);save('leaf_public_memberships.json',members);save('common_active_dates.json',dict(dates=common,selection='Intersectionofpresetmodelavailability,notoutcomes',all_three_seeds_retained=True));save('public_access.json',dict(fit_fields_forleafmembership=['X32','ids'],fit_y_weight_read_forleafsupport=False,own12cal_y_andNET_used_forfixedguardselection=True,new_forest_scores_on_oldonly_guard=0,new_cutoff_fits=0,new_policy_evaluations=0));save('public_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'union_inputs').glob('*.npz')),O/'guard_coverage.json',O/'guard_selection.json',O/'current_limiting_guards.json',O/'leaf_public_focus.json',O/'leaf_public_memberships.json',O/'common_active_dates.json',O/'public_access.json']});archive('public');print('GRID269_PUBLIC_SELECTION',json.dumps(rd(O/'guard_coverage.json')))

def analyze():
 verify();h=published('public');assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(publicselection_published=h));meta=rd(META);members=rd(O/'leaf_public_memberships.json');public=rd(O/'leaf_public_focus.json');changes={(r['seed'],r['sample_id'],r['position']):r for r in rd(N/'matched261_changes.json')};fits={};details=[]
 for r in public:
  r=dict(r)
  if r['pool'] not in fits:fits[r['pool']]=np.load(Q/'qualified_inputs'/f"{r['pool']}_fit.npz")
  fit=fits[r['pool']];fractions=[];counts=[];zeropos=0;poscalls=set();callrisk=collections.Counter()
  for key in r['leaf_keys']:
   leaf=members[key];idx=np.array(leaf['fit_indices'],dtype=int);yy=fit['y'][idx];ww=fit['weight'][idx];assert ww.sum()>0;fractions.append(float(ww[yy>0].sum()/ww.sum()));counts.append(len(idx));zeropos+=int(not np.any(yy>0))
   for ii in idx[yy>0]:
    call=meta[int(fit['ids'][ii])]['physical_state_group'];poscalls.add(call);callrisk[call]+=float(fit['weight'][ii])
  assert abs(float(np.mean(fractions))-r['forest_score'])<1e-10;c=changes[r['seed'],r['sample_id'],r['position']];r.update(actual_next_NET=c['actual_next_NET'],actual_next_gain=c['actual_next_gain'],actual_next_queries=c['actual_next_queries'],profitable=c['actual_next_NET']>0,median_leaf_training_rows=float(np.median(counts)),trees_without_positive=zeropos,positive_calls_union=len(poscalls),topcall_positive_risk_share=(max(callrisk.values())/sum(callrisk.values()) if callrisk else None));r.pop('leaf_keys');details.append(r)
 stats=lambda v:dict(min=float(min(v)),median=float(np.median(v)),max=float(max(v))) if v else None
 groups={}
 for name,positive in [('profitable_loss',True),('nonprofitable_cancel',False)]:
  rr=[r for r in details if r['profitable']==positive];groups[name]=dict(records=len(rr),forest_score=stats([r['forest_score'] for r in rr]),median_leaf_training_rows=stats([r['median_leaf_training_rows'] for r in rr]),trees_without_positive=stats([r['trees_without_positive'] for r in rr]),positive_calls_union=stats([r['positive_calls_union'] for r in rr]),topcall_positive_risk_share=stats([r['topcall_positive_risk_share'] for r in rr if r['topcall_positive_risk_share'] is not None]))
 common=rd(O/'common_active_dates.json')['dates'];newmetrics=rd(N/'search_metrics.json');oldmetrics=rd(R/'outputs/grid261_bounded_basis/search_metrics.json');commonmetrics=[]
 for seed in range(3):
  nn={r['date']:r for r in newmetrics[f'NET_FOREST_seed{seed}']['per_date']};oo={r['date']:r for r in oldmetrics[f'BOUNDED_seed{seed}']['per_date']};keys=['mean_net_utility','mean_queries','mean_gain','action_match_rate','gain_capture'];commonmetrics.append(dict(seed=seed,dates=len(common),delta={k:float(np.mean([nn[d][k]-oo[d][k] for d in common])) for k in keys},descriptive_existing_samepolicy_subset_not_deployment=True))
 coverage=rd(O/'guard_coverage.json');summary=dict(passed=True,coverage=coverage,leaf_support=groups,common_active_date_count=len(common),common_existing_policy_comparison=commonmetrics,new_model_fits=0,new_cutoff_fits=0,new_policy_evaluations=0,new_NNforward=0,new_forest_scores_for_oldonly_guard=0,branch95=True)
 save('focus_leaf_support_details.json',details);save('common_active_existing_metrics.json',commonmetrics);save('summary.json',summary);archive('analyzed');print('GRID269_ANALYZED',json.dumps(dict(coverage=coverage,leaf_support=groups,commonmetrics=commonmetrics),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
