"""Saved-change decomposition and own-fit support, no new policy, fit or alpha."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,shutil,collections,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid259_reference_diagnosis';N=R/'outputs/grid258_reference_correction';B=R/'outputs/grid250_context_residual';Q=R/'outputs/grid257_reference_labels';PUB=R/'outputs/grid249_context_bit_plan/outer_parent_public';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID259_reference_diagnosis'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 save(d/'archive_manifest.json',{p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()})
def verify():
 for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel

def prepare():
 assert not O.exists() and read(N/'complete.json')['passed'];O.mkdir()
 save(O/'design.json',dict(id='GRID259',scope='Readonly existing211changes and7misses, contribution/concentration, algebraic endpoint/step difference, ownfit8 support. Zero fits/scalarfits/NNforestforward/newpolicy predictions.',contribution='Fixed4categories restoredpositive/lostpositive/extra-nonprof/cancel-nonprof; preserveall rows, macro1/(3*16*datedenominator),rawqueries separate; allcalls andseed-datecells.',algebra='Onlyunionofpublishedchanges/residualmiss records: scoreN-scoreB = alphaB*(endpointN-endpointB)+(alphaN-alphaB)*(endpointN-b), sameX/b. Descriptive numeric terms only, nohybridarm decisions or utility.',support='All11ownfit8 pool domainrisk positivefractions, physical-call concentration/Kish descriptive count, actualfit endpoint support; public38 coordinate/range/augmentedrowL2 geometry. Focus nearest1STOP/nearest1positiveSTOP Euclidean original40,exclude samephysicalcall, fixedmetric/no learned distance/cutoff.',next_hypothesis='Onlyif continuing: separately qualify uniformparameterfree tanh on ALL38 already-standardized public coordinates,leave b/nativebit and b offset unchanged, original41paramreadout/objective/221/fitroles. No coordinate selection, learned scale, clipping cutoff, alpha/penalty tuning, or date exception. This hypothesis motivated by ownfit publicgeometry and explicit unitL2 uniform-bound gap, not chosen by outer rank. Nottrainedhere,notpresumed effective.',boundaries='Oldexposeddevelopment, correlatedprefixes/seeds, parentown12/cal4 ancestry unchanged. Notcausal/source-onlyproof,notwholeweek/actualtime. No final8/physical/grid/paper/FRAME/install.'))
 paths=[W/'run.py',O/'design.json',N/'complete.json',N/'independent_audit.json',N/'matched250_summary.json',N/'matched250_changes.json',N/'profitable_misses.json',N/'fit_receipts.json',N/'search_metrics.json',N/'trained_freeze.json',N/'scored_freeze.json',B/'complete.json',B/'fit_receipts.json',B/'trained_freeze.json',B/'scored_freeze.json',R/'outputs/grid113_relational_learning/train_metadata.json',Q/'qualification.json',Q/'qualified_freeze.json']
 for folder in ['models','outer_predictions']:paths+=list((N/folder).glob('*.npz'))+list((B/folder).glob('*.npz'))
 paths+=list(PUB.glob('*.npz'))+list((Q/'fit_inputs').glob('*.npz'))+list((Q/'calibration_inputs').glob('*.npz'))
 save(O/'source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in set(paths)});verify();archive('source');print('GRID259_SOURCE_FROZEN')

def run():
 verify();assert not (O/'started.json').exists();head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();assert subprocess.check_output(['git','-C',str(REPO),'ls-remote','lntano-src','refs/heads/main'],text=True).split()[0]==head;assert subprocess.check_output(['git','-C',str(REPO),'show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix()])==(O/'source_freeze.json').read_bytes();save(O/'started.json',dict(source_published=head,newfits=0,newpolicy=0));changes=read(N/'matched250_changes.json');misses=read(N/'profitable_misses.json');metrics=read(N/'search_metrics.json');denom={r['date']:r['cases'] for r in metrics['GATE_seed0']['per_date']};meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)}
 categories={k:[] for k in ['restored_profitable','lost_profitable','extra_nonprofitable','cancelled_nonprofitable']};callmap={};seedmap={};cellmap={}
 for row in changes:
  cat=('restored_profitable' if row['new_continue'] else 'lost_profitable') if row['actual_next_NET']>0 else ('extra_nonprofitable' if row['new_continue'] else 'cancelled_nonprofitable');v=dict(row);v.update(category=cat,macro_weight=1/(3*16*denom[row['date']]));v.update(NET_macro=v['NET_delta']*v['macro_weight'],query_macro=v['query_delta']*v['macro_weight'],gain_macro=v['gain_delta']*v['macro_weight']);categories[cat].append(v)
  for mapping,key in [(callmap,row['call']),(seedmap,str(row['seed'])),(cellmap,str(row['seed'])+'|'+row['date'])]:
   obj=mapping.setdefault(key,dict(rows=0,NET=0.,query=0.,gain=0.,raw_query=0,positive_NET=0.,negative_NET=0.));obj['rows']+=1;obj['NET']+=v['NET_macro'];obj['query']+=v['query_macro'];obj['gain']+=v['gain_macro'];obj['raw_query']+=v['query_delta'];obj['positive_NET']+=max(0.,v['NET_macro']);obj['negative_NET']+=min(0.,v['NET_macro'])
 summary={k:dict(rows=len(v),NET_macro=sum(r['NET_macro'] for r in v),query_macro=sum(r['query_macro'] for r in v),gain_macro=sum(r['gain_macro'] for r in v),raw_queries=sum(r['query_delta'] for r in v),distinct_calls=len({r['call'] for r in v}),distinct_dates=len({r['date'] for r in v}),distinct_sample_positions=len({(r['sample_id'],r['position']) for r in v})) for k,v in categories.items()};ref=read(N/'matched250_summary.json')['family_delta']
 for out,key in [('NET_macro','mean_net_utility'),('query_macro','mean_queries'),('gain_macro','mean_gain')]:assert abs(sum(v[out] for v in summary.values())-ref[key])<1e-12
 assert sum(v['rows'] for v in summary.values())==211 and sum(v['raw_queries'] for v in summary.values())==1628
 save(O/'contribution_categories.json',summary);save(O/'contribution_rows.json',[r for v in categories.values() for r in v]);save(O/'contribution_by_call.json',dict(sorted(callmap.items(),key=lambda x:-abs(x[1]['NET']))));save(O/'contribution_by_seed.json',seedmap);save(O/'contribution_by_seed_date.json',cellmap)
 modeldata={};own=[];geometries=[]
 for file in sorted((Q/'fit_inputs').glob('*.npz')):
  pool=file.stem;fit=np.load(file);cal=np.load(Q/'calibration_inputs'/file.name);new=np.load(N/'models'/file.name);old=np.load(B/'models'/file.name);pub=np.load(PUB/file.name);nscore=np.load(N/'outer_predictions'/file.name);oscore=np.load(B/'outer_predictions'/file.name);assert np.array_equal(pub['ids'],nscore['ids']) and np.array_equal(pub['ids'],oscore['ids']);modeldata[pool]=(fit,cal,new,old,pub,nscore,oscore);keep=bool(fit['fallback_KEEP'][0]);X=fit['X'];aug_norm=np.sqrt((X[:,:38]**2).sum(1)+1);theta=new['theta'];alpha=float(new['alpha'][0]);offset=fit['b'];endpoint=offset+np.sum(X*theta[:-1],axis=1)+theta[-1];fscore=offset+alpha*(endpoint-offset);nnz=len(fit['ids']);domains=[]
  for domain,mask in [('STOP',fit['domain_STOP']),('CONT',~fit['domain_STOP'])]:
   positive=fit['y']>0;w=fit['weight'][mask];risk_share=float(w[positive[mask]].sum()/w.sum()) if w.sum()>0 else None;group_weights={}
   for i,ww in zip(fit['ids'][mask],w):group_weights[meta[int(i)]['physical_state_group']]=group_weights.get(meta[int(i)]['physical_state_group'],0.)+float(ww)
   gw=np.array(list(group_weights.values()));total=float(gw.sum());domains.append(dict(domain=domain,rows=int(mask.sum()),positive=int((mask&positive).sum()),positive_risk_fraction=risk_share,positive_fit_stops=int((mask&positive&(fscore<0)).sum()),negative_fit_cancellations=int((mask&~positive&(fscore<0)).sum()),largest_physical_call_weight_share=float(gw.max()/total) if total else None,physical_call_Kish=(float(total**2/(gw@gw)) if total else None),physical_calls=len(group_weights)))
  own.append(dict(pool=pool,KEEP=keep,rows=nnz,domains=domains,alpha_new=alpha,alpha_old=float(old['alpha'][0]),theta_L2=float(np.linalg.norm(theta)),finite_current_correction_L2_bound_on_observed_X=float(np.linalg.norm(theta)*np.sqrt((X**2).sum(1)+1).max()),public38_max_abs=float(abs(X[:,:38]).max()),public38_augmented_norm_max=float(aug_norm.max()),public38_augmented_norm_median=float(np.median(aug_norm)),positive_STOP_calls=len({meta[int(i)]['physical_state_group'] for i in fit['ids'][(fit['y']>0)&fit['domain_STOP']]})))
  for k in range(38):geometries.append(dict(pool=pool,column=k,min=float(X[:,k].min()),max=float(X[:,k].max()),max_abs=float(abs(X[:,k]).max()),mean_abs=float(abs(X[:,k]).mean()),coordinate_variance=float(X[:,k].var())))
 save(O/'own_fit_risk_support.json',own);save(O/'own_fit_public_geometry.json',geometries)
 focus={}
 for r in changes:focus[(r['fold'],r['seed'],r['sample_id'],r['position'])]=dict(r,group='changed')
 for r in misses:
  key=(r['fold'],r['seed'],r['sample_id'],r['position']);focus.setdefault(key,dict(r,group='residual_miss'))
 algebra=[];supports=[];featureterms=[]
 for key,row in focus.items():
  f,s,sample,pos=key;pool=f'f{f}_s{s}';fit,cal,n,o,pub,ns,os=modeldata[pool];i=lookup[sample,pos];k=int(np.flatnonzero(pub['ids']==i)[0]);x=pub['X'][k];b=float(pub['b'][k]);an=float(n['alpha'][0]);ao=float(o['alpha'][0]);en=b+np.sum(x*n['theta'][:-1])+n['theta'][-1];eo=b+np.sum(x*o['theta'][:-1])+o['theta'][-1];direction=ao*(en-eo);strength=(an-ao)*(en-b);actual=float(ns['logit'][k]-os['logit'][k]);assert abs(direction+strength-actual)<1e-12
  identity=dict(fold=f,seed=s,sample_id=sample,position=pos,call=row['call'],date=row['date'],metadata_id=i,pool=pool,group=row['group']);algebra.append(dict(identity,old_score=float(os['logit'][k]),new_score=float(ns['logit'][k]),shared_b=b,old_alpha=ao,new_alpha=an,endpoint_difference=float(en-eo),direction_term=float(direction),step_term=float(strength),published_score_difference=actual,larger_absolute_term='direction' if abs(direction)>=abs(strength) else 'step',actual_next_NET=row['actual_next_NET'],new_continue=bool(ns['continue_'][k])))
  # All40coordinates fixed,nearest1 only; no tuned distance/threshold or policy output.
  eligible=np.array([meta[int(j)]['physical_state_group']!=row['call'] for j in fit['ids']])&fit['domain_STOP'];dist=np.sqrt(((fit['X']-x)**2).sum(1));selected=[]
  for label,mask in [('nearest_STOP',eligible),('nearest_positive_STOP',eligible&(fit['y']>0))]:
   ii=np.flatnonzero(mask)
   if len(ii):j=int(ii[np.argmin(dist[ii])]);mid=int(fit['ids'][j]);selected.append(dict(kind=label,metadata_id=mid,distance=float(dist[j]),label_positive=bool(fit['y'][j]),training_NET=float(fit['actual_NET'][j]),training_weight=float(fit['weight'][j]),date=meta[mid]['date'],call=meta[mid]['physical_state_group']))
   else:selected.append(dict(kind=label,no_support=True))
  lo=fit['X'].min(0);hi=fit['X'].max(0);outside=(x<lo)|(x>hi);stpos=fit['domain_STOP']&(fit['y']>0);bp=fit['b'][stpos];supports.append(dict(identity,neighbors=selected,outside_any_fit_coordinate=int(outside.sum()),outside_columns=np.flatnonzero(outside).tolist(),query_public38_max_abs=float(abs(x[:38]).max()),fit_public38_max_abs=float(abs(fit['X'][:,:38]).max()),fit_positive_b_min=float(bp.min()) if len(bp) else None,fit_positive_b_max=float(bp.max()) if len(bp) else None,query_b=b))
  parts=np.r_[x*n['theta'][:-1],n['theta'][-1]]
  for j,v in enumerate(parts):featureterms.append(dict(metadata_id=i,pool=pool,column=j,linear_contribution=float(v)))
 assert len(algebra)==216;save(O/'focus_score_algebra.json',algebra);save(O/'focus_own_training_support.json',supports);save(O/'focus_all41_linear_terms.json',featureterms);verify()
 result=dict(passed=True,new_fits=0,new_scalar_fits=0,new_NN_or_forest_forward=0,new_policy_predictions=0,changes=211,residual_misses=7,unique_focused_rows=216,all4category_macro_and_rawquery_conserved=True,all216algebra_identities=True,hybrid_policy_evaluation=False,own_fit_pool_summaries=12,nearest_rule='Fixed40Euclidean nearest1STOP andnearest1positiveSTOP,samephysicalcallexcluded,no fittedmetric',outer_label_tuning=False)
 save(O/'review.json',result);archive('analyzed');print('GRID259_ANALYZED',json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  if O.exists():save(O/(sys.argv[1]+'_error.json'),dict(error=repr(e),traceback=traceback.format_exc(),new_fits=0))
  raise
