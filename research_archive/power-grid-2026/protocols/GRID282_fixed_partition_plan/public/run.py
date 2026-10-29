"""GRID282 source/public routes/qualification. Never fit actual leaf tables here."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,pickle,ast,importlib.util,time
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid282_fixed_partition_plan';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID282_fixed_partition_plan';Q=R/'outputs/grid281_partition_support';F=R/'outputs/grid279_conditional_fit_plan';PART=R/'outputs/grid277_selector_free';META=R/'outputs/grid113_relational_learning/train_metadata.json';ROLES=R/'outputs/grid241_parent_prior_plan/qualified_pools.json';RUNTIME=W/'runtime282.py'
def rd(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
 for p,h in rd(O/'source_freeze.json').items():assert sha(R/p)==h,p

def path_constant(p,key):
 n=next(n.value for n in ast.parse(p.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets));assert isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div) and isinstance(n.left,ast.Name) and isinstance(n.right,ast.Constant);return {'R':R,'REPO':REPO}[n.left.id]/n.right.value

def module():
 sp=importlib.util.spec_from_file_location('fixed282_runtime',RUNTIME);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 if phase=='public':
  for folder in ['routes','original_leaf_parameters']:shutil.copytree(O/folder,d/folder)
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')

def published(phase):
 verify();h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 for rel,s in rd(A/phase/'archive_manifest.json').items():assert hashlib.sha256(git('show',h+':'+(A/phase/rel).relative_to(REPO).as_posix())).hexdigest()==s
 return h

def prepare():
 assert not O.exists() and rd(Q/'NEXT_PROTOCOL_PLAN.json')['id']=='GRID282';h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 arQ=path_constant(R/'work/grid281_partition_support/run.py','A');arF=path_constant(R/'work/grid279_conditional_fit_plan/run.py','A');arP=path_constant(R/'work/grid277_selector_free/run.py','A')
 for n in ['NEXT_PROTOCOL_PLAN.json','complete.json','decision.json','diagnosis_summary.json']:assert git('show',h+':'+(arQ/'audited'/n).relative_to(REPO).as_posix())==(Q/n).read_bytes()
 for rel,s in rd(F/'qualified_freeze.json').items():assert sha(F/rel)==s and hashlib.sha256(git('show',h+':'+(arF/'qualified'/rel).relative_to(REPO).as_posix())).hexdigest()==s
 refs=[]
 for file in sorted((PART/'models').glob('*.pkl')):
  b=pickle.loads(file.read_bytes());gp=(arP/'trained/models'/file.name).relative_to(REPO).as_posix();assert git('show',h+':'+gp)==file.read_bytes();pool=file.stem;inputs={}
  for role in ['fit','cal','base']:
   f=F/'qualified_inputs'/f'{pool}_{role}.npz';inputs[role]=dict(relative_to_R=f.relative_to(R).as_posix(),sha256=sha(f),git_path=(arF/'qualified/qualified_inputs'/f.name).relative_to(REPO).as_posix())
  refs.append(dict(pool=pool,KEEP=bool(b['KEEP']),partition_relative_to_R=file.relative_to(R).as_posix(),partition_sha256=sha(file),partition_git_path=gp,inputs=inputs))
 assert len(refs)==12 and sum(not r['KEEP'] for r in refs)==11;O.mkdir();save('input_publication.json',dict(passed=True,git_sha=h,all279input_and277partition_blobs_verified=True));save('partition_references.json',refs)
 save('design.json',dict(id='GRID282',scope='QUALIFICATION ONLY fixed277partitions conditionalleaf model.0actualleaf/model/cutofffit/actualcalprobability/newouterpolicy;publictree.apply allowednotzeroforward.',source='ALLfrozen27711forests704trees6715leaves,notbestsourceorstyleseedselection. Oldtreesvalue/geometry/nodestats immutable;newtable separate. Same279conditionalfit39D/labelsweights and543guard,16copiesretained.',routes='AfterSourcepublication exportonlyX32/ids/roles fixedleafIDs:1408batchapply=704x2,186688pointtree=64*(2410+507),KEEPzero tree. No y/weight/NET read tofit tables,no outerinput. Keepactualrouting onceandpublicfreeze beforefuturefit. Empty supportcounts rawonly;allqualifiedweightspositive.',rule='Future leafpositive=sumpositiveweight/sumallweight whenmass>0,EXACT0mass fallbackoriginal277leafpositivefrequency. No smoothing/priorstrength/grid/handgate. Staticoldleafvalues andsyntheticarraykernel tests onlyin282;do notscore actualcal routes withtablesorcalculateactualtau.',counts='2820fits;future28311wholeforest-headmodels+11cutoffs=22/one600s,0newtreefit/NNtheta,underlyingup-to6715leafprobparams/13430classsufficientstats explicit. Current168future190,old6continuations8285updates/604.703actual/678.156cap/331excluded/16copies/543guardexposure remain.',evaluation='Futureall12headtables/oldrefs/tau firstpublish,then249first38tanhcut39 routes+64means;allpredictionsfirstpublishbeforeoldoutcomes. Main280sameconditionaldata butpartitionchoice differs,277samepartitionoldvalues,261qualityanchor/GATE/allseeds16dates/adversecases. Not purecausal/honestforest:oldpartitions learnedunfilteredownfit andemptyfallbackoldvalues.',limits='Oldown12parent/internalcal4/exposedrelatedteacherpaths,notOOFblind/fullweek/time. Tree-route counts notphysicalcandidatequeries/speed. No final8/physics/grid/paper/FRAME/install;feeNETlabels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.'))
 paths=[*W.glob('*.py'),O/'design.json',O/'input_publication.json',O/'partition_references.json',Q/'NEXT_PROTOCOL_PLAN.json',F/'qualified_freeze.json',F/'runtime276.py',META,ROLES]
 for ref in refs:
  paths.append(R/ref['partition_relative_to_R']);paths.extend(R/v['relative_to_R'] for v in ref['inputs'].values())
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID282_SOURCE_FROZEN')

def route():
 verify();h=published('source');assert not (O/'routing_started.json').exists();save('routing_started.json',dict(source_published=h,real_leaf_fit0=True,tree_route_budget186688=True));(O/'routes').mkdir();(O/'original_leaf_parameters').mkdir();RT=module();meta=rd(META);roles={r['pool']:r for r in rd(ROLES)};support=[];receipts=[];batch=points=leaves=dupchecks=0;start=time.perf_counter()
 for ref in rd(O/'partition_references.json'):
  pool=ref['pool'];model=pickle.loads((R/ref['partition_relative_to_R']).read_bytes())['model'];keep=ref['KEEP'];priors,masks=([],[]) if keep else RT.original_leaf_parameters(model);payload={'KEEP':np.array([keep])}
  for j,(prior,mask) in enumerate(zip(priors,masks)):payload[f'prior_{j}']=prior;payload[f'leaf_mask_{j}']=mask;leaves+=int(mask.sum())
  np.savez_compressed(O/'original_leaf_parameters'/f'{pool}.npz',**payload);routed={}
  for role in ['fit','cal']:
   file=R/ref['inputs'][role]['relative_to_R'];z=np.load(file);X=z['X32'];ids=z['ids'];assert X.shape[1]==39 and X.dtype==np.float32 and np.isfinite(X).all();dates={meta[int(i)]['date'] for i in ids}
   if role=='fit':assert dates<=set(roles[pool]['fit_dates']) and not dates&set(roles[pool]['cal4_dates']+roles[pool]['outer_dates'])
   else:assert not dates&set(roles[pool]['outer_dates'])
   routes=np.empty((len(ids),0),np.int32) if keep else RT.route_forest(model,X)
   if not keep:RT.validate_routes(routes,masks);batch+=64;points+=routes.size
   seen={}
   for k,x in enumerate(X):
    key=x.tobytes()
    if key in seen:assert np.array_equal(routes[k],routes[seen[key]]);dupchecks+=int(role=='fit')
    else:seen[key]=k
   np.savez_compressed(O/'routes'/f'{pool}_{role}.npz',ids=ids,leaf_ids=routes,fit_role=np.ones(len(ids),bool) if role=='fit' else z['fit_role'],original_row_index=z['original_row_index'] if role=='fit' else np.arange(len(ids),dtype=np.int64));routed[role]=routes
   receipts.append(dict(pool=pool,role=role,rows=len(ids),trees=0 if keep else 64,point_tree_routes=int(routes.size),input_sha=sha(file),input_fields_used=['X32','ids','fit_role/original_row_index metadata'],target_fields_used=[],partition_sha=ref['partition_sha256']))
  for j,mask in enumerate(masks):
   raw=np.bincount(routed['fit'][:,j],minlength=len(mask));support.append(dict(pool=pool,tree=j,node_count=len(mask),leaf_ids=np.flatnonzero(mask).tolist(),raw_fit_count_on_leaves=raw[mask].tolist(),empty_leaves=int((mask&(raw==0)).sum()),leaves=int(mask.sum())))
  assert sha(R/ref['partition_relative_to_R'])==ref['partition_sha256'] and time.perf_counter()-start<600
 assert batch==1408 and points==186688 and leaves==6715 and dupchecks==16
 save('public_route_receipts.json',receipts);save('public_leaf_support.json',support);save('routing_summary.json',dict(passed=True,batched_tree_apply_calls=batch,point_tree_assignments=points,fixed_trees704=True,leaves6715=True,empty_support_leaves=sum(v['empty_leaves'] for v in support),same_input_duplicate_fit_routes16=dupchecks,wall_s=time.perf_counter()-start,realfit0=True,realprobability0=True,outer_input_read=False));save('public_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'routes').glob('*.npz')),*list((O/'original_leaf_parameters').glob('*.npz')),O/'public_route_receipts.json',O/'public_leaf_support.json',O/'routing_summary.json']});archive('public');print('GRID282_PUBLIC_ROUTES',json.dumps(rd(O/'routing_summary.json')))

def qualify():
 verify();h=published('public');assert not (O/'qualification_started.json').exists();save('qualification_started.json',dict(public_routes_published=h,real_fit0=True));RT=module();synthetic=[]
 masks=[np.array([False,True,True]),np.array([False,True,True])];priors=[np.array([0.,.25,.75]),np.array([0.,.4,.6])];routes=np.array([[1,1],[1,2],[1,1]],np.int32);y=np.array([1,0,1]);weight=np.array([1.,2.,3.]);tables,stats=RT.fit_leaf_tables(routes,y,weight,masks,priors);assert tables[0][1]==4/6 and tables[0][2]==.75 and tables[1][1]==1 and tables[1][2]==0;synthetic.append('weightedclassfrequency_and_zero_support_fallback')
 score=RT.score_routes(np.array([[1,1],[2,2]],np.int32),tables,masks);assert np.allclose(score,[5/6,3/8],rtol=0,atol=1e-15);tau,keep=RT.protect_cutoff(score,np.array([True,False]));assert not keep and tau==score[0] and np.array_equal(RT.apply_gate(np.ones(2,bool),score,tau),[True,False]);synthetic.append('mean_and_positive_tie_continue')
 unchanged,_=RT.fit_leaf_tables(routes,y,np.zeros(3),masks,priors);assert all(np.array_equal(a,b) for a,b in zip(unchanged,priors));synthetic.append('all_zero_mass_no_prior_tuning')
 try:RT.validate_routes(np.array([[0,1]],np.int32),masks)
 except AssertionError:synthetic.append('internal_node_route_rejected')
 else:raise AssertionError('invalid route not rejected')
 def nodes(file):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef)}
 before,after=nodes(F/'runtime276.py'),nodes(RUNTIME)
 for key in ['as_input','protect_cutoff','apply_gate']:assert before[key]==after[key]
 shutil.copy2(RUNTIME,O/'runtime282.py');save('runtime_freeze.json',dict(sha256=sha(RUNTIME),original_as_input_protect_apply_ASTsame=True,synthetic_examples=len(synthetic),actual_leaf_fits=0,actual_cutoff_fits=0));save('synthetic_checks.json',dict(passed=True,examples=synthetic,no_actual_training_labels_read=True,no_actual_cal_probability_or_tau=True));s=rd(O/'routing_summary.json');save('qualification.json',dict(passed=True,actual_model_or_scalar_fits=0,actual_new_probabilities=0,new_outer_policy=0,public_tree_routing_not_zero_forward=True,batch_apply_calls1408=True,point_tree_routes186688=True,oldtrees704_leaves6715=True,empty_support_leaves=s['empty_support_leaves'],source_partitions_unchanged=True,routes_published=h,future_model_fits11_cutoff11=True,future_leaf_probability_params_max6715=True,future_class_sufficient_stats13430=True,future_cap_s600=True))
 save('qualified_freeze.json',{**rd(O/'public_freeze.json'),**{n:sha(O/n) for n in ['partition_references.json','runtime282.py','runtime_freeze.json','qualification.json','synthetic_checks.json']}});archive('qualified');print('GRID282_QUALIFIED',json.dumps(rd(O/'qualification.json')))
def resume_prepare_01():
 assert O.exists() and not (O/'source_freeze.json').exists() and not (O/'routing_started.json').exists();h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h and rd(O/'input_publication.json')['git_sha']==h
 refs=rd(O/'partition_references.json')
 for ref in refs:
  f=R/ref['partition_relative_to_R'];assert sha(f)==ref['partition_sha256'] and git('show',h+':'+ref['partition_git_path'])==f.read_bytes()
  for inp in ref['inputs'].values():
   f=R/inp['relative_to_R'];assert sha(f)==inp['sha256'] and git('show',h+':'+inp['git_path'])==f.read_bytes()
 paths=[*W.glob('*.py'),O/'design.json',O/'input_publication.json',O/'partition_references.json',O/'prepare_original_01.py',O/'prepare_partial_snapshot_01.json',O/'prepare_error_01.json',Q/'NEXT_PROTOCOL_PLAN.json',F/'qualified_freeze.json',F/'runtime276.py',META,ROLES]
 for ref in refs:
  paths.append(R/ref['partition_relative_to_R']);paths.extend(R/v['relative_to_R'] for v in ref['inputs'].values())
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID282_SOURCE_FROZEN_AFTER_PREFREEZE_PATH_FIX')

if __name__=='__main__':globals()[sys.argv[1]]()
