"""GRID283 one frozen-partition conditional leaf-frequency candidate."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,pickle,shutil,subprocess,time,importlib.util
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid283_fixed_partition_leaf';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID283_fixed_partition_leaf';Q=R/'outputs/grid282_fixed_partition_plan';BASE=R/'outputs/grid208_public_amount';PUBLIC=R/'outputs/grid261_bounded_basis/outer_predictions';META=R/'outputs/grid113_relational_learning/train_metadata.json';RUNTIME=Q/'runtime282.py'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args])
def module(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
RT=module('runtime282_frozen',RUNTIME)
def verify():
 for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 for folder in {'trained':['models','leaf_statistics','calibration_predictions'],'scored':['outer_predictions']}.get(phase,[]):shutil.copytree(O/folder,d/folder)
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def publication(phase,freeze):
 verify();h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 assert git('show',h+':'+(A/phase/freeze).relative_to(REPO).as_posix())==(O/freeze).read_bytes()
 for rel,s in read(O/freeze).items():
  p=R/rel if phase=='source' else O/rel
  target=(A/phase/p.name) if phase=='source' else A/phase/rel
  if phase!='source':assert hashlib.sha256(git('show',h+':'+target.relative_to(REPO).as_posix())).hexdigest()==s
 if phase=='source':
  for rel,s in read(A/'source/archive_manifest.json').items():assert hashlib.sha256(git('show',h+':'+(A/'source'/rel).relative_to(REPO).as_posix())).hexdigest()==s
 return h

def prepare():
 assert not O.exists();h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 aq=REPO/'research_archive/power-grid-2026/protocols/GRID282_fixed_partition_plan';assert read(Q/'decision.json')['ready_for_one_fixed283'] and read(Q/'complete.json')['passed'];locations=read(Q/'qualified_artifact_locations.json');freeze=read(Q/'qualified_freeze.json');assert set(locations)==set(freeze) and len(freeze)==44
 for rel,v in locations.items():assert sha(Q/rel)==freeze[rel]==v['sha256'] and hashlib.sha256(git('show',h+':'+v['git_path'])).hexdigest()==v['sha256']
 for n in ['qualified_freeze.json','qualified_artifact_locations.json','EXECUTION_PLAN.json','decision.json']:
  phase='audited';assert git('show',h+':'+(aq/phase/n).relative_to(REPO).as_posix())==(Q/n).read_bytes()
 refs=read(Q/'partition_references.json');assert len(refs)==12 and sum(x['KEEP'] for x in refs)==1
 for ref in refs:
  assert sha(R/ref['partition_relative_to_R'])==ref['partition_sha256'] and hashlib.sha256(git('show',h+':'+ref['partition_git_path'])).hexdigest()==ref['partition_sha256']
  for v in ref['inputs'].values():assert sha(R/v['relative_to_R'])==v['sha256'] and hashlib.sha256(git('show',h+':'+v['git_path'])).hexdigest()==v['sha256']
 assert sha(RUNTIME)==read(Q/'runtime_freeze.json')['sha256']
 O.mkdir();save('pre_input_publication.json',dict(passed=True,git_sha=h,qualified_files44=True,all277models_and279inputs_specific_blobs=True,actual_fits0=True));save('partition_references.json',refs)
 save('design.json',dict(id='GRID283',candidate='ONE conditional leaf-frequency model on ALL frozen277partitions,11complete64tree heads+11same543positivecutoffs=22/one600s/1KEEP;0newtree splits/NNtheta.',inputs='Only referenced279qualifiedownfit y/weight and published282fit/cal leafIDs;2410active fit/507activeguard,KEEP134/36;16copies/331exclusions/543guard literal fields unchanged. Do NOT reroute completedfit/cal.',parameters='704oldtrees6715leaves fixed. Up to6715leafprobabilities/13430classweighted sufficientstatistics andrawcount receipts,not11numbers. Internalnodeprior entries retainedstatic andnotfit. Headstats separate,never mutateoldsklearntree.values/nodestats/files.',rule='runtime282 exact weightedpositive/total mass;EXACTzero totalmass->original277leafprior;no smoothing/priorstrength/thresholdsweep.64mean score. Unique tau=minown543positive scores,tiescontinue,KEEPidentity.',timing='Single600smanager window includeschildstartup/load/fit/cal/check;noextension/retrycomplete.',outer='All12headtables/oldrefs/tau publishbefore249first38tanh slice39 tree.apply+64tablemean. Frozen261entry/KEEP same. All12predictionspublishbeforeoldoutcomes. No newNN/thetaforward orphysics.',controls='Main280sameconditionaldata butdifferentpartitionlearning;277samepartitionoriginalvalues;261qualityanchor/GATE/all3seeds16olddates/adversecases. Notpurecausal/honestforest:partitionssaw unfilteredownfitlabels andemptyfallbackoldfrequency.',count='Current168+22=190.6oldcontinuations8285updates/NN604.703actual/user678.156cap plus16copies/543guardexposure retained.',limits='Own12parent/internalcal4/exposedrelatedteacherprefixes,notfullchainOOFblind/fullweek/time. Queryproxy notactualspeed. Final8/physics/grid/paper/FRAME/install untouched;feesNETlabels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.'))
 paths=[*W.glob('*.py'),O/'design.json',O/'partition_references.json',O/'pre_input_publication.json',RUNTIME,Q/'qualified_freeze.json',Q/'qualified_artifact_locations.json',Q/'runtime_freeze.json',Q/'EXECUTION_PLAN.json',Q/'decision.json',*[(Q/rel) for rel in freeze],META,R/'work/grid227_context_newton/join.py',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json',*PUBLIC.glob('*.npz'),*(R/'outputs/grid249_context_bit_plan/outer_parent_public').glob('*.npz')]
 for ref in refs:
  paths.append(R/ref['partition_relative_to_R']);paths.extend(R/v['relative_to_R'] for v in ref['inputs'].values())
 for folder in [R/'outputs/grid280_invocation_conditioned',R/'outputs/grid277_selector_free',R/'outputs/grid261_bounded_basis']:
  paths.extend(folder/n for n in ['gate_predictions.json','oof_trajectories.json','search_metrics.json','family_summary.json','gate_counts.json'])
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('source');print('GRID283_SOURCE_FROZEN')

def train():
 h=publication('source','source_freeze.json');assert not (O/'training_started.json').exists();start=time.perf_counter_ns();save('training_started.json',dict(source_published=h,started_monotonic_ns=start,cap_s600=True,newheads11_cutoffs11=True))
 try:
  with (O/'training_stdout.log').open('w',encoding='utf8') as out,(O/'training_stderr.log').open('w',encoding='utf8') as err:
   r=subprocess.run([sys.executable,'-X','utf8','-u',str(W/'run.py'),'train_work'],stdout=out,stderr=err,timeout=max(.001,600-(time.perf_counter_ns()-start)/1e9),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  elapsed=(time.perf_counter_ns()-start)/1e9;assert r.returncode==0 and elapsed<600,(r.returncode,elapsed);rr=read(O/'fit_receipts.json');assert len(rr)==12 and sum(x['new_head_fit'] for x in rr)==sum(x['new_cutoff_fit'] for x in rr)==11 and sum(x['guard_positive'] for x in rr)==543 and sum(x['leaves'] for x in rr)==6715 and sum(x['empty_fallback_leaves'] for x in rr)==1
  save('fits.json',dict(passed=True,new_complete_forest_heads=11,new_cutoffs=11,KEEP=1,new_tree_splits_NNtheta0=True,fixedtrees704=True,fixedleaves6715=True,leaf_probability_parameters6715=True,class_sufficient_stats13430=True,empty_fallback1=True,all543guarded=True,no_fit_cal_rerouting=True,wall_s=elapsed,cap_s=600,branch190=True,worker_exited=True))
  save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*(O/'models').iterdir(),*(O/'leaf_statistics').iterdir(),*(O/'calibration_predictions').iterdir(),O/'fit_receipts.json',O/'fits.json']});archive('trained');print('GRID283_TRAINED',json.dumps(read(O/'fits.json')))
 except Exception as e:save('training_error.json',dict(error=repr(e),partial_retained=True,no_retry=True,elapsed=(time.perf_counter_ns()-start)/1e9));raise

def train_work():
 verify();start=read(O/'training_started.json')['started_monotonic_ns'];[(O/n).mkdir() for n in ['models','leaf_statistics','calibration_predictions']];receipts=[]
 for ref in read(O/'partition_references.json'):
  assert (time.perf_counter_ns()-start)/1e9<600;pool=ref['pool'];keep=ref['KEEP'];z=np.load(R/ref['inputs']['fit']['relative_to_R']);cal=np.load(R/ref['inputs']['cal']['relative_to_R']);base=np.load(R/ref['inputs']['base']['relative_to_R']);fitroute=np.load(Q/'routes'/f'{pool}_fit.npz');calroute=np.load(Q/'routes'/f'{pool}_cal.npz');pr=np.load(Q/'original_leaf_parameters'/f'{pool}.npz');assert np.array_equal(z['ids'],fitroute['ids']) and np.array_equal(cal['ids'],calroute['ids']);t=time.perf_counter();tables=[];masks=[];stats=[];leaves=empty=0
  if not keep:
   masks=[pr[f'leaf_mask_{j}'] for j in range(64)];priors=[pr[f'prior_{j}'] for j in range(64)];assert np.array_equal(np.unique(z['y']),[0,1]) and (z['weight']>0).all();tables,stats=RT.fit_leaf_tables(fitroute['leaf_ids'],z['y'],z['weight'],masks,priors)
   for j,(mask,s) in enumerate(zip(masks,stats)):
    leaves+=int(mask.sum());empty+=int(s['empty_mask'].sum());assert int(s['count'].sum())==len(z['ids']) and abs(float(s['mass'].sum())-float(z['weight'].sum()))<1e-10 and (s['positive_mass']<=s['mass']+1e-12).all()
   score=RT.score_routes(calroute['leaf_ids'],tables,masks);tau,newkeep=RT.protect_cutoff(score,cal['y']>0);assert not newkeep and (cal['y']>0).all() and (score>=tau).all()
   if tau<1:assert np.any(score<np.nextafter(tau,np.inf))
  else:score=np.zeros(len(cal['ids']),np.float64);tau=0.;assert pool=='f0_s1' and fitroute['leaf_ids'].shape[1]==0
  parent=cal['parent261_score']>=0;cont=RT.apply_gate(parent,score,tau,KEEP=keep);assert cont.all()
  payload={}
  for j,s in enumerate(stats):
   for k,v in s.items():payload[f'{k}_{j}']=v
  np.savez_compressed(O/'leaf_statistics'/f'{pool}.npz',**payload)
  bundle=dict(pool=pool,KEEP=keep,tables=tables,masks=masks,tau=tau,partition_relative_to_R=ref['partition_relative_to_R'],partition_sha256=ref['partition_sha256'],base_model_sha=ref['inputs']['base']['sha256'],base_theta=base['theta'],base_alpha=base['alpha'],fixed_partition_conditional_leaf_model=True,runtime_sha=sha(RUNTIME))
  (O/'models'/f'{pool}.pkl').write_bytes(pickle.dumps(bundle,protocol=4));np.savez_compressed(O/'calibration_predictions'/f'{pool}.npz',ids=cal['ids'],score=score,positive=cal['y']>0,parent_continue=parent,continue_=cont)
  receipts.append(dict(pool=pool,new_head_fit=int(not keep),new_cutoff_fit=int(not keep),fit_rows=len(z['ids']),guard_positive=len(cal['ids']),fit_file_sha=sha(R/ref['inputs']['fit']['relative_to_R']),cal_file_sha=sha(R/ref['inputs']['cal']['relative_to_R']),partition_sha=ref['partition_sha256'],leaves=leaves,empty_fallback_leaves=empty,leaf_parameters=leaves,class_sufficient_stats=2*leaves,no_tree_apply=True,wall_s=time.perf_counter()-t));save('partial_fit_receipts.json',receipts)
 assert sum(r['leaves'] for r in receipts)==6715;save('fit_receipts.json',receipts)

def trained():
 verify()
 for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p

def score():
 trained();h=publication('trained','trained_freeze.json');assert not (O/'score_started.json').exists();save('pre_score_publication.json',dict(passed=True,git_sha=h,all12models_cutoffs_published=True));save('score_started.json',dict(models_published=True));(O/'outer_predictions').mkdir();receipts=[]
 for ref in read(O/'partition_references.json'):
  pool=ref['pool'];b=pickle.loads((O/'models'/f'{pool}.pkl').read_bytes());raw=np.load(R/'outputs/grid249_context_bit_plan/outer_parent_public'/f'{pool}.npz');parent=np.load(PUBLIC/f'{pool}.npz');assert np.array_equal(raw['ids'],parent['ids']);X=raw['X'].copy();X[:,:38]=np.tanh(X[:,:38]);xx=RT.as_input(X[:,:39]);routes=np.empty((len(xx),0),np.int32)
  if b['KEEP']:s=np.zeros(len(xx),np.float64)
  else:
   old=pickle.loads((R/ref['partition_relative_to_R']).read_bytes());routes=RT.route_forest(old['model'],xx);s=RT.score_routes(routes,b['tables'],b['masks'])
  cont=RT.apply_gate(parent['continue_'],s,b['tau'],KEEP=b['KEEP']);assert not np.any(cont&~parent['continue_']);np.savez_compressed(O/'outer_predictions'/f'{pool}.npz',ids=raw['ids'],X32=xx,leaf_ids=routes,forest_score=s,gate_margin=s-b['tau'],base_CONT=parent['base_CONT'],parent261_continue=parent['continue_'],parent261_logit=parent['logit'],continue_=cont)
  receipts.append(dict(pool=pool,rows=len(xx),fixed_tree_apply_calls=0 if b['KEEP'] else 64,point_tree_routes=int(routes.size),NNthetaforward0=True,model_sha=sha(O/'models'/f'{pool}.pkl'),input_source=raw.files))
 assert sum(r['rows'] for r in receipts)==4681 and sum(r['fixed_tree_apply_calls'] for r in receipts)==704;save('public_score_receipts.json',receipts);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*(O/'outer_predictions').iterdir(),O/'public_score_receipts.json']});archive('scored');print('GRID283_PUBLIC_SCORED')

def scored():
 trained()
 for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p

def score_publication():
 scored();h=publication('scored','scored_freeze.json');save('pre_join_publication.json',dict(passed=True,git_sha=h,all12predictions_published=True));print('GRID283_SCORES_PUBLISHED')

exec((W/'evaluation.py').read_text(encoding='utf8'))
if __name__=='__main__':globals()[sys.argv[1]]()
