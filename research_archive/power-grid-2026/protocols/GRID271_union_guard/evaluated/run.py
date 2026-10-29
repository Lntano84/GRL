"""GRID271: one fixed expanded empirical guard on frozen GRID268 forests."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid271_union_guard';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID271_union_guard';Q=R/'outputs/grid270_union_guard_plan';N=R/'outputs/grid268_conditional_forest';BASE=R/'outputs/grid208_public_amount';META=R/'outputs/grid113_relational_learning/train_metadata.json';RUNTIME=Q/'runtime267.py'
def module(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
RT=module('grid271_frozen_runtime267',RUNTIME)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
 for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
 for p,h in read(Q/'qualified_freeze.json').items():assert sha(Q/p)==h,p
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 for folder in {'calibrated':['cutoffs','calibration_predictions'],'scored':['outer_predictions']}.get(phase,[]):shutil.copytree(O/folder,d/folder)
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def published(phase,freeze):
 verify();h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 assert git('show',h+':'+(A/phase/freeze).relative_to(REPO).as_posix())==(O/freeze).read_bytes()
 for rel,s in read(O/freeze).items():assert sha(O/rel)==s and hashlib.sha256(git('show',h+':'+(A/phase/rel).relative_to(REPO).as_posix())).hexdigest()==s
 return h

def prepare():
 assert not O.exists() and read(Q/'decision.json')['ready_for_one_fixed271'];h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 ar=REPO/'research_archive/power-grid-2026/protocols/GRID270_union_guard_plan'
 for rel,s in read(Q/'qualified_freeze.json').items():assert sha(Q/rel)==s and hashlib.sha256(git('show',h+':'+(ar/'qualified'/rel).relative_to(REPO).as_posix())).hexdigest()==s
 for n in ['EXECUTION_PLAN.json','decision.json']:assert git('show',h+':'+(ar/'audited'/n).relative_to(REPO).as_posix())==(Q/n).read_bytes()
 assert sha(RUNTIME)==read(Q/'runtime_freeze.json')['sha256'] and read(Q/'EXECUTION_PLAN.json')['id']=='GRID271'
 refs=read(Q/'model_references.json');assert len(refs)==12
 for ref in refs:
  for rp,hp,gp in [('model_relative_to_R','model_sha256','model_git_path'),('public_prediction_relative_to_R','public_prediction_sha256','public_prediction_git_path')]:
   f=R/ref[rp];assert sha(f)==ref[hp] and git('show',h+':'+ref[gp])==f.read_bytes()
 O.mkdir();save('pre_input_publication.json',dict(passed=True,git_sha=h,all270qualified_inputs_runtime_plan_and_oldmodels_predictions_specific_blobs=True))
 save('design.json',dict(id='GRID271',hypothesis='ONE frozen-forest expanded empirical positive guard, not a new forest or threshold grid.',inputs='Exactly270positive-only543 union=current500+legacy43,507active/36KEEP;fixedperpoolown12,43 selected beforeoutcomeanalysis,no outer failureIDs.',models='ALL12immutable268bundles/704trees;0forest/theta/NNrefit. runtime267unchanged,sklearn1.7.2. actualNET12STOPbit1 preserved while43referenceNN8bit0,other39fields equal.',calibration='All11activepools once. tau=minpositiveforestscore onfixed543guard, tiescontinue, f0_s1KEEP0. Unique max feasible cutoff, newtau<=old. One600s global window including workerstartup/model loading/predict/checks, watchdog;retainpartial/no repeat or extension.',publication='New12cutoff/modelreference bundles first published. Then byte-reuse268publicforest_score/ids/parent261_continue;0outerforestNNforward. old268CONT subsetnewCONT subset261CONT. All12predictions first published then old outcomejoin.',controls='Primary268;261qualityanchor/GATE/265historical reuse;all3seeds16oldexposeddates including f0_s1KEEP4dates. No seed/subsetdeploymentchoice orparam sweep.',count='11scalarcalibrations+1KEEP/0forestNNtheta,branch95->106;prior6continuations8285updates/604.703sactual/678.156cap separate. Expanded500->543 guard exposure not samecalibrationsample.',limits='Parentown12/internalcal4/exposedcorrelateddata;empiricalguard notouterquality orNET guarantee/fullchainOOF/blindtest. No final8/physics/grid/paper/FRAME/install. Queriesnotactualcontroltime;fees/labels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.'))
 paths=[*W.glob('*.py'),O/'design.json',O/'pre_input_publication.json',Q/'qualified_freeze.json',Q/'EXECUTION_PLAN.json',Q/'decision.json',Q/'runtime_freeze.json',RUNTIME,Q/'model_references.json',*list((Q/'qualified_inputs').glob('*.npz')),META,R/'work/grid227_context_newton/join.py',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json']
 for ref in refs:paths += [R/ref['model_relative_to_R'],R/ref['public_prediction_relative_to_R']]
 for old in [N,R/'outputs/grid261_bounded_basis',R/'outputs/grid265_guarded_bias']:
  paths += [old/n for n in ['oof_trajectories.json','gate_predictions.json','family_summary.json','search_metrics.json','gate_counts.json'] if (old/n).exists()]
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('source');print('GRID271_SOURCE_FROZEN')

def calibrate():
 verify();h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h and git('show',h+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes()
 assert not (O/'calibration_started.json').exists();start=time.perf_counter_ns();save('calibration_started.json',dict(source_published=h,started_monotonic_ns=start,cap_s=600,scalar_fits=11,forest_fits=0))
 try:
  with (O/'calibration_stdout.log').open('w',encoding='utf8') as out,(O/'calibration_stderr.log').open('w',encoding='utf8') as err:
   result=subprocess.run([sys.executable,'-X','utf8','-u',str(W/'run.py'),'calibrate_work'],stdout=out,stderr=err,timeout=max(.001,600-(time.perf_counter_ns()-start)/1e9),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  wall=(time.perf_counter_ns()-start)/1e9;assert result.returncode==0 and wall<600,(result.returncode,wall)
  rr=read(O/'calibration_receipts.json');assert len(rr)==12 and sum(not r['KEEP'] for r in rr)==11 and sum(r['guard_rows'] for r in rr)==543
  save('fits.json',dict(passed=True,new_scalar_fits=11,new_forest_fits=0,newtheta_NNfits=0,KEEP=1,wall_s=wall,cap_s=600,all543_empirical_protected=True,branch_cumulative=106,worker_exited=True))
  save('calibrated_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'cutoffs').glob('*.json')),*list((O/'calibration_predictions').glob('*.npz')),O/'calibration_receipts.json',O/'fits.json']});archive('calibrated');print('GRID271_CALIBRATED',json.dumps(read(O/'fits.json')))
 except Exception as e:
  save('calibration_error.json',dict(error=repr(e),elapsed_s=(time.perf_counter_ns()-start)/1e9,partial_retained=True,no_automatic_retry=True));raise

def calibrate_work():
 verify();import sklearn;assert sklearn.__version__=='1.7.2';start=read(O/'calibration_started.json')['started_monotonic_ns'];assert not (O/'cutoffs').exists();(O/'cutoffs').mkdir();(O/'calibration_predictions').mkdir();receipts=[]
 for ref in read(Q/'model_references.json'):
  assert (time.perf_counter_ns()-start)/1e9<600;pool=ref['pool'];t=time.perf_counter();u=np.load(Q/'qualified_inputs'/f'{pool}.npz');bundle=pickle.loads((R/ref['model_relative_to_R']).read_bytes());keep=bool(bundle['KEEP']);pos=u['y']>0;assert pos.all() and keep==ref['KEEP'] and bundle['tau']==ref['old_tau'] and (u['parent261_score']>=0).all()
  if keep:assert pool=='f0_s1' and bundle['model'] is None;score=np.ones(len(pos));tau=0.
  else:
   assert len(bundle['model'].estimators_)==64;score=RT.positive_score(bundle['model'],u['X32']);tau,newkeep=RT.protect_cutoff(score,pos);assert not newkeep and tau<=ref['old_tau']
   if tau<1:assert np.any(score[pos]<float(np.nextafter(tau,np.inf)))
  cont=RT.apply_gate(u['parent261_score']>=0,score,tau,KEEP=keep);assert cont.all()
  save('cutoffs/'+pool+'.json',dict(pool=pool,tau=tau,old_tau=ref['old_tau'],KEEP=keep,model_relative_to_R=ref['model_relative_to_R'],model_sha256=ref['model_sha256'],model_git_path=ref['model_git_path'],guard_input_sha256=sha(Q/'qualified_inputs'/f'{pool}.npz'),sklearn_version=sklearn.__version__))
  np.savez_compressed(O/'calibration_predictions'/f'{pool}.npz',ids=u['ids'],score=score,positive=pos,parent_continue=u['parent261_score']>=0,continue_=cont)
  receipts.append(dict(pool=pool,KEEP=keep,new_scalar_fit=not keep,guard_rows=len(pos),added_rows=int((~u['origin_current']).sum()),tau=tau,old_tau=ref['old_tau'],all_guarded=True,forest_weight_sha256=sha(R/ref['model_relative_to_R']),wall_s=time.perf_counter()-t));save('calibration_receipts.json',receipts)
 assert (time.perf_counter_ns()-start)/1e9<600;print('GRID271_WORK_DONE_NO_FOREST_REFIT')

def trained():
 verify();assert read(O/'fits.json')['passed']
 for p,h in read(O/'calibrated_freeze.json').items():assert sha(O/p)==h

def publication():
 trained();h=published('calibrated','calibrated_freeze.json');save('pre_score_publication.json',dict(passed=True,git_sha=h,all12cutoff_modelreferences_published=True));print('GRID271_CUTOFFS_PUBLISHED')

def score():
 trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all12cutoffs_published=True,new_outer_forest_forward=0));(O/'outer_predictions').mkdir();receipts=[]
 for ref in read(Q/'model_references.json'):
  pool=ref['pool'];z=np.load(R/ref['public_prediction_relative_to_R']);b=read(O/'cutoffs'/f'{pool}.json');cont=RT.apply_gate(z['parent261_continue'],z['forest_score'],b['tau'],KEEP=b['KEEP']);assert not np.any(z['continue_']&~cont) and not np.any(cont&~z['parent261_continue'])
  np.savez_compressed(O/'outer_predictions'/f'{pool}.npz',ids=z['ids'],forest_score=z['forest_score'],gate_margin=z['forest_score']-b['tau'],continue_=cont,base_CONT=z['base_CONT'],parent261_continue=z['parent261_continue'],parent261_logit=z['parent261_logit'])
  receipts.append(dict(pool=pool,rows=len(cont),restored=int((cont&~z['continue_']).sum()),forest_score_ids_parent_bytes_reused=True,outer_forward=0))
 assert sum(r['rows'] for r in receipts)==4681;save('score_receipts.json',receipts);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'outer_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID271_SCORED_ALL12_NO_FORWARD')

def scored():
 trained()
 for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h

def score_publication():
 scored();h=published('scored','scored_freeze.json');save('pre_join_publication.json',dict(passed=True,git_sha=h,all12predictions_published=True));print('GRID271_SCORES_PUBLISHED')

def join():
    scored();assert read(O/'pre_join_publication.json')['passed'] and not (O/'join_started.json').exists();save('join_started.json',dict(all12predictions_published=True));summarize=module('grid271_saved_summary',R/'work/grid227_context_newton/join.py').summarize;meta=read(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};parents=[r for r in read(BASE/'oof_trajectories.json') if r['method'].startswith('GATE_')];task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];scores={}
    for file in (O/'outer_predictions').glob('*.npz'):
        f,s=[int(v[1:]) for v in file.stem.split('_')];z=np.load(file)
        for i,cont in zip(z['ids'],z['continue_']):scores[f,s,int(i)]=bool(cont)
    tr=list(parents);pred=[]
    for seed in range(3):
        for old in [r for r in parents if r['method']==f'GATE_seed{seed}']:
            c=cases[old['sample_id']];f=old['fold'];j=old['baseline_stop'];eligible=j<len(c['states']);append=scores[f,seed,lookup[c['sample_id'],j]] if eligible else False;assert not append or old['append'];end=j+int(append);new=dict(old);new.update(method=f'UNION_FOREST_seed{seed}',actions=[1]*end+([0] if end<len(c['states']) else []),batches=end,gain=float(c['gains'][:end].sum()),queries=int(c['queries'][:end].sum()),chosen_pool_id=c['chosen'][end],append=append);new['net_utility']=(new['gain']-fee*new['queries'])/scale;new['action_match']=new['chosen_pool_id']==c['chosen'][-1];tr.append(new)
            if eligible:pred.append(dict(fold=f,seed=seed,sample_id=c['sample_id'],position=j,date=c['date'],call=c['physical_state_group'],base_continue=old['append'],continue_=append,actual_next_NET=float((c['gains'][j]-fee*c['queries'][j])/scale),actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j])))
    assert len(tr)==14268 and len(pred)==4681;metrics={};ref=read(BASE/'search_metrics.json')
    for arm in ['GATE','UNION_FOREST']:
        for seed in range(3):
            name=f'{arm}_seed{seed}';rr=[r for r in tr if r['method']==name];assert len(rr)==2378;metrics[name]=summarize(rr)
            if arm=='GATE':
                for k,v in metrics[name]['macro'].items():assert abs(v-ref[name]['macro'][k])<1e-12
    family={arm:{k:float(np.mean([metrics[f'{arm}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{arm}_seed0']['macro']} for arm in ['GATE','UNION_FOREST']};paired=[]
    for seed in range(3):
        old,new=metrics[f'GATE_seed{seed}'],metrics[f'UNION_FOREST_seed{seed}'];paired.append(dict(seed=seed,macro_delta={k:new['macro'][k]-old['macro'][k] for k in old['macro']},per_date=[dict(date=x['date'],NET_delta=y['mean_net_utility']-x['mean_net_utility'],query_delta=y['mean_queries']-x['mean_queries'],gain_delta=y['mean_gain']-x['mean_gain']) for x,y in zip(old['per_date'],new['per_date'])]))
    counts=dict(original208_appends=sum(r['base_continue'] for r in pred),appends=sum(r['continue_'] for r in pred),profitable_retained=sum(r['continue_'] and r['actual_next_NET']>0 for r in pred),profitable_removed=sum(r['base_continue'] and not r['continue_'] and r['actual_next_NET']>0 for r in pred),unprofitable_removed=sum(r['base_continue'] and not r['continue_'] and r['actual_next_NET']<=0 for r in pred))
    for n,v in [('oof_trajectories.json',tr),('gate_predictions.json',pred),('search_metrics.json',metrics),('family_summary.json',family),('paired_summary.json',paired),('gate_counts.json',counts)]:save(n,v)
    scored();save('review.json',dict(passed=True,all14268savedsequences_checked=True,all4681publishedscores_joined=True,commonGATEreproduced=True,all3seeds16oldexposeddates_retained=True,no_new_physics=True,final8_absent=True));archive('evaluated');print('GRID271_EVALUATED',json.dumps(dict(family=family,counts=counts,paired=[dict(seed=r['seed'],NET=r['macro_delta']['mean_net_utility'],queries=r['macro_delta']['mean_queries'],gain=r['macro_delta']['mean_gain']) for r in paired])))
if __name__=='__main__':globals()[sys.argv[1]]()
