"""GRID267 zero-fit qualification of one conditional NET forest filter."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,subprocess,shutil,ast
import numpy as np
sys.dont_write_bytecode=True
import runtime267 as RT
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid267_conditional_forest_plan';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID267_conditional_forest_plan';Q=R/'outputs/grid260_bounded_basis_plan';M=R/'outputs/grid261_bounded_basis/models';PLAN=R/'outputs/grid266_guard_support/NEXT_PROTOCOL_PLAN.json';OLD=R/'work/grid203_tree_event/run.py';ROLE=R/'outputs/grid241_parent_prior_plan/qualified_pools.json';META=R/'outputs/grid113_relational_learning/train_metadata.json'
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
 if phase=='qualified':shutil.copytree(O/'qualified_inputs',d/'qualified_inputs')
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')

def prepare():
 assert not O.exists() and rd(PLAN)['id']=='GRID267';h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 for folder,ar in [(Q/'fit_inputs',REPO/'research_archive/power-grid-2026/protocols/GRID260_bounded_basis_plan/qualified/fit_inputs'),(Q/'calibration_inputs',REPO/'research_archive/power-grid-2026/protocols/GRID260_bounded_basis_plan/qualified/calibration_inputs'),(M,REPO/'research_archive/power-grid-2026/protocols/GRID261_bounded_basis/trained/models')]:
  assert len(list(folder.glob('*.npz')))==12
  for p in folder.glob('*.npz'):assert git('show',h+':'+(ar/p.name).relative_to(REPO).as_posix())==p.read_bytes()
 assert git('show',h+':research_archive/power-grid-2026/protocols/GRID266_guard_support/audited/NEXT_PROTOCOL_PLAN.json')==PLAN.read_bytes()
 O.mkdir();save('design.json',dict(id='GRID267',scope='QualificationONLY0realfit/cutoff/forestprobability/newouterpolicy;onefixedfutureNETforesthypothesisfrom266plan.',params=RT.PARAMS,seed='original0/1/2;f0_s1KEEP',training='Original260fit8 sameX40/ids/yNET>0/sampleweight/source_mass/domain. StoreX32additionalcache,notadditionalfeature;no newscaler/weights/labels.',guard='Same500own12selectedSTOPpositive(active464) and261entry. Futuretau minpositiveforestscore,tiescontinue;KEEP/emptyclass/emptyguardkeep261. No thresholdsweep.',numerics='Explicitfloat32fitandpredict fromexistingstandardizedbounded40coordinates,reportcastingerror/exactlabelcollisions;do notdroporretune collisions.',future='26811forest+11cutoff=22/one600s,1KEEP,0thetaNNretrain. All12bundlespublishedbeforeouter;all12predictionspublishedbeforeoldresults. Parent261onlyfurthercancellation,265deltaunused.',change='Differentcapacity/functionclass/giniobjective comparedto261NLL+unitL2. Weightedleafclassfrequency notcalibratedprobability;onlinecostunmeasured. Architecturefrom203only,no oldweightreuseorrestart.',count='Currentbranch73;2670,268ifexecuted separate+22=>95. Historical6continuations/8285updates/604.703sactual/678.156cap separatelyretained.',bounds='No final8/physics/grid/paper/FRAME/install;own12/internalcal4/exposedcorrelationlimitations preserved,alladversecases,no outerchosenparameters.'))
 paths=[*W.glob('*.py'),O/'design.json',PLAN,OLD,ROLE,META,*list((Q/'fit_inputs').glob('*.npz')),*list((Q/'calibration_inputs').glob('*.npz')),*list(M.glob('*.npz'))];save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('input_publication.json',dict(passed=True,git_sha=h,all36originalarrays_models_specificblobs_verified=True));archive('source');print('GRID267_SOURCE_FROZEN')

def qualify():
 verify();h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h and git('show',h+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes()
 assert not (O/'qualification_started.json').exists();save('qualification_started.json',dict(source_published=h,actualfits0=True));import sklearn
 tree=ast.parse(OLD.read_text(encoding='utf-8-sig'));node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARAMS' for t in n.targets));assert isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict';oldparams={k.arg:ast.literal_eval(k.value) for k in node.keywords};assert oldparams==RT.PARAMS==rd(PLAN)['params']
 roles={r['pool']:r for r in rd(ROLE)};meta=rd(META);records=[];(O/'qualified_inputs').mkdir();positive=activepositive=active=0
 for file in sorted((Q/'fit_inputs').glob('*.npz')):
  pool=file.stem;z=np.load(file);c=np.load(Q/'calibration_inputs'/file.name);m=np.load(M/file.name);fitdates=set(roles[pool]['fit_dates']);caldates=set(roles[pool]['cal4_dates']);outer=set(roles[pool]['outer_dates']);assert not fitdates&caldates and not (fitdates|caldates)&outer
  assert z['X'].shape[1]==c['X'].shape[1]==40 and len(z['ids'])==len(z['y']) and np.array_equal(z['y'],(z['actual_NET']>0).astype(z['y'].dtype)) and np.array_equal(c['y'],(c['actual_NET']>0).astype(c['y'].dtype))
  assert np.isfinite(z['X']).all() and np.isfinite(c['X']).all() and np.isfinite(z['weight']).all() and (z['weight']>=0).all() and z['weight'].sum()>0 and np.isfinite(z['actual_NET']).all() and np.isfinite(c['actual_NET']).all()
  assert np.array_equal(z['X'][:,38],z['b']) and np.array_equal(c['X'][:,38],c['b']) and np.array_equal(z['X'][:,39],z['domain_STOP'].astype(float)) and (c['X'][:,39]==1).all()
  for mid in z['ids']:assert meta[int(mid)]['date'] in fitdates
  for mid,isfit in zip(c['ids'],c['fit_role']):assert meta[int(mid)]['date'] in (fitdates if isfit else caldates)
  classes=np.unique(z['y'][z['weight']>0]);pos=c['y']>0;keep=bool(m['fallback_KEEP'][0]) or not np.array_equal(classes,[0,1]) or not np.any(pos);assert keep==(pool=='f0_s1')
  X32=RT.as_input(z['X']);C32=RT.as_input(c['X']);oldscore=c['b']+float(m['alpha'][0])*((c['b']+np.sum(c['X']*m['theta'][:-1],axis=1)+m['theta'][-1])-c['b']);assert np.all(oldscore[pos]>=0) and (c['b'][pos]>0).all()
  f,s=[int(v[1:]) for v in pool.split('_')];clf=RT.make_classifier(s);assert not hasattr(clf,'estimators_') and callable(clf.fit) and callable(clf.predict_proba);cp=clf.get_params();assert all(cp[k]==v for k,v in RT.PARAMS.items()) and cp['random_state']==s
  collisions=[]
  for name,xx,yy in [('fit',X32,z['y']),('cal',C32,c['y'])]:
   groups={}
   for i,row in enumerate(xx):groups.setdefault(row.tobytes(),[]).append(i)
   du=[v for v in groups.values() if len(v)>1];conf=[v for v in du if len({int(yy[i]) for i in v})>1];collisions.append(dict(role=name,duplicate_groups=len(du),opposite_label_groups=len(conf),all_records_kept=True))
  fitdata={k:z[k] for k in z.files};fitdata['X32']=X32;caldata={k:c[k] for k in c.files};caldata['X32']=C32;np.savez_compressed(O/'qualified_inputs'/f'{pool}_fit.npz',**fitdata);np.savez_compressed(O/'qualified_inputs'/f'{pool}_cal.npz',**caldata);shutil.copy2(M/file.name,O/'qualified_inputs'/f'{pool}_base.npz')
  records.append(dict(pool=pool,KEEP=keep,fit_rows=len(z['y']),fit_positive=int(z['y'].sum()),STOP_positive=int(z['y'][z['domain_STOP']].sum()),CONT_positive=int(z['y'][~z['domain_STOP']].sum()),guard_positive=int(pos.sum()),guard_fit8_positive=int((pos&c['fit_role']).sum()),guard_cal4_positive=int((pos&~c['fit_role']).sum()),weighted_classes=classes.tolist(),fit_cast_maxabs=float(abs(X32.astype(float)-z['X']).max()),cal_cast_maxabs=float(abs(C32.astype(float)-c['X']).max()),old261_all_guarded_continue=True,roles_outerexcluded=True,collisions=collisions,constructor_only_no_estimators=True));positive+=int(pos.sum());activepositive+=int(pos.sum())*int(not keep);active+=int(not keep)
 assert positive==500 and activepositive==464 and active==11
 tests=[]
 for name,scores,pos,parent in [('ties',[.3,.3,.8,.1],[1,1,1,0],[1,1,1,1]),('zero',[0.,.5,1.],[1,1,0],[1,1,0]),('one',[1.,1.,.5],[1,1,0],[1,0,1])]:
  ss=np.array(scores);pp=np.array(pos,dtype=bool);par=np.array(parent,dtype=bool);tau,keep=RT.protect_cutoff(ss,pp);out=RT.apply_gate(par,ss,tau);assert not keep and np.all(out[pp&par]) and not np.any(out&~par);tests.append(dict(name=name,passed=True,synthetic_tau=tau))
 ss=np.array([.1,.9]);assert RT.protect_cutoff(ss,np.array([False,False]))==(0.,True);assert RT.protect_cutoff(ss,np.array([True,True]),KEEP=True)==(0.,True)
 save('synthetic_checks.json',dict(passed=True,tests=tests,KEEP_andemptyguard_identity=True,actual_fit_or_real_cutoff0=True));save('qualification.json',dict(passed=True,pools=records,active11=active,KEEP1=True,guard_positive500=positive,active_guard464=activepositive,sklearn_version=sklearn.__version__,numpy_version=np.__version__,python_version=sys.version,old203_recipe_exact=True,planned_forests11_trees704_cutoffs11=True,actualfits0=True,realforestpredictions0=True,newouterinputs_or_results0=True));save('runtime_freeze.json',{'work/grid267_conditional_forest_plan/runtime267.py':sha(W/'runtime267.py')});save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'qualified_inputs').glob('*.npz')),O/'qualification.json',O/'synthetic_checks.json',O/'runtime_freeze.json']});save('complete.json',dict(passed=True,qualificationonly=True,actualfits0=True,realforestpredictions0=True,actualcutofffits0=True,qualified36files=True,branch73=True));archive('qualified');print('GRID267_QUALIFIED_NO_FITS',json.dumps(dict(active11=active,positive500=positive,active464=activepositive,sklearn=sklearn.__version__,actualfits0=True)))

if __name__=='__main__':globals()[sys.argv[1]]()
