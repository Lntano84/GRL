"""GRID279 qualification only: exact public-E row subset; preserve exclusions."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,ast,collections,importlib.util
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid279_conditional_fit_plan';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID279_conditional_fit_plan';Q=R/'outputs/grid278_invocation_support';F=R/'outputs/grid276_selector_input_plan';N=R/'outputs/grid277_selector_free';META=R/'outputs/grid113_relational_learning/train_metadata.json';ROLES=R/'outputs/grid241_parent_prior_plan/qualified_pools.json';RUNTIME=F/'runtime276.py'
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
 if phase=='qualified':
  for folder in ['qualified_inputs','excluded_inputs']:shutil.copytree(O/folder,d/folder)
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')

def prepare():
 assert not O.exists() and rd(Q/'NEXT_PROTOCOL_PLAN.json')['id']=='GRID279';h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 arQ=path_constant(R/'work/grid278_invocation_support/run.py','A');arF=path_constant(R/'work/grid276_selector_input_plan/run.py','A');arN=path_constant(R/'work/grid277_selector_free/run.py','A')
 for n in ['NEXT_PROTOCOL_PLAN.json','complete.json','decision.json','risk_scale_addendum.json']:assert git('show',h+':'+(arQ/'audited'/n).relative_to(REPO).as_posix())==(Q/n).read_bytes()
 for rel,s in rd(Q/'public_freeze.json').items():assert sha(Q/rel)==s and hashlib.sha256(git('show',h+':'+(arQ/'public'/rel).relative_to(REPO).as_posix())).hexdigest()==s
 for rel,s in rd(F/'qualified_freeze.json').items():assert sha(F/rel)==s and hashlib.sha256(git('show',h+':'+(arF/'qualified'/rel).relative_to(REPO).as_posix())).hexdigest()==s
 for phase,freeze in [('trained','trained_freeze.json'),('scored','scored_freeze.json')]:
  for rel,s in rd(N/freeze).items():assert sha(N/rel)==s and hashlib.sha256(git('show',h+':'+(arN/phase/rel).relative_to(REPO).as_posix())).hexdigest()==s
 O.mkdir();save('input_publication.json',dict(passed=True,git_sha=h,all278publicE_276inputs_and277control_specificblobs=True,actual277_archive=str(arN)))
 save('design.json',dict(id='GRID279',scope='QUALIFICATION ONLY one exactpublicE conditionalfit subset;0fit/cutoff/realforestNNprobability/outerpolicy.',selection='ALL278public_entry fit eligible E inoriginalroworder;verifyids andsourcehash. No y/sign/date/seed/outerloss selection, no newthreshold orcontext variants. Preserve original276files andall331 excludedrows withindex/reason.',arrays='MaskALLrowaligned276fitfields withE,exceptfallback_KEEP literal1-element value. X39/X32/X40original/NET/y/weight/source_mass/ids/domain/is_augmented literalvalues. No reweight/renormalize/dedup/combine;16relatedpairs retained. base/ALL543cal bytecopy unchanged.',checks='Expected2544candidate/2410active/534positive1876negative/KEEP134,11changedactivepools. Originalfit8datesexclude cal4/outer4. Allpositive retained byobservationnotselector,16copiesretained,finite classes/weights/no opposite exact39D label conflicts. Keepemptyclass/guard predeclaredfallback withrecord,not supplementdates/seeds ormodifyE.',target_change='Conditionalweightedgini risk/exposurechange explicit,notoriginalhalfhalf risk preserved/causalproof.331records12.08%countsbut.1607086%weightrisk,not331onlinequerysavings.',future='Afterqualifiedinputpublication,separate28011newforests+11same543positivecutoffs=22/one600s/1KEEP/704newtrees;runtime276shape39/64depth4leaf10maxfeatures1bootstrapFalsegini njobs1classweightNonewarmstartFalse/originalseeds unchanged,NNtheta0. Allmodelsfirst/allpredictionsbeforeoutcomes.',controls='Mainfrozen27739D,274/271historical/261qualityanchor/GATE,all3seeds16dates/KEEP/adversecases. Future249first38tanh then39D/entry261unchanged,no equalbudgetqualityoractualtimeclaim.',counts='Current146/279zero/future280+22=168,old16copies/543guardexposure and6continuations8285updates/604.703actual/678.156cap retained.',limits='278hypotheticalSTOP1 invocation onteacherprefix notphysicalreplay. Parentown12/internalcal4/exposedrelateddata,notOOFblind/fullweek/time. No final8/physics/grid/paper/FRAME/install,feeNETlabels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.'))
 paths=[W/'run.py',O/'design.json',O/'input_publication.json',Q/'NEXT_PROTOCOL_PLAN.json',Q/'public_freeze.json',Q/'public_entry_receipts.json',Q/'risk_scale_addendum.json',*list((Q/'public_entry').glob('*.npz')),F/'qualified_freeze.json',RUNTIME,*list((F/'qualified_inputs').glob('*.npz')),META,ROLES,R/'outputs/grid273_stop_context_plan/augmentation_provenance.json',*list((N/'models').glob('*.pkl')),*list((N/'outer_predictions').glob('*.npz'))]
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID279_SOURCE_FROZEN')

def qualify():
 verify();h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h and git('show',h+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes();assert not (O/'qualification_started.json').exists();save('qualification_started.json',dict(source_published=h,newfits0=True));(O/'qualified_inputs').mkdir();(O/'excluded_inputs').mkdir();meta=rd(META);roles={r['pool']:r for r in rd(ROLES)};records=[];provenance=[];selected_total=active_total=excluded_total=guard_total=active_pos=active_neg=0
 receipt={r['pool']:r for r in rd(Q/'public_entry_receipts.json') if r['role']=='fit'}
 for file in sorted((F/'qualified_inputs').glob('*_fit.npz')):
  pool=file.name[:-8];z=np.load(file);pub=np.load(Q/'public_entry'/file.name);assert np.array_equal(z['ids'],pub['ids']) and sha(file)==receipt[pool]['input_sha'];E=pub['eligible'];assert np.array_equal(E,(pub['b']>0)&(pub['parent_score']>=0));keep=bool(z['fallback_KEEP'][0]);out={};exc={};n=len(z['ids'])
  for key in z.files:
   if key=='fallback_KEEP':out[key]=z[key].copy();exc[key]=z[key].copy()
   else:
    assert len(z[key])==n;out[key]=z[key][E].copy();exc[key]=z[key][~E].copy();assert out[key].dtype==z[key].dtype and out[key].tobytes()==z[key][E].tobytes()
  idx=np.flatnonzero(E);bad=np.flatnonzero(~E);out['original_row_index']=idx;exc['original_row_index']=bad;assert len(idx)+len(bad)==n and (z['y'][bad]==0).all() and ((z['y']>0)&~E).sum()==0
  assert out['X'].shape[1]==39 and np.array_equal(out['X32'],out['X'].astype(np.float32)) and np.isfinite(out['X']).all() and np.isfinite(out['weight']).all() and (out['weight']>0).all();assert not keep or E.all()
  dates={meta[int(i)]['date'] for i in out['ids']};assert dates<=set(roles[pool]['fit_dates']) and not dates&set(roles[pool]['cal4_dates']+roles[pool]['outer_dates'])
  groups={};conflicts=0;duplicates=0
  for x,y in zip(out['X32'],out['y']):
   key=x.tobytes()
   if key in groups:duplicates+=1;conflicts+=int(groups[key]!=int(y))
   groups[key]=int(y)
  assert conflicts==0;cal=np.load(F/'qualified_inputs'/f'{pool}_cal.npz');cpublic=np.load(Q/'public_entry'/f'{pool}_cal.npz');assert cpublic['eligible'].all() and np.array_equal(cpublic['ids'],cal['ids']) and np.array_equal(cpublic['parent_score'],cal['parent261_score']);guard_total+=len(cal['ids'])
  empty=not np.array_equal(np.unique(out['y']),[0,1]) or len(cal['ids'])==0;fallback=keep or empty;assert not empty,(pool,'unexpectedemptyclass_preserve_andstop_norescue')
  np.savez_compressed(O/'qualified_inputs'/file.name,**out);np.savez_compressed(O/'excluded_inputs'/file.name,**exc)
  for role in ['cal','base']:shutil.copy2(F/'qualified_inputs'/f'{pool}_{role}.npz',O/'qualified_inputs'/f'{pool}_{role}.npz')
  for j in bad:provenance.append(dict(pool=pool,original_row_index=int(j),id=int(z['ids'][j]),reason='Outside_FIXED_published_parent261_invocation_E',parent_score=float(pub['parent_score'][j]),date=meta[int(z['ids'][j])]['date'],call=meta[int(z['ids'][j])]['physical_state_group']))
  records.append(dict(pool=pool,KEEP=fallback,changed=bool(len(bad)),old_rows=n,selected_rows=len(idx),excluded_rows=len(bad),positive=int(out['y'].sum()),negative=int((out['y']==0).sum()),retained_augmented=int(out['is_augmented'].sum()),old_risk=float(z['weight'].sum()),selected_risk=float(out['weight'].sum()),excluded_risk=float(exc['weight'].sum()),duplicate_extra_rows=duplicates,opposite_label_conflicts=conflicts));selected_total+=len(idx);excluded_total+=len(bad)
  if not keep:active_total+=len(idx);active_pos+=int(out['y'].sum());active_neg+=int((out['y']==0).sum())
 assert selected_total==2544 and active_total==2410 and excluded_total==331 and guard_total==543 and active_pos==534 and active_neg==1876 and sum(r['retained_augmented'] for r in records)==16 and sum(r['changed'] and not r['KEEP'] for r in records)==11
 for a in rd(R/'outputs/grid273_stop_context_plan/augmentation_provenance.json'):
  pub=np.load(Q/'public_entry'/f"{a['pool']}_fit.npz");assert pub['eligible'][a['original_index']] and pub['eligible'][a['new_index']]
 shutil.copy2(RUNTIME,O/'runtime276.py');save('runtime_freeze.json',dict(sha256=sha(RUNTIME),unchanged_shape39_recipe_and_math=True));save('qualified_pools.json',records);save('exclusion_provenance.json',provenance);arN=Path(rd(O/'input_publication.json')['actual277_archive']);refs=[]
 for r in records:
  p=r['pool'];refs.append(dict(pool=p,KEEP=r['KEEP'],future_newfit=not r['KEEP'],old39_model_relative_to_R=f'outputs/grid277_selector_free/models/{p}.pkl',old39_model_sha256=sha(N/'models'/f'{p}.pkl'),old39_model_git_path=(arN/'trained/models'/f'{p}.pkl').relative_to(REPO).as_posix(),old39_prediction_relative_to_R=f'outputs/grid277_selector_free/outer_predictions/{p}.npz',old39_prediction_sha256=sha(N/'outer_predictions'/f'{p}.npz'),old39_prediction_git_path=(arN/'scored/outer_predictions'/f'{p}.npz').relative_to(REPO).as_posix()))
 save('model_references.json',refs);save('qualification.json',dict(passed=True,newfits=0,newprobabilities=0,newouterstrategy=0,selected2544_active2410=True,activepositive534_negative1876=True,excluded331_preserved=True,retained16copies=True,all543cal_base_byteunchanged=True,no_reweight_renormalize_dedup=True,allfitdates_exclude_cal4_outer4=True,all11active_changed_classes_nonempty=True,KEEP1=True,future_forests11_scalars11=True,cap_s600=True))
 save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'qualified_inputs').glob('*.npz')),*list((O/'excluded_inputs').glob('*.npz')),O/'runtime276.py',O/'runtime_freeze.json',O/'qualification.json',O/'model_references.json',O/'qualified_pools.json',O/'exclusion_provenance.json']});archive('qualified');print('GRID279_QUALIFIED',json.dumps(rd(O/'qualification.json')))
if __name__=='__main__':globals()[sys.argv[1]]()
