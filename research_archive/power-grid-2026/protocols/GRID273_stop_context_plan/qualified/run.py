"""Qualification only: fixed ownfit positive actualSTOP context copies."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,collections
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid273_stop_context_plan';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID273_stop_context_plan';Q=R/'outputs/grid272_union_limits';F=R/'outputs/grid267_conditional_forest_plan';U=R/'outputs/grid270_union_guard_plan';N=R/'outputs/grid271_union_guard';META=R/'outputs/grid113_relational_learning/train_metadata.json';ROLES=R/'outputs/grid241_parent_prior_plan/qualified_pools.json';RUNTIME=U/'runtime267.py'
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
 assert not O.exists() and rd(Q/'NEXT_PROTOCOL_PLAN.json')['id']=='GRID273';h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 for file in ['NEXT_PROTOCOL_PLAN.json','ownfit_guard_source_support.json','training_context_support_summary.json','complete.json']:
  gp='research_archive/power-grid-2026/protocols/GRID272_union_limits/audited/'+file;assert git('show',h+':'+gp)==(Q/file).read_bytes()
 for source,ar in [(F,'GRID267_conditional_forest_plan'),(U,'GRID270_union_guard_plan')]:
  for rel,s in rd(source/'qualified_freeze.json').items():
   p=source/rel;assert sha(p)==s and hashlib.sha256(git('show',h+':research_archive/power-grid-2026/protocols/'+ar+'/qualified/'+rel)).hexdigest()==s
 for rel,s in rd(N/'calibrated_freeze.json').items():assert sha(N/rel)==s
 O.mkdir();save('input_publication.json',dict(passed=True,git_sha=h,all272support_267inputs_270union_published=True))
 save('design.json',dict(id='GRID273',scope='ONLYqualification fixed16ownfitpositive actualSTOPcontextcopies;0fit/cutoff/forestNNpredict/outerpolicy.',selection=rd(Q/'NEXT_PROTOCOL_PLAN.json')['selection'],target_revision='Retain originalrows/negativeexamples andoriginalweights unchanged;appendpositive copieswithoriginal y/NET/weight,270XactualSTOPbit1. Increasedpositive risk/exposure explicit,not originalhalfhalfmeanrisk unchanged. No tuned coefficient/scaler/source_massindependence.',qualification='All16 from9pools/fixedownfit8;no cal4/outer4 records. X64/X32 first39exact,bit0to1only;positiveNET/y/weight finite,existingrows exactprefix;dedup byid/context withoutdeletingoriginalCONT. Conflictingexactinputlabelsstop. All543guard unchanged;2untouchedactive+1KEEP refs frozen.',next='Separate274 nineforest+nine543positivecutoff=18/600s,notinside273. Original64/depth4/leaf10/fixedseedrecipe,2untouchedreuse+1KEEP,576newtrees/704deployed. Allweightsfirstpublished/allpredictionsbeforeoutcomes. Current106future124,0NNthetafit.',limits='Sameown12/internalcal4/exposed16developmentdates/relatedprefixseed limitations,notOOFblindfullweektime. No final8/physics/grid/paper/FRAME/install;labelsfee/controller/order/aliases/station/128/64/stagnation2/.9 unchanged. Historical6continuations8285updates/604.703actual/678.156cap separately.'))
 paths=[W/'run.py',O/'design.json',O/'input_publication.json',Q/'NEXT_PROTOCOL_PLAN.json',Q/'ownfit_guard_source_support.json',Q/'training_context_support_summary.json',F/'qualified_freeze.json',U/'qualified_freeze.json',U/'model_references.json',RUNTIME,META,ROLES,*list((F/'qualified_inputs').glob('*.npz')),*list((U/'qualified_inputs').glob('*.npz')),*list((N/'cutoffs').glob('*.json'))]
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID273_SOURCE_FROZEN')
def qualify():
 verify();h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h and git('show',h+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes();assert not (O/'qualification_started.json').exists();save('qualification_started.json',dict(source_published=h,fit0=True));(O/'qualified_inputs').mkdir();meta=rd(META);roles={r['pool']:r for r in rd(ROLES)};eligible=[r for r in rd(Q/'ownfit_guard_source_support.json') if not r['origin_current'] and r['ownfit_id_present'] and r['pool']!='f0_s1'];expected=rd(Q/'training_context_support_summary.json')['eligible_by_pool'];records=[];provenance=[]
 for file in sorted((F/'qualified_inputs').glob('*_fit.npz')):
  pool=file.name[:-8];z=np.load(file);u=np.load(U/'qualified_inputs'/f'{pool}.npz');idx={int(v):i for i,v in enumerate(z['ids'])};ui={int(v):i for i,v in enumerate(u['ids'])};selected=sorted([r for r in eligible if r['pool']==pool],key=lambda r:r['id']);old=len(z['ids']);n=len(selected);assert n==expected.get(pool,0);arr={k:z[k].copy() for k in z.files};appendix={k:[] for k in z.files if k!='fallback_KEEP'}
  for r in selected:
   mid=r['id'];j=idx[mid];k=ui[mid];m=meta[mid];assert r['fit_role'] and m['date'] in roles[pool]['fit_dates'] and m['date'] not in roles[pool]['cal4_dates']+roles[pool]['outer_dates'];assert u['fit_role'][k] and not u['origin_current'][k] and z['y'][j]==u['y'][k]==1 and z['actual_NET'][j]==u['actual_NET'][k]>0 and z['weight'][j]>0
   assert np.array_equal(z['X'][j,:39],u['X'][k,:39]) and np.array_equal(z['X32'][j,:39],u['X32'][k,:39]) and z['X'][j,39]==0 and u['X'][k,39]==1 and not z['domain_STOP'][j]
   for key in appendix:appendix[key].append(u['X'][k] if key=='X' else u['X32'][k] if key=='X32' else True if key=='domain_STOP' else z[key][j])
   provenance.append(dict(pool=pool,id=mid,original_index=j,new_index=old+len(provenance)-sum(v['pool']!=pool for v in provenance),date=m['date'],call=m['physical_state_group'],weight=float(z['weight'][j]),NET=float(z['actual_NET'][j]),oldbit=0,newbit=1))
  if n:
   for key,v in appendix.items():arr[key]=np.concatenate([arr[key],np.asarray(v,dtype=arr[key].dtype)],axis=0)
  arr['is_augmented']=np.r_[np.zeros(old,dtype=bool),np.ones(n,dtype=bool)]
  for key in z.files:
   if key=='fallback_KEEP':assert np.array_equal(arr[key],z[key])
   else:assert np.array_equal(arr[key][:old],z[key])
  assert np.isfinite(arr['X']).all() and np.isfinite(arr['weight']).all() and (arr['weight']>0).all() and np.array_equal(arr['X32'],arr['X'].astype(np.float32));assert len(set(zip(arr['ids'].tolist(),arr['X'][:,39].tolist())))==len(arr['ids'])
  exact={};conflicts=duplicates=0
  for x,y in zip(arr['X32'],arr['y']):
   key=x.tobytes()
   if key in exact:duplicates+=1;conflicts+=int(exact[key]!=int(y))
   exact[key]=int(y)
  assert conflicts==0,(pool,conflicts);np.savez_compressed(O/'qualified_inputs'/file.name,**arr);shutil.copy2(U/'qualified_inputs'/f'{pool}.npz',O/'qualified_inputs'/f'{pool}_cal.npz');shutil.copy2(F/'qualified_inputs'/f'{pool}_base.npz',O/'qualified_inputs'/f'{pool}_base.npz')
  for mid in arr['ids']:assert meta[int(mid)]['date'] in roles[pool]['fit_dates'] and meta[int(mid)]['date'] not in roles[pool]['cal4_dates']+roles[pool]['outer_dates']
  keep=bool(z['fallback_KEEP'][0]);records.append(dict(pool=pool,KEEP=keep,changed=bool(n),old_rows=old,new_rows=len(arr['ids']),added=n,old_risk_sum=float(z['weight'].sum()),new_risk_sum=float(arr['weight'].sum()),added_risk=float(arr['weight'][old:].sum()),old_positive_risk=float(z['weight'][z['y']>0].sum()),new_positive_risk=float(arr['weight'][arr['y']>0].sum()),exact_duplicate_groups=duplicates,opposite_label_conflicts=conflicts))
 assert len(records)==12 and sum(r['added'] for r in records)==16 and sum(r['changed'] for r in records)==9 and sum(r['KEEP'] for r in records)==1 and sum(r['new_rows'] for r in records)==2875 and sum(r['new_rows'] for r in records if not r['KEEP'])==2741
 for p in provenance:
  p['new_index']=next(r['old_rows'] for r in records if r['pool']==p['pool'])+[v['id'] for v in provenance if v['pool']==p['pool']].index(p['id'])
 shutil.copy2(RUNTIME,O/'runtime267.py');save('runtime_freeze.json',dict(sha256=sha(RUNTIME),copied_unmodified=True));save('augmentation_provenance.json',provenance);save('qualified_pools.json',records);refs=rd(U/'model_references.json')
 for ref in refs:ref['fixed271_cutoff']=rd(N/'cutoffs'/f"{ref['pool']}.json");ref['changed']=ref['pool'] in expected
 save('model_references.json',refs);save('qualification.json',dict(passed=True,newfits=0,newprobabilities=0,newouterstrategy=0,oldrows2859=True,newrows2875=True,active2741=True,added16_9pools=True,alloriginalrows_labels_weights_preserved=True,all16onlyownfit8_no_cal4_outer4=True,first39exact_contextonly01=True,all543calibration_inputs_byteunchanged=True,opposite_label_conflicts=0,positive_risk_addition=float(sum(r['added_risk'] for r in records)),expected_future_model_fits=9,expected_future_scalar_fits=9,cap_s=600))
 save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'qualified_inputs').iterdir()),O/'runtime267.py',O/'runtime_freeze.json',O/'qualification.json',O/'model_references.json',O/'qualified_pools.json',O/'augmentation_provenance.json']});archive('qualified');print('GRID273_QUALIFIED',json.dumps(rd(O/'qualification.json')))
if __name__=='__main__':globals()[sys.argv[1]]()
