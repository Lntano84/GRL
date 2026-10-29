"""GRID278 public fixed parent-entry support, then ownfit-only label diagnosis."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,collections,ast
import numpy as np
sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid278_invocation_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID278_invocation_support';Q=R/'outputs/grid277_selector_free';F=R/'outputs/grid276_selector_input_plan';META=R/'outputs/grid113_relational_learning/train_metadata.json';ROLES=R/'outputs/grid241_parent_prior_plan/qualified_pools.json'
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
 if phase=='public':shutil.copytree(O/'public_entry',d/'public_entry')
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')

def published(phase):
 verify();h=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h
 for rel,s in rd(A/phase/'archive_manifest.json').items():assert hashlib.sha256(git('show',h+':'+(A/phase/rel).relative_to(REPO).as_posix())).hexdigest()==s
 return h

def prepare():
 assert not O.exists() and rd(Q/'NEXT_PROTOCOL_PLAN.json')['id']=='GRID278';h=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==h;arQ=path_constant(R/'work/grid277_selector_free/run.py','A');arF=path_constant(R/'work/grid276_selector_input_plan/run.py','A')
 for n in ['NEXT_PROTOCOL_PLAN.json','complete.json','decision.json']:assert git('show',h+':'+(arQ/'audited'/n).relative_to(REPO).as_posix())==(Q/n).read_bytes()
 for rel,s in rd(F/'qualified_freeze.json').items():assert sha(F/rel)==s and hashlib.sha256(git('show',h+':'+(arF/'qualified'/rel).relative_to(REPO).as_posix())).hexdigest()==s
 O.mkdir();save('input_publication.json',dict(passed=True,git_sha=h,all276ownfit_cal_base_specificblobs_and277plan=True))
 save('design.json',dict(id='GRID278',scope='Readonly fixed deployed-context parent261entry support;0model/scalarfit/forestNNforward/newouterpolicy.',public='ALL276fit2875/cal543. TakeX40_originalcopy,forceONLYparent-evaluationbit39=1,compute p=b+alpha*((b+sum(Xdeploy40*theta[:-1])+theta[-1])-b),E=(b>0)&(p>=0). Samefloatoperationorder/frozenbase41theta+alpha. Preserve originalX40files. ExactlyONEcondition,no alternatethreshold/contextselection. Saveids/p/E/originaldomain/augmentation/roles BEFORE ownfitlabeljoin.',guard='All543cal p mustmatch savedparent261_score bitexact,Ealltrue.16relatedcopiesnotdeduped orreweighted. No39Dparentinput substitution.',label_access='Afterallpublicentryexportspublished,readONLYownfit y/NET/weight/source_mass andmetadate/call. PartitionE/notE x STOP/CONT x positive/negative,all12inclKEEP;report risk/exposure not policyreturns. No outerinput/outcomes/newteacherlabels/casepickle.',future='No filteredfitpool exported in278. Onlyifowntraininginvocation gapexists,qualifyONEfinite conditioning proposal later;no redundantfilter/refit ifallEtrue. No newfeature/weight/threshold/seed scan.',counts='Current146 actual0fits,3418fixedpubliclinear-dot evaluations incl543guard,0forestNNforward/0newphysical. Historical6continuations8285updates/604.703actual/678.156cap separate.',limits='Teacherprefixsupport underhypotheticaldeployedSTOP1,notactualindependentphysicalcontrol. Parentown12/internalcal4/exposed16dates/relatedprefixseed remain,notOOFblind/fullweek/time. Final8/physics/grid/paper/FRAME/install untouched;feeNETlabels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.'))
 paths=[W/'run.py',O/'design.json',O/'input_publication.json',Q/'NEXT_PROTOCOL_PLAN.json',F/'qualified_freeze.json',*list((F/'qualified_inputs').glob('*.npz')),META,ROLES];save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});archive('source');print('GRID278_SOURCE_FROZEN_NO_OUTER_INPUTS')

def public():
 verify();h=published('source');assert not (O/'public_started.json').exists();save('public_started.json',dict(source_published=h));(O/'public_entry').mkdir();meta=rd(META);roles={r['pool']:r for r in rd(ROLES)};records=[];total=guards=0
 for file in sorted((F/'qualified_inputs').glob('*_base.npz')):
  pool=file.name[:-9];base=np.load(file);theta=base['theta'];alpha=float(base['alpha'][0]);assert theta.shape==(41,)
  for role in ['fit','cal']:
   z=np.load(F/'qualified_inputs'/f'{pool}_{role}.npz');x=z['X40_original'].copy();assert x.shape[1]==40;x[:,39]=1;before=z['b'];score=before+alpha*((before+np.sum(x*theta[:-1],axis=1)+theta[-1])-before);E=(before>0)&(score>=0);assert np.isfinite(score).all();dates={meta[int(i)]['date'] for i in z['ids']}
   if role=='fit':assert dates<=set(roles[pool]['fit_dates']) and not dates&set(roles[pool]['cal4_dates']+roles[pool]['outer_dates']);total+=len(E)
   else:assert np.array_equal(score,z['parent261_score']) and E.all();guards+=len(E)
   fields=dict(ids=z['ids'],b=before,parent_score=score,eligible=E,fit_role=np.ones(len(E),dtype=bool) if role=='fit' else z['fit_role'],original_domain_STOP=z['domain_STOP'] if role=='fit' else np.ones(len(E),dtype=bool),is_augmented=z['is_augmented'] if role=='fit' else np.zeros(len(E),dtype=bool),original_context=z['X40_original'][:,39],evaluated_context=np.ones(len(E)))
   assert not set(fields)&{'y','actual_NET','weight','source_mass'};np.savez_compressed(O/'public_entry'/f'{pool}_{role}.npz',**fields);records.append(dict(pool=pool,role=role,rows=len(E),eligible=int(E.sum()),ineligible=int((~E).sum()),KEEP=bool(base['fallback_KEEP'][0]),input_sha=sha(F/'qualified_inputs'/f'{pool}_{role}.npz'),base_sha=sha(file),outcome_fields_read=[]))
 assert total==2875 and guards==543;save('public_entry_receipts.json',records);save('public_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'public_entry').glob('*.npz')),O/'public_entry_receipts.json']});archive('public');print('GRID278_PUBLIC3418_FIXED_LINEAR_DOTS_NO_LABELS')

def analyze():
 verify();h=published('public');assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(public_published=h));meta=rd(META);records=[];excluded=[]
 for file in sorted((F/'qualified_inputs').glob('*_fit.npz')):
  pool=file.name[:-8];z=np.load(file);pub=np.load(O/'public_entry'/file.name);base=np.load(F/'qualified_inputs'/f'{pool}_base.npz');keep=bool(base['fallback_KEEP'][0]);assert np.array_equal(pub['ids'],z['ids']);E=pub['eligible'];parts=[]
  for eligible in [True,False]:
   for stop in [True,False]:
    for positive in [True,False]:
     mask=(E==eligible)&(z['domain_STOP']==stop)&((z['y']>0)==positive);parts.append(dict(eligible=eligible,domain_STOP=stop,positive=positive,rows=int(mask.sum()),risk=float(z['weight'][mask].sum()),raw_NET_sum=float(z['actual_NET'][mask].sum()),calls=len({meta[int(i)]['physical_state_group'] for i in z['ids'][mask]}),dates=len({meta[int(i)]['date'] for i in z['ids'][mask]})))
  rec=dict(pool=pool,KEEP=keep,rows=len(E),eligible=int(E.sum()),ineligible=int((~E).sum()),original_risk=float(z['weight'].sum()),ineligible_risk=float(z['weight'][~E].sum()),eligible_risk=float(z['weight'][E].sum()),eligible_positive=int((E&(z['y']>0)).sum()),eligible_negative=int((E&(z['y']==0)).sum()),excluded_positive=int((~E&(z['y']>0)).sum()),eligible_STOP_positive=int((E&z['domain_STOP']&(z['y']>0)).sum()),partitions=parts);records.append(rec)
  for j in np.flatnonzero(~E):
   m=meta[int(z['ids'][j])];excluded.append(dict(pool=pool,index=int(j),id=int(z['ids'][j]),KEEP=keep,date=m['date'],call=m['physical_state_group'],domain_STOP=bool(z['domain_STOP'][j]),positive=bool(z['y'][j]>0),NET=float(z['actual_NET'][j]),risk=float(z['weight'][j]),parent_score=float(pub['parent_score'][j]),is_augmented=bool(z['is_augmented'][j])))
 active=[r for r in records if not r['KEEP']];actex=[r for r in excluded if not r['KEEP']];summary=dict(all_rows2875=sum(r['rows'] for r in records),active_rows2741=sum(r['rows'] for r in active),active_eligible=sum(r['eligible'] for r in active),active_ineligible=sum(r['ineligible'] for r in active),active_excluded_positive=sum(r['excluded_positive'] for r in active),active_ineligible_risk_sum=sum(r['ineligible_risk'] for r in active),changed_active_pools=sum(r['ineligible']>0 for r in active),eligible_class_empty_pools=[r['pool'] for r in active if r['eligible_positive']==0 or r['eligible_negative']==0],unique_excluded_calls=len({r['call'] for r in actex}),unique_excluded_dates=len({r['date'] for r in actex}),guard543_all_eligible=True,public_linear_evaluations3418=True,newfits0=True,newforestNNforward0=True,newouterpolicy0=True,no_filtered_fitpool_exported=True)
 save('ownfit_eligibility_support.json',records);save('ownfit_ineligible_records.json',excluded);save('diagnosis_summary.json',summary);verify();archive('analyzed');print('GRID278_READONLY_DIAGNOSED',json.dumps(summary))
if __name__=='__main__':globals()[sys.argv[1]]()
