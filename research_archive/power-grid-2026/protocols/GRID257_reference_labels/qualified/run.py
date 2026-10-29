"""Freeze then join the original teacher labels and recompute fixed structural risk."""
from pathlib import Path
import os,sys,json,hashlib,pickle,shutil,subprocess,collections,time,importlib.util,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid257_reference_labels';PUBLIC=R/'outputs/grid256_reference_public';ROLE=R/'outputs/grid241_parent_prior_plan';OLD=R/'outputs/grid214_inner_head_generation';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID257_reference_labels'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def published(files):
 head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','-C',str(REPO),'ls-remote','lntano-src','refs/heads/main'],text=True).split()[0];assert head==remote
 for p in files:assert subprocess.check_output(['git','-C',str(REPO),'show',head+':'+p.relative_to(REPO).as_posix()])==p.read_bytes(),str(p)
 return head

def verify():
 for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel
 for rel,h in read(PUBLIC/'generated_freeze.json').items():assert sha(PUBLIC/rel)==h,rel

def archive(phase,folders=[]):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 for name in folders:shutil.copytree(O/name,d/name)
 save(d/'archive_manifest.json',{p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()})

def prepare():
 assert not O.exists() and read(PUBLIC/'complete.json')['passed'];O.mkdir();barrier=read(PUBLIC/'pre_target_publication.json');assert barrier['passed'] and barrier['all_public_exports_published'];head=published([REPO/'research_archive/power-grid-2026/protocols/GRID256_reference_public/generated/generated_freeze.json',REPO/'research_archive/power-grid-2026/protocols/GRID256_reference_public/delivery/pre_target_publication.json'])
 for rel,h in read(PUBLIC/'generated_freeze.json').items():
  pp=PUBLIC/rel;ap=REPO/'research_archive/power-grid-2026/protocols/GRID256_reference_public/generated'/rel;assert sha(pp)==h;assert hashlib.sha256(subprocess.check_output(['git','-C',str(REPO),'show',head+':'+ap.relative_to(REPO).as_posix()])).hexdigest()==h
 save(O/'pre_label_publication.json',dict(passed=True,all69_public_exports_published_at=head,teacher_labels_not_parsed_in_prepare=True))
 save(O/'design.json',dict(id='GRID257',scope='Label-provenance/coverage/structural-risk and future support qualification ONLY; no optimizer/scalar/network/forest prediction or newouterpolicy.',teacher_NET='Original102cases train gains float64/scale minus originalfee*intquery/scale in the exact original196/213/214 operation order; compare every old214 same-row NET exactly before newlabel export. No use of NN float32 target.',access='AfterGRID256publicbarrier,deserialize historical cases container but only index gains/queries for respective own12 metadata rows. Do not claim unopened physical outer-storage; validation/final8 no indexed labels. Per-poolrole access logged.',structure='STOP: equal representeddate then physicalgroup then eligible sample/prefix; CONT: same then uniform preSTOP positions within sample. Each domain from full newlyselectedown12 before parentb/fit8 filtering,half mass each. Verify identical rule reproduces old192 stored structural masses.',risk='On fit8 parentCONT-eligible newselection,each domain mass*absNET renormalized .5; same total scaling by NfitSTOP as242. No riskratio/fee/label/threshold modification.',roles='Fixed241fit8/cal4. Label fit8 risk and ALL own12 selectedSTOPparentCONT empirical constraints; not independentcal4.',KEEP='f0_s1 OR nofit8STOPpositive OR emptySTOP/CONT -> predeclaredKEEP, no data/date/seed replacement; invalid zero-risk domain halts instead of inventing rule.',future='Same25040input41param anchored correction;<=11linear+11original221positive scalar once600s. Sourceinputpublished then allweights/steps then scores then labeljoin. Not training here.',limits='OriginalNN/heads own12 and cal4 ancestor limitations remain, old exposed development. No final8/physical/grid/paper/FRAME/install.'))
 paths=[W/'run.py',O/'design.json',O/'pre_label_publication.json',PUBLIC/'generated_freeze.json',PUBLIC/'pre_target_publication.json',ROLE/'qualified_pools.json',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',OLD/'bundles.json',R/'work/grid189_baseline_extension/run.py',R/'work/grid192_gate_support/run.py',R/'work/grid196_gain_readout/run.py',R/'work/grid213_inner_head_plan/run.py',R/'work/grid242_parent_prior_inputs/run.py',R/'work/grid219_prior_objective/run.py',R/'work/grid221_step_plan/run.py']
 paths += [PUBLIC/rel for rel in read(PUBLIC/'generated_freeze.json')]
 for p in read(ROLE/'qualified_pools.json'):paths.append(R/'outputs/grid192_gate_support/fit_features'/f"{p['pool']}.npz")
 paths += list((OLD/'labelled_samples').glob('*.npz'));save(O/'source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in set(paths)});verify();archive('source');print('GRID257_SOURCE_FROZEN_NO_LABEL_JOIN')

def mass(ids,meta):
 if len(ids)==0:return np.zeros(0)
 mm=[meta[int(i)] for i in ids];dates={r['date'] for r in mm};gd={d:{r['physical_state_group'] for r in mm if r['date']==d} for d in dates};sc=collections.defaultdict(set);positions=collections.Counter(r['sample_id'] for r in mm)
 for r in mm:sc[r['physical_state_group']].add(r['sample_id'])
 x=np.array([1/(len(dates)*len(gd[r['date']])*len(sc[r['physical_state_group']])*positions[r['sample_id']]) for r in mm]);assert abs(x.sum()-1)<1e-12;return x

def qualify():
 verify();assert not (O/'join_started.json').exists();head=published([A/'source/source_freeze.json',A/'source/run.py']);save(O/'join_started.json',dict(source_published=head,fit_calls=0));meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];assert fee>0 and scale>0
 container=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cases={r['sample_id']:r for r in container if r['partition']=='train'};assert {m['sample_id'] for m in meta}<=set(cases);bundles=read(OLD/'bundles.json');records=[];access=[];align=[];struct=[];totalfit=0;totalcal=0;positivecal=0
 for name in ['fit_inputs','calibration_inputs','public_mass','role_labels']:(O/name).mkdir()
 spec=importlib.util.spec_from_file_location('g257_original219objective',R/'work/grid219_prior_objective/run.py');loss=importlib.util.module_from_spec(spec);spec.loader.exec_module(loss)
 for role in read(ROLE/'qualified_pools.json'):
  pool=role['pool'];f,seed=[int(v[1:]) for v in pool.split('_')];full=np.load(PUBLIC/'public_full'/f'{pool}.npz');ids=full['ids'].astype(int);own=set(role['fit_dates'])|set(role['cal4_dates']);assert {meta[int(i)]['date'] for i in ids}==own and not own&set(role['outer_dates']);pos={int(i):k for k,i in enumerate(ids)};g=[];queries=[]
  for i in ids:
   m=meta[int(i)];c=cases[m['sample_id']];j=m['position'];assert c['date']==m['date'] and c['date'] in own and c['partition']=='train';g.append(float(c['gains'][j]));queries.append(int(c['queries'][j]));assert int(c['queries'][j])==m['next_queries']
  gain=np.array(g,float);queries=np.array(queries,int);net=gain/scale-fee*queries/scale;assert np.isfinite(net).all() and (queries>0).all();access.append(dict(pool=pool,indexed_dates=sorted(own),indexed_metadata_ids=ids.tolist(),indexed_fields=['gains[position]','queries[position]'],fit8_dates=role['fit_dates'],constraint_cal4_dates=role['cal4_dates'],outer_dates=role['outer_dates'],outer_label_indices_used=0,container_deserialized_in_full=True,unparsed_outer_storage_claim=False))
  # Old saved double precision targets must match exactly; no float32 substitution.
  oldids=[];maxerr=0.;checkrows=0
  for b in [b for b in bundles if b['outer_fold']==f and b['seed']==seed]:
   z=np.load(OLD/'labelled_samples'/f"{b['bundle']}.npz");ii=z['meta_ids'].astype(int);assert set(meta[int(i)]['date'] for i in ii)<=own;nn=net[[pos[int(i)] for i in ii]];err=float(abs(nn-z['actual_NET']).max());maxerr=max(maxerr,err);assert np.array_equal(nn,z['actual_NET']),('OriginalNET precision/source mismatch',pool,b['bundle'],err);oldids.extend(ii.tolist());checkrows+=len(ii)
  old=np.load(R/'outputs/grid192_gate_support/fit_features'/f'{pool}.npz');assert set(oldids)==set(old['ids'].tolist());n=int(old['original_STOP_count']);oldmass=np.r_[.5*mass(old['ids'][:n],meta),.5*mass(old['ids'][n:],meta)];masserr=float(abs(oldmass-old['mass']).max());assert masserr<1e-15;align.append(dict(pool=pool,old214_rows=checkrows,old_NET_bit_exact=True,max_abs_error=maxerr));struct.append(dict(pool=pool,old192_rows=len(old['ids']),same_structural_rule_max_abs_error=masserr))
  selected=full['selected'];stop=full['domain_STOP'];source_mass=np.zeros(len(ids));source_mass[selected&stop]=.5*mass(ids[selected&stop],meta);source_mass[selected&~stop]=.5*mass(ids[selected&~stop],meta);assert abs(source_mass.sum()-1)<1e-12;np.savez_compressed(O/'public_mass'/f'{pool}.npz',ids=ids,selected=selected,domain_STOP=stop,source_mass=source_mass)
  # Role label file is own12 only, not a new held-policy evaluation.
  np.savez_compressed(O/'role_labels'/f'{pool}.npz',ids=ids,actual_NET=net,gain_normalized=gain/scale,known_charge=fee*queries/scale,fit_role=full['fit_role'])
  fit=np.load(PUBLIC/'fit_public'/f'{pool}.npz');fi=fit['ids'].astype(int);fp=np.array([pos[int(i)] for i in fi]);assert all(meta[int(i)]['date'] in role['fit_dates'] for i in fi);fn=net[fp];fm=source_mass[fp];st=fit['domain_STOP'];y=fn>0;fallback=pool=='f0_s1' or not np.any(y&st) or not st.any() or not (~st).any();weight=np.zeros(len(fi));domain_risk=[]
  for mask in [st,~st]:
   risk=fm[mask]*abs(fn[mask]);total=float(risk.sum());domain_risk.append(total)
   if total>0:weight[mask]=.5*risk/total
  if not fallback:
   assert all(t>0 for t in domain_risk) and (weight>0).all() and abs(weight[st].sum()-.5)<1e-12 and abs(weight[~st].sum()-.5)<1e-12;weight*=int(st.sum())
  np.savez_compressed(O/'fit_inputs'/f'{pool}.npz',X=fit['X'],b=fit['b'],ids=fi,y=y.astype(int),weight=weight,actual_NET=fn,source_mass=fm,domain_STOP=st,fallback_KEEP=np.array([fallback]))
  cal=np.load(PUBLIC/'calibration_public'/f'{pool}.npz');ci=cal['ids'].astype(int);cp=np.array([pos[int(i)] for i in ci]);cn=net[cp];assert cal['domain_STOP'].all();np.savez_compressed(O/'calibration_inputs'/f'{pool}.npz',X=cal['X'],b=cal['b'],ids=ci,y=(cn>0).astype(int),actual_NET=cn,fit_role=cal['fit_role']);assert fallback or (cn>0).any()
  graderr=0.
  if not fallback:
   theta=np.linspace(-.02,.02,41);val,grad=loss.objective(theta,fit['X'],y,weight,fit['b']);assert np.isfinite(val) and np.isfinite(grad).all()
   for k in range(41):
    delta=np.zeros(41);delta[k]=1e-6;fd=(loss.objective(theta+delta,fit['X'],y,weight,fit['b'])[0]-loss.objective(theta-delta,fit['X'],y,weight,fit['b'])[0])/2e-6;graderr=max(graderr,abs(fd-grad[k]))
   assert graderr<1e-6
  totalfit+=len(fi);totalcal+=len(ci);positivecal+=int((cn>0).sum());records.append(dict(pool=pool,fit8_rows=len(fi),fit8_STOP=int(st.sum()),fit8_STOP_positive=int((y&st).sum()),fit8_CONT=int((~st).sum()),fit8_CONT_positive=int((y&~st).sum()),own12_selectedSTOP_rows=len(ci),own12_selectedSTOP_positive=int((cn>0).sum()),positive_STOP_physical_calls=len({meta[int(i)]['physical_state_group'] for i,nv in zip(ci,cn) if nv>0}),risk_STOP=domain_risk[0],risk_CONT=domain_risk[1],future_linear_fit=not fallback,future_scalar_fit=not fallback,fallback_KEEP=fallback,numerical41_gradient_max_error=graderr))
 active=sum(r['future_linear_fit'] for r in records);assert active<=11 and totalfit==2859 and totalcal==2054;save(O/'target_access.json',dict(historical_container_deserialized=True,NN_float32_targets_read=False,new_outer_policy_score=False,pools=access));save(O/'old_label_alignment.json',align);save(O/'old_structure_alignment.json',struct);save(O/'qualification.json',dict(passed=True,pools=records,fit8_rows=totalfit,own12_STOP_rows=totalcal,own12_STOP_positive=positivecal,linear_fits_planned=active,scalar_fits_planned=active,total_future_fits=2*active,future_fit_window_s=600,actual_fits=0,new_network_forward=0,new_outer_policy_score=0,old_NET_bit_exact_all_pools=True,old_structure_rule_reproduced=True,new_structure_and_risk_recomputed=True,all_roles_retained=True))
 files=[p for name in ['fit_inputs','calibration_inputs','public_mass','role_labels'] for p in (O/name).iterdir()]+[O/'qualification.json',O/'target_access.json',O/'old_label_alignment.json',O/'old_structure_alignment.json'];save(O/'qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in files});verify();archive('qualified',['fit_inputs','calibration_inputs','public_mass','role_labels']);print('GRID257_QUALIFIED',totalfit,totalcal,positivecal,active)

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  if O.exists():save(O/(sys.argv[1]+'_error.json'),dict(error=repr(e),traceback=traceback.format_exc(),no_auto_retry=True,actual_fits=0))
  raise
