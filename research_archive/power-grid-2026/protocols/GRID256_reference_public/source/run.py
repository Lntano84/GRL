"""Fixed reference STOP-source public export only. Never joins teacher outcomes."""
from pathlib import Path
import os,sys,time,json,hashlib,subprocess,pickle,shutil,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid256_reference_public';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID256_reference_public';P=R/'outputs/grid254_reference_preflight';NN=R/'outputs/grid255_reference_stop_nn/recovery_run01';PLAN=R/'outputs/grid253_stop_source_plan';ROLE=R/'outputs/grid241_parent_prior_plan';RAW41=R/'outputs/grid214_inner_head_generation/public_raw41.npz';FULL=R/'outputs/grid192_gate_support';ACCESS=[]
sys.path.insert(0,str(R/'work/grid254_reference_preflight'));import runtime254 as C
KEEP=[k for k in range(41) if k not in [14,18,19]]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):C.save(p,v)
def fields(p,names):
 allowed={'raw','values','x','ids','mean','scale','raw_mean','raw_scale','mean41','scale41','fit_ids','score_ids','normalized'};assert set(names)<=allowed
 with np.load(p) as z:out={n:z[n].copy() for n in names}
 ACCESS.append(dict(path=p.relative_to(R).as_posix(),fields=list(names)));return out

def verify():
 for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel
 for rel,h in read(NN/'trained_freeze.json').items():assert sha(R/rel)==h,rel

def published(files):
 head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','-C',str(REPO),'ls-remote','lntano-src','refs/heads/main'],text=True).split()[0];assert head==remote
 for p in files:assert subprocess.check_output(['git','-C',str(REPO),'show',head+':'+p.relative_to(REPO).as_posix()])==p.read_bytes(),str(p)
 return head

def archive(phase,folders=[]):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 for name in folders:shutil.copytree(O/name,d/name)
 save(d/'archive_manifest.json',{p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()})

def prepare():
 assert not O.exists() and read(NN/'complete.json')['passed'];O.mkdir();idx=read(NN/'model_index.json');assert len(idx)==18
 # Check publication barrier without executing a network or fitting a head.
 head=published([REPO/'research_archive/power-grid-2026/protocols/GRID255_reference_stop_nn/recovery_run01/completed/trained_freeze.json'])
 base=REPO/'research_archive/power-grid-2026/protocols/GRID255_reference_stop_nn'
 for m in idx:
  path=base/m['archive_model'];assert sha(R/m['model'])==sha(path)==m['model_sha256'];blob=subprocess.check_output(['git','-C',str(REPO),'show',head+':'+path.relative_to(REPO).as_posix()]);assert hashlib.sha256(blob).hexdigest()==m['model_sha256']
 save(O/'pre_reference_publication.json',dict(passed=True,all18weight_blobs_published_at=head,network_forward=0))
 save(O/'design.json',dict(id='GRID256',scope='Public reference scoring, first native STOP/preSTOP CONT selection and original-owner parent inputs ONLY. No outcomes or correction training/policy evaluation.',reference='18 frozen 8date networks,32238 excluded-date public rows,32 batch; original GRID121 raw41 and fixed six pair fit8mean/scale/clip20, not raw publicdict global41.',selection='For each fixed teacher prefix path first q1<=q0 is STOP, all earlier q1>q0 auxiliary CONT, afterSTOP excluded; allCONT terminal has no STOPexample. No cases pickle read.',parent='All own12 public rows: reuse22653 oldraw83 rows bit-for-bit; at most9585 missing rows forward ONLY originalownerf NN via qualified254interface. Frozen24GRID208heads/scalers; p*mu and originalcharge produce b; no new forest.',residual='Fixed213inner8 public41scaler,drop14/18/19 to38,append parentb and native referenceSTOPbit ->40inputs same250. Native deploy NN12 unchanged.',roles='Original241fit8/cal4; fit export selected&parentCONT&fit8; constraints export selectedSTOP&parentCONT across own12, labels later.',public_only='NPZ field allowlist; no y/NET/mass/event labels; heads andweights reused only. No teacher targetjoin/fits/newouterpolicy.',generation_cap_s=600,no_retry_completed_generation=True,boundaries='All seeds/dates retained. Reference records/outer date excluded; parent/heads own12,internalcal4 not whole-chainOOF. No final8/newphysical/grid/paper/FRAME/install. No quality/latency claim.'))
 paths=[W/'run.py',O/'design.json',O/'pre_reference_publication.json',NN/'model_index.json',NN/'trained_freeze.json',PLAN/'reference_nodes.json',PLAN/'reference_uses.json',ROLE/'qualified_pools.json',RAW41,R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid113_relational_learning/train_public.pkl',R/'outputs/grid121_date_transfer/train_arrays.npz',R/'work/grid254_reference_preflight/parent_interface.py',P/'jobs.json',P/'source_freeze_recovery01.json']
 paths += [R/rel for rel in read(P/'source_freeze_recovery01.json')]
 for m in idx:paths.append(R/m['model'])
 for p in read(ROLE/'qualified_pools.json'):
  paths += [R/p['parent_scaler'],R/p['residual_public41_scaler'],FULL/'fit_features'/f"{p['pool']}.npz",R/'outputs/grid242_parent_prior_inputs/public_full'/f"{p['pool']}.npz"]+[R/x for x in p['parent_heads']]
 save(O/'source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in set(paths)});verify();archive('source');print('GRID256_SOURCE_FROZEN_ZERO_FORWARD')

def generate():
 verify();assert not (O/'generation_started.json').exists();head=published([A/'source/source_freeze.json',A/'source/run.py']);C.claim(O/'generation_started.json',dict(pid=os.getpid(),source_published=head,no_target_join=True));start=time.monotonic();deadline=start+600
 for folder in ['reference_scores','original_parent83','public_full','fit_public','calibration_public']:(O/folder).mkdir()
 meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');public=pickle.loads((R/'outputs/grid113_relational_learning/train_public.pkl').read_bytes());ACCESS.append(dict(path='outputs/grid113_relational_learning/train_public.pkl',fields=['context_action','context_feedback','next_action','anchor_action','known_query_cost'],global41_replaced_by_original_training_raw_normalization=True));raw=fields(R/'outputs/grid121_date_transfer/train_arrays.npz',['raw'])['raw'];r41=fields(RAW41,['values'])['values'];assert raw.shape==r41.shape==(3582,41)
 import torch
 S=C.orig();index={m['node']:m for m in read(NN/'model_index.json')};jobs={j['node']:j for j in read(P/'jobs.json')};nodeq={};receipts=[]
 for n in read(PLAN/'reference_nodes.json'):
  C.before(deadline);name=n['node'];j=jobs[name];z=fields(P/j['package'],['fit_ids','score_ids','mean','scale']);ids=z['score_ids'].astype(int);assert np.array_equal(ids,n['score_ids']) and set(meta[int(i)]['date'] for i in ids)==set(n['reference_score_dates']);assert set(n['fit_dates']).isdisjoint(n['reference_score_dates']);cp=torch.load(R/index[name]['model'],map_location='cpu',weights_only=False);net=S.build(n['seed'],'FACTOR');net.load_state_dict(cp['state_dict']);net.eval();before=S.digest(net);assert cp['updates']==7250;qq=[]
  with torch.inference_mode():
   for off in range(0,len(ids),32):
    C.before(deadline);ii=ids[off:off+32];normalized=np.clip((raw[ii]-z['mean'])/z['scale'],-20,20).astype('f');rows=[dict(public[int(i)],global41=normalized[k]) for k,i in enumerate(ii)];q=net(**S.G.old_api().collate(rows)).numpy();assert q.shape==(len(ii),2) and np.isfinite(q).all();qq.append(q)
  q=np.concatenate(qq);assert S.digest(net)==before and sha(R/index[name]['model'])==index[name]['model_sha256'];np.savez_compressed(O/'reference_scores'/f'{name}.npz',ids=ids,q=q);nodeq[name]={int(i):q[k] for k,i in enumerate(ids)};receipts.append(dict(node=name,rows=len(ids),weights_unchanged=True,record_and_outer_exclusion_fixed=True,weight_sha256=index[name]['model_sha256']));save(O/'reference_progress.json',receipts)
 assert sum(r['rows'] for r in receipts)==32238;save(O/'reference_receipts.json',receipts)
 from parent_interface import extract_original,SOURCES
 pools=[];uses=read(PLAN/'reference_uses.json');total_cache=0;total_missing=0;groups={}
 for i,m in enumerate(meta):groups.setdefault(m['sample_id'],[]).append(i)
 for groupids in groups.values():groupids.sort(key=lambda i:meta[i]['position']);assert [meta[i]['position'] for i in groupids]==list(range(len(groupids)))
 for p in read(ROLE/'qualified_pools.json'):
  C.before(deadline);pool=p['pool'];f,seed=[int(v[1:]) for v in pool.split('_')];arr=fields(SOURCES[f]/'fold_arrays.npz',['fit_ids']);ids=arr['fit_ids'].astype(int);pos={int(i):k for k,i in enumerate(ids)};own_dates=set(p['fit_dates'])|set(p['cal4_dates']);assert {meta[int(i)]['date'] for i in ids}==own_dates
  qq=np.full((len(ids),2),np.nan,np.float32);node_names=np.full(len(ids),'',dtype='<U12')
  for use in [u for u in uses if u['gate_pool']==pool]:
   assert set(use['fit_label_dates']).isdisjoint(use['record_dates']) and set(use['fit_label_dates']).isdisjoint(use['outer_dates']);date_set=set(use['record_dates'])
   for i in ids:
    if meta[int(i)]['date'] in date_set:qq[pos[int(i)]]=nodeq[use['reference_node']][int(i)];node_names[pos[int(i)]]=use['reference_node']
  assert np.isfinite(qq).all() and (node_names!='').all();selected=np.zeros(len(ids),bool);stop=np.zeros(len(ids),bool);allcont=0
  for groupids in groups.values():
   if meta[groupids[0]]['date'] not in own_dates:continue
   rr=[pos[i] for i in groupids];native=qq[rr,1]<=qq[rr,0];first=int(np.flatnonzero(native)[0]) if native.any() else len(rr);selected[np.array(rr[:first],int)]=True
   if first<len(rr):selected[rr[first]]=True;stop[rr[first]]=True
   else:allcont+=1
  cache=fields(FULL/'fit_features'/f'{pool}.npz',['ids','x']);cid=cache['ids'].astype(int);assert len(set(cid.tolist()))==len(cid);full83=np.full((len(ids),83),np.nan);cache_positions=np.array([pos[int(i)] for i in cid]);full83[cache_positions]=cache['x'];missing=[int(i) for i in ids if int(i) not in set(cid.tolist())];proof=dict(rows=0,original_model_unchanged=True,new_reference_hidden_never_used=True)
  if missing:full83[[pos[i] for i in missing]],proof=extract_original(f,seed,missing)
  assert np.array_equal(full83[cache_positions],cache['x']) and sha(FULL/'fit_features'/f'{pool}.npz')==read(O/'source_freeze.json')[(FULL/'fit_features'/f'{pool}.npz').relative_to(R).as_posix()];assert np.isfinite(full83).all();total_cache+=len(cid);total_missing+=len(missing);assert total_missing<=9585;np.savez_compressed(O/'original_parent83'/f'{pool}.npz',ids=ids,x=full83,cached_mask=np.isin(ids,cid))
  norm=fields(R/p['parent_scaler'],['mean','scale','raw_mean','raw_scale']);scaler=fields(R/p['residual_public41_scaler'],['mean41','scale41']);x124=np.c_[(full83-norm['mean'])/norm['scale'],(r41[ids]-norm['raw_mean'])/norm['raw_scale']];epath,apath=[R/x for x in p['parent_heads']];before=[sha(epath),sha(apath)];event=pickle.loads(epath.read_bytes());amount=pickle.loads(apath.read_bytes());ep=event.predict_proba(x124)[:,1];mu=amount.predict(x124);charge=full83[:,82];assert (charge>0).all() and (mu>0).all();gain=ep*mu;b=(gain-charge)/(gain+charge);assert before==[sha(epath),sha(apath)];old=fields(R/'outputs/grid242_parent_prior_inputs/public_full'/f'{pool}.npz',['ids']);oldz=np.load(R/'outputs/grid242_parent_prior_inputs/public_full'/f'{pool}.npz');oldb=oldz['b'];ACCESS.append(dict(path=f'outputs/grid242_parent_prior_inputs/public_full/{pool}.npz',fields=['b']));oldlookup={int(i):k for k,i in enumerate(old['ids'])};err=max(abs(b[pos[int(i)]]-oldb[oldlookup[int(i)]]) for i in cid);assert err<1e-12
  public38=((r41[ids]-scaler['mean41'])/scaler['scale41'])[:,KEEP];native=qq[:,1]<=qq[:,0];X=np.c_[public38,b,native.astype(float)];assert X.shape==(len(ids),40) and np.isfinite(X).all();assert np.array_equal(native[selected],stop[selected]);fit=np.array([meta[int(i)]['date'] in p['fit_dates'] for i in ids]);cal=~fit;assert np.array_equal(b>0,gain>charge);eligible=selected&(b>0)
  np.savez_compressed(O/'public_full'/f'{pool}.npz',ids=ids,X=X,b=b,reference_q=qq,reference_node=node_names,selected=selected,domain_STOP=stop,fit_role=fit,cal_role=cal,parent_event_p=ep,parent_mu=mu,charge=charge,parent_CONT=b>0)
  for folder,mask in [('fit_public',fit&eligible),('calibration_public',stop&eligible)]:np.savez_compressed(O/folder/f'{pool}.npz',ids=ids[mask],X=X[mask],b=b[mask],domain_STOP=stop[mask],fit_role=fit[mask])
  pools.append(dict(pool=pool,own12_rows=len(ids),reference_selected_STOP=int(stop.sum()),reference_selected_CONT=int((selected&~stop).sum()),all_CONT_paths_terminal_no_STOP_example=allcont,fit8_eligible_STOP=int((fit&eligible&stop).sum()),fit8_eligible_CONT=int((fit&eligible&~stop).sum()),own12_eligible_STOP=int((eligible&stop).sum()),cached_rows_reused=len(cid),missing_original_owner_rows=len(missing),cached_parent_b_max_abs_error=float(err),missing_parent_proof=proof,targets_read=False));save(O/'pool_progress.json',pools)
 assert total_cache==22653 and total_missing<=9585;assert all(not {'y','NET','actual_NET','mass','target','target_fit','event_y'}&set(a['fields']) for a in ACCESS);C.before(deadline);verify();save(O/'public_access.json',ACCESS);save(O/'generation_receipts.json',dict(passed=True,reference_rows=32238,pools=pools,cached_original83_rows=total_cache,new_original_owner83_rows=total_missing,reused_heads=24,new_fits=0,new_teacher_target_join=0,new_outer_policy_scores=0,generation_wall_s=time.monotonic()-start));files=[p for folder in ['reference_scores','original_parent83','public_full','fit_public','calibration_public'] for p in (O/folder).iterdir()]+[O/'reference_receipts.json',O/'generation_receipts.json',O/'public_access.json'];save(O/'generated_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in files});archive('generated',['reference_scores','original_parent83','public_full','fit_public','calibration_public']);print('GRID256_PUBLIC_GENERATED',total_cache,total_missing,time.monotonic()-start)

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  if O.exists():save(O/(sys.argv[1]+'_error.json'),dict(error=repr(e),traceback=traceback.format_exc(),no_automatic_retry=True))
  raise
