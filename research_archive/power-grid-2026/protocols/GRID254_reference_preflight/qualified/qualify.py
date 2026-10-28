"""Zero-optimizer qualification of datasets, six-worker resources and lifecycle."""
from pathlib import Path
import sys,os,time,threading,subprocess,json,pickle,shutil
import numpy as np
import runtime254 as C
import manager as M
from prepare import archive

def lifecycle_tests():
 qa=C.O/'lifecycle';qa.mkdir(exist_ok=False);C.claim(qa/'claim.json',dict(first=True))
 try:C.claim(qa/'claim.json',dict(second=True));raise AssertionError('Duplicate claim accepted')
 except FileExistsError:pass
 # Expired deadline launches nothing; this exercises the same pool check as training.
 specs=[dict(node=f'fake{i}',args=['fake',qa/'deadline'/f'fake{i}'],logdir=qa/'deadline'/f'fake{i}') for i in range(9)];expired=M.run_pool(specs,time.monotonic()-1,qa/'none','fake');assert not expired['passed'] and not expired['launched']
 timed=M.run_pool(specs,time.monotonic()+1.5,qa/'none','fake');assert not timed['passed'] and timed['all_workers_exited'] and len(timed['launched'])==6 and all((qa/'deadline'/f'fake{i}'/'partial.json').exists() for i in range(6))
 stop=qa/'stop_fixture'/'STOP_REQUESTED';specs=[dict(node=f'fake{i}',args=['fake',qa/'stop_fixture'/f'fake{i}'],logdir=qa/'stop_fixture'/f'fake{i}') for i in range(9)]
 def request_stop():
  limit=time.monotonic()+10
  while not all((qa/'stop_fixture'/f'fake{i}'/'partial.json').exists() for i in range(6)):C.before(limit);time.sleep(.02)
  C.save(stop,dict(synthetic=True))
 t=threading.Thread(target=request_stop);t.start();stopped=M.run_pool(specs,time.monotonic()+15,stop,'fake');t.join();assert not stopped['passed'] and stopped['all_workers_exited'] and len(stopped['launched'])==6 and len(stopped['pending'])==3 and 'InterruptedError' in stopped['error']
 # Intentional denial: future training has no authorization or qualification yet.
 p=M.launch(['train',C.read(C.O/'jobs.json')[0]['node'],C.TRAIN_OUT],qa/'authorization_guard');code=p.wait(timeout=20);assert code!=0 and not C.TRAIN_OUT.exists()
 result=dict(passed=True,atomic_duplicate_rejected=True,expired_deadline=expired,live_deadline=timed,user_stop=stopped,expected_training_denial_exit=code,no_real_NNfits=True);C.save(qa/'tests.json',result);return result

def main():
 C.verify();assert not (C.O/'qualification_started.json').exists() and not C.TRAIN_OUT.exists();head=subprocess.check_output(['git','-C',str(C.REPO),'rev-parse','HEAD']).decode().strip();remote=subprocess.check_output(['git','-C',str(C.REPO),'ls-remote','lntano-src','refs/heads/main']).decode().split()[0];assert remote==head;blob=subprocess.check_output(['git','-C',str(C.REPO),'show',head+':'+(C.A/'source/source_freeze.json').relative_to(C.REPO).as_posix()]);assert blob==(C.O/'source_freeze.json').read_bytes();C.save(C.O/'qualification_started.json',dict(sourcepublished=head,optimizer_instances=0,training_disabled=True));jobs=C.read(C.O/'jobs.json');meta=C.read(C.R/'outputs/grid113_relational_learning/train_metadata.json');src=np.load(C.R/'outputs/grid121_date_transfer/train_arrays.npz');raw=src['raw'];y=src['target'];S=C.orig();public=pickle.loads((C.R/'outputs/grid113_relational_learning/train_public.pkl').read_bytes());packages={};data_checks=[]
 for j in jobs:
  z=np.load(C.O/j['package']);fi,hi=z['fit_ids'],z['score_ids'];assert len(set(meta[int(i)]['date'] for i in fi))==8 and {meta[int(i)]['date'] for i in fi}.isdisjoint(j['score_dates']);assert np.isfinite(z['normalized_fit']).all() and np.array_equal(y[fi],z['target_fit']);expected=np.load(C.O/j['expected_exposure'])['state_draws'];assert expected[hi].sum()==0 and expected.sum()==232000
  if j['package'] not in packages:
   mean,scale=S.normalize(raw,meta,fi);assert np.array_equal(mean,z['mean']) and np.array_equal(scale,z['scale']);poison=raw.copy();poison[hi]=np.nan;py=y.copy();py[hi]=np.nan;pm,ps=S.normalize(poison,meta,fi);assert np.array_equal(pm,mean) and np.array_equal(ps,scale) and np.array_equal(py[fi],z['target_fit'])
   n=max(len(public[int(i)]['context_action']) for i in fi);m=max(len(public[int(i)]['next_action']) for i in fi);estimated=len(fi)*(n*76*4+m*68*4+n+m+(68+41+1)*4);packages[j['package']]=estimated
  data_checks.append(dict(node=j['node'],fit_dates8=True,no_other8date_optimizerdraws=True,ownfit_targets_only=True,normalization_unchanged_if_excluded_values_poisoned=True))
 C.save(C.O/'input_checks.json',data_checks);tests=lifecycle_tests();ordered=sorted(jobs,key=lambda j:(-packages[j['package']],j['node']));probes=[];resources=[]
 for wave in range(3):
  batch=ordered[wave*6:(wave+1)*6];release=C.O/'worker_probes'/f'release_wave{wave}';active=[];started=time.monotonic();before=C.memory()
  try:
   for j in batch:
    out=C.O/'worker_probes'/j['node'];p=M.launch(['preflight',j['node'],out,release],out);active.append((p,j,out))
   limit=time.monotonic()+120
   while not all((out/'ready.json').exists() for _,_,out in active):
    C.before(limit)
    for p,j,out in active:
     if p.poll() is not None and p.returncode!=0:raise RuntimeError('Preflight worker failed: '+j['node'])
    time.sleep(.1)
   current=C.memory();assert current['logical_cpus']>=6;reserve=1024**3+6*1024**2;assert current['available_physical']>=reserve,'Six-worker memory reserve not qualified; preserve records, do not start training.';rr=[C.read(out/'ready.json') for _,_,out in active];assert all(r['optimizer_instances']==r['optimizer_steps']==0 and r['weights_unchanged'] and r['torch_threads']==1 for r in rr);resources.append(dict(wave=wave,nodes=[j['node'] for j in batch],before=before,all6resident=current,reserve_required=reserve,sum_peak_worker_workingset=sum(r['memory']['peak_working_set'] for r in rr),note='Working sets may share DLL pages; global availableRAM is the admission check.'));C.save(release,dict(released=True))
   for p,j,out in active:assert p.wait(timeout=30)==0;probes.append(C.read(out/'complete.json'))
  finally:M.stop_processes([(p,j) for p,j,out in active])
 C.save(C.O/'worker_checks.json',probes);C.save(C.O/'resource_checks.json',resources);assert len(probes)==18
 # Generic parent interface is checked only on own-fit dates, never with a reference NN.
 from parent_interface import extract_original,SOURCES
 parent=[];qualified={r['pool']:r for r in C.read(C.R/'outputs/grid241_parent_prior_plan/qualified_pools.json')}
 for f,root in enumerate(SOURCES):
  for seed in [0,1,2]:
   pool=f'f{f}_s{seed}';z=np.load(C.R/f'outputs/grid192_gate_support/fit_features/{pool}.npz');origstop=np.arange(len(z['ids']))<int(z['original_STOP_count']);fitdates=set(qualified[pool]['fit_dates']);eligible=np.array([meta[int(i)]['date'] in fitdates for i in z['ids']]);pick=[]
   for role in [True,False]:
    options=np.flatnonzero(eligible&(origstop==role));assert len(options);k=int(options[np.argmax(abs(z['x'][options,81]-z['x'][options,80]))]);pick.append(int(z['ids'][k]))
   arrays=np.load(root/'fold_arrays.npz');published=np.load(root/'state_scores.npz')[f'FACTOR_seed{seed}'];missing=[int(i) for i in arrays['fit_ids'] if meta[int(i)]['date'] in fitdates and int(i) not in set(z['ids'].tolist())]
   if missing:pick.append(max(missing,key=lambda i:abs(float(published[i,1]-published[i,0]))))
   out,proof=extract_original(f,seed,pick);lookup={int(i):k for k,i in enumerate(z['ids'])}
   for k,i in enumerate(pick[:2]):assert np.allclose(out[k],z['x'][lookup[i]],atol=3e-5,rtol=3e-5)
   proof['oldcached_rows_reproduced']=2;proof['missing_parent_interface_tested']=len(pick)>2;proof['only_fit8_public_rows']=True;parent.append(proof)
 C.save(C.O/'parent_interface_checks.json',parent);C.verify();passed=dict(passed=True,execution_qualified=True,optimizer_instances=0,optimizer_steps=0,new_trained_NN=0,new_reference_unseen_forward=0,new_policy_scores=0,fit_side_synthetic_forward_rows=18*32,synthetic_backwards=18,original_parent_fit_only_forward_rows=sum(r['rows'] for r in parent),six_input_packages=True,eighteen_streams_andinitialhashes=True,all18_initialweights_unchanged=True,six_concurrent_resource_waves=3,all_worker_processes_exited=True,lifecycle_deadline_stop_dedup_qualified=True,expected_authorization_guard_denial=True,reference_fits_next=18,global_fit_window_s=600,max_workers=6,readout_laterfits_max=22,complete_pipeline_fits_max=40);C.save(C.O/'qualification.json',passed);paths=[*list((C.O/'pairs').iterdir()),*list((C.O/'expected').iterdir()),C.O/'source_freeze.json',C.O/'jobs.json',C.O/'qualification.json',C.O/'input_checks.json',C.O/'worker_checks.json',C.O/'resource_checks.json',C.O/'parent_interface_checks.json',C.O/'lifecycle/tests.json'];C.save(C.O/'qualified_freeze.json',{p.relative_to(C.O).as_posix():C.sha(p) for p in paths});C.save(C.O/'complete.json',dict(passed=True,qualification_only=True,optimizer_instances=0,optimizer_steps=0,new_trained_NN=0,NN_reference18_not_started=True));archive('qualified');print('GRID254_EXECUTION_QUALIFIED',json.dumps(passed));print('MEMORY_MIN_AVAILABLE',min(r['all6resident']['available_physical'] for r in resources))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback;C.save(C.O/'qualification_error.json',dict(error=repr(e),traceback=traceback.format_exc(),optimizer_instances=0,optimizer_steps=0));raise
