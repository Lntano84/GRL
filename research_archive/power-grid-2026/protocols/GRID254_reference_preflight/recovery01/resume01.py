"""Resume only incomplete resource/interface qualification after a Win32 pointer fix."""
from pathlib import Path
import json,pickle,subprocess,time,sys
import numpy as np
import runtime254 as C
import manager as M
from prepare import archive

def main():
 assert not (C.O/'complete.json').exists() and not C.TRAIN_OUT.exists() and not (C.O/'qualification_resume01_started.json').exists();C.verify();repair=C.read(C.O/'engineering_recovery01.json');assert repair['passed'] and repair['optimizer_instances']==0;assert C.sha(C.O/'input_checks.json')==repair['reused_input_checks_sha256'] and C.sha(C.O/'lifecycle/tests.json')==repair['reused_lifecycle_tests_sha256'];assert C.read(C.O/'lifecycle/tests.json')['passed'];head=subprocess.check_output(['git','-C',str(C.REPO),'rev-parse','HEAD']).decode().strip();assert subprocess.check_output(['git','-C',str(C.REPO),'ls-remote','lntano-src','refs/heads/main']).decode().split()[0]==head;assert subprocess.check_output(['git','-C',str(C.REPO),'show',head+':'+(C.A/'recovery01/source_freeze_recovery01.json').relative_to(C.REPO).as_posix()])==(C.O/'source_freeze_recovery01.json').read_bytes();C.save(C.O/'qualification_resume01_started.json',dict(repair_published=head,optimizer_instances=0,complete_input_lifecycle_checks_reused=True));jobs=C.read(C.O/'jobs.json');meta=C.read(C.R/'outputs/grid113_relational_learning/train_metadata.json');public=pickle.loads((C.R/'outputs/grid113_relational_learning/train_public.pkl').read_bytes());packages={}
 for j in jobs:
  if j['package'] in packages:continue
  z=np.load(C.O/j['package']);fi=z['fit_ids'];n=max(len(public[int(i)]['context_action']) for i in fi);m=max(len(public[int(i)]['next_action']) for i in fi);packages[j['package']]=len(fi)*(n*76*4+m*68*4+n+m+(68+41+1)*4)
 ordered=sorted(jobs,key=lambda j:(-packages[j['package']],j['node']));probes=[];resources=[]
 for wave in range(3):
  batch=ordered[wave*6:(wave+1)*6];release=C.O/'worker_probes'/f'release_wave{wave}';active=[];before=C.memory()
  try:
   for j in batch:
    out=C.O/'worker_probes'/j['node'];p=M.launch(['preflight',j['node'],out,release],out);active.append((p,j,out))
   limit=time.monotonic()+120
   while not all((out/'ready.json').exists() for _,_,out in active):
    C.before(limit)
    for p,j,out in active:
     if p.poll() is not None and p.returncode!=0:raise RuntimeError('Preflight worker failed: '+j['node'])
    time.sleep(.1)
   current=C.memory();reserve=1024**3+6*1024**2;assert current['logical_cpus']>=6 and current['available_physical']>=reserve,'Six-worker RAM reserve not qualified; preserve records, do not train.';rr=[C.read(out/'ready.json') for _,_,out in active];assert all(r['optimizer_instances']==r['optimizer_steps']==0 and r['weights_unchanged'] and r['torch_threads']==1 for r in rr);resources.append(dict(wave=wave,nodes=[j['node'] for j in batch],before=before,all6resident=current,reserve_required=reserve,sum_peak_worker_workingset=sum(r['memory']['peak_working_set'] for r in rr)));C.save(C.O/'resource_checks_partial.json',resources);C.save(release,dict(released=True))
   for p,j,out in active:assert p.wait(timeout=30)==0;probes.append(C.read(out/'complete.json'))
  finally:M.stop_processes([(p,j) for p,j,out in active])
 C.save(C.O/'worker_checks.json',probes);C.save(C.O/'resource_checks.json',resources);assert len(probes)==18
 from parent_interface import extract_original,SOURCES
 parent=[];qualified={r['pool']:r for r in C.read(C.R/'outputs/grid241_parent_prior_plan/qualified_pools.json')}
 for f,root in enumerate(SOURCES):
  for seed in [0,1,2]:
   pool=f'f{f}_s{seed}';z=np.load(C.R/f'outputs/grid192_gate_support/fit_features/{pool}.npz');origstop=np.arange(len(z['ids']))<int(z['original_STOP_count']);fitdates=set(qualified[pool]['fit_dates']);eligible=np.array([meta[int(i)]['date'] in fitdates for i in z['ids']]);pick=[]
   for role in [True,False]:
    options=np.flatnonzero(eligible&(origstop==role));assert len(options);k=int(options[np.argmax(abs(z['x'][options,81]-z['x'][options,80]))]);pick.append(int(z['ids'][k]))
   arrays=np.load(root/'fold_arrays.npz');published=np.load(root/'state_scores.npz')[f'FACTOR_seed{seed}'];known=set(z['ids'].tolist());missing=[int(i) for i in arrays['fit_ids'] if meta[int(i)]['date'] in fitdates and int(i) not in known]
   if missing:pick.append(max(missing,key=lambda i:abs(float(published[i,1]-published[i,0]))))
   out,proof=extract_original(f,seed,pick);lookup={int(i):k for k,i in enumerate(z['ids'])}
   for k,i in enumerate(pick[:2]):assert np.allclose(out[k],z['x'][lookup[i]],atol=3e-5,rtol=3e-5)
   proof.update(oldcached_rows_reproduced=2,missing_parent_interface_tested=len(pick)>2,only_fit8_public_rows=True);parent.append(proof)
 C.save(C.O/'parent_interface_checks.json',parent);C.verify();passed=dict(passed=True,execution_qualified=True,optimizer_instances=0,optimizer_steps=0,new_trained_NN=0,new_reference_unseen_forward=0,new_policy_scores=0,fit_side_synthetic_forward_rows_successful=18*32,synthetic_backwards_successful=18,earlier_interrupted_synthetic_probe_upperbound=6,original_parent_fit_only_forward_rows=sum(r['rows'] for r in parent),six_input_packages=True,eighteen_streams_andinitialhashes=True,all18_initialweights_unchanged=True,six_concurrent_resource_waves=3,all_workers_exited=True,input_and_lifecycle_checks_reused_after_nonnumerical_memory_API_fix=True,expected_authorization_guard_denial=True,recovery01_passed=True,reference_fits_next=18,global_fit_window_s=600,max_workers=6,readout_laterfits_max=22,complete_pipeline_fits_max=40);C.save(C.O/'qualification.json',passed);paths=[*list((C.O/'pairs').iterdir()),*list((C.O/'expected').iterdir()),C.O/'source_freeze.json',C.O/'source_freeze_recovery01.json',C.O/'engineering_recovery01.json',C.O/'jobs.json',C.O/'qualification.json',C.O/'input_checks.json',C.O/'worker_checks.json',C.O/'resource_checks.json',C.O/'parent_interface_checks.json',C.O/'lifecycle/tests.json'];C.save(C.O/'qualified_freeze.json',{p.relative_to(C.O).as_posix():C.sha(p) for p in paths});C.save(C.O/'complete.json',dict(passed=True,qualification_only=True,optimizer_instances=0,optimizer_steps=0,new_trained_NN=0,NN_reference18_not_started=True,recovery01_passed=True));archive('qualified');print('GRID254_EXECUTION_QUALIFIED',json.dumps(passed));print('MEMORY_MIN_AVAILABLE',min(r['all6resident']['available_physical'] for r in resources))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback;C.save(C.O/'qualification_resume01_error.json',dict(error=repr(e),traceback=traceback.format_exc(),optimizer_instances=0,optimizer_steps=0));raise
