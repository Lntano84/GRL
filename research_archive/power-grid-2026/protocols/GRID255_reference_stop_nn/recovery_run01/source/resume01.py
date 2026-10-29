"""GRID255 explicitly authorized single-window checkpoint continuation, not a new fit."""
from pathlib import Path
import os,sys,time,json,hashlib,subprocess,shutil,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.dont_write_bytecode=True
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid255_reference_stop_nn';P=R/'outputs/grid254_reference_preflight';Q=O/'recovery_qualification01';T=O/'recovery_run01';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID255_reference_stop_nn/recovery_run01'
sys.path.insert(0,str(R/'work/grid254_reference_preflight'));import runtime254 as C

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):C.save(p,v)
def digest_state(sd):
 h=hashlib.sha256()
 for k,v in sd.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def verify():
 for rel,h in read(T/'source_freeze.json').items():assert sha(R/rel)==h,rel
 C.verify()
def publication(files):
 head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','-C',str(REPO),'ls-remote','lntano-src','refs/heads/main'],text=True).split()[0];assert head==remote
 for p in files:assert subprocess.check_output(['git','-C',str(REPO),'show',head+':'+p.relative_to(REPO).as_posix()])==p.read_bytes(),str(p)
 return head

def prepare():
 assert not T.exists() and not A.exists();q=read(Q/'qualification.json');assert q['passed'] and q['remaining_updates_total']==8285;T.mkdir();A.mkdir(parents=True)
 save(T/'design.json',dict(id='GRID255_RECOVERY01',human_authorization='允许120秒恢复窗口',resume_window_s=120,old_used_s=558.1559999999954,cumulative_cap_s=678.1559999999954,workers=6,torch_threads=1,continued_trajectories=6,completed_models_not_retrained=12,additional_updates=8285,final_updates_per_model=7250,model_data_seed_optimizer_unchanged=True,old_stop_and_errors_preserved=True,prefix_sampler_replay_only=True,missing_old_partial_losses_disclosed=True,zero_new_reference_scoring=True,no_automatic_extension=True))
 save(T/'jobs.json',[r for r in q['rows'] if not r['complete']]);paths=[W/'resume01.py',T/'design.json',T/'jobs.json',Q/'qualification.json',Q/'resume_proposal.json',O/'source_freeze.json',O/'shutdown_receipt.json',P/'jobs.json',P/'qualified_freeze.json',P/'source_freeze_recovery01.json']
 paths += [R/rel for rel in read(O/'source_freeze.json')]
 for r in q['rows']:paths += [O/'jobs'/r['node']/('model.pt' if r['complete'] else 'partial_checkpoint.pt')]
 paths += list(Q.glob('*_prefix_exposure.npz'));save(T/'source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in set(paths)})
 verify();d=A/'source';d.mkdir()
 for p in [W/'resume01.py',T/'design.json',T/'jobs.json',T/'source_freeze.json']:shutil.copy2(p,d/p.name)
 save(d/'archive_manifest.json',{p.name:sha(p) for p in d.iterdir() if p.is_file()});print('RESUME_SOURCE_READY_ZERO_UPDATES')

def authorize():
 verify();assert not (T/'authorization.json').exists();head=publication([p for p in (A/'source').iterdir() if p.is_file()]);save(T/'authorization.json',dict(passed=True,human_explicitly_authorized=True,human_reply='允许120秒恢复窗口',window_s=120,models=6,updates=8285,source_published_head=head,source_freeze_sha256=sha(T/'source_freeze.json'),jobs_sha256=sha(T/'jobs.json'),qualification_sha256=sha(Q/'qualification.json')));d=A/'authorization';d.mkdir();shutil.copy2(T/'authorization.json',d/'authorization.json');print('RESUME_AUTH_READY',head)

def worker(name):
 verify();import numpy as np,torch
 j=C.job(name);r=next(r for r in read(T/'jobs.json') if r['node']==name);ctx=read(T/'run_context.json');auth=read(T/'authorization.json');assert auth['passed'] and auth['window_s']==120 and ctx['deadline']-ctx['started_monotonic']==120;out=T/'jobs'/name;C.claim(out/'started.json',dict(node=name,pid=os.getpid(),start_updates=r['checkpoint_updates']));C.before(ctx['deadline'])
 checkpoint=O/'jobs'/name/'partial_checkpoint.pt';assert sha(checkpoint)==r['checkpoint_sha256'];S=C.orig();cp=torch.load(checkpoint,map_location='cpu',weights_only=False);z,rows=C.load_public(j);take=S.G.old_api().tensor_cache(rows);ids=z['fit_ids'];prob=z['fit_probability'];target=z['target_fit'];yt=torch.tensor(np.column_stack([np.zeros_like(target),target]));net=S.build(j['seed'],'FACTOR');assert S.digest(net)==j['initial_hash'];net.load_state_dict(cp['state_dict'],strict=True);assert S.digest(net)==r['final_weight_hash'];opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);opt.load_state_dict(cp['optimizer']);start_step=cp['updates'];assert start_step==r['checkpoint_updates'];assert all(int(v['step'])==start_step for v in opt.state.values());assert torch.get_num_threads()==1
 rng=np.random.default_rng(20261022+j['seed']);counts=np.zeros(3582,np.int64);stream=hashlib.sha256()
 for _ in range(start_step):
  C.before(ctx['deadline']);ix=rng.choice(len(ids),32,replace=True,p=prob);gi=ids[ix];np.add.at(counts,gi,1);stream.update(gi.astype('<i8').tobytes())
 assert rng.bit_generator.state==cp['rng_state'];assert np.array_equal(counts,np.load(Q/(name+'_prefix_exposure.npz'))['state_draws']);rng.bit_generator.state=cp['rng_state'];torch.set_rng_state(cp['torch_rng']);assert torch.equal(torch.get_rng_state(),cp['torch_rng']);net.train();last=start_step;losses=[];fs=time.monotonic();save(out/'restored.json',dict(passed=True,checkpoint_sha256=sha(checkpoint),restored_weight_hash=S.digest(net),original_initial_hash=j['initial_hash'],restored_updates=start_step,restored_adam_steps=True,restored_numpy_torch_rng=True,prefix_replay_was_sampler_only=True))
 def write_checkpoint(path):
  tmp=path.with_suffix('.tmp');torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),seed=j['seed'],pair=j['pair'],updates=last,rng_state=rng.bit_generator.state,torch_rng=torch.get_rng_state(),continued_from_sha256=r['checkpoint_sha256'],continued_from_updates=start_step),tmp);tmp.replace(path)
 try:
  for update in range(start_step+1,7251):
   C.before(ctx['deadline'])
   if (T/'STOP_REQUESTED').exists():raise InterruptedError('User stop requested during authorized recovery.')
   ix=rng.choice(len(ids),32,replace=True,p=prob);gi=ids[ix];stream.update(gi.astype('<i8').tobytes());np.add.at(counts,gi,1);pred=net(**take(ix));loss=torch.nn.functional.smooth_l1_loss(pred,yt[ix],beta=1.);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(gn);opt.step();last=update;losses.append(float(loss.detach()))
   if update%500==0 or update==7250:write_checkpoint(out/'partial_checkpoint.pt');save(out/'progress.json',dict(node=name,update=update,new_updates=last-start_step,resume_elapsed_s=time.monotonic()-ctx['started_monotonic']))
  assert np.array_equal(counts,np.load(P/j['expected_exposure'])['state_draws']) and stream.hexdigest()==j['sampling_hash'] and counts[z['score_ids']].sum()==0;assert not net.base.q_net[0].weight[:,41].detach().count_nonzero();C.before(ctx['deadline']);write_checkpoint(out/'model.pt');np.savez_compressed(out/'exposure.npz',state_draws=counts,continuation_losses=np.array(losses),loss_first_update=start_step+1);C.before(ctx['deadline']);save(out/'complete.json',dict(passed=True,node=name,seed=j['seed'],pair=j['pair'],updates=last,new_updates=last-start_step,start_updates=start_step,initial_hash=j['initial_hash'],restored_hash=r['final_weight_hash'],final_hash=S.digest(net),checkpoint_sha256=sha(out/'model.pt'),sampling_hash=stream.hexdigest(),excluded_draws=0,resume_elapsed_s=time.monotonic()-ctx['started_monotonic'],finished_monotonic=time.monotonic(),continued_fit_wall_s=time.monotonic()-fs,prefix_losses_not_saved=True))
 except BaseException as e:
  write_checkpoint(out/'partial_checkpoint.pt');np.savez_compressed(out/'partial_exposure.npz',state_draws=counts,continuation_losses=np.array(losses));save(out/'error.json',dict(error=repr(e),updates_completed=last,traceback=traceback.format_exc()));raise

def manager():
 verify();auth=read(T/'authorization.json');assert auth['human_explicitly_authorized'] and auth['window_s']==120 and auth['source_freeze_sha256']==sha(T/'source_freeze.json') and auth['jobs_sha256']==sha(T/'jobs.json');assert not (T/'run_context.json').exists();head=publication([A/'authorization/authorization.json',A/'source/resume01.py',A/'source/source_freeze.json']);assert (T/'authorization.json').read_bytes()==(A/'authorization/authorization.json').read_bytes();C.claim(T/'started.json',dict(pid=os.getpid(),published_head=head,one_resume_only=True));start=time.monotonic();ctx=dict(started_monotonic=start,deadline=start+120,max_workers=6,nodes=[r['node'] for r in read(T/'jobs.json')]);save(T/'run_context.json',ctx);active=[];completed=[];error=None
 try:
  for r in read(T/'jobs.json'):
   C.before(ctx['deadline']);out=T/'jobs'/r['node'];out.mkdir(parents=True);fo=(out/'stdout.log').open('wb');fe=(out/'stderr.log').open('wb');p=subprocess.Popen([sys.executable,'-X','utf8','-u',str(W/'resume01.py'),'worker',r['node']],stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);fo.close();fe.close();active.append((p,r['node']))
  while active:
   C.before(ctx['deadline'])
   if (T/'STOP_REQUESTED').exists():raise InterruptedError('User stop requested; no extension.')
   for p,name in list(active):
    if p.poll() is None:continue
    active.remove((p,name));assert p.returncode==0,f'Worker {name} failed exit={p.returncode}';assert read(T/'jobs'/name/'complete.json')['passed'];completed.append(name)
   if active:time.sleep(.05)
 except BaseException as e:error=repr(e)
 finally:
  for p,name in active:
   if p.poll() is None:p.terminate()
  for p,name in active:
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
 result=dict(passed=error is None and len(completed)==6,error=error,completed=completed,all_workers_exited=all(p.poll() is not None for p,name in active),resume_wall_s=time.monotonic()-start,original_used_s=558.1559999999954,cumulative_actual_s=558.1559999999954+time.monotonic()-start,authorized_resume_window_s=120,new_reference_forward=0)
 save(T/('finished.json' if result['passed'] else 'error.json'),result);print('RECOVERY_MANAGER_FINISHED',result['passed']);sys.exit(0 if result['passed'] else 1)

if __name__=='__main__':
 try:
  mode=sys.argv[1]
  if mode=='worker':worker(sys.argv[2])
  else:globals()[mode]()
 except Exception as e:
  if T.exists():save(T/('entry_'+sys.argv[1]+'_error.json'),dict(error=repr(e),traceback=traceback.format_exc()))
  raise
