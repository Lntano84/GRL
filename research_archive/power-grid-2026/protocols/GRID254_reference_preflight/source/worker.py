"""One immutable job: preflight has no optimizer; training requires future authorization."""
from pathlib import Path
import sys,os,time,hashlib,traceback
import runtime254 as C
import numpy as np

def fake(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);C.save(out/'partial.json',dict(pid=os.getpid(),fake=True,optimizer_steps=0))
 while True:time.sleep(.05)

def preflight(name,out,release):
 out=Path(out);C.claim(out/'started.json',dict(mode='preflight',pid=os.getpid(),optimizer_instances=0));C.verify();j=C.job(name);S=C.orig();import torch
 def denied(*a,**kw):raise AssertionError('Optimizer construction forbidden during GRID254 qualification.')
 torch.optim.Adam=denied;z,rows=C.load_public(j);take=S.G.old_api().tensor_cache(rows);net=S.build(j['seed'],'FACTOR');before=S.digest(net);assert before==j['initial_hash'];assert sum(p.numel() for p in net.parameters())==12980
 # Fixed label-free worst-shape engineering batch; actual training uses all weighted draws.
 order=sorted(range(len(rows)),key=lambda k:(-len(rows[k]['context_action']),-len(rows[k]['next_action']),k));ix=order[:32];mq=max(range(len(rows)),key=lambda k:len(rows[k]['next_action']))
 if mq not in ix:ix[-1]=mq
 net.eval();q=net(**take(np.array(ix)));assert q.shape==(32,2) and torch.isfinite(q).all();loss=torch.nn.functional.smooth_l1_loss(q,torch.zeros_like(q),beta=1.);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(gn);assert S.digest(net)==before
 info=dict(passed=True,node=name,pid=os.getpid(),parameters=12980,initial_hash=before,post_probe_hash=S.digest(net),weights_unchanged=True,synthetic_public_fit_forward_rows=32,synthetic_backwards=1,optimizer_instances=0,optimizer_steps=0,training_labels_loaded=False,fit_rows=len(rows),memory=C.process_memory(),torch_threads=torch.get_num_threads(),public_probe_ids=z['fit_ids'][ix].astype(int).tolist());C.save(out/'ready.json',info)
 limit=time.monotonic()+120
 while not Path(release).exists():C.before(limit);time.sleep(.05)
 C.save(out/'complete.json',info)

def train(name,root):
 j=C.job(name);ctx=C.validate_training(root,j);C.verify();root=Path(root);out=root/'jobs'/name;C.claim(out/'started.json',dict(mode='train',node=name,pid=os.getpid(),deadline=ctx['deadline']));C.before(ctx['deadline']);S=C.orig();import torch
 z,rows=C.load_public(j);take=S.G.old_api().tensor_cache(rows);ids=z['fit_ids'];prob=z['fit_probability'];target=z['target_fit'];yt=torch.tensor(np.column_stack([np.zeros_like(target),target]));net=S.build(j['seed'],'FACTOR');initial=S.digest(net);assert initial==j['initial_hash'];assert sum(p.numel() for p in net.parameters())==12980;C.before(ctx['deadline']);opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);rng=np.random.default_rng(20261022+j['seed']);counts=np.zeros(3582,np.int64);stream=hashlib.sha256();losses=[];last=0;start=time.monotonic();net.train()
 def checkpoint(path):
  tmp=path.with_suffix('.tmp');torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),seed=j['seed'],pair=j['pair'],updates=last,rng_state=rng.bit_generator.state,torch_rng=torch.get_rng_state()),tmp);tmp.replace(path)
 try:
  for update in range(1,7251):
   C.before(ctx['deadline'])
   if (root/'STOP_REQUESTED').exists():raise InterruptedError('User stop requested; preserve partial model.')
   ix=rng.choice(len(ids),32,replace=True,p=prob);globalix=ids[ix];stream.update(globalix.astype('<i8').tobytes());np.add.at(counts,globalix,1);q=net(**take(ix));loss=torch.nn.functional.smooth_l1_loss(q,yt[ix],beta=1.);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(gn);opt.step();last=update;losses.append(float(loss.detach()))
   if update%500==0 or update==7250:
    checkpoint(out/'partial_checkpoint.pt');C.save(out/'progress.json',dict(node=name,update=update,fit_wall_s=time.monotonic()-start,global_elapsed_s=time.monotonic()-ctx['started_monotonic'],mean_last_loss=float(np.mean(losses[-500:]))))
  exp=np.load(C.O/j['expected_exposure']);assert np.array_equal(counts,exp['state_draws']) and stream.hexdigest()==j['sampling_hash'];assert counts[z['score_ids']].sum()==0;assert not net.base.q_net[0].weight[:,41].detach().count_nonzero();C.before(ctx['deadline']);checkpoint(out/'model.pt');np.savez_compressed(out/'exposure.npz',state_draws=counts,losses=np.array(losses));C.before(ctx['deadline']);C.save(out/'complete.json',dict(passed=True,node=name,seed=j['seed'],pair=j['pair'],updates=last,optimizer_instances=1,optimizer_steps=last,parameters=12980,initial_hash=initial,final_hash=S.digest(net),checkpoint_sha256=C.sha(out/'model.pt'),sampling_hash=stream.hexdigest(),excluded_optimizer_draws=int(counts[z['score_ids']].sum()),fit_wall_s=time.monotonic()-start,finished_monotonic=time.monotonic(),global_elapsed_s=time.monotonic()-ctx['started_monotonic']))
 except BaseException as e:
  checkpoint(out/'partial_checkpoint.pt');np.savez_compressed(out/'partial_exposure.npz',state_draws=counts,losses=np.array(losses));C.save(out/'error.json',dict(error=repr(e),traceback=traceback.format_exc(),updates_completed=last,optimizer_steps=last));raise

if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='fake':fake(sys.argv[2])
 elif mode=='preflight':preflight(sys.argv[2],sys.argv[3],sys.argv[4])
 elif mode=='train':train(sys.argv[2],sys.argv[3])
 else:raise ValueError(mode)
