"""Thirty fixed serial fits, only fit indices enter gradients."""
import time,traceback
from common123 import *

def main():
    assert not (O/'training_started.json').exists();verify();configure();assert read(O/'frozen_push.json')['passed'];D=read(O/'design.json');P=read(O/'preflight.json');original_stream=read(S/'preflight.json')['sampling_streams'];fold_stream=read(CV/'preflight.json')['sampling_streams']
    (O/'models').mkdir(exist_ok=False);started=time.perf_counter();save(O/'training_started.json',dict(pid=os.getpid(),unix=time.time(),wall_cap_s=1800));records=[];net=None;opt=None;losses=[]
    try:
        for split,seed,arm in D['training']['fit_order']:
            if time.perf_counter()-started>=1800:raise TimeoutError('Original30fit1800s cap')
            x,y,event,ids,prob=getdata(split);X=torch.from_numpy(x);Y=torch.from_numpy(y);E=torch.from_numpy(event);net=build(seed);initial=digest(net);assert initial==P['original_initial_hashes'][str(seed)]
            opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);rng=np.random.default_rng(20261022+seed);h=hashlib.sha256();counts=np.zeros(len(y),np.int64);losses=[];norms=[];diagnostics=[];name=f'{split}__{arm}_seed{seed}';fs=time.perf_counter()
            for update in range(1,7251):
                if time.perf_counter()-started>=1800:raise TimeoutError('Original30fit1800s cap')
                ix=ids[rng.choice(len(ids),32,replace=True,p=prob)];np.add.at(counts,ix,1);h.update(ix.astype('<i8').tobytes());net.train();q=net(X[ix]);main,aux=components(q,Y[ix],E[ix],arm);loss=main+aux;assert torch.isfinite(loss)
                if update in [1,7250]:diagnostics.append(dict(update=update,**diagnostic_gradients(net,main,aux)))
                opt.zero_grad(set_to_none=True);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters());norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(norm);opt.step();losses.append([float(main.detach()),float(aux.detach())]);norms.append(float(norm))
                if update%2000==0 or update==7250:
                    progress=dict(stage='training',split=split,seed=seed,arm=arm,updates=update,completed_fits=len(records),wall_s=time.perf_counter()-started);save(O/'state.json',progress);print(json.dumps(progress),flush=True)
            key=f'fold{split[1:]}_seed{seed}' if split!='full' else str(seed)
            expected_hash=original_stream[key]['index_sha256'] if split=='full' else fold_stream[key]
            expected=np.load(OLD/f'training_exposure_seed{seed}.npz')['state_draws'] if split=='full' else np.load(CV/f'{key}_expected_exposure.npz')['state_draws']
            assert h.hexdigest()==expected_hash and np.array_equal(counts,expected)
            path=O/'models'/f'{name}.pt';torch.save(dict(split=split,seed=seed,arm=arm,state_dict=net.state_dict(),optimizer=opt.state_dict(),updates=7250,initial_hash=initial,final_hash=digest(net),rng=rng.bit_generator.state),path)
            np.savez_compressed(O/f'exposure_{name}.npz',state_draws=counts,half_main_aux_losses=np.asarray(losses),gradient_norm_before_clip=np.asarray(norms));save(O/f'gradient_components_{name}.json',diagnostics)
            records.append(dict(split=split,seed=seed,arm=arm,updates=7250,parameters=6978,initial_hash=initial,final_hash=digest(net),checkpoint_sha256=sha(path),index_sha256=h.hexdigest(),state_draws=232000,fit_wall_s=time.perf_counter()-fs,gradient_diagnostics_inside_budget=True,clipped_updates=int(np.sum(np.array(norms)>10))))
            save(O/'fit_progress.json',dict(completed=records,wall_s=time.perf_counter()-started))
        save(O/'fits.json',dict(passed=True,new_fits=30,records=records,wall_s=time.perf_counter()-started,validation_and_holdout_predictions_during_fit=0,new_physical_calls=0));verify();save(O/'state.json',dict(stage='all30_fits_complete_awaiting_evaluation',fits=30))
    except BaseException as e:
        if net is not None:torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict() if opt else None,split=split,seed=seed,arm=arm,updates_completed=len(losses)),O/'partial_checkpoint.pt')
        save(O/'training_error.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=records,wall_s=time.perf_counter()-started));raise
if __name__=='__main__':main()
