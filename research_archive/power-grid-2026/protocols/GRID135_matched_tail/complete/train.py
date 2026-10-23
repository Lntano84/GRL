"""Exactly twelve serial final-only fits, then calibrate/freeze before validation."""
from common135 import *
import time,traceback
def main():
    verify();configure();assert read(O/'preflight.json')['passed'];assert (O/'frozen_push.json').exists();assert not (O/'training_started.json').exists()
    D=read(O/'design.json');P=read(O/'preflight.json');T=D['budget'];cache={k:tensor_cache(pickle.loads((O/f'fit_{k}.pkl').read_bytes())) for k in ['FIRST','FULL']}
    z=np.load(O/'fit_targets.npz');y=torch.from_numpy(z['positive'].astype(np.int64));prob=z['probability'];assert len(y)==445
    (O/'models').mkdir(exist_ok=False);start=time.perf_counter();fits=[];save('training_started.json',dict(pid=os.getpid(),planned=12,updates=T['updates'],wall_cap_s=T['wall_cap_s']))
    net=None;opt=None;losses=[]
    try:
        for arm,seed in D['fit_order']:
            net=net_for(arm,seed);assert digest(net)==P['initial_hashes'][str(seed)];opt=torch.optim.Adam(net.parameters(),lr=T['lr'],betas=T['betas'],eps=T['eps'],weight_decay=T['weight_decay'])
            rg=np.random.default_rng(20261022+seed);counts=np.zeros(445,np.int64);h=hashlib.sha256();losses=[];fs=time.perf_counter()
            for update in range(1,T['updates']+1):
                if time.perf_counter()-start>=T['wall_cap_s']:raise TimeoutError('Fixed12fit total wall cap; preserve partial,do not extend')
                ids=rg.choice(445,size=32,replace=True,p=prob);np.add.at(counts,ids,1);h.update(ids.astype('<i8').tobytes());net.train();logits=net(**cache[arm.split('_')[0]](ids))
                loss=torch.nn.functional.cross_entropy(logits,y[ids]);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach()))
                if update%300==0 or update==T['updates']:
                    progress=dict(arm=arm,seed=seed,update=update,mean_recent_loss=float(np.mean(losses[-100:])),completed=len(fits),total_wall_s=time.perf_counter()-start)
                    save('state.json',dict(stage='training',**progress));print(json.dumps(progress),flush=True)
            assert h.hexdigest()==P['sampling_streams'][str(seed)]['sha256'];assert counts.tolist()==P['sampling_streams'][str(seed)]['counts']
            name=f'{arm}_seed{seed}';path=O/'models'/f'{name}.pt';torch.save(dict(arm=arm,seed=seed,state_dict=net.state_dict(),optimizer=opt.state_dict(),updates=T['updates'],initial_hash=digest(net_for(arm,seed)),final_hash=digest(net)),path)
            np.savez_compressed(O/f'exposure_{name}.npz',draws=counts,losses=np.array(losses))
            fits.append(dict(name=name,arm=arm,seed=seed,updates=T['updates'],parameters=5938,index_sha256=h.hexdigest(),initial_hash=P['initial_hashes'][str(seed)],final_hash=digest(net),checkpoint_sha256=sha(path),fit_wall_s=time.perf_counter()-fs,first100_loss=float(np.mean(losses[:100])),last100_loss=float(np.mean(losses[-100:]))))
            save('fit_progress.json',dict(records=fits,completed=len(fits)));print('FIT_DONE '+name,flush=True)
        save('fits.json',dict(passed=True,records=fits,new_fits=12,wall_s=time.perf_counter()-start,no_calibration_or_validation_used_for_fit=True));verify()
        # All fits completed before any calibration outcome readout/new validation scoring.
        cal={k:pickle.loads((O/f'calibration_{k}.pkl').read_bytes()) for k in ['FIRST','FULL']};fitrows={k:pickle.loads((O/f'fit_{k}.pkl').read_bytes()) for k in ['FIRST','FULL']}
        witness=np.load(O/'calibration_targets.npz')['witness'];thresholds={};pred={}
        for rec in fits:
            name=rec['name'];arm=rec['arm'];cp=torch.load(O/'models'/f'{name}.pt',weights_only=False,map_location='cpu');net=net_for(arm,rec['seed']);net.load_state_dict(cp['state_dict'])
            for part,rr in [('fit',fitrows),('calibration',cal)]:pred[name+'__'+part]=predict(net,rr[arm.split('_')[0]])
            ss=no_gain(pred[name+'__calibration']);thresholds[name]=threshold(ss,witness);assert not ((ss>thresholds[name])&witness).any()
        np.savez_compressed(O/'train_calibration_logits.npz',**pred);save('thresholds.json',thresholds)
        freeze={p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').glob('*.pt')),O/'thresholds.json',O/'train_calibration_logits.npz',O/'fits.json']};save('evaluation_freeze.json',freeze)
        save('state.json',dict(stage='all12models_and_cutoffs_frozen_before_validation',fits=12));print('ALL12_FROZEN_BEFORE_VALIDATION',flush=True)
    except BaseException as e:
        if net is not None:torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict() if opt else None,completed_updates=len(losses)),O/'partial_checkpoint.pt')
        save('training_error.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=fits,partial_preserved=True));raise
if __name__=='__main__':main()
