"""Seal two losses and two budgets before fitting; reuse all old Huber anchors."""
import copy,datetime,inspect,shutil
import numpy as np,torch
from stable_baselines3 import DQN
from common108 import ROOT,W,OUT,ARCHIVE,read,save,sha,sealed,carrier,weight_digest
def main():
    assert not OUT.exists();sealed();source=ROOT/'outputs/grid107_direct_value'
    assert read(source/'complete.json')['passed'] and read(source/'review.json')['passed']
    code=read(source/'code_freeze.json');data=read(source/'data_freeze.json')
    for p,h in {**code,**data}.items():assert sha(ROOT/p)==h,p
    old=read(source/'design.json');D=copy.deepcopy(old)
    D.update(id='GRID108_LOSS_X_BUDGET',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User: continue investigating training sufficiency and loss. Notified fixed2x2 with no data/network/physical expansion.',
        losses=['HUBER','MSE'],budgets=[7250,29000],seeds=[0,1,2],
        source_train='outputs/grid107_direct_value/train.npz',source_training_rows='outputs/grid107_direct_value/training_rows.json',
        experiment='2losses x2 fixed update budgets. Huber7250 existing checkpoints reused.3 Huber continuations plus3 fresh MSE paths, all seeds retained.',
        caps=dict(fitting_wall_s=600,one_path_s=150,new_training_paths=6,new_saved_checkpoints=9),
        loss_definition=dict(HUBER='torch smooth_l1_loss beta1, mean over32x2 targets',MSE='torch mse_loss, mean over32x2 targets; unchanged optimizer and clip10'),
        matching='Identical initial net, Adam, LR, targets, feature normalization, minibatch indices and scalar-label budget at each matched update. Huber resumes Adam and exact next minibatch from old7250.',
        budget_definition='7250 and fixed4x29000 chosen before new evaluation based on still-decreasing prior TRAIN loss; no validation-selected duration.',
        interpretation='Paired loss effects at each budget, paired budget effects at each loss, all3 seed means/dates versus DQN and BATCH1. No claiming convergence at cap or selecting winning configuration.',
        limits='No larger network, new observations/scenarios, positive oversampling, fee/threshold tuning, final8 test access, DQN refits, physical closed-loop or automatic additional experiments.')
    D['evaluation']['methods']=[f'{loss}_{budget}_{seed}' for loss in D['losses'] for budget in D['budgets'] for seed in D['seeds']]+[f'DQN0_{s}' for s in D['seeds']]+['RF','STOP','FULL_LOCAL','BATCH1']
    D['training'].update(updates=D['budgets'],loss='Two prespecified loss functions in loss_definition.',
        fitting='3 exact Huber continuations and3 fresh MSE paths; both endpoints retained, original3 Huber short anchors reused.',
        intentionally_changed='Loss only in matched-budget pairs; number of updates only in same-loss pairs. All other inputs and sample streams fixed.')
    D['evaluation']['interpretation']='Report paired loss/budget effects, all3 seed family means and per-date reversals versus fixedBATCH1. No selecting best configuration or convergence claim at cap.'
    models=[]
    for loss in D['losses']:
        for budget in D['budgets']:
            for seed in D['seeds']:
                reused=loss=='HUBER' and budget==7250
                p=f'outputs/grid107_direct_value/models/direct_seed{seed}_final.zip' if reused else f'outputs/grid108_loss_budget/models/{loss}_{budget}_{seed}.zip'
                models.append(dict(method=f'{loss}_{budget}_{seed}',loss=loss,updates=budget,seed=seed,path=p,reused=reused))
    D['models']=models
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with np.load(ROOT/D['source_train']) as z:prob=z['sampling_probability'];target=z['target']
    checks=[]
    for seed in D['seeds']:
        p=source/f'models/direct_seed{seed}_final.zip';m=DQN.load(p,device='cpu')
        assert m._n_updates==7250 and m.num_timesteps==0
        assert not any(isinstance(layer,(torch.nn.Dropout,torch.nn.BatchNorm1d)) for layer in m.q_net.modules())
        assert all(float(v['step'])==7250 for v in m.policy.optimizer.state.values())
        assert weight_digest(m.q_net)==read(source/'fits.json')['records'][seed]['final_q_hash']
        rng=np.random.default_rng(old['training']['sampler_seed_offset']+seed);counts=np.zeros(len(prob),dtype=np.int64)
        for _ in range(7250):np.add.at(counts,rng.choice(len(prob),32,replace=True,p=prob),1)
        with np.load(source/f'training_exposure_seed{seed}.npz') as z:assert np.array_equal(counts,z['state_draws'])
        initial=carrier(seed,D['dqn_initialization']);assert weight_digest(initial.q_net)==read(source/'preflight.json')['initial_q_hashes'][str(seed)]
        checks.append(dict(seed=seed,optimizer_resume_step=7250,exact_old_draw_counts=True,matched_initialization=True));initial.env.close()
        code[p.relative_to(ROOT).as_posix()]=sha(p)
    # Qualify loss formulas, including their different treatment of large residuals.
    a=torch.tensor([[0.,-2.],[.1,.5]]);b=torch.tensor([[0.,3.],[0.,-.5]])
    diff=(a-b).numpy()
    assert np.isclose(float(torch.nn.functional.mse_loss(a,b)),np.mean(diff**2))
    assert np.isclose(float(torch.nn.functional.smooth_l1_loss(a,b)),np.mean(np.where(np.abs(diff)<1,.5*diff**2,np.abs(diff)-.5)))
    OUT.mkdir();save(OUT/'design.json',D)
    shutil.copy2(source/'probe_sources.json',OUT/'probe_sources.json')
    save(OUT/'preflight.json',dict(passed=True,optimizer_rng_resume_checks=checks,loss_formulas_checked=True,
        train_states=len(target),positive_targets=int(np.sum(target>0)),new_fits=0,final_test_materialized=False))
    data[(source/'train.npz').relative_to(ROOT).as_posix()]=sha(source/'train.npz')
    data[(source/'training_rows.json').relative_to(ROOT).as_posix()]=sha(source/'training_rows.json')
    for name in ['validation_results.json','validation_trajectories.json','preflight.json','fits.json']:
        p=source/name;data[p.relative_to(ROOT).as_posix()]=sha(p)
    for seed in D['seeds']:
        p=source/f'training_exposure_seed{seed}.npz';data[p.relative_to(ROOT).as_posix()]=sha(p)
    data[(OUT/'probe_sources.json').relative_to(ROOT).as_posix()]=sha(OUT/'probe_sources.json')
    for p in list(W.glob('*.py'))+[OUT/'design.json',OUT/'preflight.json']:
        code[p.relative_to(ROOT).as_posix()]=sha(p)
    p=__import__('pathlib').Path(inspect.getsourcefile(torch.nn.functional.smooth_l1_loss));code[p.as_posix()]=sha(p)
    save(OUT/'code_freeze.json',code);save(OUT/'data_freeze.json',data)
    dest=ARCHIVE/'frozen';dest.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+list(OUT.iterdir()):
        if p.is_file():shutil.copy2(p,dest/p.name)
    save(dest/'archive_manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
    print(read(OUT/'preflight.json'),flush=True)
if __name__=='__main__':main()
