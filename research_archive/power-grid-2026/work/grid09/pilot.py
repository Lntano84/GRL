"""Bounded PPO auxiliary pilot. Immutable author baseline; episode checkpoints survive interruption."""
import argparse,traceback,gzip
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'grid08'))
from common import *
from trace_math import advantages_trace
OUT=ROOT/'outputs/grid09'
from networks_v1 import ActorCritic
from torch.distributions import Categorical

parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['train','evaluate'],required=True)
parser.add_argument('--kind',choices=['gnn'],default='gnn');parser.add_argument('--version',choices=['physical','decision'],required=True)
args=parser.parse_args()
D=json.loads((OUT/'design.json').read_text()); CFG=D['ppo']; START=time.perf_counter()
assert json.loads((OUT/'preflight.json').read_text())['passed']
torch.manual_seed(D['training_seed']);np.random.seed(D['training_seed'])
RUN=OUT/args.version/args.phase/(args.kind or 'baselines');RUN.mkdir(parents=True,exist_ok=True)
assert not (RUN/'finished.json').exists(), 'Refuse to overwrite finished phase'
env=make_env(); physical=0; updates=0; opt_wall=0.; summaries=[]

def update(model,optimizer,buffer,bootstrap):
    global updates,opt_wall
    t=time.perf_counter();n=len(buffer)
    values=np.array([b['value'] for b in buffer]+[bootstrap],dtype=float)
    adv=advantages_trace([b['reward'] for b in buffer],values,
        [b['terminal'] for b in buffer],[bool(b['features'][2][6]>.5) for b in buffer],
        CFG['gamma'],CFG['gae_lambda'],args.version)
    ret=torch.from_numpy(adv+values[:-1].astype(np.float32))
    active=torch.tensor([bool(b['features'][2][6]>.5) for b in buffer],dtype=torch.bool)
    advantages=torch.zeros(n)
    if active.any():
        a=torch.from_numpy(adv)[active]
        advantages[active]=(a-a.mean())/(a.std(unbiased=False)+1e-8)
    old_lp=torch.tensor([b['logp'] for b in buffer]);actions=torch.tensor([b['action'] for b in buffer])
    metrics=[]
    for epoch in range(CFG['epochs']):
        for ids in torch.randperm(n).split(CFG['minibatch']):
            selected=[buffer[int(i)] for i in ids]
            xs=torch.stack([torch.from_numpy(b['features'][0]) for b in selected])
            es=[torch.from_numpy(b['features'][1]) for b in selected]
            gs=torch.stack([torch.from_numpy(b['features'][2]) for b in selected])
            logits,v=model(xs,es,gs);dist=Categorical(logits=logits)
            ratio=(dist.log_prob(actions[ids])-old_lp[ids]).exp()
            local=active[ids]
            if local.any():
                pg=-torch.minimum(ratio[local]*advantages[ids][local],ratio[local].clamp(1-CFG['clip'],1+CFG['clip'])*advantages[ids][local]).mean()
                ent=dist.entropy()[local].mean()
            else:
                pg=logits.sum()*0.;ent=logits.sum()*0.
            # Bounded error avoids huge critic gradients from rare terminal safety penalty.
            vf=torch.nn.functional.smooth_l1_loss(v,ret[ids])
            loss=pg+.5*vf-CFG['entropy']*ent
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5);torch.nn.utils.clip_grad_norm_(model.value.parameters(),.5);optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            metrics.append([float(pg.detach()),float(vf.detach()),float(ent.detach())])
    updates+=1;elapsed=time.perf_counter()-t;opt_wall+=elapsed
    with (RUN/'updates.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'update':updates,'samples':n,'offered':sum(b['features'][2][6]>.5 for b in buffer),
            'mean_losses':np.asarray(metrics).mean(0).tolist(),'wall_s':elapsed},default=json_default)+'\n')

def episode(scenario,policy,model=None,optimizer=None,label=''):
    global physical
    path=RUN/(label or f'{scenario}__{policy}');path.mkdir(parents=True,exist_ok=False)
    env.seed(D['environment_seed']);env.set_id(scenario);obs=env.reset()
    ctl=ResidualControl(env,obs);horizon=int(env.max_episode_duration());reward=0.;cost=0.
    offered=selected=eligible=0;illegal=ambiguous=exceptions=0;sim_total=0;buffer=[];teacher=[]
    arrays={k:[] for k in ['action','rho','gen_p','load_p','actual_dispatch','curtailment_mw','storage_power']}
    rows=[];t=time.perf_counter(); inference_s=0.
    metadata={'scenario':scenario,'policy':policy,'horizon':horizon,'seed':D['environment_seed'],
        'gen_cost_per_MW':env.gen_cost_per_MW.tolist(),'dt_hours':float(env.delta_time_seconds)/3600,
        'cadence':D['cadence'],'model_kind':args.kind,'version':args.version}
    write_json(path/'metadata.json',metadata)
    with gzip.open(path/'steps.jsonl.gz','wt',encoding='utf-8') as log:
        for step in range(1,horizon+1):
            assert time.perf_counter()-START<D['caps']['process_wall_s']
            oldcurt=float(obs.curtailment_mw.sum(dtype=np.float64));pre=digest(obs.to_vect());ns=int(env.nb_highres_called)
            base,rest,p=ctl.propose(obs,reward,D['cadence']);feat=graph_features(obs,env,p)
            action=0;lp=value=0.
            if model is not None:
                ti=time.perf_counter()
                with torch.no_grad():
                    logits,v=model(*tensor_features(feat));dist=Categorical(logits=logits)
                    action=int(dist.sample().item()) if optimizer is not None else int(logits.argmax(-1).item())
                    lp=float(dist.log_prob(torch.tensor([action])).item());value=float(v.item())
                inference_s+=time.perf_counter()-ti
            elif policy=='ALWAYS_RESTORE':action=int(p['offered'])
            elif policy=='IMMEDIATE_COST' and p['offered']:
                action=int(p['predictions'][1]['cost']<p['predictions'][0]['cost'])
            if p['offered']:
                assert rest is not None
                teacher.append({'features':feat,'action':int(p['predictions'][1]['cost']<p['predictions'][0]['cost'])})
            if action:assert p['offered']
            act=rest if action else base
            obs,reward,done,info=env.step(act);physical+=1
            fl=flags(info);illegal+=fl['illegal'];ambiguous+=fl['ambiguous'];exceptions+=bool(fl['exceptions'])
            raw=float(info['rewards'][f'{PREFIX}_grid_operational_cost']);cost+=raw
            valid=bool((obs.gen_p>0).any()) and not fl['exceptions']
            ledger=public_cost_ledger(obs,env,oldcurt) if valid else None
            tol=float(cost_rounding_tolerance(obs,env)) if valid else None
            if ledger:assert abs(ledger['recomputed_raw_cost']-raw)<=tol
            sim=int(env.nb_highres_called)-ns;sim_total+=sim
            complete=bool(obs.current_step>=horizon or env.chronics_handler.done())
            train_reward=-raw/CFG['cost_scale']
            if done and not complete:train_reward-=CFG['blackout_penalty']
            if optimizer is not None:
                buffer.append({'features':feat,'action':action,'logp':lp,'value':value,
                    'reward':train_reward,'terminal':bool(done)})
                if len(buffer)>=CFG['rollout'] or done:
                    bootstrap=0.
                    if not done:
                        # Current observed state only; no future action/proposal or physics for bootstrap.
                        blank={'offered':False,'predictions':[None,None]}
                        with torch.no_grad():_,bv=model(*tensor_features(graph_features(obs,env,blank)))
                        bootstrap=float(bv.item())
                    update(model,optimizer,buffer,bootstrap);buffer=[]
            offered+=p['offered'];eligible+=p['eligible'];selected+=action
            row={'step':step,'current_step':int(obs.current_step),'raw_cost':raw,'done':bool(done),
                'complete':complete,'ledger':ledger,'ledger_tolerance':tol,'before_hash':pre,
                'after_hash':digest(obs.to_vect()),'action_hash':digest(action_vector(act)),
                'choice':action,'proposal':p,'simulations':sim,**fl}
            log.write(json.dumps(row,allow_nan=False,default=json_default)+'\n');log.flush()
            for k in arrays:arrays[k].append(action_vector(act) if k=='action' else np.array(getattr(obs,k),copy=True))
            if step%256==0 or done:
                print(json.dumps({'scenario':scenario,'policy':policy,'step':step,'selected':selected,
                    'offered':offered,'cost':cost,'wall_s':time.perf_counter()-t}),flush=True)
                write_json(path/'progress.json',{'step':step,'selected':selected,'offered':offered,'cost':cost})
            if done:break
    np.savez_compressed(path/'vectors.npz',**{k:np.stack(v) for k,v in arrays.items()})
    if teacher:
        np.savez_compressed(path/'teacher.npz',x=np.stack([z['features'][0] for z in teacher]),
            g=np.stack([z['features'][2] for z in teacher]),actions=np.array([z['action'] for z in teacher]),
            **{f'e{i}':z['features'][1] for i,z in enumerate(teacher)})
    summary={'scenario':scenario,'policy':policy,'steps':step,'complete':complete,'raw_cost':cost,
        'eligible':eligible,'offered':offered,'selected':selected,'simulations':sim_total,
        'illegal':illegal,'ambiguous':ambiguous,'exceptions':exceptions,'wall_s':time.perf_counter()-t,
        'neural_inference_wall_s':inference_s,'path':str(path.relative_to(ROOT))}
    write_json(path/'summary.json',summary);return summary

try:
    if args.phase=='train':
        assert args.kind is not None
        model=ActorCritic(args.kind,2*env.n_sub,max_edges=2*env.n_line)
        checkpoint=ROOT/D['source_checkpoint']['path']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==D['source_checkpoint']['sha256']
        saved=torch.load(checkpoint,weights_only=False)
        model.load_state_dict(saved['state'])
        optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])
        write_json(RUN/'warm_start.json',D['source_checkpoint'])
        torch.save({'state':model.state_dict(),'kind':args.kind,'n_nodes':2*env.n_sub,'max_edges':2*env.n_line},RUN/'initial.pt')
        write_json(RUN/'model_metadata.json',{'kind':args.kind,'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,'device':'cpu'})
        for epoch in range(D['training_epochs']):
            for j,scenario in enumerate(D['training_weeks']):
                summary=episode(scenario,args.kind,model,optimizer,label=f'e{epoch}_{scenario}');summaries.append(summary)
                checkpoint=RUN/f'epoch{epoch}_week{j}.pt'
                torch.save({'state':model.state_dict(),'optimizer':optimizer.state_dict(),'kind':args.kind,
                    'n_nodes':2*env.n_sub,'max_edges':2*env.n_line,'episode':len(summaries),'torch_rng':torch.get_rng_state()},checkpoint)
                write_json(RUN/'manifest.json',summaries)
        torch.save({'state':model.state_dict(),'kind':args.kind,'n_nodes':2*env.n_sub,'max_edges':2*env.n_line},RUN/'final.pt')
    else:
        model=None
        if args.kind:
            file=OUT/args.version/'train'/args.kind/'final.pt'; saved=torch.load(file,weights_only=False)
            model=ActorCritic(args.kind,saved['n_nodes'],max_edges=saved['max_edges']);model.load_state_dict(saved['state']);model.eval()
        for scenario in D['evaluation_weeks']:
            for policy in ([args.kind] if args.kind else D['baselines']):
                summaries.append(episode(scenario,policy,model));write_json(RUN/'manifest.json',summaries)
    write_json(RUN/'finished.json',{'passed':True,'summaries':summaries,'physical_steps':physical,
        'wall_s':time.perf_counter()-START,'optimizer_wall_s':opt_wall,'updates':updates})
except Exception:
    write_json(RUN/'failure.json',{'traceback':traceback.format_exc(),'physical_steps':physical,
        'wall_s':time.perf_counter()-START});raise
finally:env.close()
