"""Paired cheap-helper electrical inputs; fixed old cross-fitted baseline and rule."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,datetime,hashlib,importlib.util,json,math,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid174_power_helper';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID174_power_helper'
sp=importlib.util.spec_from_file_location('g174_oldfold',R/'work/grid170_remaining_date_folds/run.py');T=importlib.util.module_from_spec(sp);sp.loader.exec_module(T)
S=T.S;G=T.G;H=T.H;read=T.read;sha=T.sha;digest=T.digest;ARMS=['ZERO8','POWER8'];SEEDS=[0,1,2]
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def sources():return [T.SOURCE0,*[T.O/f'fold{f}' for f in [1,2,3]]]
def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    if (O/'models').exists():shutil.copytree(O/'models',dest/'models')
    (dest/'archive_manifest.json').write_text(json.dumps({p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    T.verify_trained()
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    H.sealed()
def build(seed):
    import torch
    from torch import nn
    old=H.build(seed)[0]
    class Q(nn.Module):
        def __init__(self):super().__init__();self.q_net=nn.Sequential(nn.Linear(50,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
        def forward(self,x):return self.q_net(x)
    net=Q()
    with torch.no_grad():
        net.q_net[0].weight[:,:42].copy_(old.q_net[0].weight);net.q_net[0].weight[:,42:].zero_();net.q_net[0].bias.copy_(old.q_net[0].bias)
        for j in [2,4]:net.q_net[j].weight.copy_(old.q_net[j].weight);net.q_net[j].bias.copy_(old.q_net[j].bias)
    return net,old
def xdata(fold,arm):
    z=np.load(sources()[fold]/'fold_arrays.npz');p=np.load(O/f'fold{fold}_power.npz')['normalized']
    return np.concatenate([z['normalized'],z['cost'],p if arm=='POWER8' else np.zeros_like(p)],1).astype('f'),z
def prepare():
    assert not O.exists();O.mkdir(parents=True);T.verify_trained();assert read(T.O/'complete.json')['passed_audit'];assert read(R/'outputs/grid173_public_state/complete.json')['passed_audit'];G.configure()
    meta=read(T.SOURCE0/'metadata.json');newmeta=read(R/'outputs/grid173_public_state/metadata.json');assert [(r['sample_id'],r['position']) for r in meta]==[(r['sample_id'],r['position']) for r in newmeta]
    power=np.load(R/'outputs/grid173_public_state/features.npz')['physical8'];assert power.shape==(3582,8)
    old=read(T.O/'design.json');d=dict(id='GRID174_CHEAP_HELPER_POWER8',authorization='User continuation; explicit pretraining notice24pairedfits, no newphysics.',
        hypothesis='Query-before-known electrical state andaction relations may help the cheap helper recognize profitable next batches missed by current summaries; keep positivequerysavings and allcounterexamples.',
        arms=ARMS,seeds=SEEDS,folds=[0,1,2,3],fit_order=[dict(fold=f,seed=s,arm=a) for f in range(4) for s in SEEDS for a in ARMS],fits_permitted=24,
        network='50->64->64->2 ReLU,7554parameters both. First42columns/laterlayers transplant existing RANDOM42constructor;8extracolumns0. No fittedweight initialization.',
        intervention='Onlycheaphelper8inputcolumns differ:allzeros versus jointlyfixed GRID173physical8. Old cross-fittedFACTOR andROUTED weights frozen. Same F AND(R OR C) rule;do not add a new safetygate/threshold.',
        normalization='Original41foldnormalization andknowncost unchanged;physical8 date-balancedmean/scale on12fitdatesonly,zero-variance scale1,clip[-20,20]. Duplicatedmaxrho retained explicitly,not all8newinformation.',
        task='Same immediate NET[0,y]Huberbeta1,price/gainscale/updates/sampler. Original4dev notscored/final8sealed. Historicallyfixedtaskunits andexposed16dates,not independenttest.',
        training=dict(updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip=10.,wall_cap_s=600,sampler_seed_offset=20261022),
        cost_per_query=old['cost_per_query'],gain_scale=old['gain_scale'],evaluation='All24finalweights published before any heldscores. PairedPOWER8vsZERO8 helper andcompleteone-sidedsearch,plus oldFACTOR/oldONE_SIDED. Allseeds,16dates,physicalgroups,query/gain/net andfirstdifferences retained.',
        interpretation='Small recurringpositivechanges worthresearch;no all-win gate. Capacitymatchedcomparison,not proof eachfieldnecessary or futureflowaccurate. No newfullcontroltimeclaim.',
        forbidden=['extraiterations/features/fields/losses/thresholds/fees/seed/foldselection','newdata/physics/final8','original4devscores','paper','automatic scale'])
    save('design.json',d);checks=[]
    for fold,path in enumerate(sources()):
        z=np.load(path/'fold_arrays.npz');fi=z['fit_ids'];hi=z['held_ids'];mean,scale=S.normalize(power,meta,fi);normalized=np.clip((power-mean)/scale,-20,20).astype('f');np.savez_compressed(O/f'fold{fold}_power.npz',normalized=normalized,mean=mean,scale=scale,fit_ids=fi,held_ids=hi)
        assert np.isfinite(normalized).all();assert not {meta[i]['date'] for i in fi}&{meta[i]['date'] for i in hi}
        checks.append(dict(fold=fold,fit_states=len(fi),held_states=len(hi),fit_dates=sorted({meta[i]['date'] for i in fi}),held_dates=sorted({meta[i]['date'] for i in hi})))
    import torch
    initial=[]
    for seed in SEEDS:
        net,oldnet=build(seed);paired,_=build(seed);assert digest(net)==digest(paired);assert sum(p.numel() for p in net.parameters())==7554
        x,z=xdata(0,'POWER8');a,z=xdata(0,'ZERO8');ii=z['fit_ids'][:16]
        with torch.no_grad():assert torch.equal(net(torch.tensor(x[ii])),paired(torch.tensor(a[ii])));assert torch.allclose(net(torch.tensor(x[ii])),oldnet(torch.tensor(x[ii,:42])),atol=1e-6,rtol=1e-6)
        initial.append(dict(seed=seed,hash=digest(net)))
    save('preflight.json',dict(passed=True,folds=checks,initial_hashes=initial,parameters=7554,optimizer_steps=0,new_fits=0))
    files=[R/'outputs/grid173_public_state'/n for n in ['design.json','features.npz','metadata.json','source_hashes.json','review.json','complete.json']]+[T.O/n for n in ['design.json','review.json','complete.json','all_oof_trajectories.json','all7134_paired_events.json','key_changes.json']]+[R/'outputs/grid102_search_rl/cases.pkl',O/'design.json',O/'preflight.json',*O.glob('fold*_power.npz')]
    files +=[path/n for path in sources() for n in ['metadata.json','fold_arrays.npz','design.json','preflight.json','state_scores.npz','held_trajectories.json','trained_freeze.json']]
    files +=[R/f'outputs/grid121_date_transfer/fold{f}_seed{s}_expected_exposure.npz' for f in range(4) for s in SEEDS]
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in files});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [*W.glob('*.py'),R/'work/grid173_public_state/feature_core.py',R/'work/grid164_explicit_cost/run.py',R/'work/grid168_one_sided_transfer/run.py',R/'work/grid170_remaining_date_folds/run.py']});verify();archive('frozen');save('state.json',dict(stage='frozen',new_fits=0));print('GRID174_FROZEN')
def train():
    verify();assert not (O/'training_started.json').exists();G.configure();import torch
    save('training_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_fits=24));(O/'models').mkdir();start=time.perf_counter();receipts=[]
    for fold,path in enumerate(sources()):
        xx={a:xdata(fold,a)[0] for a in ARMS};z=np.load(path/'fold_arrays.npz');fi=z['fit_ids'];hi=z['held_ids'];prob=z['fit_probability'];y=z['target'][fi];xt={a:torch.tensor(xx[a][fi]) for a in ARMS};yt=torch.tensor(np.column_stack([np.zeros_like(y),y]))
        for seed in SEEDS:
            expected=next(p for p in read(path/'preflight.json')['paired'] if p['seed']==seed)
            for arm in ARMS:
                net,_=build(seed);initial=digest(net);assert initial==next(v['hash'] for v in read(O/'preflight.json')['initial_hashes'] if v['seed']==seed)
                opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);rng=np.random.default_rng(20261022+seed);count=np.zeros(3582,np.int64);stream=hashlib.sha256();losses=[];fs=time.perf_counter();net.train()
                for update in range(1,7251):
                    if time.perf_counter()-start>600:raise TimeoutError('Preserve24fixedfitpartialoutputs;noextension/repeat')
                    ix=rng.choice(len(fi),32,replace=True,p=prob);gix=fi[ix];np.add.at(count,gix,1);stream.update(gix.astype('<i8').tobytes());q=net(xt[arm][ix]);loss=torch.nn.functional.smooth_l1_loss(q,yt[ix],beta=1.);assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach()))
                assert stream.hexdigest()==expected['sampling_hash'];assert count[hi].sum()==0;assert np.array_equal(count,np.load(R/f'outputs/grid121_date_transfer/fold{fold}_seed{seed}_expected_exposure.npz')['state_draws'])
                name=f'fold{fold}_{arm}_seed{seed}';p=O/'models'/f'{name}.pt';torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),fold=fold,seed=seed,arm=arm,updates=7250),p);np.savez_compressed(O/f'exposure_{name}.npz',state_draws=count,losses=np.asarray(losses))
                rec=dict(fold=fold,seed=seed,arm=arm,updates=7250,initial_hash=initial,final_hash=digest(net),checkpoint_sha256=sha(p),sampling_hash=stream.hexdigest(),held_draws=0,fit_wall_s=time.perf_counter()-fs);receipts.append(rec);save('fit_receipts.json',receipts);save('fit_progress.json',dict(completed=len(receipts),fold=fold,seed=seed,arm=arm,total_wall_s=time.perf_counter()-start));print(json.dumps(read(O/'fit_progress.json')),flush=True)
    save('fits.json',dict(passed=True,new_fits=24,records=receipts,total_wall_s=time.perf_counter()-start,held_used_for_fitting=False));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),*O.glob('exposure_*.npz'),O/'fits.json',O/'fit_receipts.json']});verify();archive('trained');save('state.json',dict(stage='trained_before_held_scores'));print('GRID174_24_FITS_COMPLETE')
def verify_trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
def publication():
    verify_trained();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote
    path=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{path}'],cwd=REPO)).hexdigest()==sha(O/'trained_freeze.json');save('preheld_publication.json',dict(passed=True,local_sha=local,remote_sha=remote));print('GRID174_ALL_WEIGHTS_PUBLISHED')
def evaluate():
    verify_trained();assert read(O/'preheld_publication.json')['passed'];assert not (O/'evaluation_started.json').exists();G.configure();import torch
    save('evaluation_started.json',dict(all24published=True));cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'};tr=[];pred={};checks=collections.Counter();maxerr=0.;controlchecks=[]
    for fold,path in enumerate(sources()):
        meta=read(path/'metadata.json');lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};z=np.load(path/'fold_arrays.npz');hi=z['held_ids'];held=set(read(path/'design.json')['fold']['held_dates']);old=np.load(path/'state_scores.npz');foldcases=[c for c in cases.values() if c['date'] in held]
        for seed in SEEDS:
            F=old[f'FACTOR_seed{seed}'];U=old[f'ROUTED_seed{seed}'];base=F[:,1]>F[:,0];route=U[:,1]>U[:,0]
            for arm in ARMS:
                name=f'fold{fold}_{arm}_seed{seed}';rec=next(v for v in read(O/'fits.json')['records'] if (v['fold'],v['seed'],v['arm'])==(fold,seed,arm));p=O/'models'/f'{name}.pt';assert sha(p)==rec['checkpoint_sha256'];cp=torch.load(p,weights_only=False,map_location='cpu');assert all(int(v['step'])==7250 for v in cp['optimizer']['state'].values());net,_=build(seed);net.load_state_dict(cp['state_dict']);net.eval();assert digest(net)==rec['final_hash'];x,_=xdata(fold,arm)
                with torch.no_grad():q=net(torch.tensor(x)).numpy()
                w={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()};nq=S.numpy_mlp(w,x,'');maxerr=max(maxerr,float(abs(q-nq).max()));assert np.allclose(q,nq,atol=3e-5,rtol=3e-5);assert np.array_equal(q[:,1]>q[:,0],nq[:,1]>nq[:,0]);checks['independent_numpy_rows']+=len(q);pred[name]=q
                go=q[:,1]>q[:,0];new=base&(route|go);assert np.all(new<=base);checks['state_query_direction']+=len(new)
                if arm=='ZERO8':controlchecks.append(dict(fold=fold,seed=seed,max_abs_score_difference=float(abs(q-old[f'COST42_seed{seed}']).max()),decision_differences=int(np.sum(go!=(old[f'COST42_seed{seed}'][:,1]>old[f'COST42_seed{seed}'][:,0])))))
                for mode,dec in [(arm+'_HELPER',go),(arm+'_GATE',new)]:
                    for c in foldcases:
                        ids=[lookup[c['sample_id'],j] for j in range(len(c['states']))];r=H.trajectory(c,dec[ids],f'{mode}_seed{seed}',read(O/'design.json')['cost_per_query'],read(O/'design.json')['gain_scale']);tr.append(r)
            for mode,dec in [('OLD_GATE',old[f'ONE_SIDED_seed{seed}']),('FACTOR',base)]:
                for c in foldcases:
                    ids=[lookup[c['sample_id'],j] for j in range(len(c['states']))];tr.append(H.trajectory(c,dec[ids],f'{mode}_seed{seed}',read(O/'design.json')['cost_per_query'],read(O/'design.json')['gain_scale']))
    np.savez_compressed(O/'state_scores.npz',**pred);save('control_reproduction.json',controlchecks);save('oof_trajectories.json',tr)
    methods=['ZERO8_HELPER','POWER8_HELPER','ZERO8_GATE','POWER8_GATE','OLD_GATE','FACTOR'];summary={m:H.summarize([r for r in tr if r['method']==m]) for m in {r['method'] for r in tr}};assert all(v['cases']==2378 and len(v['per_date'])==16 for v in summary.values())
    family={a:{k:float(np.mean([summary[f'{a}_seed{s}']['macro'][k] for s in SEEDS])) for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_group_mean_utility']} for a in methods};by={(r['method'],r['sample_id']):r for r in tr};events=[];d=read(O/'design.json');fee=d['cost_per_query'];scale=d['gain_scale'];datecounts=collections.Counter(c['date'] for c in cases.values());prior={(e['seed'],e['sample_id']):e for e in read(T.O/'all7134_paired_events.json')}
    for seed in SEEDS:
        for c in cases.values():
            a=by[f'POWER8_GATE_seed{seed}',c['sample_id']];b=by[f'ZERO8_GATE_seed{seed}',c['sample_id']];F=by[f'FACTOR_seed{seed}',c['sample_id']]
            assert a['queries']<=F['queries'] and b['queries']<=F['queries'];checks['search_query_bound']+=2
            for r in [a,b]:assert r['gain']<=F['gain']+1e-12
            dg=a['gain']-b['gain'];qs=b['queries']-a['queries'];nd=a['net_utility']-b['net_utility'];assert abs(nd-(dg+fee*qs)/scale)<1e-12
            events.append(dict(seed=seed,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],gain_delta=dg,saved_queries=qs,net_delta=nd,same_choice=a['chosen_pool_id']==b['chosen_pool_id'],power_batches=a['batches'],zero_batches=b['batches'],old_category=prior[seed,c['sample_id']]['category'],power_delta_vs_factor=a['net_utility']-F['net_utility'],zero_delta_vs_factor=b['net_utility']-F['net_utility'],weight=1/(3*16*datecounts[c['date']])))
    pairs=[]
    for seed in SEEDS:
        a=summary[f'POWER8_GATE_seed{seed}'];b=summary[f'ZERO8_GATE_seed{seed}'];pairs.append(dict(seed=seed,net_delta=a['macro']['mean_net_utility']-b['macro']['mean_net_utility'],per_date=[dict(date=x['date'],net_delta=x['mean_net_utility']-y['mean_net_utility']) for x,y in zip(a['per_date'],b['per_date'])]))
    dates=[dict(date=dt,mean_delta=float(np.mean([next(x['net_delta'] for x in p['per_date'] if x['date']==dt) for p in pairs]))) for dt in sorted(datecounts)]
    nd=family['POWER8_GATE']['mean_net_utility']-family['ZERO8_GATE']['mean_net_utility'];assert abs(nd-math.fsum(e['weight']*e['net_delta'] for e in events))<1e-12
    key=dict(net_delta_power_vs_zero=nd,relative_power_vs_zero=nd/family['ZERO8_GATE']['mean_net_utility'],relative_power_vs_factor=family['POWER8_GATE']['mean_net_utility']/family['FACTOR']['mean_net_utility']-1,query_reduction_power_vs_factor=1-family['POWER8_GATE']['mean_queries']/family['FACTOR']['mean_queries'],positive_seeds=sum(p['net_delta']>0 for p in pairs),positive_dates=sum(p['mean_delta']>0 for p in dates),old_loss_pairs=15,old_losses_improved=sum(e['old_category']=='LOST_GAIN_TOO_MUCH' and e['net_delta']>0 for e in events),old_same_gain_savings_preserved=sum(e['old_category']=='SAME_GAIN_LESS_QUERY' and e['power_delta_vs_factor']>0 for e in events),zero8_decision_differences_from_old_cost=sum(e['decision_differences'] for e in controlchecks))
    save('summary.json',summary);save('family_summary.json',family);save('all7134_paired_events.json',events);save('paired_summary.json',dict(pairs=pairs,per_date=dates));save('key_changes.json',key)
    verify_trained();save('review.json',dict(passed=True,checks=dict(checks),max_numpy_error=maxerr,all24weights_before_held=True,all_dates_seeds_retained=True,fit_only_power_normalization=True,new_physics=0,original4dev_not_scored=True,final8_absent=True));save('state.json',dict(stage='audited'));print('GRID174_AUDITED',json.dumps(key,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
