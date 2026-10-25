"""GRID168: first existing date fold; fixed one-sided rule; nine fresh fits."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import collections,datetime,hashlib,importlib.util,json,math,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent
O=R/'outputs/grid168_one_sided_transfer';REPO=R/'work/GRL-results-integration'
A=REPO/'research_archive/power-grid-2026/protocols/GRID168_one_sided_transfer'
spec=importlib.util.spec_from_file_location('g168_builders',R/'work/grid165_factor_cost_routing/run.py')
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
H=G.H;read=H.read;sha=H.sha;digest=H.digest
ARMS=['FACTOR','ROUTED','COST42'];SEEDS=[0,1,2]

def save(name,obj):
    (O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')

def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    if (O/'models').exists():shutil.copytree(O/'models',dest/'models')
    manifest={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()}
    (dest/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8',newline='\n')

def verify():
    for name in ['source_hashes.json','code_freeze.json']:
        for path,h in read(O/name).items():assert sha(R/path)==h,path
    H.sealed()

def verify_trained():
    verify()
    for path,h in read(O/'trained_freeze.json').items():assert sha(O/path)==h,path

def build(seed,arm):
    return H.build(seed)[0] if arm=='COST42' else G.build(seed,'ZERO_MAIN' if arm=='FACTOR' else 'COST_MAIN')[0]

def data():
    arrays=np.load(O/'fold_arrays.npz');meta=read(O/'metadata.json')
    rows=pickle.loads((R/'outputs/grid113_relational_learning/train_public.pkl').read_bytes())
    rows=[dict(row,global41=arrays['normalized'][i]) for i,row in enumerate(rows)]
    return arrays,meta,rows

def normalize(raw,meta,ids):
    counts=collections.Counter(meta[i]['date'] for i in ids)
    weights=np.asarray([1/counts[meta[i]['date']] for i in ids]);weights/=weights.mean()
    x=raw[ids].astype(float);mean=np.average(x,axis=0,weights=weights)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weights));scale[scale<1e-8]=1.
    return mean,scale

def prepare():
    assert not O.exists();O.mkdir(parents=True);H.sealed();G.configure()
    assert read(R/'outputs/grid167_one_sided_stop/complete.json')['passed_audit']
    fold=read(R/'outputs/grid121_date_transfer/design.json')['folds'][0]
    meta=read(R/'outputs/grid113_relational_learning/train_metadata.json')
    oldmeta=read(R/'outputs/grid121_date_transfer/metadata.json')
    assert [(r['sample_id'],r['position']) for r in meta]==[(r['sample_id'],r['position']) for r in oldmeta]
    z=np.load(R/'outputs/grid121_date_transfer/train_arrays.npz');raw=z['raw'];target=z['target'];prob0=z['original_probability']
    assert np.array_equal(target,np.load(R/'outputs/grid113_relational_learning/train_targets.npz')['target'])
    fit=np.array([m['date'] in fold['fit_dates'] for m in meta]);held=np.array([m['date'] in fold['held_dates'] for m in meta])
    assert np.all(fit^held);fi=np.flatnonzero(fit);hi=np.flatnonzero(held)
    assert len(fi)==2478 and len(hi)==1104
    assert not {meta[i]['physical_state_group'] for i in fi}&{meta[i]['physical_state_group'] for i in hi}
    mean,scale=normalize(raw,meta,fi);prior=np.load(R/'outputs/grid121_date_transfer/fold0_normalization.npz')
    assert np.allclose(mean,prior['mean'],atol=1e-12,rtol=1e-12)
    assert np.allclose(scale,prior['scale'],atol=1e-12,rtol=1e-12)
    normalized=np.clip((raw-mean)/scale,-20,20).astype('f');prob=prob0[fi]/prob0[fi].sum()
    task=read(G.O/'design.json');fee=task['cost_per_query'];gain_scale=task['gain_scale']
    allcases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())
    cases=[c for c in allcases if c['partition']=='train'];cm={(c['sample_id'],j):(c,j) for c in cases for j in range(len(c['states']))}
    public=G.inputs('train');cost=[]
    for i,m in enumerate(meta):
        c,j=cm[m['sample_id'],m['position']]
        assert np.array_equal(raw[i],c['states'][j])
        assert target[i]==np.float32((c['gains'][j]-fee*c['queries'][j])/gain_scale)
        assert len(public[i]['next_action'])==c['queries'][j]
        cc=np.float32(fee*c['queries'][j]/gain_scale)
        assert np.allclose(public[i]['known_query_cost'],cc,atol=1e-7,rtol=1e-7)
        cost.append(public[i]['known_query_cost'])
    cost=np.stack(cost)
    save('metadata.json',meta)
    np.savez_compressed(O/'fold_arrays.npz',normalized=normalized,mean=mean,scale=scale,fit_ids=fi,held_ids=hi,fit_probability=prob,target=target,cost=cost)
    d=dict(id='GRID168_ONE_SIDED_DATE_TRANSFER',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User sustained continuation; pretraining notice: first existing whole-date fold, three families x three seeds, no new physical data.',
        purpose='Check whether the small GRID167 earlier-stop benefit transfers to one fixed existing group of dates excluded from all constituent fits.',
        selection='GRID121 fold0, fixed first fold; do not inspect/select alternative folds. All dates historically exposed; conditional date-transfer development, not fresh independent test.',
        fold=fold,arms=ARMS,seeds=SEEDS,fit_order=[dict(seed=s,arm=a) for s in SEEDS for a in ARMS],fits_permitted=9,
        rule='go_ONE_SIDED = go_FACTOR AND (go_ROUTED OR go_COST42); strict q1>q0. Never reopen original STOP. No weights/threshold or rule search.',
        model=dict(FACTOR='12980 parameters: ZERO_MAIN from GRID165,42-column main with zero dummy plus original FACTOR correction.',ROUTED='Same12980 with known cost in main, correction unchanged.',COST42='7042parameter small42-64-64-2 MLP with known cost.'),
        random_initialization='Existing constructors only; no fitted weights loaded. Same main41 randomseed and zero addedcolumn, FACTOR correction randomoffset unchanged.',
        normalization='Date-balanced raw41 mean/scale computed only on12fitdates, clipped[-20,20]. Independently reproduce oldGRID121fold0 normalization; known cost unchanged.',
        sampling='Original date/prefix/position probabilities restricted and renormalized tofitIDs; same global-index stream perseed acrossall3families; reproduce oldGRID121fold0 exposure.',
        training=dict(updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip=10.,loss='mean SmoothL1(q,[0,net]),beta1',sampler_seed_offset=20261022,wall_cap_s=1800),
        cost_per_query=fee,gain_scale=gain_scale,
        historical_dependency='Fee/gain_scale and public feedback transforms remain historical task units originally defined using all16trainingdates. Not a fully nested reconstruction of task design.',
        population=dict(fit_states=len(fi),held_states=len(hi),fit_dates=12,held_dates=4,fit_starts=sum(c['date'] in fold['fit_dates'] for c in cases),held_starts=sum(c['date'] in fold['held_dates'] for c in cases)),
        evaluation='Publish ALL9final weights before held-date forward. All3seeds/all4helddates; utility,rawgain,queries,capture,physical-groupmean,allpairedchanges. Fit-side descriptive separately. No all-seedwin gate for research value.',
        numpy_audit='All1104heldstates plus fixed every151st/first/last and everypositive fitstate for structured families; all3582states smallMLP. Verify original targets and exact querybound each saved search.',
        limitations='Old samegrid localNET task, correlated prefixes/date replicas; no fullweek quality or measuredcontroltime claim. Three-network inference cost excluded from proxyqueryfee.',
        prohibited=['morefits/checkpoint/seed/fold/rule/threshold/fee searches','heldlabels for fitting or normalization','newdata/physics/test8','original4dev predictions','paper','automatic scale'])
    save('design.json',d);paired=[]
    import torch
    fitrows=[dict(public[i],global41=normalized[i]) for i in fi[:2]];pub=G.old_api().collate(fitrows)
    for seed in SEEDS:
        factor=build(seed,'FACTOR');routed=build(seed,'ROUTED');small=build(seed,'COST42')
        assert digest(factor)==digest(routed)
        assert sum(p.numel() for p in factor.parameters())==12980
        assert sum(p.numel() for p in small.parameters())==7042
        with torch.no_grad():assert torch.equal(factor(**pub),routed(**pub))
        rng=np.random.default_rng(20261022+seed);stream=hashlib.sha256();count=np.zeros(len(meta),np.int64)
        for _ in range(7250):
            ix=fi[rng.choice(len(fi),32,replace=True,p=prob)];stream.update(ix.astype('<i8').tobytes());np.add.at(count,ix,1)
        assert count[hi].sum()==0
        assert stream.hexdigest()==read(R/'outputs/grid121_date_transfer/preflight.json')['sampling_streams'][f'fold0_seed{seed}']
        assert np.array_equal(count,np.load(R/f'outputs/grid121_date_transfer/fold0_seed{seed}_expected_exposure.npz')['state_draws'])
        paired.append(dict(seed=seed,initial_hashes={a:digest(n) for a,n in zip(ARMS,[factor,routed,small])},sampling_hash=stream.hexdigest()))
    save('preflight.json',dict(passed=True,fold0_normalization_reproduced=True,labels_and_queries_checked=len(meta),rawstates_checked=len(meta),date_and_physical_groups_disjoint=True,held_optimizer_draws=0,paired=paired,optimizer_instances=0,new_fits=0))
    files=[R/'outputs/grid113_relational_learning'/n for n in ['train_public.pkl','train_metadata.json','train_targets.npz']]
    files +=[R/'outputs/grid121_date_transfer'/n for n in ['design.json','preflight.json','metadata.json','train_arrays.npz','fold0_normalization.npz',*[f'fold0_seed{s}_expected_exposure.npz' for s in SEEDS]]]
    files +=[R/'outputs/grid102_search_rl/cases.pkl',G.O/'design.json',R/'outputs/grid167_one_sided_stop/design.json',R/'outputs/grid167_one_sided_stop/complete.json',O/'design.json',O/'preflight.json',O/'metadata.json',O/'fold_arrays.npz']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in files})
    code=[*W.glob('*.py')]+[R/'work'/n for n in ['grid165_factor_cost_routing/run.py','grid164_explicit_cost/run.py','grid116_matched_module/common116.py','grid115_matched_module_plan/matched_model.py','grid113_relational_learning/common.py','grid113_relational_learning/model.py','grid113_relational_learning/audit.py','grid112_relational_plan/schema.py','grid139_global_feasibility/run.py']]
    save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in code});verify();save('state.json',dict(stage='frozen',fits=0));archive('frozen');print('GRID168_FROZEN',d['population'])

def train():
    verify();assert not (O/'training_started.json').exists();G.configure();import torch
    d=read(O/'design.json');z,meta,rows=data();fi=z['fit_ids'];hi=z['held_ids'];prob=z['fit_probability'];fitrows=[rows[i] for i in fi]
    take=G.old_api().tensor_cache(fitrows);x=torch.tensor(np.concatenate([z['normalized'][fi],z['cost'][fi]],1));target=z['target'][fi];yt=torch.tensor(np.column_stack([np.zeros_like(target),target]))
    start=time.perf_counter();save('training_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fits_permitted=9,wall_cap_s=1800));(O/'models').mkdir();receipts=[]
    for item in d['fit_order']:
        seed,arm=item['seed'],item['arm'];name=f'{arm}_seed{seed}';net=build(seed,arm)
        expected=next(v for v in read(O/'preflight.json')['paired'] if v['seed']==seed);initial=digest(net);assert initial==expected['initial_hashes'][arm]
        opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        rng=np.random.default_rng(20261022+seed);count=np.zeros(len(meta),np.int64);stream=hashlib.sha256();losses=[];fs=time.perf_counter();net.train()
        for update in range(1,7251):
            if time.perf_counter()-start>1800:raise TimeoutError('Preserve fixed9-fit partial outputs; no automatic extension/retry')
            ix=rng.choice(len(fi),32,replace=True,p=prob);global_ix=fi[ix];np.add.at(count,global_ix,1);stream.update(global_ix.astype('<i8').tobytes())
            q=net(x[ix]) if arm=='COST42' else net(**take(ix))
            loss=torch.nn.functional.smooth_l1_loss(q,yt[ix],beta=1.);assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach()))
            if update%500==0 or update==7250:
                progress=dict(seed=seed,arm=arm,update=update,completed=len(receipts),fit_wall_s=time.perf_counter()-fs,total_wall_s=time.perf_counter()-start,mean_last_loss=float(np.mean(losses[-500:])))
                save('fit_progress.json',progress);print(json.dumps(progress),flush=True)
        assert stream.hexdigest()==expected['sampling_hash'];assert count[hi].sum()==0
        assert np.array_equal(count,np.load(R/f'outputs/grid121_date_transfer/fold0_seed{seed}_expected_exposure.npz')['state_draws'])
        path=O/'models'/f'{name}.pt';torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),seed=seed,arm=arm,updates=7250),path)
        np.savez_compressed(O/f'exposure_{name}.npz',state_draws=count,losses=np.asarray(losses))
        rec=dict(seed=seed,arm=arm,updates=7250,initial_hash=initial,final_hash=digest(net),checkpoint_sha256=sha(path),sampling_hash=stream.hexdigest(),parameters=sum(p.numel() for p in net.parameters()),fit_wall_s=time.perf_counter()-fs,held_draws=int(count[hi].sum()))
        receipts.append(rec);save('fit_receipts.json',receipts)
    assert len(receipts)==9;save('fits.json',dict(passed=True,new_fits=9,records=receipts,total_wall_s=time.perf_counter()-start,held_used_for_fit=False))
    save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),*O.glob('exposure_*.npz'),O/'fits.json',O/'fit_receipts.json']})
    verify_trained();save('state.json',dict(stage='trained_before_held_forward',fits=9));archive('trained');print('GRID168_NINE_FITS_COMPLETE')

def publication():
    verify_trained();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote
    path=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix()
    assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{path}'],cwd=REPO)).hexdigest()==sha(O/'trained_freeze.json')
    save('preheld_publication.json',dict(passed=True,local_sha=local,remote_sha=remote,all9_weights_before_held_forward=True));print('GRID168_WEIGHTS_PUBLISHED')

def numpy_mlp(w,x,prefix):
    x=x.astype(float)
    for j in [0,2,4]:
        x=x@w[f'{prefix}q_net.{j}.weight'].T+w[f'{prefix}q_net.{j}.bias']
        if j!=4:x=np.maximum(x,0)
    return x

def evaluate():
    verify_trained();assert read(O/'preheld_publication.json')['passed'];assert not (O/'evaluation_started.json').exists()
    save('evaluation_started.json',dict(all9_weights_published=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));G.configure();import torch
    spec=importlib.util.spec_from_file_location('g168_numpy',R/'work/grid113_relational_learning/audit.py');aud=importlib.util.module_from_spec(spec);spec.loader.exec_module(aud)
    z,meta,rows=data();d=read(O/'design.json');fee=d['cost_per_query'];scale=d['gain_scale'];Old=G.old_api();scores={};checks=collections.Counter();maxerr=0.;state_metrics=[]
    for rec in read(O/'fits.json')['records']:
        seed,arm=rec['seed'],rec['arm'];name=f'{arm}_seed{seed}';path=O/'models'/f'{name}.pt';assert sha(path)==rec['checkpoint_sha256']
        cp=torch.load(path,weights_only=False,map_location='cpu');assert cp['updates']==7250
        assert all(int(v['step'])==7250 for v in cp['optimizer']['state'].values())
        net=build(seed,arm);net.load_state_dict(cp['state_dict']);net.eval();assert digest(net)==rec['final_hash']
        w={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()};x=np.concatenate([z['normalized'],z['cost']],1)
        if arm=='COST42':
            with torch.no_grad():q=net(torch.tensor(x)).numpy()
            manual=numpy_mlp(w,x,'');assert np.allclose(manual,q,atol=3e-5,rtol=3e-5)
            assert np.array_equal(manual[:,1]>manual[:,0],q[:,1]>q[:,0]);maxerr=max(maxerr,float(abs(manual-q).max()));checks['numpy_rows']+=len(q)
        else:
            q=Old.predict(net,rows,batch=32)
            xx=x.copy()
            if arm=='FACTOR':xx[:,-1]=0
            main=numpy_mlp(w,xx,'base.');dw={k.removeprefix('delta.'):v for k,v in w.items() if k.startswith('delta.')}
            ids=sorted(set(z['held_ids'].tolist()+[0,len(rows)-1]+list(range(0,len(rows),151))+np.flatnonzero(z['target']>0).tolist()))
            for i in ids:
                nq=main[i]+aud.numpy_forward(dw,rows[i],'FACTOR');maxerr=max(maxerr,float(abs(nq-q[i]).max()))
                assert np.allclose(nq,q[i],atol=3e-5,rtol=3e-5),(name,i,nq,q[i])
                assert bool(nq[1]>nq[0])==bool(q[i,1]>q[i,0]);checks['numpy_rows']+=1
        scores[name]=q
        for part,key in [('fit','fit_ids'),('held','held_ids')]:
            ids=z[key];y=z['target'][ids];go=q[ids,1]>q[ids,0];pos=y>0
            state_metrics.append(dict(method=name,partition=part,states=len(ids),positive_states=int(pos.sum()),false_stop=int(np.sum(pos&~go)),unprofitable_continue=int(np.sum(~pos&go)),positive_recall=float(go[pos].mean()) if pos.any() else None,sign_regret=float(np.mean(np.maximum(y,0)-go*y))))
    for seed in SEEDS:
        a=scores[f'FACTOR_seed{seed}'];b=scores[f'ROUTED_seed{seed}'];c=scores[f'COST42_seed{seed}']
        base=a[:,1]>a[:,0];stop=base&((b[:,1]>b[:,0])|(c[:,1]>c[:,0]));assert np.all(stop<=base)
        scores[f'ONE_SIDED_seed{seed}']=stop;checks['state_no_reopening']+=len(base)
    np.savez_compressed(O/'state_scores.npz',**scores);save('state_metrics.json',state_metrics)
    cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train']
    lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};summaries={};alltr={};families={};pair_summaries={};all_events={}
    for part,dates in [('fit',d['fold']['fit_dates']),('held',d['fold']['held_dates'])]:
        cc=[c for c in cases if c['date'] in dates];out=[]
        for seed in SEEDS:
            for arm in ARMS+['ONE_SIDED']:
                name=f'{arm}_seed{seed}';v=scores[name];go=v if arm=='ONE_SIDED' else v[:,1]>v[:,0];rr=[]
                for case in cc:
                    ids=[lookup[case['sample_id'],j] for j in range(len(case['states']))]
                    t=H.trajectory(case,go[ids],name,fee,scale)
                    # Independent walk, separate accounting and action selection.
                    pos=next((j for j,i in enumerate(ids) if not go[i]),len(ids))
                    assert t['batches']==pos and t['queries']==sum(int(x) for x in case['queries'][:pos])
                    assert abs(t['gain']-math.fsum(float(x) for x in case['gains'][:pos]))<1e-12
                    assert t['chosen_pool_id']==case['chosen'][pos]
                    rr.append(t);checks['independent_search_walks']+=1
                summaries[name+'__'+part]=H.summarize(rr);out+=rr
        tm={(r['method'],r['sample_id']):r for r in out};events=[];datecounts=collections.Counter(c['date'] for c in cc)
        for seed in SEEDS:
            for case in cc:
                aa=tm[f'ONE_SIDED_seed{seed}',case['sample_id']];bb=tm[f'FACTOR_seed{seed}',case['sample_id']]
                assert aa['batches']<=bb['batches'] and aa['queries']<=bb['queries'];assert aa['gain']<=bb['gain']+1e-12
                checks['query_bound_searches']+=1;qs=bb['queries']-aa['queries'];dg=aa['gain']-bb['gain'];nd=aa['net_utility']-bb['net_utility']
                assert abs(nd-(dg+fee*qs)/scale)<1e-12
                cat='UNCHANGED' if not qs else ('SAME_GAIN_LESS_QUERY' if abs(dg)<1e-15 else ('GAIN_LOSS_NET_POSITIVE' if nd>0 else 'LOST_GAIN_TOO_MUCH'))
                event=dict(seed=seed,sample_id=case['sample_id'],date=case['date'],physical_state_group=case['physical_state_group'],category=cat,saved_queries=qs,gain_delta=dg,net_delta=nd,same_choice=aa['chosen_pool_id']==bb['chosen_pool_id'],base_batches=bb['batches'],new_batches=aa['batches'],weight=1/(3*len(dates)*datecounts[case['date']]))
                if qs:
                    j=aa['batches'];i=lookup[case['sample_id'],j]
                    event.update(first_difference_position=j,next_queries=int(case['queries'][j]),next_gain=float(case['gains'][j]),next_net=float(z['target'][i]),scores={a:scores[f'{a}_seed{seed}'][i].tolist() for a in ARMS})
                events.append(event)
        ff={a:{k:float(np.mean([summaries[f'{a}_seed{s}__{part}']['macro'][k] for s in SEEDS])) for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_group_mean_utility']} for a in ARMS+['ONE_SIDED']}
        pairs=[]
        for seed in SEEDS:
            aa=summaries[f'ONE_SIDED_seed{seed}__{part}'];bb=summaries[f'FACTOR_seed{seed}__{part}']
            pairs.append(dict(seed=seed,one_sided=aa['macro']['mean_net_utility'],factor=bb['macro']['mean_net_utility'],net_delta=aa['macro']['mean_net_utility']-bb['macro']['mean_net_utility'],per_date=[dict(date=x['date'],net_delta=x['mean_net_utility']-y['mean_net_utility']) for x,y in zip(aa['per_date'],bb['per_date'])]))
        perdate=[dict(date=dt,mean_delta=float(np.mean([next(r['net_delta'] for r in p['per_date'] if r['date']==dt) for p in pairs]))) for dt in sorted(dates)]
        nd=ff['ONE_SIDED']['mean_net_utility']-ff['FACTOR']['mean_net_utility'];assert abs(nd-math.fsum(e['weight']*e['net_delta'] for e in events))<1e-12
        es=[]
        for cat in sorted({e['category'] for e in events}):
            ee=[e for e in events if e['category']==cat]
            es.append(dict(category=cat,pairs=len(ee),starts=len({e['sample_id'] for e in ee}),physical_groups=len({e['physical_state_group'] for e in ee}),dates=sorted({e['date'] for e in ee}),same_choice_pairs=sum(e['same_choice'] for e in ee),net_contribution=math.fsum(e['weight']*e['net_delta'] for e in ee)))
        pair_summaries[part]=dict(pairs=pairs,per_date=perdate,mean_net_delta=nd,relative_change=nd/ff['FACTOR']['mean_net_utility'] if ff['FACTOR']['mean_net_utility'] else None,query_reduction=1-ff['ONE_SIDED']['mean_queries']/ff['FACTOR']['mean_queries'],event_summary=es,positive_seeds=sum(p['net_delta']>0 for p in pairs))
        families[part]=ff;alltr[part]=out;all_events[part]=events
        save(part+'_trajectories.json',out);save(part+'_paired_events.json',events)
    save('summary.json',summaries);save('family_summary.json',families);save('paired_summary.json',pair_summaries)
    verify_trained();save('review.json',dict(passed=True,checks=dict(checks),maximum_numpy_error=maxerr,nine_fresh_fits_all_final_weights_published=True,fit_only_normalization_and_draws=True,all_dates_and_seeds_retained=True,original4dev_not_scored=True,new_physics=0,new_data=0,final8_absent=True,local_task_not_wholeweek_quality=True))
    save('state.json',dict(stage='analysed',fits=9));print(json.dumps(dict(held=families['held'],paired=pair_summaries['held']),ensure_ascii=False))

if __name__=='__main__':globals()[sys.argv[1]]()
