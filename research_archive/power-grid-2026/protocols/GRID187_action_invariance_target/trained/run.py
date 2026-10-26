"""Fixed target-only comparison: signed action-change labels, original FACTOR training."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,datetime,hashlib,importlib.util,json,math,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid187_action_invariance_target';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID187_action_invariance_target'
sp=importlib.util.spec_from_file_location('grid187_original_factor',R/'work/grid168_one_sided_transfer/run.py');S=importlib.util.module_from_spec(sp);sp.loader.exec_module(S)
G=S.G;H=S.H;read=S.read;sha=S.sha;digest=S.digest
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def sources():return [S.O,*[R/f'outputs/grid170_remaining_date_folds/fold{f}' for f in [1,2,3]]]
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    if (O/'models').exists():shutil.copytree(O/'models',d/'models')
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    H.sealed()
def verify_trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
def data(f):
    with np.load(sources()[f]/'fold_arrays.npz') as z:arrays={k:z[k].copy() for k in z.files}
    public=G.inputs('train');rows=[dict(row,global41=arrays['normalized'][i]) for i,row in enumerate(public)]
    return arrays,rows
def prepare():
    assert not O.exists();O.mkdir(parents=True);H.sealed();assert read(R/'outputs/grid186_invariance_labels/complete.json')['passed']
    meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'];bycase={c['sample_id']:c for c in cases}
    inv=np.load(R/'outputs/grid186_invariance_labels/train_labels.npz')['action_invariant'];assert len(inv)==len(meta)==3582
    for i,m in enumerate(meta):assert bool(inv[i])==(bycase[m['sample_id']]['chosen'][m['position']]==bycase[m['sample_id']]['chosen'][-1])
    signed=np.where(inv,-1.,1.).astype('f');np.savez_compressed(O/'target_labels.npz',signed_change=signed,action_invariant=inv)
    old=read(S.O/'design.json');save('design.json',dict(id='GRID187_SIGNED_ACTION_CHANGE_TARGET',authorization='User:ok继续吧;pretrainingnotice12fixedfits(original4datefoldsx3seeds). No physicalrun authorization implied.',
        hypothesis='Predictwhetherremainingobservedlocalsuffixchangescurrentbestaction, ratherthanlearnprice-weightednext-batchNET. Retain realaction-invariantquerysavings;keepallerrors/tradeoffs.',
        target='STOPhead target0;CONTINUEhead target+1 if finalobservedchosen differsfromcurrentchosen,else-1. Same meanHuberbeta1. Strictq1>q0 unchanged. Scoresare signedclassregression,NOT calibratedprobabilities orcurrency.',
        changed='Only CONTINUE supervision target. Architecture/input/normalization/seeds/initialparameters/sampling/updates/optimizer/loss/comparisonrule unchanged. No classreweight/threshold tuning.',
        new_fits=12,fit_order=[dict(fold=f,seed=s) for f in range(4) for s in [0,1,2]],
        model='Original12980parameterFACTOR=42columnmain(extra0)+5938FACTORcorrection. Knownquerycostretainedasoriginalinput;no addedcheapmodules or newfeatures.',
        training=dict(updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip=10.,loss='meanSmoothL1beta1(q,[0,signedchange])',sampler_seed_offset=20261022,wall_cap_s=2400),
        baseline='Reuse all12frozenFACTOR NET checkpoints fromGRID168fold0/GRID170fold1-3,verifyinitialhash/exposure/normalization andsavedoutputs. No baseline retraining.',
        population='All16originaltrainingdates held out in their pre-existing12fit/4helddate fold;2378starts/3582states/222physicalgroups. Original4dev not scored;final8sealed.',
        freeze='All12finalweights published before any helddateforward. Final7250only;no interimheldlabel/model/seed/foldselection.',
        evaluation='Primary:wrongSTOPonaction-change states,STOPprecision,full-recorded-localactionmatch andactualboughtlocalqueries in savedsearch. Also oldNET/rawgain/capture/physical-group metrics, allseed/date pairs. STOP_ALL/BATCH1/FULL_LOCAL deterministic reference once,not3independentseeds.',
        cost_per_query=old['cost_per_query'],gain_scale=old['gain_scale'],
        limits='Historicallyexposed samegrid teacherpaths,not blind/FRAME-path/newclosedloopquality or realcontroltime. Labels/fee taskdesign arehistorical;normalization andoptimizer draws excludecorrespondinghelddates. Actionmatchingdoesnotrepair referencecontrollerweeklycost.',
        prohibited=['extraweights/epochs/seed/fold/threshold/classweight/targetsearch','original4devforward/final8','newphysics/data','newcheapensemble','paper/figures','automatic scale']))
    G.configure();import torch
    checks=[];paths=[O/'design.json',O/'target_labels.npz',R/'outputs/grid186_invariance_labels/train_labels.npz',R/'outputs/grid186_invariance_labels/review.json',R/'outputs/grid186_invariance_labels/label_rows.json',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid113_relational_learning/train_public.pkl',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid121_date_transfer/design.json']
    for f,path in enumerate(sources()):
        z,rows=data(f);fi=z['fit_ids'];hi=z['held_ids'];assert not {meta[i]['date'] for i in fi}&{meta[i]['date'] for i in hi};assert len(np.unique(signed[fi]))==len(np.unique(signed[hi]))==2
        for seed in [0,1,2]:
            expected=next(x for x in read(path/'preflight.json')['paired'] if x['seed']==seed);net=S.build(seed,'FACTOR');assert digest(net)==expected['initial_hashes']['FACTOR'];assert sum(v.numel() for v in net.parameters())==12980
            rec=next(v for v in read(path/'fits.json')['records'] if v['arm']=='FACTOR' and v['seed']==seed);assert sha(path/'models'/f'FACTOR_seed{seed}.pt')==rec['checkpoint_sha256']
            rng=np.random.default_rng(20261022+seed);h=hashlib.sha256();count=np.zeros(3582,np.int64)
            for _ in range(7250):ix=fi[rng.choice(len(fi),32,replace=True,p=z['fit_probability'])];h.update(ix.astype('<i8').tobytes());np.add.at(count,ix,1)
            assert h.hexdigest()==expected['sampling_hash']==rec['sampling_hash'];assert not count[hi].any();assert np.array_equal(count,np.load(R/f'outputs/grid121_date_transfer/fold{f}_seed{seed}_expected_exposure.npz')['state_draws'])
            checks.append(dict(fold=f,seed=seed,initial_hash=digest(net),sampling_hash=h.hexdigest(),old_checkpoint_sha256=rec['checkpoint_sha256'],fit_states=len(fi),held_states=len(hi)))
            paths.append(path/'models'/f'FACTOR_seed{seed}.pt');paths.append(R/f'outputs/grid121_date_transfer/fold{f}_seed{seed}_expected_exposure.npz')
        paths +=[path/n for n in ['design.json','fold_arrays.npz','preflight.json','fits.json','state_scores.npz']]
    save('preflight.json',dict(passed=True,paired=checks,old_metadata_label_alignment=True,all_four_folds_fixed=True,new_fits=0,optimizer_instances=0));paths.append(O/'preflight.json')
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});code=[*W.glob('*.py')]+[R/p for p in read(S.O/'code_freeze.json')];save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in code});verify();archive('frozen');print('GRID187_12_FITS_FROZEN')
def train():
    verify();assert not (O/'training_started.json').exists();G.configure();import torch
    save('training_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fits_permitted=12,wall_cap_s=2400));(O/'models').mkdir();label=np.load(O/'target_labels.npz')['signed_change'];start=time.perf_counter();receipts=[]
    for f,path in enumerate(sources()):
        z,rows=data(f);fi=z['fit_ids'];hi=z['held_ids'];take=G.old_api().tensor_cache([rows[i] for i in fi]);yt=torch.tensor(np.column_stack([np.zeros(len(fi),'f'),label[fi]]))
        for seed in [0,1,2]:
            expected=next(v for v in read(O/'preflight.json')['paired'] if (v['fold'],v['seed'])==(f,seed));net=S.build(seed,'FACTOR');initial=digest(net);assert initial==expected['initial_hash'];opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);rng=np.random.default_rng(20261022+seed);count=np.zeros(3582,np.int64);stream=hashlib.sha256();losses=[];fs=time.perf_counter();net.train()
            for update in range(1,7251):
                if time.perf_counter()-start>2400:raise TimeoutError('Keep originalpartialfits;no automaticretraining/windowextension')
                ix=rng.choice(len(fi),32,replace=True,p=z['fit_probability']);globalix=fi[ix];np.add.at(count,globalix,1);stream.update(globalix.astype('<i8').tobytes());loss=torch.nn.functional.smooth_l1_loss(net(**take(ix)),yt[ix],beta=1.);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach()))
                if update%1000==0 or update==7250:
                    progress=dict(fold=f,seed=seed,update=update,completed=len(receipts),total_wall_s=time.perf_counter()-start,mean_last_loss=float(np.mean(losses[-500:])));save('fit_progress.json',progress);print(json.dumps(progress),flush=True)
            assert stream.hexdigest()==expected['sampling_hash'] and not count[hi].any();assert np.array_equal(count,np.load(R/f'outputs/grid121_date_transfer/fold{f}_seed{seed}_expected_exposure.npz')['state_draws']);name=f'fold{f}_CHANGE_seed{seed}';p=O/'models'/f'{name}.pt';torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),fold=f,seed=seed,arm='SIGNED_CHANGE',updates=7250),p);np.savez_compressed(O/f'exposure_{name}.npz',state_draws=count,losses=np.asarray(losses));receipts.append(dict(fold=f,seed=seed,initial_hash=initial,final_hash=digest(net),checkpoint_sha256=sha(p),sampling_hash=stream.hexdigest(),fit_wall_s=time.perf_counter()-fs,held_draws=int(count[hi].sum())));save('fit_receipts.json',receipts)
    save('fits.json',dict(passed=True,new_fits=12,records=receipts,total_wall_s=time.perf_counter()-start));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),*O.glob('exposure_*.npz'),O/'fits.json',O/'fit_receipts.json']});verify();archive('trained');print('GRID187_ALL12_FINAL_WEIGHTS_TRAINED')
def publication():
    verify_trained();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote;rel=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=REPO)).hexdigest()==sha(O/'trained_freeze.json');save('preheld_publication.json',dict(passed=True,git_sha=local,remote_sha=remote,all12weights_published=True));print('GRID187_ALL_WEIGHTS_PUBLISHED')
def state_metrics(go,changed,meta,ids):
    out=[]
    for date in sorted({meta[i]['date'] for i in ids}):
        ii=np.array([i for i in ids if meta[i]['date']==date]);t=changed[ii];p=go[ii];stop=~p
        out.append(dict(date=date,states=len(ii),must_change=int(t.sum()),predicted_STOP=int(stop.sum()),wrong_STOP=int(np.sum(t&stop)),wrong_STOP_rate=float(np.mean(stop[t])) if t.any() else None,STOP_precision=float(np.mean(~t[stop])) if stop.any() else None,unnecessary_CONTINUE=int(np.sum(~t&p)),accuracy=float(np.mean(t==p))))
    return dict(per_date=out,macro={k:float(np.mean([v[k] for v in out if v[k] is not None])) for k in ['wrong_STOP_rate','STOP_precision','accuracy']},counts={k:sum(v[k] for v in out) for k in ['states','must_change','predicted_STOP','wrong_STOP','unnecessary_CONTINUE']})
def search_summary(rr):
    base=H.summarize(rr)
    for row in base['per_date']:
        sub=[x for x in rr if x['date']==row['date']];row['action_match_rate']=float(np.mean([x['action_matches_full_local'] for x in sub]));row['wrong_STOP_searches']=sum(not x['action_matches_full_local'] for x in sub)
    base['macro']['action_match_rate']=float(np.mean([x['action_match_rate'] for x in base['per_date']]));return base
def evaluate():
    verify_trained();assert read(O/'preheld_publication.json')['passed'];assert not (O/'evaluation_started.json').exists();save('evaluation_started.json',dict(all12weights_published=True));G.configure();import torch
    Old=G.old_api();sp=importlib.util.spec_from_file_location('grid187_independent_numpy',R/'work/grid113_relational_learning/audit.py');Aud=importlib.util.module_from_spec(sp);sp.loader.exec_module(Aud)
    meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');changed=~np.load(O/'target_labels.npz')['action_invariant'];lookup={(v['sample_id'],v['position']):i for i,v in enumerate(meta)};cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'];d=read(O/'design.json');fee=d['cost_per_query'];scale=d['gain_scale'];score={f'{mode}_seed{s}':np.full((3582,2),np.nan,np.float32) for mode in ['NET','CHANGE'] for s in [0,1,2]};checks=collections.Counter();maxerr=0.
    for f,path in enumerate(sources()):
        z,rows=data(f);hi=z['held_ids'];saved=np.load(path/'state_scores.npz')
        for seed in [0,1,2]:
            for mode in ['NET','CHANGE']:
                net=S.build(seed,'FACTOR');p=path/'models'/f'FACTOR_seed{seed}.pt' if mode=='NET' else O/'models'/f'fold{f}_CHANGE_seed{seed}.pt';cp=torch.load(p,map_location='cpu',weights_only=False);net.load_state_dict(cp['state_dict']);net.eval();assert all(int(v['step'])==7250 for v in cp['optimizer']['state'].values());q=Old.predict(net,rows,batch=32)
                if mode=='NET':
                    ref=saved[f'FACTOR_seed{seed}'];assert np.allclose(q,ref,atol=3e-5,rtol=3e-5) and np.array_equal(q[:,1]>q[:,0],ref[:,1]>ref[:,0]);checks['baseline_saved_output_rows']+=len(q)
                else:
                    rec=next(v for v in read(O/'fits.json')['records'] if (v['fold'],v['seed'])==(f,seed));assert digest(net)==rec['final_hash'];w={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()};dw={k.removeprefix('delta.'):v for k,v in w.items() if k.startswith('delta.')}
                    for i in hi:
                        x=np.r_[rows[i]['global41'],0.].astype(float)
                        for j in [0,2,4]:
                            x=x@w[f'base.q_net.{j}.weight'].T+w[f'base.q_net.{j}.bias']
                            if j!=4:x=np.maximum(x,0)
                        manual=x+Aud.numpy_forward(dw,rows[i],'FACTOR');maxerr=max(maxerr,float(abs(manual-q[i]).max()));assert np.allclose(manual,q[i],atol=3e-5,rtol=3e-5) and bool(manual[1]>manual[0])==bool(q[i,1]>q[i,0]);checks['independent_numpy_held_rows']+=1
                assert np.isnan(score[f'{mode}_seed{seed}'][hi]).all();score[f'{mode}_seed{seed}'][hi]=q[hi]
    assert all(np.isfinite(q).all() for q in score.values());np.savez_compressed(O/'oof_scores.npz',**score);tr=[];states={};results={}
    for name,q in score.items():
        go=q[:,1]>q[:,0];states[name]=state_metrics(go,changed,meta,np.arange(3582));rr=[]
        for c in cases:
            ids=[lookup[c['sample_id'],j] for j in range(len(c['states']))];x=H.trajectory(c,go[ids],name,fee,scale);x['action_matches_full_local']=x['chosen_pool_id']==c['chosen'][-1];rr.append(x)
        tr+=rr;results[name]=search_summary(rr)
    for name in ['STOP_ALL','BATCH1','FULL_LOCAL']:
        rr=[]
        for c in cases:
            go=np.zeros(len(c['states']),bool) if name=='STOP_ALL' else np.ones(len(c['states']),bool) if name=='FULL_LOCAL' else np.arange(len(c['states']))==0;x=H.trajectory(c,go,name,fee,scale);x['action_matches_full_local']=x['chosen_pool_id']==c['chosen'][-1];rr.append(x)
        tr+=rr;results[name]=search_summary(rr)
    family={}
    for mode in ['NET','CHANGE']:
        family[mode]=dict(search_macro={k:float(np.mean([results[f'{mode}_seed{s}']['macro'][k] for s in [0,1,2]])) for k in results[f'{mode}_seed0']['macro']},state_macro={k:float(np.mean([states[f'{mode}_seed{s}']['macro'][k] for s in [0,1,2]])) for k in states[f'{mode}_seed0']['macro']},state_counts={k:sum(states[f'{mode}_seed{s}']['counts'][k] for s in [0,1,2]) for k in states[f'{mode}_seed0']['counts']})
    pairs=[]
    for seed in [0,1,2]:
        a=results[f'CHANGE_seed{seed}'];b=results[f'NET_seed{seed}'];pairs.append(dict(seed=seed,action_match_delta=a['macro']['action_match_rate']-b['macro']['action_match_rate'],query_change=a['macro']['mean_queries']/b['macro']['mean_queries']-1,net_utility_delta=a['macro']['mean_net_utility']-b['macro']['mean_net_utility'],per_date=[dict(date=x['date'],action_match_delta=x['action_match_rate']-y['action_match_rate'],query_change=x['mean_queries']/y['mean_queries']-1,net_utility_delta=x['mean_net_utility']-y['mean_net_utility']) for x,y in zip(a['per_date'],b['per_date'])]))
    save('oof_trajectories.json',tr);save('state_metrics.json',states);save('search_metrics.json',results);save('family_summary.json',family);save('paired_summary.json',dict(pairs=pairs,all_seeds_retained=True,all16dates_retained=True,exploratory_no_allwin_gate=True))
    verify_trained();save('review.json',dict(passed=True,checks=dict(checks),max_numpy_error=maxerr,new_fits=12,optimizer_steps_each=7250,all_weights_published_before_held=True,complete_OOF_states_per_seed=3582,saved_searches_checked=len(tr),no_held_optimizer_draws=True,old4dev_new_scores=0,new_physics=0,final8_absent=True));archive('evaluated');print('GRID187_EVALUATED',json.dumps(dict(families=family,pairs=pairs),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
