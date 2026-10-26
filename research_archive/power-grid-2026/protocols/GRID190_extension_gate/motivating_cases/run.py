"""Twelve frozen-NET logistic extension gates. No physical environment."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,datetime,hashlib,importlib.metadata,importlib.util,json,pickle,shutil,subprocess,sys,time,warnings
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid190_extension_gate';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID190_extension_gate';PRE=R/'outputs/grid189_baseline_extension'
sp=importlib.util.spec_from_file_location('grid190_frozen_factor',R/'work/grid168_one_sided_transfer/run.py');S=importlib.util.module_from_spec(sp);sp.loader.exec_module(S)
G=S.G;H=S.H;read=S.read;sha=S.sha;digest=S.digest
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item())+'\n',encoding='utf8',newline='\n')
def sources():return [S.O,*[R/f'outputs/grid170_remaining_date_folds/fold{f}' for f in [1,2,3]]]
def archive(phase):
    p=A/phase;p.mkdir(parents=True,exist_ok=False)
    for f in [*W.glob('*.py'),*O.iterdir()]:
        if f.is_file():shutil.copy2(f,p/f.name)
    for name in ['fit_features','models']:
        if (O/name).exists():shutil.copytree(O/name,p/name)
    (p/'archive_manifest.json').write_text(json.dumps({f.relative_to(p).as_posix():sha(f) for f in p.rglob('*') if f.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for n in ['source_freeze.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    H.sealed()
def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
def load():
    meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');lookup={(v['sample_id'],v['position']):i for i,v in enumerate(meta)}
    cases=[v for v in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if v['partition']=='train']
    return meta,lookup,cases
def features(f,seed,ids,tag):
    """Only listed partition rows are forwarded; hooks do not alter frozen net."""
    import torch
    path=sources()[f];z=np.load(path/'fold_arrays.npz');public=G.inputs('train');rows=[dict(public[i],global41=z['normalized'][i]) for i in ids]
    net=S.build(seed,'FACTOR');cp=torch.load(path/'models'/f'FACTOR_seed{seed}.pt',map_location='cpu',weights_only=False);net.load_state_dict(cp['state_dict']);net.eval();before=digest(net)
    rec=next(v for v in read(path/'fits.json')['records'] if v['arm']=='FACTOR' and v['seed']==seed);assert before==rec['final_hash']
    got={};handles=[net.base.q_net[3].register_forward_hook(lambda m,i,o:got.__setitem__('main',o.detach().numpy().copy())),net.delta.head[1].register_forward_hook(lambda m,i,o:got.__setitem__('delta',o.detach().numpy().copy()))];xx=[];qq=[];maxreconstruct=0.
    try:
        with torch.inference_mode():
            for p in range(0,len(rows),32):
                batch=rows[p:p+32];q=net(**G.old_api().collate(batch)).numpy();h0=got['main'];h1=got['delta'];assert h0.shape==(len(batch),64) and h1.shape==(len(batch),16)
                manual=h0.astype(float)@cp['state_dict']['base.q_net.4.weight'].numpy().astype(float).T+cp['state_dict']['base.q_net.4.bias'].numpy()+h1.astype(float)@cp['state_dict']['delta.head.2.weight'].numpy().astype(float).T+cp['state_dict']['delta.head.2.bias'].numpy();maxreconstruct=max(maxreconstruct,float(abs(manual-q).max()));assert np.allclose(manual,q,atol=3e-5,rtol=3e-5)
                xx.append(np.concatenate([h0,h1,q,np.stack([r['known_query_cost'] for r in batch])],1));qq.append(q.copy())
    finally:
        for h in handles:h.remove()
    q=np.concatenate(qq);x=np.concatenate(xx);expected=np.load(path/'state_scores.npz')[f'FACTOR_seed{seed}'][ids]
    assert x.shape==(len(ids),83) and np.isfinite(x).all();assert np.allclose(q,expected,atol=3e-5,rtol=3e-5)
    assert np.array_equal(q[:,1]>q[:,0],expected[:,1]>expected[:,0]),('NativeSTOP mismatch',f,seed,tag)
    assert not (q[:,1]>q[:,0]).any();assert digest(net)==before
    return x.astype(float),dict(fold=f,seed=seed,partition=tag,rows=len(ids),max_saved_score_error=float(abs(q-expected).max()),max_output_reconstruction_error=maxreconstruct,weights_unchanged=True,decisions_unchanged=True)
def prepare():
    assert not O.exists();O.mkdir(parents=True);H.sealed();assert read(PRE/'complete.json')['passed']
    save('design.json',dict(id='GRID190_FROZEN_NET_EXTENSION_GATE',authorization='User explicitly:开始训练把. Pretraining notice delivered; exactly12fixed84parameter logistic gates underGRID189 specification.',spec=read(PRE/'next_probe_spec.json'),new_fits=12,wall_cap_s=600,all_final_weights_before_held=True,
        runtime=dict(python=sys.version,numpy=np.__version__,torch=importlib.metadata.version('torch'),sklearn=importlib.metadata.version('scikit-learn')),
        controls='No NET changes/retraining/physics/data/final8/new thresholds/extra seeds/restarts. Existing old4dev not scored. Histories exposed;OOFconditional notblindtest.',
        evaluation='NET unchanged to first STOP;boundedgate/alwaysADD1 return best after zero/one batch. Retain all16dates/3seeds. Quality increment and addedcost separately;no allwin gate. Fixed candidate monotonicity not fullweek guarantee.'))
    files=[O/'design.json',PRE/'next_probe_spec.json',PRE/'fit_stop_rows.json',PRE/'fit_support.json',PRE/'audit.json',PRE/'complete.json',R/'outputs/grid113_relational_learning/train_public.pkl',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid187_action_invariance_target/oof_trajectories.json',R/'outputs/grid187_action_invariance_target/search_metrics.json',R/'outputs/grid123_event_auxiliary/decision.json']
    for p in sources():files += [p/n for n in ['fold_arrays.npz','state_scores.npz','fits.json']]+list((p/'models').glob('FACTOR_seed*.pt'))
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files});code=[*W.glob('*.py')]+[R/p for p in read(R/'outputs/grid187_action_invariance_target/code_freeze.json')];save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in code});verify();archive('planned');print('GRID190_PLANNED')
def qualify():
    verify();assert not (O/'preflight.json').exists();G.configure();(O/'fit_features').mkdir();rows=read(PRE/'fit_stop_rows.json');meta,lookup,cases=load();checks=[]
    for f in range(4):
        fitids=set(np.load(sources()[f]/'fold_arrays.npz')['fit_ids'].tolist())
        for seed in range(3):
            rr=[v for v in rows if v['fold']==f and v['seed']==seed];ids=[v['metadata_index'] for v in rr];assert set(ids)<=fitids
            x,rec=features(f,seed,ids,'fit');y=np.array([v['private_label']['next_action_changes'] for v in rr],np.int64);mass=np.array([v['proposed_sampling_mass'] for v in rr]);assert abs(mass.sum()-1)<1e-10 and set(y.tolist())=={0,1}
            mean=np.sum(x*mass[:,None],axis=0);var=np.sum((x-mean)**2*mass[:,None],axis=0);scale=np.sqrt(var);scale[scale<1e-8]=1.;standard=(x-mean)/scale;assert np.isfinite(standard).all()
            np.savez_compressed(O/'fit_features'/f'f{f}_s{seed}.npz',x=x,standard=standard,y=y,mass=mass,sample_weight=mass*len(y),mean=mean,scale=scale,ids=np.array(ids));checks.append(rec);save('qualification_progress.json',dict(completed=len(checks),total=12,checks=checks));print('QUALIFIED_FEATURE_POOL',f,seed,len(ids),flush=True)
    save('preflight.json',dict(passed=True,pools=checks,total_fit_rows=sum(v['rows'] for v in checks),all_frozen_NET_weights_and_STOP_decisions_unchanged=True,feature_width=83,gate_parameters=84,held_feature_rows=0,new_fits=0,new_physics=0));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_features').iterdir()),O/'preflight.json']});verify();archive('qualified');print('GRID190_QUALIFIED')
def train():
    verify();assert read(O/'preflight.json')['passed'];assert not (O/'training_started.json').exists()
    for p,h in read(O/'qualified_freeze.json').items():assert sha(O/p)==h
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    save('training_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fits=12,wall_cap_s=600));(O/'models').mkdir();receipts=[];start=time.perf_counter()
    for f in range(4):
        for seed in range(3):
            if time.perf_counter()-start>600:raise TimeoutError('Preservepartialfixed12fits;noautomaticextension')
            z=np.load(O/'fit_features'/f'f{f}_s{seed}.npz');model=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',max_iter=2000)
            with warnings.catch_warnings(record=True) as ww:
                warnings.simplefilter('always');t=time.perf_counter();model.fit(z['standard'],z['y'],sample_weight=z['sample_weight']);elapsed=time.perf_counter()-t
            manual=z['standard']@model.coef_[0]+model.intercept_[0];assert np.allclose(manual,model.decision_function(z['standard']),atol=1e-12,rtol=1e-12);assert np.array_equal(manual>0,model.predict(z['standard'])==1)
            assert model.coef_.size+model.intercept_.size==84;name=f'f{f}_s{seed}'
            np.savez_compressed(O/'models'/f'{name}.npz',coef=model.coef_,intercept=model.intercept_,mean=z['mean'],scale=z['scale'],classes=model.classes_,n_iter=model.n_iter_)
            (O/'models'/f'{name}.pkl').write_bytes(pickle.dumps(model,protocol=4));receipts.append(dict(fold=f,seed=seed,rows=len(z['y']),positive=int(z['y'].sum()),n_iter=int(model.n_iter_[0]),converged=not any(issubclass(v.category,ConvergenceWarning) for v in ww),warnings=[str(v.message) for v in ww],fit_wall_s=elapsed,weight_sha256=sha(O/'models'/f'{name}.npz'),fit_logit_min=float(manual.min()),fit_logit_max=float(manual.max())));save('fit_receipts.json',receipts);print('FITTED_GATE',f,seed,int(model.n_iter_[0]),flush=True)
    save('fits.json',dict(passed=True,records=receipts,new_fits=12,total_wall_s=time.perf_counter()-start,all_converged=all(v['converged'] for v in receipts)));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'fit_receipts.json']});verify();archive('trained');print('GRID190_ALL12_TRAINED')
def publication():
    trained();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote;rel=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=REPO)).hexdigest()==sha(O/'trained_freeze.json');save('preheld_publication.json',dict(passed=True,git_sha=local,remote_sha=remote,all12weights_published=True));print('GRID190_ALL_WEIGHTS_PUBLISHED')
def summarize(rr):
    x=H.summarize(rr)
    for d in x['per_date']:
        r=[v for v in rr if v['date']==d['date']];d['action_match_rate']=float(np.mean([v['action_match'] for v in r]));d['eligible_gate_fraction']=float(np.mean([v['eligible'] for v in r]));d['append_fraction']=float(np.mean([v['append'] for v in r]));d['added_queries']=float(np.mean([v['queries']-v['baseline_queries'] for v in r]));d['added_gain']=float(np.mean([v['gain']-v['baseline_gain'] for v in r]))
    for k in ['action_match_rate','eligible_gate_fraction','append_fraction','added_queries','added_gain']:x['macro'][k]=float(np.mean([d[k] for d in x['per_date']]))
    return x
def evaluate():
    trained();assert read(O/'preheld_publication.json')['passed'];assert not (O/'evaluation_started.json').exists();save('evaluation_started.json',dict(all12weights_published=True));G.configure();meta,lookup,cases=load();historical={}
    for v in read(R/'outputs/grid187_action_invariance_target/oof_trajectories.json'):
        if v['method'].startswith('NET_seed'):historical[v['method'],v['sample_id']]=v
    task=read(R/'outputs/grid102_search_rl/design.json');fee=task['cost_per_query'];scale=task['gain_scale'];tr=[];pred=[];featurechecks=[];checked=0
    for f in range(4):
        z=np.load(sources()[f]/'fold_arrays.npz');hi=set(z['held_ids'].tolist());helddates={meta[i]['date'] for i in hi};cc=[c for c in cases if c['date'] in helddates];cached=np.load(sources()[f]/'state_scores.npz')
        for seed in range(3):
            stop={};eligible=[];ids=[]
            for c in cc:
                ix=[lookup[c['sample_id'],j] for j in range(len(c['states']))];assert set(ix)<=hi;q=cached[f'FACTOR_seed{seed}'][ix];p=next((j for j,b in enumerate(q[:,1]>q[:,0]) if not b),len(ix));stop[c['sample_id']]=p
                if p<len(ix):eligible.append(c);ids.append(ix[p])
            x,rec=features(f,seed,ids,'held');featurechecks.append(rec);cp=np.load(O/'models'/f'f{f}_s{seed}.npz');model=pickle.loads((O/'models'/f'f{f}_s{seed}.pkl').read_bytes());standard=(x-cp['mean'])/cp['scale'];logit=standard@cp['coef'][0]+cp['intercept'][0];native=model.decision_function(standard);prob=np.exp(-np.logaddexp(0.,-logit));assert np.allclose(native,logit,atol=1e-11,rtol=1e-11);assert np.array_equal(logit>0,model.predict(standard)==1);assert np.allclose(prob,model.predict_proba(standard)[:,1],atol=1e-12,rtol=1e-12);gate={c['sample_id']:(bool(v>0),float(p)) for c,v,p in zip(eligible,logit,prob)}
            np.savez_compressed(O/f'held_f{f}_s{seed}.npz',ids=np.array(ids),features=x,logit=logit,probability=prob)
            for c in eligible:
                p=stop[c['sample_id']];pred.append(dict(fold=f,seed=seed,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=p,gate_continue=gate[c['sample_id']][0],probability=gate[c['sample_id']][1],actual_next_change=c['chosen'][p+1]!=c['chosen'][p],actual_next_gain=float(c['gains'][p]),actual_next_queries=int(c['queries'][p]),actual_next_NET=float((c['gains'][p]-fee*c['queries'][p])/scale)))
            for c in cc:
                p=stop[c['sample_id']];can=p<len(c['states']);base_gain=float(c['gains'][:p].sum());base_queries=int(c['queries'][:p].sum());fullmatch=c['chosen'][p]==c['chosen'][-1]
                for method in ['NET','ADD1','GATE']:
                    add=bool(can and (method=='ADD1' or method=='GATE' and gate[c['sample_id']][0]));end=p+int(add);name=f'{method}_seed{seed}';v=H.trajectory(c,np.arange(len(c['states']))<end,name,fee,scale);v.update(fold=f,baseline_stop=p,baseline_gain=base_gain,baseline_queries=base_queries,eligible=can,append=add,gate_probability=gate[c['sample_id']][1] if can and method=='GATE' else None,action_match=v['chosen_pool_id']==c['chosen'][-1]);assert base_queries<=v['queries']<=int(c['queries'].sum()) and v['queries']-base_queries<=64;assert base_gain<=v['gain']<=float(c['gains'].sum());assert not fullmatch or v['action_match']
                    if method=='NET':
                        old=historical[name,c['sample_id']]
                        for k in ['gain','queries','chosen_pool_id','actions']:assert v[k]==old[k]
                    tr.append(v);checked+=1
    results={}
    for method in ['NET','ADD1','GATE']:
        for seed in range(3):
            name=f'{method}_seed{seed}';rr=[v for v in tr if v['method']==name];assert len(rr)==2378;results[name]=summarize(rr)
    families={method:{k:float(np.mean([results[f'{method}_seed{s}']['macro'][k] for s in range(3)])) for k in results[f'{method}_seed0']['macro']} for method in ['NET','ADD1','GATE']}
    paired=[]
    for seed in range(3):
        b=results[f'NET_seed{seed}'];g=results[f'GATE_seed{seed}'];a=results[f'ADD1_seed{seed}'];paired.append(dict(seed=seed,match_delta_pp=100*(g['macro']['action_match_rate']-b['macro']['action_match_rate']),query_delta=g['macro']['mean_queries']-b['macro']['mean_queries'],NET_delta=g['macro']['mean_net_utility']-b['macro']['mean_net_utility'],per_date=[dict(date=x['date'],match_delta_pp=100*(x['action_match_rate']-y['action_match_rate']),query_delta=x['mean_queries']-y['mean_queries'],NET_delta=x['mean_net_utility']-y['mean_net_utility'],query_vs_ADD1=x['mean_queries']-z['mean_queries']) for x,y,z in zip(g['per_date'],b['per_date'],a['per_date'])]))
    gate_counts=dict(eligible=len(pred),append=sum(v['gate_continue'] for v in pred),append_changes=sum(v['gate_continue'] and v['actual_next_change'] for v in pred),append_no_change=sum(v['gate_continue'] and not v['actual_next_change'] for v in pred),missed_changes=sum(not v['gate_continue'] and v['actual_next_change'] for v in pred),all_positive_changes=sum(v['actual_next_change'] for v in pred),append_positive_NET=sum(v['gate_continue'] and v['actual_next_NET']>0 for v in pred))
    save('oof_trajectories.json',tr);save('gate_predictions.json',pred);save('search_metrics.json',results);save('family_summary.json',families);save('paired_summary.json',paired);save('gate_counts.json',gate_counts);save('held_feature_checks.json',featurechecks)
    trained();save('review.json',dict(passed=True,trajectory_rows_checked=checked,unchanged_NET_trajectories=7134,held_gate_rows=len(pred),frozen_NET_and_decisions_unchanged=True,gate_native_and_manual_predictions_match=True,local_gain_actionmatch_and_query_bounds_hold=True,all12_gates_published_before_held=True,new_fits=12,all16dates_all3seeds=True,new_physics=0,old4dev_new_scores=0,final8_absent=True));archive('evaluated');print('GRID190_EVALUATED',json.dumps(dict(families=families,gate_counts=gate_counts),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
