"""Only deepen the fixed conditional amount forest from four to five levels."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import datetime,hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid210_amount_depth';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID210_public_amount';OLD=R/'outputs/grid208_public_amount';PRIOR=R/'outputs/grid206_public_event';LINEAR=R/'outputs/grid198_conditional_gain';FIT=R/'outputs/grid192_gate_support';LAB=LINEAR/'fit_labels';CACHE=R/'outputs/grid190_extension_gate'
sp=importlib.util.spec_from_file_location('grid210_fixed_base',R/'work/grid208_public_amount/run.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P);B=P.B
PARAMS=dict(n_estimators=64,max_depth=5,min_samples_leaf=10,max_features=1.0,bootstrap=False,n_jobs=1,criterion='squared_error',warm_start=False)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    for name in ['fit_features','models']:
        if (O/name).exists():shutil.copytree(O/name,d/name)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    P.trained()
def raw_inputs():return P.raw_inputs()
def prepare():
    assert not O.exists();O.mkdir(parents=True);P.trained();assert read(R/'outputs/grid209_lost_gain/independent_audit.json')['passed']
    save('design.json',dict(id='GRID210_AMOUNT_DEPTH',authorization='Usercontinue;pretrainingnotice exactly12fixedamountforests maxdepth4to5 only delivered.',change='Only conditionalforest max_depth4->5. Entire124input,scalers,positive198targets/weights/rows,64trees,minleaf10,allotherestimatorparams andseeds unchanged. Event208bytefrozen andNETfrozen.',motivation='GRID209 all4681paths depth4and legallysplittable mean97.93percent;lost4mean89.45percent. Structuralroomnotguarantee generalization;testoneextra level forallstates,notrescuespecificcases.',model=dict(type='ExtraTreesRegressor',params=PARAMS,seeds=[0,1,2],new_amount_forests=12,new_amount_trees=768,cap_s=600),capacity='Exactlyoneestimatorhyperparameter changed,capacityincreases;notfixedcapacity. Frozen768eventtrees+new768amounttrees,total1536active across12pairs.',labels='Exactlyold208positivefitarrays and198conditionaltargetweights. Nozeroeventmu0 labels,scaling changes,classweights ornewfeatures.',action='Samefrozenscorep * newconditionalmu > originalknowncharge buysone<=64thenreturnsbest. No fee/threshold/rank/batch/stationchanges.',evaluation='All12pairs published beforeheld. Primary208isolatesdepth;secondary206cumulativeamountinputanddepth,notpureablation.198rawreference. All4681STOProws/21402traces/all16exposedtrain dates/all3seeds. Prespecifiedsameexpectedqueryvs208and206,randomADD1;alladversekept,noallwin/5percentcriteria.',focus='Original10mainrecovery plus4old208lostprofitablepairs andallnewlosses,notonlywhether4rescued. Seed0actualnegativeandwastefractionretained.',metrics='PositiveconditionalMSEMAE,expectedgainerrors,eventAUC/Brierunchanged,actionmatch/gaincapture/query/proxyNET. Localteacherpathonly,notfullweekquality/time.',limits='No event/NETfits/forwards/newphysics/generation/final8/paper/newFRAME. No depth/leaf/treenumber/fee/thresholdsearch orautomaticextrapolation. Onefixed600s cap;preserveallpartialfailures, no extra fitting.'))
    files=[W/'run.py',W/'build_source.py',W/'source_derivation.json',O/'design.json',R/'work/grid208_public_amount/run.py',R/'work/grid208_public_amount/finalize.py',R/'work/grid203_tree_event/finalize.py',R/'outputs/grid209_lost_gain/complete.json',R/'outputs/grid209_lost_gain/independent_audit.json',R/'outputs/grid209_lost_gain/summary.json',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']
    for d in [OLD,PRIOR,LINEAR]:files += [d/n for n in ['complete.json','gate_predictions.json','oof_trajectories.json','search_metrics.json','family_summary.json','predictive_metrics.json']]
    files+=list((OLD/'models').iterdir())+list((OLD/'fit_features').glob('*.npz'))+list((FIT/'fit_features').glob('*.npz'))+list(LAB.glob('*.npz'))+list(CACHE.glob('held_f*.npz'));save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID210_PLANNED')
def qualify():
    verify();assert not (O/'preflight.json').exists();meta,_,_,raw=raw_inputs();(O/'fit_features').mkdir();records=[]
    from sklearn.ensemble import ExtraTreesRegressor
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';z=np.load(OLD/'fit_features'/f'{name}.npz');lab=np.load(LAB/f'{name}.npz');oldmodel=pickle.loads((OLD/'models'/f'{name}_amount.pkl').read_bytes());candidate=ExtraTreesRegressor(**PARAMS,random_state=s);assert not hasattr(candidate,'estimators_');oldparams=oldmodel.get_params();newparams=candidate.get_params();diff=[k for k in oldparams if oldparams[k]!=newparams[k]];assert diff==['max_depth'] and oldparams['max_depth']==4 and newparams['max_depth']==5;assert np.array_equal(z['target'],lab['target']) and np.array_equal(z['sample_weight'],lab['sample_weight']) and np.array_equal(z['metadata_ids'],lab['metadata_ids']);assert z['standard'].shape==(len(lab['target']),124) and (z['target']>0).all()
            hz=np.load(CACHE/f'held_{name}.npz');fd={meta[int(i)]['date'] for i in z['metadata_ids']};hd={meta[int(i)]['date'] for i in hz['ids']};assert len(fd)==12 and len(hd)==4 and not fd&hd;shutil.copy2(OLD/'fit_features'/f'{name}.npz',O/'fit_features'/f'{name}.npz');assert sha(O/'fit_features'/f'{name}.npz')==sha(OLD/'fit_features'/f'{name}.npz');records.append(dict(fold=f,seed=s,positive_rows=len(lab['target']),parameter_differences=diff,event_sha256=sha(OLD/'models'/f'{name}_event.pkl'),fit_array_sha256=sha(O/'fit_features'/f'{name}.npz')))
    save('preflight.json',dict(passed=True,pools=records,only_max_depth_changes_and_fitarrays_byte_identical=True,new_fits=0,new_NET_forwards=0));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_features').iterdir()),O/'preflight.json']});verify();archive('qualified');print('GRID210_QUALIFIED')
def train():
    verify();assert read(O/'preflight.json')['passed'] and not (O/'training_started.json').exists()
    for p,h in read(O/'qualified_freeze.json').items():assert sha(O/p)==h
    from sklearn.ensemble import ExtraTreesRegressor
    save('training_started.json',dict(new_amount_forests=12,new_trees=768,cap_s=600,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));(O/'models').mkdir();records=[];start=time.perf_counter()
    for f in range(4):
        for s in range(3):
            if time.perf_counter()-start>600:raise TimeoutError('Preservepartialfits; original600s cap')
            name=f'f{f}_s{s}';z=np.load(O/'fit_features'/f'{name}.npz');x,y,w=z['standard'],z['target'],z['sample_weight'];t=time.perf_counter();model=ExtraTreesRegressor(**PARAMS,random_state=s);model.fit(x,y,sample_weight=w);nodes=leaves=0;worst=0.;manual=[]
            for tree in model.estimators_:
                leaf=tree.apply(x);manual.append(tree.tree_.value[leaf,0,0]);assert tree.tree_.max_depth<=5;nodes+=tree.tree_.node_count;leaves+=int(np.sum(tree.tree_.children_left==-1))
                for j in np.unique(leaf):
                    mask=leaf==j;assert mask.sum()>=10;err=abs(float(w[mask]@y[mask]/w[mask].sum())-float(tree.tree_.value[j,0,0]));worst=max(worst,err);assert err<1e-8
            mu=model.predict(x);assert np.allclose(mu,np.mean(manual,axis=0),atol=1e-10,rtol=1e-10) and mu.min()>=y.min()-1e-10 and mu.max()<=y.max()+1e-10 and mu.min()>0;assert model.n_features_in_==124
            (O/'models'/f'{name}_amount.pkl').write_bytes(pickle.dumps(model,protocol=4));shutil.copy2(OLD/'models'/f'{name}_event.pkl',O/'models'/f'{name}_event.pkl');shutil.copy2(OLD/'models'/f'{name}.npz',O/'models'/f'{name}.npz');records.append(dict(fold=f,seed=s,positive_rows=len(y),features=124,nodes=nodes,leaves=leaves,max_leaf_mean_error=worst,fit_positive_MSE=float(w@((mu-y)**2)/w.sum()),fit_wall_s=time.perf_counter()-t,amount_sha256=sha(O/'models'/f'{name}_amount.pkl'),frozen_event_sha256=sha(O/'models'/f'{name}_event.pkl')));save('fit_receipts.json',records);print('GRID210_FIT',f,s,nodes,leaves,flush=True)
    save('fits.json',dict(passed=True,new_amount_forests=12,new_amount_trees=768,event_and_NET_fits=0,total_wall_s=time.perf_counter()-start,records=records));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'fit_receipts.json']});verify();archive('trained');print('GRID210_TRAINED')
def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
def publication():
    trained();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert head==remote;p=A/'trained/trained_freeze.json';assert subprocess.check_output(['git','show','HEAD:'+p.relative_to(REPO).as_posix()],cwd=REPO)==p.read_bytes();save('preheld_publication.json',dict(passed=True,git_sha=head,remote_main=head,all12pairs_published=True));print('GRID210_PUBLISHED')
def evaluate():
    trained();assert read(O/'preheld_publication.json')['passed'] and not (O/'evaluation_started.json').exists();save('evaluation_started.json',dict(all12pairs_published=True));meta,lookup,cases,raw=raw_inputs();task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];bases={(r['method'],r['sample_id']):r for r in read(OLD/'oof_trajectories.json')};oldpred={(r['seed'],r['sample_id']):r for r in read(OLD/'gate_predictions.json')};pred=[];tr=[];checks=[]
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';cp=np.load(O/'models'/f'{name}.npz');hz=np.load(CACHE/f'held_{name}.npz');x=np.c_[(hz['features']-cp['mean'])/cp['scale'],(raw[hz['ids'].astype(int)]-cp['raw_mean'])/cp['raw_scale']];event=pickle.loads((O/'models'/f'{name}_event.pkl').read_bytes());amount=pickle.loads((O/'models'/f'{name}_amount.pkl').read_bytes());p=event.predict_proba(x)[:,1];mu=amount.predict(x);manual=np.mean([tree.tree_.value[tree.apply(x),0,0] for tree in amount.estimators_],axis=0);assert np.allclose(mu,manual,atol=1e-10,rtol=1e-10) and mu.min()>0;lab=np.load(LAB/f'{name}.npz');assert mu.min()>=lab['target'].min()-1e-10 and mu.max()<=lab['target'].max()+1e-10;byidx={int(i):(bool(pp*mm>q),float(pp),float(mm),float(q)) for i,pp,mm,q in zip(hz['ids'],p,mu,hz['features'][:,82])}
            for c in [c for c in cases if bases[f'NET_seed{s}',c['sample_id']]['fold']==f]:
                base=bases[f'NET_seed{s}',c['sample_id']];j=base['baseline_stop'];eligible=j<len(c['states']);d,pp,mm,q=byidx[lookup[c['sample_id'],j]] if eligible else (False,0.,0.,0.)
                if eligible:
                    assert np.isclose(pp,oldpred[s,c['sample_id']]['event_probability'],atol=1e-11,rtol=1e-10);assert np.isclose(q,fee*c['queries'][j]/scale,atol=1e-8,rtol=2e-6);pred.append(dict(fold=f,seed=s,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=j,gate_continue=d,event_probability=pp,conditional_normalized_gain=mm,predicted_normalized_gain=pp*mm,known_charge=q,predicted_net=pp*mm-q,actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j]),actual_next_NET=float((c['gains'][j]-fee*c['queries'][j])/scale),actual_next_change=bool(c['chosen'][j]!=c['chosen'][j+1])))
                for method in ['NET','ADD1','GATE']:
                    add=bool(eligible and (method=='ADD1' or method=='GATE' and d));v=B.H.trajectory(c,np.arange(len(c['states']))<j+int(add),f'{method}_seed{s}',fee,scale);v.update(fold=f,baseline_stop=j,baseline_gain=base['gain'],baseline_queries=base['queries'],eligible=eligible,append=add,action_match=v['chosen_pool_id']==c['chosen'][-1]);assert base['queries']<=v['queries']<=int(c['queries'].sum()) and v['queries']-base['queries']<=64 and base['gain']<=v['gain'];assert not base['action_match'] or v['action_match']
                    if method in ['NET','ADD1']:
                        for k in ['gain','queries','chosen_pool_id','actions']:assert v[k]==bases[f'{method}_seed{s}',c['sample_id']][k]
                    tr.append(v)
            checks.append(dict(fold=f,seed=s,held_rows=len(p),positive_native_mu_checked=True,event_probability_frozen=True))
    metrics={f'{m}_seed{s}':B.summarize([r for r in tr if r['method']==f'{m}_seed{s}']) for m in ['NET','ADD1','GATE'] for s in range(3)};family={m:{k:float(np.mean([metrics[f'{m}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{m}_seed0']['macro']} for m in ['NET','ADD1','GATE']};counts=dict(eligible=len(pred),append=sum(r['gate_continue'] for r in pred),append_changes=sum(r['gate_continue'] and r['actual_next_change'] for r in pred),append_no_change=sum(r['gate_continue'] and not r['actual_next_change'] for r in pred),append_positive_NET=sum(r['gate_continue'] and r['actual_next_NET']>0 for r in pred));assert len(pred)==4681 and len(tr)==21402
    for n,v in [('gate_predictions.json',pred),('oof_trajectories.json',tr),('search_metrics.json',metrics),('family_summary.json',family),('gate_counts.json',counts),('held_native_checks.json',checks)]:save(n,v)
    trained();save('review.json',dict(passed=True,all4681_native_mu_and_frozen_probabilities_checked=True,all21402_NET_ADD1_controls_reproduced=True,all12weights_beforeheld=True,new_amount_forests=12,event_and_NET_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('evaluated');print('GRID210_EVALUATED',json.dumps(dict(family=family,counts=counts)))
if __name__=='__main__':globals()[sys.argv[1]]()
