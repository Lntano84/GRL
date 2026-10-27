"""Replace only event readout by fixed shallow classification forests."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import datetime,hashlib,importlib.metadata,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid203_tree_event';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID203_tree_event';OLD=R/'outputs/grid201_tree_amount';FIT=R/'outputs/grid192_gate_support';LAB=R/'outputs/grid196_gain_readout/fit_labels';CACHE=R/'outputs/grid190_extension_gate';LINEAR=R/'outputs/grid198_conditional_gain';DIRECT=R/'outputs/grid196_gain_readout'
sp=importlib.util.spec_from_file_location('grid203_frozen_helpers',R/'work/grid201_tree_amount/run.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M);B=M.B
PARAMS=dict(n_estimators=64,max_depth=4,min_samples_leaf=10,max_features=1.0,bootstrap=False,n_jobs=1,criterion='gini',class_weight=None,warm_start=False)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    if (O/'models').exists():shutil.copytree(O/'models',d/'models')
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    B.H.sealed()
def prepare():
    assert not O.exists();O.mkdir(parents=True);B.H.sealed()
    save('design.json',dict(id='GRID203_SHALLOW_TREE_EVENT',authorization='User继续 internallearning;pretrainingnotice12fixedeventforests,64trees/depth4/minleaf10,frozen201amount/NET.',
        hypothesis='GRID202eventfitSTOPAUC.895 vsheld.635 andmissedprofitmass61percent event-onlysinglehindsightrepair. Test event readout transfer,not claimcausalproof.',
        fixed='Exactly192fullmixture83standardizedfeatures/halfSTOPhalfactualCONT/originalweights. Target nextgain>0 equals originalnextactionchange;no fee/gainmagnitudeclassweight. Freeze byteidentical201conditionalforests and originalNET. No datashift/threshold/fee/global12864/.9/rank/station changes.',
        model=dict(type='ExtraTreesClassifier',params=PARAMS,random_state='existingNETseed0/1/2',new_event_head_fits=12,new_event_trees=768,cap_s=600),
        capacity='Neweventforest replaces84parameterlogisticreadout;pairedamountforest64treesfrozen. Total1536active trees across12pairs,new768eventtrees. Changedclass/capacity/inductivebias,not samecapacityablation. Originalneuralnetworknotexpanded.',
        action='Nativeweightedleafeventfrequency averaged64trees times frozen201positiveconditionalmean > knownnextquerycharge buysone<=64thenreturnsbest. No .5eventcutoff/calibration/search.',
        qualification='Zero-fitchecklabels/gainidentity/sampling/dateisolation/float32/amountSHA. Allactualforestsleafweightedclassfrequencies/probability/nativepredictions/depth/minleafcheckedinsideoriginalfitcap. Copiedamountweightsmustbyte-match201.',
        evaluation='Publishall12pairedweights beforeheld. All16historicaldatesOOF/all3seeds;primary201isolates changedhead,198higherreward reference and196rawreference. Prespecifiedcost-onlysameexpectedquerycomparisonsvs201and198,ADD1randomreference,alladversekept.',
        metrics='EventBrier/AUC/BCE,amountMSEunchanged,expectedgainMSE/MAE,actionmatch/gaincapture/query/proxyNET. BCEclip1e-15fornumericmetriconly,notinference.',
        limits='Exposedteacherpathdevelopment,notsignificance/newclosedloopquality. No neworiginalFRAMEcomparison/NNforward/physics/newdata/final8/paper. No furtherfits/retries/parametersearch/automaticexpansion.'))
    files=[W/'run.py',R/'work/grid201_tree_amount/run.py',O/'design.json',R/'outputs/grid202_factor_bottleneck/complete.json',R/'outputs/grid202_factor_bottleneck/summary.json',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid113_relational_learning/train_metadata.json']
    for d in [OLD,LINEAR,DIRECT]:files += [d/n for n in ['complete.json','oof_trajectories.json','search_metrics.json','family_summary.json']]
    files += list((OLD/'models').iterdir())+list((FIT/'fit_features').glob('*.npz'))+list(LAB.glob('*.npz'))+list(CACHE.glob('held_f*.npz'))+[R/p for p in read(CACHE/'code_freeze.json')];save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID203_PLANNED')
def qualify():
    verify();assert not (O/'preflight.json').exists();from sklearn.ensemble import ExtraTreesClassifier
    candidate=ExtraTreesClassifier(**PARAMS,random_state=0);assert not hasattr(candidate,'estimators_');meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');base=read(OLD/'oof_trajectories.json');records=[]
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';z=np.load(FIT/'fit_features'/f'{name}.npz');lab=np.load(LAB/f'{name}.npz');cp=np.load(OLD/'models'/f'{name}.npz');assert np.array_equal(z['y']>0,lab['target']>0);assert np.isfinite(z['standard'].astype(np.float32)).all();assert np.array_equal(cp['mean'],z['mean']) and np.array_equal(cp['scale'],z['scale']);assert abs(z['sample_weight'].sum()-int(z['original_STOP_count']))<1e-8;assert set(z['y'].tolist())=={0,1};fd={meta[int(i)]['date'] for i in z['ids']};hd={r['date'] for r in base if r['fold']==f and r['method']==f'NET_seed{s}'};assert len(fd)==12 and len(hd)==4 and not fd&hd;records.append(dict(fold=f,seed=s,rows=len(z['y']),positives=int(z['y'].sum()),weighted_event_prior=float(z['sample_weight']@z['y']/z['sample_weight'].sum()),amount_sha256=sha(OLD/'models'/f'{name}.pkl')))
    save('preflight.json',dict(passed=True,params=PARAMS,sklearn_version=importlib.metadata.version('scikit-learn'),all12eventlabels_weights_date_splits_and_amounts_checked=True,pools=records,new_event_fits=0,new_NET_forwards=0));save('qualified_freeze.json',{'preflight.json':sha(O/'preflight.json')});verify();archive('qualified');print('GRID203_QUALIFIED')
def train():
    verify();assert read(O/'preflight.json')['passed'] and not (O/'training_started.json').exists();from sklearn.ensemble import ExtraTreesClassifier
    for p,h in read(O/'qualified_freeze.json').items():assert sha(O/p)==h
    save('training_started.json',dict(new_event_heads=12,new_event_trees=768,cap_s=600,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));(O/'models').mkdir();records=[];start=time.perf_counter()
    for f in range(4):
        for s in range(3):
            if time.perf_counter()-start>600:raise TimeoutError('Original600s cap;preservepartialfits')
            name=f'f{f}_s{s}';z=np.load(FIT/'fit_features'/f'{name}.npz');x,y,w=z['standard'],z['y'].astype(int),z['sample_weight'];t=time.perf_counter();clf=ExtraTreesClassifier(**PARAMS,random_state=s);clf.fit(x,y,sample_weight=w);assert np.array_equal(clf.classes_,[0,1]);manual=[];maxerr=0.;nodes=0;leaves=0
            for tree in clf.estimators_:
                leaf=tree.apply(x);v=tree.tree_.value[leaf,0,:];prob=v[:,1]/v.sum(axis=1);manual.append(prob);assert tree.tree_.max_depth<=4
                for j in np.unique(leaf):
                    mask=leaf==j;assert mask.sum()>=10;truth=float(w[mask]@y[mask]/w[mask].sum());val=tree.tree_.value[j,0,:];err=abs(truth-float(val[1]/val.sum()));maxerr=max(maxerr,err);assert err<1e-9
                nodes+=tree.tree_.node_count;leaves+=int(np.sum(tree.tree_.children_left==-1))
            native=clf.predict_proba(x)[:,1];assert np.allclose(native,np.mean(manual,axis=0),atol=1e-11,rtol=1e-10);assert native.min()>=0 and native.max()<=1;assert abs(float(w@native/w.sum())-float(w@y/w.sum()))<1e-10
            (O/'models'/f'{name}_event.pkl').write_bytes(pickle.dumps(clf,protocol=4));shutil.copy2(OLD/'models'/f'{name}.pkl',O/'models'/f'{name}_amount.pkl');assert sha(OLD/'models'/f'{name}.pkl')==sha(O/'models'/f'{name}_amount.pkl');cp=np.load(OLD/'models'/f'{name}.npz');np.savez_compressed(O/'models'/f'{name}.npz',mean=cp['mean'],scale=cp['scale'],positive_min=cp['positive_min'],positive_max=cp['positive_max']);records.append(dict(fold=f,seed=s,rows=len(y),trees=64,nodes=nodes,leaves=leaves,fit_Brier=float(w@((native-y)**2)/w.sum()),max_leaf_probability_error=maxerr,fit_wall_s=time.perf_counter()-t,event_bytes=(O/'models'/f'{name}_event.pkl').stat().st_size,event_sha256=sha(O/'models'/f'{name}_event.pkl'),frozen_amount_sha256=sha(O/'models'/f'{name}_amount.pkl')));save('fit_receipts.json',records);print('FIT_EVENT_TREE',f,s,nodes,leaves,flush=True)
    save('fits.json',dict(passed=True,new_event_forests=12,new_event_trees=768,amount_and_NET_fits=0,total_wall_s=time.perf_counter()-start,records=records));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'fit_receipts.json']});verify();archive('trained');print('GRID203_ALL12_TRAINED')
def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h
def publication():
    trained();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert head==remote;p=A/'trained/trained_freeze.json';assert subprocess.check_output(['git','show','HEAD:'+p.relative_to(REPO).as_posix()],cwd=REPO)==p.read_bytes();save('preheld_publication.json',dict(passed=True,git_sha=head,remote_main=head,all12event_amount_pairs_published=True));print('GRID203_PUBLISHED')
def evaluate():
    trained();assert read(O/'preheld_publication.json')['passed'] and not (O/'evaluation_started.json').exists();save('evaluation_started.json',dict(all12pairedweights_published=True));meta,lookup,cases=B.load();task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];bases={(r['method'],r['sample_id']):r for r in read(OLD/'oof_trajectories.json')};oldpred={(r['seed'],r['sample_id']):r for r in read(OLD/'gate_predictions.json')};pred=[];tr=[];checks=[]
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';cp=np.load(O/'models'/f'{name}.npz');clf=pickle.loads((O/'models'/f'{name}_event.pkl').read_bytes());amount=pickle.loads((O/'models'/f'{name}_amount.pkl').read_bytes());hz=np.load(CACHE/f'held_{name}.npz');x=(hz['features']-cp['mean'])/cp['scale'];prob=clf.predict_proba(x)[:,1];mu=amount.predict(x);manual=[]
            for tree in clf.estimators_:v=tree.tree_.value[tree.apply(x),0,:];manual.append(v[:,1]/v.sum(axis=1))
            assert np.allclose(prob,np.mean(manual,axis=0),atol=1e-11,rtol=1e-10);assert prob.min()>=0 and prob.max()<=1 and mu.min()>0;gain=prob*mu;charge=hz['features'][:,82];byidx={int(i):(bool(g>q),float(p),float(m),float(g),float(q)) for i,p,m,g,q in zip(hz['ids'],prob,mu,gain,charge)};checks.append(dict(fold=f,seed=s,held_rows=len(prob),native_leaf_probabilities_match=True,amount_sha256=sha(O/'models'/f'{name}_amount.pkl')))
            for c in [c for c in cases if bases[f'NET_seed{s}',c['sample_id']]['fold']==f]:
                base=bases[f'NET_seed{s}',c['sample_id']];j=base['baseline_stop'];eligible=j<len(c['states']);d,p,m,g,q=byidx[lookup[c['sample_id'],j]] if eligible else (False,0.,0.,0.,0.)
                if eligible:
                    assert np.isclose(q,fee*c['queries'][j]/scale,atol=1e-8,rtol=2e-6);assert np.isclose(m,oldpred[s,c['sample_id']]['conditional_normalized_gain'],atol=1e-10,rtol=1e-10);pred.append(dict(fold=f,seed=s,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=j,gate_continue=d,event_probability=p,conditional_normalized_gain=m,predicted_normalized_gain=g,known_charge=q,predicted_net=g-q,actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j]),actual_next_NET=float((c['gains'][j]-fee*c['queries'][j])/scale),actual_next_change=bool(c['chosen'][j]!=c['chosen'][j+1])))
                for method in ['NET','ADD1','GATE']:
                    add=bool(eligible and (method=='ADD1' or method=='GATE' and d));end=j+int(add);name1=f'{method}_seed{s}';v=B.H.trajectory(c,np.arange(len(c['states']))<end,name1,fee,scale);v.update(fold=f,baseline_stop=j,baseline_gain=base['gain'],baseline_queries=base['queries'],eligible=eligible,append=add,action_match=v['chosen_pool_id']==c['chosen'][-1]);assert base['queries']<=v['queries']<=int(c['queries'].sum()) and v['queries']-base['queries']<=64 and base['gain']<=v['gain'];assert not base['action_match'] or v['action_match']
                    if method in ['NET','ADD1']:
                        for k in ['gain','queries','chosen_pool_id','actions']:assert v[k]==bases[name1,c['sample_id']][k]
                    tr.append(v)
    metrics={f'{method}_seed{s}':B.summarize([r for r in tr if r['method']==f'{method}_seed{s}']) for method in ['NET','ADD1','GATE'] for s in range(3)};families={method:{k:float(np.mean([metrics[f'{method}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{method}_seed0']['macro']} for method in ['NET','ADD1','GATE']};counts=dict(eligible=len(pred),append=sum(r['gate_continue'] for r in pred),append_changes=sum(r['gate_continue'] and r['actual_next_change'] for r in pred),append_no_change=sum(r['gate_continue'] and not r['actual_next_change'] for r in pred),append_positive_NET=sum(r['gate_continue'] and r['actual_next_NET']>0 for r in pred));assert len(pred)==4681 and len(tr)==21402;save('gate_predictions.json',pred);save('oof_trajectories.json',tr);save('search_metrics.json',metrics);save('family_summary.json',families);save('gate_counts.json',counts);save('held_native_checks.json',checks);trained();save('review.json',dict(passed=True,all21402trajectories_NET_ADD1_controls_reproduced=True,all4681event_probabilities_and_frozen_amounts_checked=True,all12pairs_before_held=True,new_event_heads=12,new_event_trees=768,amount_and_NET_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('evaluated');print('GRID203_EVALUATED',json.dumps(dict(families=families,counts=counts),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
