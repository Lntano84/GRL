"""Fixed shallow ExtraTrees positive-conditional means with frozen event heads."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import datetime,hashlib,importlib.metadata,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid201_tree_amount';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID201_tree_amount';LINEAR=R/'outputs/grid198_conditional_gain';CONSTANT=R/'outputs/grid199_constant_amount';FIT=R/'outputs/grid192_gate_support';EVENT=R/'outputs/grid193_gate_precondition';CACHE=R/'outputs/grid190_extension_gate';DIRECT=R/'outputs/grid196_gain_readout'
sp=importlib.util.spec_from_file_location('grid201_frozen_helpers',R/'work/grid198_conditional_gain/run.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M);B=M.B
PARAMS=dict(n_estimators=64,max_depth=4,min_samples_leaf=10,max_features=1.0,bootstrap=False,n_jobs=1,criterion='squared_error',warm_start=False)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    for name in ['fit_labels','models']:
        if (O/name).exists():shutil.copytree(O/name,d/name)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    B.H.sealed()
def prepare():
    assert not O.exists();O.mkdir(parents=True);B.H.sealed()
    save('design.json',dict(id='GRID201_SHALLOW_TREE_CONDITIONAL_AMOUNT',authorization='User prioritize internal learning;pretrainingnotice fixed12heads each64trees/depth4/minleaf10,allotherparts frozen.',
        hypothesis='GRID200 found284/612 truepositiveSTOPs get negative conditional linearprediction,thenclippedzero. Test positive leaf-average conditionalmodel withoutlinearnegativeextrapolation;not claimingextrapolationsolecause.',
        fixed='Same198positive fitrows/targets/conditionalweights/83features/originalSTOPnormalizer. Same193frozen eventheads,fees,global12864/.9,ranking,station,aliases,baseNET. No newdatasets or inputs.',
        model=dict(type='ExtraTreesRegressor',params=PARAMS,random_state='Correspondingexisting NETseed0/1/2;notextraseedselection',fit_cap_s=600,total_head_fits=12,total_trees=768),
        capacity='Changes modelclass,capacityandinductivebias vs84parameterlinearconditionalhead. Reportactual nodes/leaves/bytes,not samecapacityorjust-outputclippingablation. Originalneuralnetworknotexpanded/retrained.',
        target='Conditionalpositive gain/gain_scale. Nativeleafpredictor weightedaverage positivefitlabels;ensemble average infitpositive labelrange. No new deploymentclip or logtransform.',
        action='sigmoid(frozen193eventlogit)*tree_conditional_mean > knownnextquerycharge buysone<=64thenreturnsbest. No .5eventcutoff/calibration/thresholdsearch.',
        qualification='Checkoldfitpositive IDs/weights/targets/normalizers/frozen eventcoefficients/dateisolation andfinitefloat32inputs without fitting. Firstandallactualforests must passleafmean/positive range/depth/minleafnativechecks insidefixedfitwindow.',
        evaluation='All12weights publishedbeforeheldscoring. All16oldtrainingdates OOF/all3NETseeds;zero newNNforwards or physics. Primary198linearconditional and199constantconditional;196directgainrawreference. Sameexpectedquerycomparison separatelyvs198and199,cost-onlyuniformthinningperdate/seed,ADD1randomreference.',
        user_boundary='Improvelearninginternallyfirst;do not launch neworiginalFRAMEclosedloopcomparison.',
        prohibited=['parametersearch/fitretries/newseeds','eventorNETtraining','newphysicalrollout/newdates/final8/paper','posthocensembleorautomaticexpansion'],
        limits='Sharedteacherprefixes/seedsnotindependent. Localreward/proxyqueryfee not wholeweekquality/reallatency. Preservealladverse outcomes,smallrepeatedgainworthretaining.'))
    files=[W/'run.py',R/'work/grid198_conditional_gain/run.py',O/'design.json',R/'outputs/grid200_amount_role/complete.json',R/'outputs/grid200_amount_role/summary.json',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid113_relational_learning/train_metadata.json']
    for d in [LINEAR,CONSTANT,DIRECT]:files += [d/n for n in ['complete.json','oof_trajectories.json','search_metrics.json','family_summary.json']]
    files+=list((LINEAR/'fit_labels').glob('*.npz'))+list((FIT/'fit_features').glob('*.npz'))+list((EVENT/'models').glob('*.npz'))+list(CACHE.glob('held_f*.npz'))+[R/p for p in read(CACHE/'code_freeze.json')];save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID201_PLANNED')
def qualify():
    verify();assert not (O/'preflight.json').exists();from sklearn.ensemble import ExtraTreesRegressor
    model=ExtraTreesRegressor(**PARAMS,random_state=0);assert not hasattr(model,'estimators_');meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');base=read(CACHE/'oof_trajectories.json');(O/'fit_labels').mkdir();records=[]
    for f in range(4):
        for s in range(3):
            source=LINEAR/'fit_labels'/f'f{f}_s{s}.npz';shutil.copy2(source,O/'fit_labels'/source.name);lab=np.load(source);z=np.load(FIT/'fit_features'/source.name);cp=np.load(EVENT/'models'/source.name);assert np.array_equal(cp['mean'],z['mean']) and np.array_equal(cp['scale'],z['scale']);assert np.array_equal(z['ids'][lab['indices']],lab['metadata_ids']);assert (lab['target']>0).all() and (lab['sample_weight']>0).all();assert np.isfinite(z['standard'][lab['indices']].astype(np.float32)).all();fd={meta[int(i)]['date'] for i in z['ids']};hd={r['date'] for r in base if r['fold']==f and r['method']==f'NET_seed{s}'};assert len(fd)==12 and len(hd)==4 and not fd&hd;records.append(dict(fold=f,seed=s,positive_rows=len(lab['target']),conditional_dates=len({meta[int(i)]['date'] for i in lab['metadata_ids']}),conditional_calls=len({meta[int(i)]['physical_state_group'] for i in lab['metadata_ids']}),positive_target_min=float(lab['target'].min()),positive_target_max=float(lab['target'].max()),source_target_sha256=sha(source)))
    save('preflight.json',dict(passed=True,params=PARAMS,sklearn_version=importlib.metadata.version('scikit-learn'),all12positive_targets_weights_date_splits_event_normalizers_match=True,pools=records,new_forest_fits=0,new_NET_forwards=0));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_labels').iterdir()),O/'preflight.json']});verify();archive('qualified');print('GRID201_QUALIFIED')
def train():
    verify();assert read(O/'preflight.json')['passed'] and not (O/'training_started.json').exists();from sklearn.ensemble import ExtraTreesRegressor
    for p,h in read(O/'qualified_freeze.json').items():assert sha(O/p)==h
    save('training_started.json',dict(head_fits=12,trees_per_head=64,total_trees=768,wall_cap_s=600,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));(O/'models').mkdir();start=time.perf_counter();records=[]
    for f in range(4):
        for s in range(3):
            if time.perf_counter()-start>600:raise TimeoutError('Original600s cap;preservepartials,noautomaticretry')
            name=f'f{f}_s{s}';z=np.load(FIT/'fit_features'/f'{name}.npz');lab=np.load(O/'fit_labels'/f'{name}.npz');event=np.load(EVENT/'models'/f'{name}.npz');x=z['standard'][lab['indices']];y,w=lab['target'],lab['sample_weight'];t=time.perf_counter();model=ExtraTreesRegressor(**PARAMS,random_state=s);model.fit(x,y,sample_weight=w);manual=[];leafcount=0;nodecount=0;maxerr=0.
            for tree in model.estimators_:
                leaf=tree.apply(x);value=tree.tree_.value[leaf,0,0];manual.append(value);assert tree.tree_.max_depth<=4
                for j in np.unique(leaf):
                    mask=leaf==j;assert mask.sum()>=10;expected=float(w[mask]@y[mask]/w[mask].sum());err=abs(expected-float(tree.tree_.value[j,0,0]));maxerr=max(maxerr,err);assert err<1e-8;assert float(y.min())-1e-10<=expected<=float(y.max())+1e-10
                leafcount+=int(np.sum(tree.tree_.children_left==-1));nodecount+=tree.tree_.node_count
            native=model.predict(x);assert np.allclose(native,np.mean(manual,axis=0),atol=1e-10,rtol=1e-10);assert native.min()>0 and native.max()<=y.max()+1e-8
            (O/'models'/f'{name}.pkl').write_bytes(pickle.dumps(model,protocol=4));np.savez_compressed(O/'models'/f'{name}.npz',event_coef=event['coef'][0],event_intercept=event['intercept'],mean=z['mean'],scale=z['scale'],positive_min=np.array([y.min()]),positive_max=np.array([y.max()]));records.append(dict(fold=f,seed=s,positive_rows=len(y),trees=len(model.estimators_),total_nodes=nodecount,total_leaves=leafcount,fit_wall_s=time.perf_counter()-t,fit_positive_MSE=float(w@((native-y)**2)/w.sum()),max_leaf_mean_error=maxerr,forest_bytes=(O/'models'/f'{name}.pkl').stat().st_size,forest_sha256=sha(O/'models'/f'{name}.pkl'),frozen_event_sha256=sha(EVENT/'models'/f'{name}.npz')));save('fit_receipts.json',records);print('FIT_POSITIVE_TREE',f,s,nodecount,leafcount,flush=True)
    save('fits.json',dict(passed=True,new_forest_heads=12,new_trees=768,event_and_NET_fits=0,total_wall_s=time.perf_counter()-start,records=records));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'fit_receipts.json']});verify();archive('trained');print('GRID201_ALL12_TRAINED')
def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h
def publication():
    trained();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert head==remote;p=A/'trained/trained_freeze.json';assert subprocess.check_output(['git','show','HEAD:'+p.relative_to(REPO).as_posix()],cwd=REPO)==p.read_bytes();save('preheld_publication.json',dict(passed=True,git_sha=head,remote_main=head,all12forest_eventpairs_published=True));print('GRID201_PUBLISHED')
def evaluate():
    trained();assert read(O/'preheld_publication.json')['passed'] and not (O/'evaluation_started.json').exists();save('evaluation_started.json',dict(all12pairedweights_published=True));meta,lookup,cases=B.load();task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];old=read(LINEAR/'oof_trajectories.json');bases={(r['method'],r['sample_id']):r for r in old};pred=[];tr=[];checks=[]
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';cp=np.load(O/'models'/f'{name}.npz');model=pickle.loads((O/'models'/f'{name}.pkl').read_bytes());hz=np.load(CACHE/f'held_{name}.npz');x=(hz['features']-cp['mean'])/cp['scale'];logit=x@cp['event_coef']+cp['event_intercept'][0];p=np.exp(-np.logaddexp(0.,-logit));mu=model.predict(x);manual=np.mean([tree.tree_.value[tree.apply(x),0,0] for tree in model.estimators_],axis=0);assert np.allclose(mu,manual,atol=1e-10,rtol=1e-10);assert mu.min()>=cp['positive_min'][0]-1e-8 and mu.max()<=cp['positive_max'][0]+1e-8 and (mu>0).all();g=p*mu;charge=hz['features'][:,82];byidx={int(i):(bool(gg>q),float(pp),float(mm),float(gg),float(q)) for i,pp,mm,gg,q in zip(hz['ids'],p,mu,g,charge)};checks.append(dict(fold=f,seed=s,held_rows=len(mu),native_leafmeans_match=True,positive_and_in_fit_label_range=True))
            for c in [c for c in cases if bases[f'NET_seed{s}',c['sample_id']]['fold']==f]:
                base=bases[f'NET_seed{s}',c['sample_id']];j=base['baseline_stop'];eligible=j<len(c['states']);d,prob,amount,gain,cost=byidx[lookup[c['sample_id'],j]] if eligible else (False,0.,0.,0.,0.)
                if eligible:
                    assert np.isclose(cost,fee*c['queries'][j]/scale,atol=1e-8,rtol=2e-6);pred.append(dict(fold=f,seed=s,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=j,gate_continue=d,event_probability=prob,conditional_normalized_gain=amount,predicted_normalized_gain=gain,known_charge=cost,predicted_net=gain-cost,actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j]),actual_next_NET=float((c['gains'][j]-fee*c['queries'][j])/scale),actual_next_change=bool(c['chosen'][j]!=c['chosen'][j+1])))
                for method in ['NET','ADD1','GATE']:
                    add=bool(eligible and (method=='ADD1' or method=='GATE' and d));end=j+int(add);name1=f'{method}_seed{s}';v=B.H.trajectory(c,np.arange(len(c['states']))<end,name1,fee,scale);v.update(fold=f,baseline_stop=j,baseline_gain=base['gain'],baseline_queries=base['queries'],eligible=eligible,append=add,action_match=v['chosen_pool_id']==c['chosen'][-1]);assert base['queries']<=v['queries']<=int(c['queries'].sum()) and v['queries']-base['queries']<=64 and base['gain']<=v['gain'];assert not base['action_match'] or v['action_match']
                    if method in ['NET','ADD1']:
                        for k in ['gain','queries','chosen_pool_id','actions']:assert v[k]==bases[name1,c['sample_id']][k]
                    tr.append(v)
    metrics={f'{method}_seed{s}':B.summarize([r for r in tr if r['method']==f'{method}_seed{s}']) for method in ['NET','ADD1','GATE'] for s in range(3)};families={method:{k:float(np.mean([metrics[f'{method}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{method}_seed0']['macro']} for method in ['NET','ADD1','GATE']};counts=dict(eligible=len(pred),append=sum(p['gate_continue'] for p in pred),append_changes=sum(p['gate_continue'] and p['actual_next_change'] for p in pred),append_no_change=sum(p['gate_continue'] and not p['actual_next_change'] for p in pred),append_positive_NET=sum(p['gate_continue'] and p['actual_next_NET']>0 for p in pred));assert len(pred)==4681 and len(tr)==21402
    save('gate_predictions.json',pred);save('oof_trajectories.json',tr);save('search_metrics.json',metrics);save('family_summary.json',families);save('gate_counts.json',counts);save('held_leaf_checks.json',checks);trained();save('review.json',dict(passed=True,all21402trajectories_and_controls_reproduced=True,all4681_native_positiveleafmeans_and_query_bounds_checked=True,all12weights_before_held=True,new_forest_heads=12,new_trees=768,event_and_NET_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('evaluated');print('GRID201_EVALUATED',json.dumps(dict(families=families,counts=counts),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
