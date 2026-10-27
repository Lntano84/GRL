"""Only add original public41 to conditional amount input; freeze GRID206 event."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import datetime,hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid208_public_amount';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID208_public_amount';OLD=R/'outputs/grid206_public_event';PRIOR=R/'outputs/grid203_tree_event';LINEAR=R/'outputs/grid198_conditional_gain';FIT=R/'outputs/grid192_gate_support';LAB=LINEAR/'fit_labels';CACHE=R/'outputs/grid190_extension_gate'
sp=importlib.util.spec_from_file_location('grid208_fixed_base',R/'work/grid206_public_event/run.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P);B=P.B
PARAMS=dict(n_estimators=64,max_depth=4,min_samples_leaf=10,max_features=1.0,bootstrap=False,n_jobs=1,criterion='squared_error',warm_start=False)
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
    assert not O.exists();O.mkdir(parents=True);P.trained();assert read(R/'outputs/grid207_amount_cost/independent_audit.json')['passed']
    save('design.json',dict(id='GRID208_PUBLIC_AMOUNT',authorization='Usercontinue;pretrainingnotice exactly12sameconditionalforests addingoriginal41,freezing206event delivered.',change='Only amountinput: original83hidden plus entire original41prefixsummary=124. Reuse206fit-only scalers exactly. Event206forestsbytefrozen,NETfrozen. No25diagnosticextras/featureselection.',labels='Exactly198positive-only targetgain/gain_scale,positiveindices,conditionalweights;never add mu=0targets for zeroevent. Fullfittingpopulation and currentnormalizations unchanged.',model=dict(type='ExtraTreesRegressor',params=PARAMS,seeds=[0,1,2],new_amount_forests=12,new_amount_trees=768,cap_s=600),motivation='207descriptive diagnosis:10newpositiveunprofitable ratiosmedian134;94zero caseslargercost. Test amountinput only,notcausalproof orguarantee.',capacity='Same64trees/depth4/leaf10 but124splitdimensions changefunctionclass/randomfeaturechoice. Total1536activetrees across12pairs,new768amount+frozen768event. Notstrictsamecapacity.',action='Native frozen206eventp * newpositiveamountmu > sameknownquerycharge buysone<=64 thenreturnsbestobserved. No fee/threshold/batchchange.',qualification='All124fitfeatures fromfrozen206,positiveindicesandweightsmatch198,fit12held4isolated,scalersunchanged,eventhashes. Nativeleafweightedpositive means andrange checked insideoriginal600s cap.',evaluation='Publish all12pairs beforeheld. Primary206isolates amountinput;secondary203bothheadsdiffer from208,notpureablation. 198rawreference. All4681stops/21402trajectories/all16exposedtrain dates/all3seeds;prespecifiedsameexpectedqueryvs206and203,randomADD1. Alladverse retained, noallwinor5percentthreshold.',metrics='ConditionalpositiveMSEMAE,expectedgainMSEMAE,eventBrierAUCunchanged;actionmatch/gaincapture/query/proxyNET. Preserve prior10restoredpairs and allchangeddecisions. Notfullweekquality/time.',limits='Zeroevent/NETfits,zeroNETforwards/newphysics/generation/final8/paper/newFRAME. Noextra fits/parametersearch/seedselection/capextension/automaticexpansion. Teacherprefixesandseedsnotindependent.'))
    files=[W/'run.py',O/'design.json',R/'work/grid206_public_event/run.py',R/'work/grid203_tree_event/finalize.py',R/'outputs/grid207_amount_cost/complete.json',R/'outputs/grid207_amount_cost/independent_audit.json',R/'outputs/grid207_amount_cost/summary.json',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']
    for d in [OLD,PRIOR,LINEAR]:files+=[d/n for n in ['complete.json','oof_trajectories.json','gate_predictions.json','search_metrics.json','family_summary.json','predictive_metrics.json']]
    files+=list((OLD/'models').iterdir())+list((OLD/'fit_features').glob('*.npz'))+list((FIT/'fit_features').glob('*.npz'))+list(LAB.glob('*.npz'))+list(CACHE.glob('held_f*.npz'));save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID208_PLANNED')
def qualify():
    verify();assert not (O/'preflight.json').exists();meta,_,_,raw=raw_inputs();(O/'fit_features').mkdir();records=[]
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';full=np.load(OLD/'fit_features'/f'{name}.npz');lab=np.load(LAB/f'{name}.npz');pos=lab['indices'].astype(int);ids=full['ids'][pos];cp=np.load(OLD/'models'/f'{name}.npz');x=full['standard'][pos];assert x.shape==(len(pos),124) and np.array_equal(ids,lab['metadata_ids']);assert (lab['target']>0).all() and abs(lab['sample_weight'].sum()-full['sample_weight'].sum())<1e-8;assert np.allclose(x[:,83:],(raw[ids.astype(int)]-cp['raw_mean'])/cp['raw_scale'],atol=1e-12,rtol=1e-12);assert np.isfinite(x.astype(np.float32)).all();hz=np.load(CACHE/f'held_{name}.npz');fd={meta[int(i)]['date'] for i in full['ids']};hd={meta[int(i)]['date'] for i in hz['ids']};assert len(fd)==12 and len(hd)==4 and not fd&hd
            np.savez_compressed(O/'fit_features'/f'{name}.npz',standard=x,target=lab['target'],sample_weight=lab['sample_weight'],metadata_ids=ids,positive_indices=pos);records.append(dict(fold=f,seed=s,positive_rows=len(pos),features=124,conditional_weight_sum=float(lab['sample_weight'].sum()),event_sha256=sha(OLD/'models'/f'{name}_event.pkl'),positive_min=float(lab['target'].min()),positive_max=float(lab['target'].max())))
    save('preflight.json',dict(passed=True,pools=records,all12_positive_targets_weights_scalers_eventhashes_checked=True,new_fits=0,new_NET_forwards=0));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_features').iterdir()),O/'preflight.json']});verify();archive('qualified');print('GRID208_QUALIFIED')
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
                leaf=tree.apply(x);manual.append(tree.tree_.value[leaf,0,0]);assert tree.tree_.max_depth<=4;nodes+=tree.tree_.node_count;leaves+=int(np.sum(tree.tree_.children_left==-1))
                for j in np.unique(leaf):
                    mask=leaf==j;assert mask.sum()>=10;err=abs(float(w[mask]@y[mask]/w[mask].sum())-float(tree.tree_.value[j,0,0]));worst=max(worst,err);assert err<1e-8
            mu=model.predict(x);assert np.allclose(mu,np.mean(manual,axis=0),atol=1e-10,rtol=1e-10) and mu.min()>=y.min()-1e-10 and mu.max()<=y.max()+1e-10 and mu.min()>0;assert model.n_features_in_==124
            (O/'models'/f'{name}_amount.pkl').write_bytes(pickle.dumps(model,protocol=4));shutil.copy2(OLD/'models'/f'{name}_event.pkl',O/'models'/f'{name}_event.pkl');shutil.copy2(OLD/'models'/f'{name}.npz',O/'models'/f'{name}.npz');records.append(dict(fold=f,seed=s,positive_rows=len(y),features=124,nodes=nodes,leaves=leaves,max_leaf_mean_error=worst,fit_positive_MSE=float(w@((mu-y)**2)/w.sum()),fit_wall_s=time.perf_counter()-t,amount_sha256=sha(O/'models'/f'{name}_amount.pkl'),frozen_event_sha256=sha(O/'models'/f'{name}_event.pkl')));save('fit_receipts.json',records);print('GRID208_FIT',f,s,nodes,leaves,flush=True)
    save('fits.json',dict(passed=True,new_amount_forests=12,new_amount_trees=768,event_and_NET_fits=0,total_wall_s=time.perf_counter()-start,records=records));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'fit_receipts.json']});verify();archive('trained');print('GRID208_TRAINED')
def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
def publication():
    trained();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert head==remote;p=A/'trained/trained_freeze.json';assert subprocess.check_output(['git','show','HEAD:'+p.relative_to(REPO).as_posix()],cwd=REPO)==p.read_bytes();save('preheld_publication.json',dict(passed=True,git_sha=head,remote_main=head,all12pairs_published=True));print('GRID208_PUBLISHED')
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
    trained();save('review.json',dict(passed=True,all4681_native_mu_and_frozen_probabilities_checked=True,all21402_NET_ADD1_controls_reproduced=True,all12weights_beforeheld=True,new_amount_forests=12,event_and_NET_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('evaluated');print('GRID208_EVALUATED',json.dumps(dict(family=family,counts=counts)))
if __name__=='__main__':globals()[sys.argv[1]]()
