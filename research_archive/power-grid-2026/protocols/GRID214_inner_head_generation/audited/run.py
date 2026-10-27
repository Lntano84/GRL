"""Execute the fixed GRID213 head-only generation contract; no router/physics."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import datetime,hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid214_inner_head_generation';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID214_inner_head_generation';PLAN=R/'outputs/grid213_inner_head_plan';FIT=R/'outputs/grid192_gate_support';PUBLIC=R/'outputs/grid205_public_separation/public_descriptors.json';META=R/'outputs/grid113_relational_learning/train_metadata.json'
sp=importlib.util.spec_from_file_location('grid214_frozen_plan',R/'work/grid213_inner_head_plan/run.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
PHASE='import';ACCESS=[]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def array(p,kind):
    if kind=='score_targets':assert PHASE=='join','Score targets cannot be parsed before all predictions are sealed.'
    ACCESS.append(dict(kind=kind,path=p.relative_to(R).as_posix()));return np.load(p)
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    folders={'trained':['models'],'scored':['predictions'],'labelled':['labelled_samples']}.get(phase,[])
    for name in folders:shutil.copytree(O/name,d/name)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    P.verify()
    for p,h in read(PLAN/'qualified_freeze.json').items():assert sha(PLAN/p)==h,p
def matrix(bundle,role):
    assert role in ['fit','score'];q=array(PLAN/'qualified_inputs'/f'{bundle}.npz','qualified_inputs');record=next(r for r in read(O/'bundles.json') if r['bundle']==bundle);name=f"f{record['outer_fold']}_s{record['seed']}";z=array(FIT/'fit_features'/f'{name}.npz','public_features83');raw41=array(O/'public_raw41.npz','public_features41')['values'];pos=q[f'{role}_positions'];ids=q[f'{role}_meta_ids'];x=np.c_[(z['x'][pos]-q['mean83'])/q['scale83'],(raw41[ids.astype(int)]-q['mean41'])/q['scale41']];assert x.shape==(len(pos),124) and np.isfinite(x.astype(np.float32)).all();assert hashlib.sha256(x.tobytes()).hexdigest()==str(q[f'{role}_X_sha256'][0]);return x,ids.copy(),z['x'][pos,82].copy(),q
def prepare():
    assert not O.exists();O.mkdir(parents=True);P.verify();assert read(PLAN/'independent_audit.json')['passed'] and read(PLAN/'complete.json')['actual_fits']==0;public=read(PUBLIC);raw=np.asarray(public['values'],float)[:,:41];assert raw.shape==(3582,41) and public['metadata_indices']==list(range(3582));np.savez_compressed(O/'public_raw41.npz',values=raw);save('jobs.json',read(PLAN/'future_jobs.json'));save('bundles.json',read(PLAN/'qualified_bundles.json'));save('configs.json',read(PLAN/'estimator_configs.json'))
    save('design.json',dict(id='GRID214_INNER_HEAD_GENERATION',authorization='User开始吧 authorizes fixedGRID213 108headfits,theninnerscoringandsamplecoverageonly.',scope='Exactly108smallforests,6912trees in213ordinalorder;36bundles8fit4scorewithinouter12;eventdepth4/amountdepth4/depth5,64trees,minleaf10,currentconfigs. No router/combinedpolicy/outerevaluation/NNfitsorforwards/physics.',inputs='Consume213qualifiedindices/scalers/weights/fitlabels unchanged. Feature-onlyraw41from205descriptorfirst41,hash-qualifiedagainstall213fit/scorematrixhashes. No casespickleparsed or scoretargetsparsedinfit/scoreworker.',barriers='All108weights archivedandGitpublished beforeinnerscores. All36predictionfiles archivedandGitpublishedbefore scoretargetsparsed/joined. Inputparseaccesslogs perphase.',fit_cap_s=600,failures='Originalone600s windowincludesfit/nativefitchecks,neverautomaticextension/refit/retrycompletedmodels. Preserveallpartialoutputsanderrors.',evaluation='Generationquality/coverage only:alloriginalSTOP/preSTOPCONT separate,allouterfolds/seeds/innerdates,all cancellationlabelsandcontinuousactualNET. Notcancelpolicyrewardorfullweek/time;nomodelselection.',dependency='Head-onlydatecrossgeneration,notfullynestedpipeline:baseNETandSTOPselectionfrozenouter12inclinner4,outer4ancestorsdisjointper213graph. No reuseotherouterOOFmodels.',prohibited=['routerfitting/deployment/combination','newfeatures/fees/thresholds/batches/ranks','extraheadfits/seeds/depths','newscenarios/final8/paper/FRAME']))
    files=[W/'run.py',O/'design.json',O/'jobs.json',O/'bundles.json',O/'configs.json',O/'public_raw41.npz',PUBLIC,META,R/'work/grid213_inner_head_plan/run.py',PLAN/'design.json',PLAN/'qualification.json',PLAN/'independent_audit.json',PLAN/'dependency_graph.json',PLAN/'qualified_freeze.json',R/'outputs/grid212_cancellation_support/summary.json']+[PLAN/p for p in read(PLAN/'qualified_freeze.json')]+list((FIT/'fit_features').glob('*.npz'));save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID214_PLANNED')
def qualify():
    verify();assert not (O/'preflight.json').exists();from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
    meta=read(META);bundles=read(O/'bundles.json');jobs=read(O/'jobs.json');assert len(bundles)==36 and len(jobs)==108;assert jobs==read(PLAN/'future_jobs.json');assert [j['ordinal'] for j in jobs]==list(range(108));assert len({(j['bundle'],j['head']) for j in jobs})==108;records=[]
    for r in bundles:
        bundle=r['bundle'];x,ids,charge,q=matrix(bundle,'fit');xh,hi,_,_=matrix(bundle,'score');fit=array(PLAN/'fit_targets'/f'{bundle}.npz','fit_targets');assert np.array_equal(fit['event_y']>0,np.isin(np.arange(len(x)),fit['amount_positive_positions']));assert len(set(fit['event_y'].tolist()))==2 and len(fit['amount_y'])>=10 and (fit['amount_y']>0).all();assert abs(q['event_weights'].sum()-int(q['inner_STOP_count'][0]))<1e-8;assert abs(fit['amount_weights'].sum()-q['event_weights'].sum())<1e-8;assert set(meta[int(i)]['date'] for i in ids)==set(r['fit_dates']);assert set(meta[int(i)]['date'] for i in hi)==set(r['score_dates']);assert not set(meta[int(i)]['date'] for i in np.r_[ids,hi])&set(r['outer_evaluation_dates']);name=f"f{r['outer_fold']}_s{r['seed']}"
        for head in ['event','amount_depth4','amount_depth5']:
            params=read(O/'configs.json')[name][head];model=(ExtraTreesClassifier if head=='event' else ExtraTreesRegressor)(**params);assert not hasattr(model,'estimators_');assert params['n_estimators']==64 and params['min_samples_leaf']==10 and params['random_state']==r['seed']
        records.append(dict(bundle=bundle,fit_rows=len(x),score_rows=len(xh),positive_fit_rows=len(fit['amount_y']),all_qualified_matrix_hashes_reproduced=True))
    assert not any(a['kind']=='score_targets' for a in ACCESS);save('preflight.json',dict(passed=True,bundles=records,all108configs_and_36matrix_hash_pairs_checked=True,score_target_parses=0,new_fits=0,new_predictions=0));save('qualify_access.json',ACCESS);verify();archive('qualified');print('GRID214_QUALIFIED')
def train():
    verify();assert read(O/'preflight.json')['passed'] and not (O/'training_started.json').exists();from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
    save('training_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fits=108,cap_s=600));(O/'models').mkdir();records=[];start=time.perf_counter();data={}
    for job in read(O/'jobs.json'):
        if time.perf_counter()-start>600:raise TimeoutError('Original600s cap: preserve partialmodels, noautomaticextension/refit.')
        bundle,head=job['bundle'],job['head'];name=f"f{job['outer_fold']}_s{job['seed']}";params=read(O/'configs.json')[name][head]
        if bundle not in data:
            x,ids,_,q=matrix(bundle,'fit');fit=array(PLAN/'fit_targets'/f'{bundle}.npz','fit_targets');data[bundle]=(x,q,fit)
        x,q,fit=data[bundle];event=head=='event';xx=x if event else x[fit['amount_positive_positions']];y=fit['event_y'] if event else fit['amount_y'];w=q['event_weights'] if event else fit['amount_weights'];t=time.perf_counter();model=(ExtraTreesClassifier if event else ExtraTreesRegressor)(**params);model.fit(xx,y,sample_weight=w);nodes=leaves=0;worst=0.;manual=[]
        assert model.n_features_in_==124 and len(model.estimators_)==64
        for tree in model.estimators_:
            leaf=tree.apply(xx);v=tree.tree_.value[leaf,0,:];manual.append(v[:,1]/v.sum(1) if event else v[:,0]);assert tree.tree_.max_depth<=params['max_depth'];nodes+=tree.tree_.node_count;leaves+=int(np.sum(tree.tree_.children_left==-1))
            for j in np.unique(leaf):
                m=leaf==j;assert m.sum()>=10;truth=float(w[m]@y[m]/w[m].sum());v=tree.tree_.value[j,0,:];value=float(v[1]/v.sum()) if event else float(v[0]);worst=max(worst,abs(value-truth));assert abs(value-truth)<1e-8
        native=model.predict_proba(xx)[:,1] if event else model.predict(xx);assert np.allclose(native,np.mean(manual,axis=0),atol=1e-10,rtol=1e-10)
        if event:assert np.array_equal(model.classes_,[0,1]) and native.min()>=0 and native.max()<=1
        else:assert native.min()>0 and native.min()>=y.min()-1e-10 and native.max()<=y.max()+1e-10
        p=O/'models'/f'{bundle}_{head}.pkl';p.write_bytes(pickle.dumps(model,protocol=4));records.append(dict(ordinal=job['ordinal'],bundle=bundle,head=head,fit_rows=len(xx),nodes=nodes,leaves=leaves,max_leaf_error=worst,fit_weighted_error=float(w@((native-y)**2)/w.sum()),target_min=float(y.min()),target_max=float(y.max()),fit_wall_s=time.perf_counter()-t,model_sha256=sha(p)));save('fit_receipts.json',records)
        if len(records)%9==0:print('GRID214_FITS',len(records),'/108',flush=True)
    assert len(records)==108 and len(list((O/'models').glob('*.pkl')))==108 and not any(a['kind']=='score_targets' for a in ACCESS);save('fits.json',dict(passed=True,new_forests=108,new_trees=6912,new_NN_fits=0,total_fit_and_native_check_wall_s=time.perf_counter()-start,records=records));save('train_access.json',ACCESS);save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'fit_receipts.json',O/'train_access.json']});verify();archive('trained');print('GRID214_ALL108_TRAINED')
def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
def publication():
    trained();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert head==remote;p=A/'trained/trained_freeze.json';assert subprocess.check_output(['git','show','HEAD:'+p.relative_to(REPO).as_posix()],cwd=REPO)==p.read_bytes();save('pre_score_publication.json',dict(passed=True,git_sha=head,remote_main=head,all108weights_published=True));print('GRID214_WEIGHTS_PUBLISHED')
def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all108weights_published=True));(O/'predictions').mkdir();records=[];fitrec={(r['bundle'],r['head']):r for r in read(O/'fit_receipts.json')}
    for r in read(O/'bundles.json'):
        bundle=r['bundle'];x,ids,charge,_=matrix(bundle,'score');scores={}
        for head in ['event','amount_depth4','amount_depth5']:
            event=head=='event';model=pickle.loads((O/'models'/f'{bundle}_{head}.pkl').read_bytes());value=model.predict_proba(x)[:,1] if event else model.predict(x);manual=[]
            for tree in model.estimators_:v=tree.tree_.value[tree.apply(x),0,:];manual.append(v[:,1]/v.sum(1) if event else v[:,0])
            assert np.allclose(value,np.mean(manual,axis=0),atol=1e-10,rtol=1e-10)
            if event:assert value.min()>=0 and value.max()<=1
            else:assert value.min()>0 and value.min()>=fitrec[bundle,head]['target_min']-1e-10 and value.max()<=fitrec[bundle,head]['target_max']+1e-10
            scores[head]=value
        a=scores['event']*scores['amount_depth4']>charge;b=scores['event']*scores['amount_depth5']>charge;np.savez_compressed(O/'predictions'/f'{bundle}.npz',meta_ids=ids,event_p=scores['event'],mu4=scores['amount_depth4'],mu5=scores['amount_depth5'],public_charge=charge,continue4=a,continue5=b,candidate_cancel=a&~b);records.append(dict(bundle=bundle,score_rows=len(ids),candidate_cancel_rows=int((a&~b).sum()),targets_parsed=False))
    assert len(records)==36 and not any(a['kind']=='score_targets' for a in ACCESS);save('score_receipts.json',records);save('score_access.json',ACCESS);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'predictions').iterdir()),O/'score_receipts.json',O/'score_access.json']});archive('scored');print('GRID214_ALL36_SCORED_NO_TARGET_JOIN')
def scored():
    trained()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p
def score_publication():
    scored();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert head==remote;p=A/'scored/scored_freeze.json';assert subprocess.check_output(['git','show','HEAD:'+p.relative_to(REPO).as_posix()],cwd=REPO)==p.read_bytes();save('pre_join_publication.json',dict(passed=True,git_sha=head,remote_main=head,all36predictions_published_before_labels=True));print('GRID214_PREDICTIONS_PUBLISHED')
def join():
    scored();assert read(O/'pre_join_publication.json')['passed'] and not (O/'join_started.json').exists();save('join_started.json',dict(all108weights_and36predictionfiles_published=True));(O/'labelled_samples').mkdir();meta=read(META);records=[];candidate_rows=[]
    for r in read(O/'bundles.json'):
        bundle=r['bundle'];p=array(O/'predictions'/f'{bundle}.npz','sealed_predictions');target=array(PLAN/'score_targets'/f'{bundle}.npz','score_targets');q=array(PLAN/'qualified_inputs'/f'{bundle}.npz','qualified_inputs');ids=p['meta_ids'].astype(int);assert np.array_equal(ids,q['score_meta_ids']) and np.array_equal(p['public_charge'],target['public_charge']);assert np.array_equal(target['event_y']>0,target['gain_normalized']>0);assert np.allclose(target['actual_NET'],target['gain_normalized']-target['known_charge'],atol=1e-12,rtol=1e-12);assert {meta[int(i)]['date'] for i in ids}==set(r['score_dates']) and not {meta[int(i)]['date'] for i in ids}&set(r['outer_evaluation_dates']);source=array(FIT/'fit_features'/f"f{r['outer_fold']}_s{r['seed']}.npz",'original_structure');mass=source['mass'][q['score_positions']];mask=p['candidate_cancel'];actual=target['actual_NET'];domains=[]
        for domain,dm in [('actual_STOP',target['original_STOP_mask']),('preSTOP_CONT',~target['original_STOP_mask'])]:
            take=mask&dm;prof=take&(actual>0);unprof=take&(actual<=0);domains.append(dict(domain=domain,score_rows=int(dm.sum()),candidate_rows=int(take.sum()),profitable=int(prof.sum()),unprofitable=int(unprof.sum()),zero_gain=int((take&(target['gain_normalized']==0)).sum()),profitable_calls=len({meta[int(ids[k])]['physical_state_group'] for k in np.flatnonzero(prof)}),profitable_dates=len({meta[int(ids[k])]['date'] for k in np.flatnonzero(prof)})))
            for k in np.flatnonzero(take):
                m=meta[int(ids[k])];candidate_rows.append(dict(bundle=bundle,outer_fold=r['outer_fold'],inner_fold=r['inner_fold'],seed=r['seed'],index=int(ids[k]),sample_id=m['sample_id'],position=m['position'],date=m['date'],call=m['physical_state_group'],domain=domain,event_p=float(p['event_p'][k]),mu4=float(p['mu4'][k]),mu5=float(p['mu5'][k]),public_charge=float(p['public_charge'][k]),actual_NET=float(actual[k]),gain_normalized=float(target['gain_normalized'][k]),source_mass=float(mass[k])))
        np.savez_compressed(O/'labelled_samples'/f'{bundle}.npz',meta_ids=ids,event_p=p['event_p'],mu4=p['mu4'],mu5=p['mu5'],public_charge=p['public_charge'],continue4=p['continue4'],continue5=p['continue5'],candidate_cancel=mask,event_y=target['event_y'],gain_normalized=target['gain_normalized'],actual_NET=actual,true_fee_normalized=target['known_charge'],original_STOP_mask=target['original_STOP_mask'],original_source_mass=mass);records.append(dict(bundle=bundle,outer_fold=r['outer_fold'],inner_fold=r['inner_fold'],seed=r['seed'],score_rows=len(ids),domains=domains))
    assert sum(a['kind']=='score_targets' for a in ACCESS)==36;save('candidate_rows.json',candidate_rows);save('label_join_receipts.json',records);save('join_access.json',ACCESS);save('labelled_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'labelled_samples').iterdir()),O/'candidate_rows.json',O/'label_join_receipts.json',O/'join_access.json']});archive('labelled');print('GRID214_LABELS_JOINED',len(candidate_rows))
if __name__=='__main__':
    PHASE=sys.argv[1];globals()[PHASE]()
