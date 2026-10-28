"""Fixed cancellation-only forests over GRID215 cross-head training rows."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import collections,datetime,hashlib,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid216_cancellation_gate';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID216_cancellation_gate';D=R/'outputs/grid215_cancellation_domain';S=R/'outputs/grid214_inner_head_generation';BASE=R/'outputs/grid208_public_amount';FULL=R/'outputs/grid206_public_event';CACHE=R/'outputs/grid190_extension_gate';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args])
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    for p,h in read(D/'target_freeze.json').items():assert sha(D/p)==h,p
    for name in ['trained_freeze.json','scored_freeze.json','labelled_freeze.json']:
        for p,h in read(S/name).items():assert sha(S/p)==h,p
    for name in ['trained_freeze.json']:
        for p,h in read(BASE/name).items():assert sha(BASE/p)==h,p

def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    for folder in {'qualified':['fit_inputs'],'trained':['models'],'scored':['held_predictions']}.get(phase,[]):shutil.copytree(O/folder,dest/folder)
    (dest/'archive_manifest.json').write_text(json.dumps({p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')

def prepare():
    assert not O.exists();assert read(D/'independent_audit.json')['passed'];O.mkdir();(O/'fit_inputs').mkdir();meta=read(META);params=read(R/'outputs/grid213_inner_head_plan/estimator_configs.json');configs={};records=[]
    save('design.json',dict(id='GRID216_CANCELLATION_GATE',authorization='User好继续 plus standing autonomous targeted improvement; pretraining notice11 forests+1constantKEEP delivered2026-10-28.',change='Frozen208 continuation decisions followed by optional cancellation classifier. Training domain only inner head-unfitted depth4CONT at originalNETSTOP from215; all16exposed dates/4outerfolds/3seeds retained. No rescue, no next-extra batch.',target='Keep if originalactualNET>0; sampleweight originalstructural_mass*absNET globally renormalizedtosum selectedrows. No classbalance/clipping/oversampling/feechange. Costweightedclassifier output not calibrated nativeprobability.',input='Same208public124scalers and frozen outer12NET features, no newscalers/NN forwards/innerhead probabilities. Original206full fit feature matrix supplies only selected215ids. Held cached83plus original41 applies identical208scalers.',models='Exactly11ExtraTreesClassifier forests:64trees/depth4/minleaf10/bootstrapFalse/maxfeatures1/Gini/n_jobs1/seed0,1,2. f0_s1 keepconstantnoFit from215noPositive. Original600s totalfit+nativechecks, noextension/refits/capacitysearch.',readout='Nativecostweightedkeep>=0.5 retains original208candidate, otherwisecancel. Equal0.5keep; no probability/threshold tuning, all seeds retained.',publication='All11weights+1KEEP published before anyheldrouter scores. Score all4681 cachedSTOPfeatures without outcomes; publish all12scorefiles before joining208decisions/cases/heldlabels.',assessment='One fixed primary208 control. Reconstruct fullNETprefix+atmostsameonebatch on originalteacher sequences only. All16date/3seeds, actual queries/localgain/actionmatch/proxyNET, perseed/date/removedzeroandprofitablecalls; noallwinor5percentnewgate. No newpolicychosenbyheldresults.',limits='Selectedtrainingheads8dates vs original20812date gate creates entrance shift. BaseNET/STOPselection in-sample on innerdates; no innerroutervalidationclaim. Allancestors disjointownouter4. Not fullynestedNET, newphysics, fullweekquality/controltime, final8, paper or newFRAME. GRID120costweightedBCEnegativeprecedent retained.11learners+1KEEP notclaim12learnedmodels.'))
    for f in range(4):
        for s in range(3):
            key=f'f{f}_s{s}';t=np.load(D/'qualified_targets'/f'{key}.npz');full=np.load(FULL/'fit_features'/f'{key}.npz');ids=t['meta_ids'].astype(int);idx={int(i):j for j,i in enumerate(full['ids'])};pos=np.array([idx[int(i)] for i in ids]);x=full['standard'][pos];assert x.shape==(len(ids),124) and np.isfinite(x.astype(np.float32)).all();fd={meta[int(i)]['date'] for i in full['ids']};hz=np.load(CACHE/f'held_{key}.npz');hd={meta[int(i)]['date'] for i in hz['ids']};assert len(fd)==12 and len(hd)==4 and not fd&hd;assert set(meta[int(i)]['date'] for i in ids)<=fd
            np.savez_compressed(O/'fit_inputs'/f'{key}.npz',X=x,ids=ids,y=t['keep_label'].astype(int),weight=t['normalized_fit_weight'],structural_mass=t['structural_mass'],actual_NET=t['actual_NET'],fallback_keep=t['fallback_keep']);configs[key]=params[key]['event'];assert configs[key]['max_depth']==4 and configs[key]['n_estimators']==64 and configs[key]['min_samples_leaf']==10 and configs[key]['random_state']==s
            records.append(dict(pool=key,fit_dates=sorted(fd),evaluation_dates=sorted(hd),rows=len(ids),positive_rows=int(t['keep_label'].sum()),fallback_keep=bool(t['fallback_keep'][0])))
    save('configs.json',configs);save('fit_qualification.json',dict(passed=True,records=records,forests=11,constant_KEEP=1,feature_dimensions=124,allownouter_dates_isolated=True))
    paths=[W/'run.py',O/'design.json',O/'configs.json',O/'fit_qualification.json',D/'independent_audit.json',D/'target_freeze.json',S/'public_raw41.npz',S/'bundles.json',META,BASE/'trained_freeze.json',BASE/'gate_predictions.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json',BASE/'family_summary.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'work/grid208_public_amount/run.py']+list((D/'qualified_targets').glob('*.npz'))+list((O/'fit_inputs').glob('*.npz'))+list((FULL/'fit_features').glob('*.npz'))+list((BASE/'models').glob('*.npz'))+list(CACHE.glob('held_f*.npz'))
    save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('qualified');print('GRID216_QUALIFIED_11PLUS1')

def train():
    verify();assert not (O/'training_started.json').exists();from sklearn.ensemble import ExtraTreesClassifier
    save('training_started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),forests=11,constant_KEEP=1,cap_s=600));(O/'models').mkdir();start=time.perf_counter();receipts=[]
    for key,params in read(O/'configs.json').items():
        assert time.perf_counter()-start<600,'Original600s cap; preserve partial files and do not retry.';z=np.load(O/'fit_inputs'/f'{key}.npz');x,y,w=z['X'],z['y'],z['weight'];fallback=bool(z['fallback_keep'][0]);t=time.perf_counter()
        if fallback:
            assert key=='f0_s1' and not y.any();model=dict(type='KEEP',reason='No profitable support in fixed own outer/seed pool');nodes=leaves=0;maxerror=0.
        else:
            assert set(y.tolist())=={0,1};model=ExtraTreesClassifier(**params);model.fit(x,y,sample_weight=w);assert model.n_features_in_==124 and len(model.estimators_)==64 and np.array_equal(model.classes_,[0,1]);manual=[];nodes=leaves=0;maxerror=0.
            for tree in model.estimators_:
                ids=tree.apply(x);v=tree.tree_.value[ids,0,:];manual.append(v[:,1]/v.sum(1));assert tree.tree_.max_depth<=4;nodes+=tree.tree_.node_count;leaves+=int(np.sum(tree.tree_.children_left==-1))
                for leaf in np.unique(ids):
                    mask=ids==leaf;assert mask.sum()>=10;truth=float(w[mask]@y[mask]/w[mask].sum());v=tree.tree_.value[leaf,0,:];error=abs(truth-v[1]/v.sum());maxerror=max(maxerror,error);assert error<1e-8
            assert np.allclose(model.predict_proba(x)[:,1],np.mean(manual,axis=0),atol=1e-10,rtol=1e-10)
        path=O/'models'/f'{key}.pkl';path.write_bytes(pickle.dumps(model,protocol=4));receipts.append(dict(pool=key,fitted=not fallback,rows=len(y),positive_rows=int(y.sum()),nodes=nodes,leaves=leaves,max_leaf_weighted_mean_error=float(maxerror),fit_and_native_check_wall_s=time.perf_counter()-t,model_sha256=sha(path)));save('fit_receipts.json',receipts)
    assert sum(r['fitted'] for r in receipts)==11 and len(receipts)==12;save('fits.json',dict(passed=True,new_forests=11,new_trees=704,constant_KEEP=1,total_fit_and_native_check_wall_s=time.perf_counter()-start,new_NN_fits=0));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fit_receipts.json',O/'fits.json']});verify();archive('trained');print('GRID216_TRAINED_11FORESTS')

def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p

def publication():
    trained();head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote;assert git('show','HEAD:'+(A/'trained/trained_freeze.json').relative_to(REPO).as_posix())==(O/'trained_freeze.json').read_bytes();save('pre_score_publication.json',dict(passed=True,git_sha=head,all11forests_and1constant_published=True));print('GRID216_WEIGHTS_PUBLISHED')

def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(allweights_published=True));(O/'held_predictions').mkdir();raw=np.load(S/'public_raw41.npz')['values'];records=[]
    for key in read(O/'configs.json'):
        cp=np.load(BASE/'models'/f'{key}.npz');h=np.load(CACHE/f'held_{key}.npz');ids=h['ids'].astype(int);x=np.c_[(h['features']-cp['mean'])/cp['scale'],(raw[ids]-cp['raw_mean'])/cp['raw_scale']];assert x.shape==(len(ids),124) and np.isfinite(x.astype(np.float32)).all();model=pickle.loads((O/'models'/f'{key}.pkl').read_bytes())
        if isinstance(model,dict):assert model['type']=='KEEP';p=np.ones(len(ids))
        else:
            p=model.predict_proba(x)[:,1];manual=[]
            for tree in model.estimators_:v=tree.tree_.value[tree.apply(x),0,:];manual.append(v[:,1]/v.sum(1))
            assert np.allclose(p,np.mean(manual,axis=0),atol=1e-10,rtol=1e-10)
        np.savez_compressed(O/'held_predictions'/f'{key}.npz',ids=ids,keep_score=p,keep=p>=.5);records.append(dict(pool=key,held_rows=len(ids),held_outcomes_parsed=False))
    assert sum(r['held_rows'] for r in records)==4681;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'held_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID216_SCORED_4681_BEFORE_OUTCOMES')

def scored():
    trained()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p

def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes();save('pre_join_publication.json',dict(passed=True,git_sha=head,all12predictionfiles_published=True));print('GRID216_PREDICTIONS_PUBLISHED')
if __name__=='__main__':globals()[sys.argv[1]]()
