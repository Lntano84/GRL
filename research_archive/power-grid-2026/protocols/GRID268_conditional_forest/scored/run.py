"""Single preregistered parent-coherent residual and empirical positive protection."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid268_conditional_forest';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID268_conditional_forest';Q=R/'outputs/grid267_conditional_forest_plan';PUBLIC=R/'outputs/grid261_bounded_basis/outer_predictions';PLAN=Q;BASE=R/'outputs/grid208_public_amount';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def module(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
N=R/'outputs/grid261_bounded_basis';RUNTIME=R/'work/grid267_conditional_forest_plan/runtime267.py';RT=module('grid268_qualified_runtime',RUNTIME)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    for name in ['qualified_freeze.json']:
        for p,h in read(Q/name).items():assert sha(Q/p)==h,p
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    for folder in {'trained':['models','calibration_predictions'],'scored':['outer_predictions']}.get(phase,[]):shutil.copytree(O/folder,d/folder)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def prepare():
    assert not O.exists() and read(Q/'decision.json')['ready_for_one_fixed268'];head=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
    ar=REPO/'research_archive/power-grid-2026/protocols/GRID267_conditional_forest_plan/qualified'
    for rel,h in read(Q/'qualified_freeze.json').items():assert hashlib.sha256(git('show',head+':'+(ar/rel).relative_to(REPO).as_posix())).hexdigest()==h
    assert git('show',head+':'+(ar/'runtime267.py').relative_to(REPO).as_posix())==RUNTIME.read_bytes() and sha(RUNTIME)==read(Q/'runtime_freeze.json')['work/grid267_conditional_forest_plan/runtime267.py']
    for name in ['EXECUTION_PLAN.json','decision.json']:assert git('show',head+':'+(REPO/'research_archive/power-grid-2026/protocols/GRID267_conditional_forest_plan/audited'/name).relative_to(REPO).as_posix())==(Q/name).read_bytes()
    oldar=REPO/'research_archive/power-grid-2026/protocols/GRID261_bounded_basis/scored/outer_predictions'
    for f in PUBLIC.glob('*.npz'):assert git('show',head+':'+(oldar/f.name).relative_to(REPO).as_posix())==f.read_bytes()
    assert len(list(PUBLIC.glob('*.npz')))==12;O.mkdir();save('pre_input_publication.json',dict(passed=True,git_sha=head,qualified36inputs_runtime_plan_and12publicentries_published=True))
    save('design.json',dict(id='GRID268',candidate='ONEfixedconditionalNETforest filteronfrozen261entry;11ExtraTrees+11cutoff=22/one600s/704trees/1KEEP/0thetaNNrefits.',params=RT.PARAMS,fit='Only267qualifiedownfit8 X32/yNET>0/originalsampleweight;same40features/no newscaler/labels/weights. Seedsoriginal0/1/2;f0_s1KEEP;emptyclassorpositiveguardKEEP. sklearn1.7.2 locked.',guard='tau=min forestclass1score onALLsameown12selectedSTOPpositives;tiescontinue. 500empiricalpositives(active464/KEEP36). Riskweightedleafscore notcalibratedprobability;empiricalguard notouter guarantee.',checks='All64trees/depth<=4/minleaf10/weightedclassfrequencies vsnativeprobabilities inside600s withloading/fit/calpredict. 0outerbeforeallmodelspublished.',outer='AfterALL12bundlespublished:249raw40 first38tanh thenruntime267as_inputfloat32. Frozen261ids/continue/logit reuse;newCONT=261CONT & score>=tau,KEEPsame261,no265delta/no NNthetaforward. All12predictionspublishedbeforeoldresults.',change='ExtraTreesfunctionclass/capacity/giniobjective differs from261linearNLL+unitL2,not samecapacity orpurecalibration. Additionalinferencecostnotmeasured,queriesnotactualtime.',controls='Primary261,sharedGATE/265historicaltradeoff reuse,norefit. All3seeds16exposeddevelopmentdates andadversecases,no parameterorcutoffsearch/seedselection/Jan5exception.',count='Separate22fits,branch73->95. Historical6continuations8285updates/NN604.703sactual/678.156cap retained.',limits='Parentown12/internalcal4ancestry andoldexposedcorrelateddata remain,not wholechainOOF/blind/fullweekquality/time. Final8/physics/grid/paper/FRAME/install untouched;fee/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.'))
    paths=[W/'run.py',O/'design.json',O/'pre_input_publication.json',RUNTIME,Q/'decision.json',Q/'EXECUTION_PLAN.json',Q/'qualification.json',Q/'qualified_freeze.json',Q/'runtime_freeze.json',*list((Q/'qualified_inputs').glob('*.npz')),*list(PUBLIC.glob('*.npz')),*list((R/'outputs/grid249_context_bit_plan/outer_parent_public').glob('*.npz')),META,R/'work/grid227_context_newton/join.py',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json',N/'oof_trajectories.json',N/'gate_predictions.json',N/'family_summary.json',N/'search_metrics.json',N/'gate_counts.json',R/'outputs/grid265_guarded_bias/family_summary.json']
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('source');print('GRID268_SOURCE_FROZEN')

def train():
    verify();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head and git('show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes()
    assert not (O/'training_started.json').exists();save('training_started.json',dict(forests11_cutoffs11=True,cap_s600=True,source_published=head));start=time.perf_counter();import sklearn;assert sklearn.__version__==read(Q/'qualification.json')['sklearn_version'];(O/'models').mkdir();(O/'calibration_predictions').mkdir();receipts=[]
    for file in sorted((Q/'qualified_inputs').glob('*_fit.npz')):
        assert time.perf_counter()-start<600;pool=file.name[:-8];f,s=[int(v[1:]) for v in pool.split('_')];z=np.load(file);cal=np.load(Q/'qualified_inputs'/f'{pool}_cal.npz');base=np.load(Q/'qualified_inputs'/f'{pool}_base.npz');pos=cal['y']>0;keep=bool(base['fallback_KEEP'][0]) or not np.array_equal(np.unique(z['y'][z['weight']>0]),[0,1]) or not pos.any();t=time.perf_counter();clf=None;tau=0.;nodes=leaves=0;maxerr=0.;minleaf=None
        if keep:calprob=np.ones(len(cal['y']),dtype=float)
        else:
            clf=RT.make_classifier(s);assert all(clf.get_params()[k]==v for k,v in RT.PARAMS.items());clf.fit(z['X32'],z['y'],sample_weight=z['weight']);assert len(clf.estimators_)==64 and np.array_equal(clf.classes_,[0,1]);manual=[];minleaf=10**9
            for tree in clf.estimators_:
                assert tree.tree_.max_depth<=4;lid=tree.apply(z['X32']);v=tree.tree_.value[lid,0,:];manual.append(v[:,1]/v.sum(axis=1));leafnodes=np.flatnonzero(tree.tree_.children_left==-1);assert (tree.tree_.n_node_samples[leafnodes]>=10).all();minleaf=min(minleaf,int(tree.tree_.n_node_samples[leafnodes].min()));nodes+=tree.tree_.node_count;leaves+=len(leafnodes)
                for leaf in leafnodes:
                    mask=lid==leaf;ww=z['weight'][mask];total=float(ww.sum());assert total>0;actual=float(ww[z['y'][mask]==1].sum()/total);stored=tree.tree_.value[leaf,0,:];native=float(stored[1]/stored.sum());assert abs(actual-native)<1e-10
            fitted=RT.positive_score(clf,z['X32']);maxerr=float(abs(np.mean(manual,axis=0)-fitted).max());assert maxerr<1e-12;calprob=RT.positive_score(clf,cal['X32']);tau,unexpectedkeep=RT.protect_cutoff(calprob,pos);assert not unexpectedkeep and np.all(calprob[pos]>=tau)
            if tau<1:assert np.any(calprob[pos]<float(np.nextafter(tau,np.inf)))
        parent_score=cal['b']+float(base['alpha'][0])*((cal['b']+np.sum(cal['X']*base['theta'][:-1],axis=1)+base['theta'][-1])-cal['b']);parent_continue=(cal['b']>0)&(parent_score>=0);cont=RT.apply_gate(parent_continue,calprob,tau,KEEP=keep);assert cont[pos].all()
        bundle=dict(pool=pool,model=clf,tau=tau,KEEP=keep,seed=s,params=RT.PARAMS,base_model_sha=sha(Q/'qualified_inputs'/f'{pool}_base.npz'),base_theta=base['theta'],base_alpha=base['alpha'],sklearn_version=sklearn.__version__);(O/'models'/f'{pool}.pkl').write_bytes(pickle.dumps(bundle,protocol=4));np.savez_compressed(O/'calibration_predictions'/f'{pool}.npz',ids=cal['ids'],score=calprob,positive=pos,parent_continue=parent_continue,continue_=cont);receipts.append(dict(pool=pool,fitted=not keep,new_forest_fit=not keep,new_cutoff_fit=not keep,tau=tau,fit_rows=len(z['y']),fit_positive=int(z['y'].sum()),guard_positive=int(pos.sum()),all_empirical_guarded_continue=True,trees=0 if keep else 64,nodes=int(nodes),leaves=int(leaves),min_leaf_samples=minleaf,manual_native_probability_maxerr=maxerr,fit_file_sha=sha(file),cal_file_sha=sha(Q/'qualified_inputs'/f'{pool}_cal.npz'),wall_s=time.perf_counter()-t));save('fit_receipts.json',receipts)
        assert time.perf_counter()-start<600
    elapsed=time.perf_counter()-start;assert len(receipts)==12 and sum(r['fitted'] for r in receipts)==11 and sum(r['guard_positive'] for r in receipts)==500 and sum(r['trees'] for r in receipts)==704;save('fits.json',dict(passed=True,new_forest_fits=11,new_scalar_fits=11,constantKEEP=1,newtheta_NNfits=0,newtrees704=True,wall_s=elapsed,all500_empirical_guarded=True,branchcumulative95=True));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),*list((O/'calibration_predictions').iterdir()),O/'fit_receipts.json',O/'fits.json']});archive('trained');print('GRID268_TRAINED',json.dumps(read(O/'fits.json')))

def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
    assert read(O/'fits.json')['passed']
def publication():
    trained();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'trained/trained_freeze.json').relative_to(REPO).as_posix())==(O/'trained_freeze.json').read_bytes()
    for rel,h in read(O/'trained_freeze.json').items():assert hashlib.sha256(git('show',head+':'+(A/'trained'/rel).relative_to(REPO).as_posix())).hexdigest()==h
    save('pre_score_publication.json',dict(passed=True,git_sha=head,all12weights_steps_published=True));print('GRID268_MODELS_PUBLISHED')
def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all12model_cutoffbundles_published=True));(O/'outer_predictions').mkdir();records=[]
    for file in sorted(PUBLIC.glob('*.npz')):
        parent=np.load(file);raw=np.load(R/'outputs/grid249_context_bit_plan/outer_parent_public'/file.name);assert np.array_equal(parent['ids'],raw['ids']);bundle=pickle.loads((O/'models'/f'{file.stem}.pkl').read_bytes());X=raw['X'].copy();X[:,:38]=np.tanh(X[:,:38]);X32=RT.as_input(X);prob=np.ones(len(X32)) if bundle['KEEP'] else RT.positive_score(bundle['model'],X32);cont=RT.apply_gate(parent['continue_'],prob,bundle['tau'],KEEP=bundle['KEEP']);assert not np.any(cont&~parent['continue_']);np.savez_compressed(O/'outer_predictions'/file.name,ids=parent['ids'],forest_score=prob,gate_margin=prob-bundle['tau'],continue_=cont,base_CONT=parent['base_CONT'],parent261_continue=parent['continue_'],parent261_logit=parent['logit']);records.append(dict(pool=file.stem,rows=len(prob),continues=int(cont.sum()),outcome_fields_read=[]))
    assert sum(r['rows'] for r in records)==4681;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'outer_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID268_SCORED_ALL12')

def scored():
    trained()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p
def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes()
    for rel,h in read(O/'scored_freeze.json').items():assert hashlib.sha256(git('show',head+':'+(A/'scored'/rel).relative_to(REPO).as_posix())).hexdigest()==h
    save('pre_join_publication.json',dict(passed=True,git_sha=head,all12predictions_published=True));print('GRID268_SCORES_PUBLISHED')
def join():
    scored();assert read(O/'pre_join_publication.json')['passed'] and not (O/'join_started.json').exists();save('join_started.json',dict(all12predictions_published=True));summarize=module('grid268_saved_summary',R/'work/grid227_context_newton/join.py').summarize;meta=read(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};parents=[r for r in read(BASE/'oof_trajectories.json') if r['method'].startswith('GATE_')];task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];scores={}
    for file in (O/'outer_predictions').glob('*.npz'):
        f,s=[int(v[1:]) for v in file.stem.split('_')];z=np.load(file)
        for i,cont in zip(z['ids'],z['continue_']):scores[f,s,int(i)]=bool(cont)
    tr=list(parents);pred=[]
    for seed in range(3):
        for old in [r for r in parents if r['method']==f'GATE_seed{seed}']:
            c=cases[old['sample_id']];f=old['fold'];j=old['baseline_stop'];eligible=j<len(c['states']);append=scores[f,seed,lookup[c['sample_id'],j]] if eligible else False;assert not append or old['append'];end=j+int(append);new=dict(old);new.update(method=f'NET_FOREST_seed{seed}',actions=[1]*end+([0] if end<len(c['states']) else []),batches=end,gain=float(c['gains'][:end].sum()),queries=int(c['queries'][:end].sum()),chosen_pool_id=c['chosen'][end],append=append);new['net_utility']=(new['gain']-fee*new['queries'])/scale;new['action_match']=new['chosen_pool_id']==c['chosen'][-1];tr.append(new)
            if eligible:pred.append(dict(fold=f,seed=seed,sample_id=c['sample_id'],position=j,date=c['date'],call=c['physical_state_group'],base_continue=old['append'],continue_=append,actual_next_NET=float((c['gains'][j]-fee*c['queries'][j])/scale),actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j])))
    assert len(tr)==14268 and len(pred)==4681;metrics={};ref=read(BASE/'search_metrics.json')
    for arm in ['GATE','NET_FOREST']:
        for seed in range(3):
            name=f'{arm}_seed{seed}';rr=[r for r in tr if r['method']==name];assert len(rr)==2378;metrics[name]=summarize(rr)
            if arm=='GATE':
                for k,v in metrics[name]['macro'].items():assert abs(v-ref[name]['macro'][k])<1e-12
    family={arm:{k:float(np.mean([metrics[f'{arm}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{arm}_seed0']['macro']} for arm in ['GATE','NET_FOREST']};paired=[]
    for seed in range(3):
        old,new=metrics[f'GATE_seed{seed}'],metrics[f'NET_FOREST_seed{seed}'];paired.append(dict(seed=seed,macro_delta={k:new['macro'][k]-old['macro'][k] for k in old['macro']},per_date=[dict(date=x['date'],NET_delta=y['mean_net_utility']-x['mean_net_utility'],query_delta=y['mean_queries']-x['mean_queries'],gain_delta=y['mean_gain']-x['mean_gain']) for x,y in zip(old['per_date'],new['per_date'])]))
    counts=dict(original208_appends=sum(r['base_continue'] for r in pred),appends=sum(r['continue_'] for r in pred),profitable_retained=sum(r['continue_'] and r['actual_next_NET']>0 for r in pred),profitable_removed=sum(r['base_continue'] and not r['continue_'] and r['actual_next_NET']>0 for r in pred),unprofitable_removed=sum(r['base_continue'] and not r['continue_'] and r['actual_next_NET']<=0 for r in pred))
    for n,v in [('oof_trajectories.json',tr),('gate_predictions.json',pred),('search_metrics.json',metrics),('family_summary.json',family),('paired_summary.json',paired),('gate_counts.json',counts)]:save(n,v)
    scored();save('review.json',dict(passed=True,all14268savedsequences_checked=True,all4681publishedscores_joined=True,commonGATEreproduced=True,all3seeds16oldexposeddates_retained=True,no_new_physics=True,final8_absent=True));archive('evaluated');print('GRID268_EVALUATED',json.dumps(dict(family=family,counts=counts,paired=[dict(seed=r['seed'],NET=r['macro_delta']['mean_net_utility'],queries=r['macro_delta']['mean_queries'],gain=r['macro_delta']['mean_gain']) for r in paired])))
if __name__=='__main__':globals()[sys.argv[1]]()
