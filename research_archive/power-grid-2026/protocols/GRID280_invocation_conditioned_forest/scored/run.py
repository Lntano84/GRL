"""Single preregistered parent-coherent residual and empirical positive protection."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid280_invocation_conditioned';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID280_invocation_conditioned_forest';Q=R/'outputs/grid279_conditional_fit_plan';PUBLIC=R/'outputs/grid261_bounded_basis/outer_predictions';PLAN=Q;BASE=R/'outputs/grid208_public_amount';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def module(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
N=R/'outputs/grid261_bounded_basis';RUNTIME=Q/'runtime276.py';RT=module('grid268_qualified_runtime',RUNTIME)
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
    assert not O.exists() and read(Q/'decision.json')['ready_for_one_fixed280'];head=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
    ar=REPO/'research_archive/power-grid-2026/protocols/GRID279_conditional_fit_plan'
    for rel,h in read(Q/'qualified_freeze.json').items():assert sha(Q/rel)==h and hashlib.sha256(git('show',head+':'+(ar/'qualified'/rel).relative_to(REPO).as_posix())).hexdigest()==h
    for name in ['EXECUTION_PLAN.json','decision.json']:assert git('show',head+':'+(ar/'audited'/name).relative_to(REPO).as_posix())==(Q/name).read_bytes()
    assert sha(RUNTIME)==read(Q/'runtime_freeze.json')['sha256'];refs=read(Q/'model_references.json');assert sum(r['future_newfit'] for r in refs)==11 and sum(r['KEEP'] for r in refs)==1
    for ref in refs:
        for rp,gp in [('old39_model_relative_to_R','old39_model_git_path'),('old39_prediction_relative_to_R','old39_prediction_git_path')]:assert git('show',head+':'+ref[gp])==(R/ref[rp]).read_bytes()
    O.mkdir();save('pre_input_publication.json',dict(passed=True,git_sha=head,all279qualified_inputs_excluded_evidence_runtime_plan_controls_published=True))
    save('design.json',dict(id='GRID280',candidate='ONE fixed parent-invocation-conditioned39Dforest,11freshforests+11same543cutoffs=22/one600s/1KEEP/704newtrees,NNtheta0.',params=RT.PARAMS,fit='Only279qualified_inputs fitX32/y/weight.2544candidate/2410active=534positive1876negative;16copiesandallpositive retained,331excludedrows onlyevidence NEVERfit. Ewaspublic beforelabels,notnegative/outerIDselection. Literalweights/source_mass/NET preserved,no normalization/dedup or furtherfilter.',target_change='Conditionalweightedgini risk/sampleexposurechanges vs277.331excludedrecords12.08%countsbut.1607086%oldrisk,notmaincauseoronlinequerysaveproof. ParentE basedteacherprefixhypotheticalSTOP1 notnewphysicalreplay.',guard='Same543own12unionpositivecal/base arrays bytecopied. Unique tau=minscore, tiescontinue.Original39Drecipe/seed,no newguardIDs/weights/riskorparametersearch.',timing='Supervisorone600s includeschildstartup/load/fit/calibration/weightedleafchecks;retainpartial/noextension/repeat.',outer='Afterall12modelcutoffs published,raw249publicX first38tanh slice39,frozen261entry/KEEP,0NNforward. All12predictionspublishbeforeoldoutcomes.',controls='Mainfrozen277 same39D butriskexposurechange,274/271historical/261qualityanchor/GATE reused,all3seeds16olddevelopmentdates/KEEP/adversecases,no bestseed/subsetdeployment.',count='Independent22/current146->168;16relatedcopies/500to543guardexposure andold6continuations8285updates/604.703actual/678.156cap separate.',limits='Parentown12/internalcal4/exposedrelateddata remain,notOOFblind/fullweek/time. Queryproxy notactualtime or equalbudgetquality. No final8/physics/grid/paper/FRAME/install;feeNETlabels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.'))
    control=R/'outputs/grid277_selector_free'
    paths=[*W.glob('*.py'),O/'design.json',O/'pre_input_publication.json',RUNTIME,Q/'decision.json',Q/'EXECUTION_PLAN.json',Q/'qualification.json',Q/'qualified_freeze.json',Q/'runtime_freeze.json',Q/'model_references.json',Q/'exclusion_provenance.json',*list((Q/'qualified_inputs').glob('*.npz')),*list((Q/'excluded_inputs').glob('*.npz')),*list(PUBLIC.glob('*.npz')),*list((R/'outputs/grid249_context_bit_plan/outer_parent_public').glob('*.npz')),META,R/'work/grid227_context_newton/join.py',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json']
    for ref in refs:paths.append(R/ref['old39_model_relative_to_R'])
    for old in [control,R/'outputs/grid274_stop_context',R/'outputs/grid271_union_guard',N,R/'outputs/grid265_guarded_bias']:
        paths += [old/n for n in ['oof_trajectories.json','gate_predictions.json','family_summary.json','search_metrics.json','gate_counts.json'] if (old/n).exists()]
    paths += list((control/'outer_predictions').glob('*.npz'))+list((control/'calibration_predictions').glob('*.npz'))
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('source');print('GRID280_SOURCE_FROZEN')




def train():
    verify();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head and git('show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes()
    assert not (O/'training_started.json').exists();start=time.perf_counter_ns();save('training_started.json',dict(forests11_cutoffs11=True,cap_s600=True,source_published=head,started_monotonic_ns=start))
    try:
        with (O/'training_stdout.log').open('w',encoding='utf8') as out,(O/'training_stderr.log').open('w',encoding='utf8') as err:
            result=subprocess.run([sys.executable,'-X','utf8','-u',str(W/'run.py'),'train_work'],stdout=out,stderr=err,timeout=max(.001,600-(time.perf_counter_ns()-start)/1e9),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        elapsed=(time.perf_counter_ns()-start)/1e9;assert result.returncode==0 and elapsed<600,(result.returncode,elapsed)
        rr=read(O/'fit_receipts.json');assert len(rr)==12 and sum(r['new_forest_fit'] for r in rr)==11 and sum(r['new_cutoff_fit'] for r in rr)==11 and sum(r['guard_positive'] for r in rr)==543
        save('fits.json',dict(passed=True,new_forest_fits=11,new_scalar_fits=11,constantKEEP=1,reused_active_models_cutoffs=0,newtheta_NNfits=0,new_trees704=True,deployed_trees704=True,wall_s=elapsed,cap_s=600,all543_empirical_guarded=True,branch_cumulative168=True,worker_exited=True))
        save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),*list((O/'calibration_predictions').iterdir()),O/'fit_receipts.json',O/'fits.json']});archive('trained');print('GRID280_TRAINED',json.dumps(read(O/'fits.json')))
    except Exception as e:save('training_error.json',dict(error=repr(e),partial_retained=True,no_retry=True,elapsed=(time.perf_counter_ns()-start)/1e9));raise

def train_work():
    verify();import sklearn;assert sklearn.__version__=='1.7.2';start=read(O/'training_started.json')['started_monotonic_ns'];(O/'models').mkdir();(O/'calibration_predictions').mkdir();receipts=[]
    for ref in read(Q/'model_references.json'):
        assert (time.perf_counter_ns()-start)/1e9<600;pool=ref['pool'];seed=int(pool[-1]);changed=ref['future_newfit'];keep=ref['KEEP'];z=np.load(Q/'qualified_inputs'/f'{pool}_fit.npz');cal=np.load(Q/'qualified_inputs'/f'{pool}_cal.npz');base=np.load(Q/'qualified_inputs'/f'{pool}_base.npz');old=pickle.loads((R/ref['old39_model_relative_to_R']).read_bytes());t=time.perf_counter();clf=old['model'];tau=old['tau'];nodes=leaves=0;maxerr=0.;minleaf=None
        if changed:
            assert not keep and np.array_equal(np.unique(z['y'][z['weight']>0]),[0,1]);clf=RT.make_classifier(seed);clf.fit(z['X32'],z['y'],sample_weight=z['weight']);assert len(clf.estimators_)==64;manual=[];minleaf=10**9
            for tree in clf.estimators_:
                assert tree.tree_.max_depth<=4;lid=tree.apply(z['X32']);v=tree.tree_.value[lid,0,:];manual.append(v[:,1]/v.sum(axis=1));leafnodes=np.flatnonzero(tree.tree_.children_left==-1);assert (tree.tree_.n_node_samples[leafnodes]>=10).all();minleaf=min(minleaf,int(tree.tree_.n_node_samples[leafnodes].min()));nodes+=tree.tree_.node_count;leaves+=len(leafnodes)
                for leaf in leafnodes:
                    mask=lid==leaf;ww=z['weight'][mask];assert ww.sum()>0;value=tree.tree_.value[leaf,0,:];assert abs(float(ww[z['y'][mask]==1].sum()/ww.sum())-float(value[1]/value.sum()))<1e-10
            fitprob=RT.positive_score(clf,z['X32']);maxerr=float(abs(np.mean(manual,axis=0)-fitprob).max());assert maxerr<1e-12;calprob=RT.positive_score(clf,cal['X32']);tau,newkeep=RT.protect_cutoff(calprob,cal['y']>0);assert not newkeep and (calprob>=tau).all()
            if tau<1:assert np.any(calprob<np.nextafter(tau,np.inf))
        else:
            saved=np.load(R/'outputs/grid277_selector_free/calibration_predictions'/f'{pool}.npz');assert np.array_equal(saved['ids'],cal['ids']);calprob=saved['score'];assert tau==old['tau'];assert keep==(clf is None)
            if not keep:nodes=sum(t.tree_.node_count for t in clf.estimators_);leaves=sum(int((t.tree_.children_left==-1).sum()) for t in clf.estimators_)
        parent=cal['parent261_score']>=0;cont=RT.apply_gate(parent,calprob,tau,KEEP=keep);assert cont.all()
        bundle=dict(pool=pool,model=clf,tau=tau,KEEP=keep,seed=seed,params=RT.PARAMS,base_model_sha=sha(Q/'qualified_inputs'/f'{pool}_base.npz'),base_theta=base['theta'],base_alpha=base['alpha'],sklearn_version=sklearn.__version__,invocation_conditioned_fit=changed,original_model_sha=ref['old39_model_sha256'])
        (O/'models'/f'{pool}.pkl').write_bytes(pickle.dumps(bundle,protocol=4));np.savez_compressed(O/'calibration_predictions'/f'{pool}.npz',ids=cal['ids'],score=calprob,positive=cal['y']>0,parent_continue=parent,continue_=cont)
        receipts.append(dict(pool=pool,fitted=changed,new_forest_fit=changed,new_cutoff_fit=changed,KEEP=keep,tau=tau,fit_rows=len(z['ids']),augmentation_rows=int(z['is_augmented'].sum()),guard_positive=len(cal['ids']),trees=0 if keep else 64,new_trees=64 if changed else 0,nodes=int(nodes),leaves=int(leaves),min_leaf_samples=minleaf,manual_native_probability_maxerr=maxerr,fit_file_sha=sha(Q/'qualified_inputs'/f'{pool}_fit.npz'),cal_file_sha=sha(Q/'qualified_inputs'/f'{pool}_cal.npz'),wall_s=time.perf_counter()-t));save('fit_receipts.json',receipts)
    assert len(receipts)==12 and sum(r['new_trees'] for r in receipts)==704 and sum(r['trees'] for r in receipts)==704 and (time.perf_counter_ns()-start)/1e9<600;print('GRID280_ELEVENFIT_ELEVENCUTOFF_DONE')


def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
    assert read(O/'fits.json')['passed']
def publication():
    trained();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'trained/trained_freeze.json').relative_to(REPO).as_posix())==(O/'trained_freeze.json').read_bytes()
    for rel,h in read(O/'trained_freeze.json').items():assert hashlib.sha256(git('show',head+':'+(A/'trained'/rel).relative_to(REPO).as_posix())).hexdigest()==h
    save('pre_score_publication.json',dict(passed=True,git_sha=head,all12weights_steps_published=True));print('GRID280_MODELS_PUBLISHED')
def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all12model_cutoffbundles_published=True,NNforward0=True));(O/'outer_predictions').mkdir();records=[];refs={r['pool']:r for r in read(Q/'model_references.json')}
    for file in sorted(PUBLIC.glob('*.npz')):
        pool=file.stem;parent=np.load(file);bundle=pickle.loads((O/'models'/f'{pool}.pkl').read_bytes());changed=refs[pool]['future_newfit']
        if changed:
            raw=np.load(R/'outputs/grid249_context_bit_plan/outer_parent_public'/file.name);assert np.array_equal(parent['ids'],raw['ids']);X=raw['X'].copy();X[:,:38]=np.tanh(X[:,:38]);prob=RT.positive_score(bundle['model'],RT.as_input(X[:,:39]))
        else:
            old=np.load(R/'outputs/grid277_selector_free/outer_predictions'/file.name);assert np.array_equal(parent['ids'],old['ids']);prob=old['forest_score']
        cont=RT.apply_gate(parent['continue_'],prob,bundle['tau'],KEEP=bundle['KEEP']);assert not np.any(cont&~parent['continue_'])
        np.savez_compressed(O/'outer_predictions'/file.name,ids=parent['ids'],forest_score=prob,gate_margin=prob-bundle['tau'],continue_=cont,base_CONT=parent['base_CONT'],parent261_continue=parent['continue_'],parent261_logit=parent['logit']);records.append(dict(pool=pool,rows=len(prob),continues=int(cont.sum()),new_forest_forward=changed,NN_forward=False,outcome_fields_read=[]))
    assert sum(r['rows'] for r in records)==4681 and sum(r['new_forest_forward'] for r in records)==11;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'outer_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID280_SCORED_ALL12')


def scored():
    trained()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p
def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes()
    for rel,h in read(O/'scored_freeze.json').items():assert hashlib.sha256(git('show',head+':'+(A/'scored'/rel).relative_to(REPO).as_posix())).hexdigest()==h
    save('pre_join_publication.json',dict(passed=True,git_sha=head,all12predictions_published=True));print('GRID280_SCORES_PUBLISHED')
def join():
    scored();assert read(O/'pre_join_publication.json')['passed'] and not (O/'join_started.json').exists();save('join_started.json',dict(all12predictions_published=True));summarize=module('grid268_saved_summary',R/'work/grid227_context_newton/join.py').summarize;meta=read(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};parents=[r for r in read(BASE/'oof_trajectories.json') if r['method'].startswith('GATE_')];task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];scores={}
    for file in (O/'outer_predictions').glob('*.npz'):
        f,s=[int(v[1:]) for v in file.stem.split('_')];z=np.load(file)
        for i,cont in zip(z['ids'],z['continue_']):scores[f,s,int(i)]=bool(cont)
    tr=list(parents);pred=[]
    for seed in range(3):
        for old in [r for r in parents if r['method']==f'GATE_seed{seed}']:
            c=cases[old['sample_id']];f=old['fold'];j=old['baseline_stop'];eligible=j<len(c['states']);append=scores[f,seed,lookup[c['sample_id'],j]] if eligible else False;assert not append or old['append'];end=j+int(append);new=dict(old);new.update(method=f'ENTRY_CONDITIONED_seed{seed}',actions=[1]*end+([0] if end<len(c['states']) else []),batches=end,gain=float(c['gains'][:end].sum()),queries=int(c['queries'][:end].sum()),chosen_pool_id=c['chosen'][end],append=append);new['net_utility']=(new['gain']-fee*new['queries'])/scale;new['action_match']=new['chosen_pool_id']==c['chosen'][-1];tr.append(new)
            if eligible:pred.append(dict(fold=f,seed=seed,sample_id=c['sample_id'],position=j,date=c['date'],call=c['physical_state_group'],base_continue=old['append'],continue_=append,actual_next_NET=float((c['gains'][j]-fee*c['queries'][j])/scale),actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j])))
    assert len(tr)==14268 and len(pred)==4681;metrics={};ref=read(BASE/'search_metrics.json')
    for arm in ['GATE','ENTRY_CONDITIONED']:
        for seed in range(3):
            name=f'{arm}_seed{seed}';rr=[r for r in tr if r['method']==name];assert len(rr)==2378;metrics[name]=summarize(rr)
            if arm=='GATE':
                for k,v in metrics[name]['macro'].items():assert abs(v-ref[name]['macro'][k])<1e-12
    family={arm:{k:float(np.mean([metrics[f'{arm}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{arm}_seed0']['macro']} for arm in ['GATE','ENTRY_CONDITIONED']};paired=[]
    for seed in range(3):
        old,new=metrics[f'GATE_seed{seed}'],metrics[f'ENTRY_CONDITIONED_seed{seed}'];paired.append(dict(seed=seed,macro_delta={k:new['macro'][k]-old['macro'][k] for k in old['macro']},per_date=[dict(date=x['date'],NET_delta=y['mean_net_utility']-x['mean_net_utility'],query_delta=y['mean_queries']-x['mean_queries'],gain_delta=y['mean_gain']-x['mean_gain']) for x,y in zip(old['per_date'],new['per_date'])]))
    counts=dict(original208_appends=sum(r['base_continue'] for r in pred),appends=sum(r['continue_'] for r in pred),profitable_retained=sum(r['continue_'] and r['actual_next_NET']>0 for r in pred),profitable_removed=sum(r['base_continue'] and not r['continue_'] and r['actual_next_NET']>0 for r in pred),unprofitable_removed=sum(r['base_continue'] and not r['continue_'] and r['actual_next_NET']<=0 for r in pred))
    for n,v in [('oof_trajectories.json',tr),('gate_predictions.json',pred),('search_metrics.json',metrics),('family_summary.json',family),('paired_summary.json',paired),('gate_counts.json',counts)]:save(n,v)
    scored();save('review.json',dict(passed=True,all14268savedsequences_checked=True,all4681publishedscores_joined=True,commonGATEreproduced=True,all3seeds16oldexposeddates_retained=True,no_new_physics=True,final8_absent=True));archive('evaluated');print('GRID280_EVALUATED',json.dumps(dict(family=family,counts=counts,paired=[dict(seed=r['seed'],NET=r['macro_delta']['mean_net_utility'],queries=r['macro_delta']['mean_queries'],gain=r['macro_delta']['mean_gain']) for r in paired])))
if __name__=='__main__':globals()[sys.argv[1]]()
