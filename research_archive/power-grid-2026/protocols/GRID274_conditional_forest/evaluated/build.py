from pathlib import Path
import ast
R=Path('D:/Codex/GRL研究');W=R/'work/grid274_stop_context';s=(R/'work/grid268_conditional_forest/run.py').read_text(encoding='utf-8-sig');s=s.replace('GRID268','GRID274').replace('grid268_conditional_forest','grid274_stop_context').replace("Q=R/'outputs/grid267_conditional_forest_plan'","Q=R/'outputs/grid273_stop_context_plan'").replace("RUNTIME=R/'work/grid267_conditional_forest_plan/runtime267.py'","RUNTIME=Q/'runtime267.py'").replace('NET_FOREST','CONTEXT_FOREST')
prepare=r'''def prepare():
    assert not O.exists() and read(Q/'decision.json')['ready_for_one_fixed274'];head=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
    ar=REPO/'research_archive/power-grid-2026/protocols/GRID273_stop_context_plan'
    for rel,h in read(Q/'qualified_freeze.json').items():assert sha(Q/rel)==h and hashlib.sha256(git('show',head+':'+(ar/'qualified'/rel).relative_to(REPO).as_posix())).hexdigest()==h
    for name in ['EXECUTION_PLAN.json','decision.json']:assert git('show',head+':'+(ar/'audited'/name).relative_to(REPO).as_posix())==(Q/name).read_bytes()
    assert sha(RUNTIME)==read(Q/'runtime_freeze.json')['sha256'];refs=read(Q/'model_references.json');assert sum(r['changed'] for r in refs)==9 and sum(r['KEEP'] for r in refs)==1
    for r in refs:
        for rp,gp in [('model_relative_to_R','model_git_path'),('public_prediction_relative_to_R','public_prediction_git_path')]:assert git('show',head+':'+r[gp])==(R/r[rp]).read_bytes()
    control=R/'outputs/grid271_union_guard';controlar=REPO/'research_archive/power-grid-2026/protocols/GRID271_union_guard'
    for rel,h in read(control/'scored_freeze.json').items():assert sha(control/rel)==h and hashlib.sha256(git('show',head+':'+(controlar/'scored'/rel).relative_to(REPO).as_posix())).hexdigest()==h
    O.mkdir();save('pre_input_publication.json',dict(passed=True,git_sha=head,all273qualifiedinputs_runtime_plan_and_frozen_models_published=True))
    save('design.json',dict(id='GRID274',candidate='ONE ownfit actualSTOP context augmentation;9newforests+9same543guardcutoffs=18/one600s. 2unchangedactive model/cutoff reuse+1KEEP. 576newtrees/704deployment;0NNthetafits.',params=RT.PARAMS,fit='Only273qualifiedfitX32/y/weight;retainold2725active+16copies=2741active. Original CONTrows andnegatives retained;copyinheritspositiveNET/y/weight,onlybit0to1. Explicitpositive-risk/sampleexposureaddition,not samehalfhalfobjective orpurebitablation. No param/risk/weightmultipliersearch.',guard='Same273bytecopied543positiveunion(active507/KEEP36);unique tau=minscore, equalityCONT,9changedpoolscalibrateonce. Two unaffected active reuse271tau andsavedcalibration,KEEPidentity. No additional guardIDs.',timing='Manager watchdog one600s includes childstartup/imports/load/fit/calibration/checks;partialretained/no repeat/extension.',outer='AfterALL12modelorrefcutoffbundlespublished:only9changedforests infer249publicX first38tanh thenfloat32. Two unchangedactive+KEEP old271scorebytesreuse;newCONT=261CONT & score>=tau. All12predictionspublishbeforeoldresults.',control='Primaryfrozen271;261qualityanchor/GATE reused,all3seeds16olddevelopmentdates andadversecases/KEEPscope,neverbestseed/subset deployment.',count='Independent18/current106->124;original6continuations8285updates/604.703actual/678.156cap separate. Copies16from6calls4dates arecorrelatednotnewindependentdata.',limits='own12parent/internalcal4ancestry andexposedrelateddata,empiricalguardnotouter guarantee/wholechainOOF/blind/fullweek/time. No final8/physics/grid/paper/FRAME/install;fee/NETlabels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.'))
    paths=[*W.glob('*.py'),O/'design.json',O/'pre_input_publication.json',RUNTIME,Q/'decision.json',Q/'EXECUTION_PLAN.json',Q/'qualification.json',Q/'qualified_freeze.json',Q/'runtime_freeze.json',Q/'model_references.json',Q/'qualified_pools.json',*list((Q/'qualified_inputs').glob('*.npz')),*list(PUBLIC.glob('*.npz')),*list((R/'outputs/grid249_context_bit_plan/outer_parent_public').glob('*.npz')),META,R/'work/grid227_context_newton/join.py',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json']
    for r in refs:paths.append(R/r['model_relative_to_R'])
    for old in [control,N,R/'outputs/grid265_guarded_bias']:
        paths += [old/n for n in ['oof_trajectories.json','gate_predictions.json','family_summary.json','search_metrics.json','gate_counts.json'] if (old/n).exists()]
    paths += list((control/'outer_predictions').glob('*.npz'))+list((control/'calibration_predictions').glob('*.npz'))
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('source');print('GRID274_SOURCE_FROZEN')
'''
train=r'''def train():
    verify();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head and git('show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes()
    assert not (O/'training_started.json').exists();start=time.perf_counter_ns();save('training_started.json',dict(forests9_cutoffs9=True,cap_s600=True,source_published=head,started_monotonic_ns=start))
    try:
        with (O/'training_stdout.log').open('w',encoding='utf8') as out,(O/'training_stderr.log').open('w',encoding='utf8') as err:
            result=subprocess.run([sys.executable,'-X','utf8','-u',str(W/'run.py'),'train_work'],stdout=out,stderr=err,timeout=max(.001,600-(time.perf_counter_ns()-start)/1e9),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        elapsed=(time.perf_counter_ns()-start)/1e9;assert result.returncode==0 and elapsed<600,(result.returncode,elapsed)
        rr=read(O/'fit_receipts.json');assert len(rr)==12 and sum(r['new_forest_fit'] for r in rr)==9 and sum(r['new_cutoff_fit'] for r in rr)==9 and sum(r['guard_positive'] for r in rr)==543
        save('fits.json',dict(passed=True,new_forest_fits=9,new_scalar_fits=9,constantKEEP=1,reused_active_models_cutoffs=2,newtheta_NNfits=0,new_trees576=True,deployed_trees704=True,wall_s=elapsed,cap_s=600,all543_empirical_guarded=True,branch_cumulative124=True,worker_exited=True))
        save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),*list((O/'calibration_predictions').iterdir()),O/'fit_receipts.json',O/'fits.json']});archive('trained');print('GRID274_TRAINED',json.dumps(read(O/'fits.json')))
    except Exception as e:save('training_error.json',dict(error=repr(e),partial_retained=True,no_retry=True,elapsed=(time.perf_counter_ns()-start)/1e9));raise

def train_work():
    verify();import sklearn;assert sklearn.__version__=='1.7.2';start=read(O/'training_started.json')['started_monotonic_ns'];(O/'models').mkdir();(O/'calibration_predictions').mkdir();receipts=[]
    for ref in read(Q/'model_references.json'):
        assert (time.perf_counter_ns()-start)/1e9<600;pool=ref['pool'];seed=int(pool[-1]);changed=ref['changed'];keep=ref['KEEP'];z=np.load(Q/'qualified_inputs'/f'{pool}_fit.npz');cal=np.load(Q/'qualified_inputs'/f'{pool}_cal.npz');base=np.load(Q/'qualified_inputs'/f'{pool}_base.npz');old=pickle.loads((R/ref['model_relative_to_R']).read_bytes());t=time.perf_counter();clf=old['model'];tau=ref['fixed271_cutoff']['tau'];nodes=leaves=0;maxerr=0.;minleaf=None
        if changed:
            assert not keep and np.array_equal(np.unique(z['y'][z['weight']>0]),[0,1]);clf=RT.make_classifier(seed);clf.fit(z['X32'],z['y'],sample_weight=z['weight']);assert len(clf.estimators_)==64;manual=[];minleaf=10**9
            for tree in clf.estimators_:
                assert tree.tree_.max_depth<=4;lid=tree.apply(z['X32']);v=tree.tree_.value[lid,0,:];manual.append(v[:,1]/v.sum(axis=1));leafnodes=np.flatnonzero(tree.tree_.children_left==-1);assert (tree.tree_.n_node_samples[leafnodes]>=10).all();minleaf=min(minleaf,int(tree.tree_.n_node_samples[leafnodes].min()));nodes+=tree.tree_.node_count;leaves+=len(leafnodes)
                for leaf in leafnodes:
                    mask=lid==leaf;ww=z['weight'][mask];assert ww.sum()>0;value=tree.tree_.value[leaf,0,:];assert abs(float(ww[z['y'][mask]==1].sum()/ww.sum())-float(value[1]/value.sum()))<1e-10
            fitprob=RT.positive_score(clf,z['X32']);maxerr=float(abs(np.mean(manual,axis=0)-fitprob).max());assert maxerr<1e-12;calprob=RT.positive_score(clf,cal['X32']);tau,newkeep=RT.protect_cutoff(calprob,cal['y']>0);assert not newkeep and (calprob>=tau).all()
            if tau<1:assert np.any(calprob<np.nextafter(tau,np.inf))
        else:
            saved=np.load(R/'outputs/grid271_union_guard/calibration_predictions'/f'{pool}.npz');assert np.array_equal(saved['ids'],cal['ids']);calprob=saved['score'];assert tau==ref['fixed271_cutoff']['tau'];assert keep==(clf is None)
            if not keep:nodes=sum(t.tree_.node_count for t in clf.estimators_);leaves=sum(int((t.tree_.children_left==-1).sum()) for t in clf.estimators_)
        parent=cal['parent261_score']>=0;cont=RT.apply_gate(parent,calprob,tau,KEEP=keep);assert cont.all()
        bundle=dict(pool=pool,model=clf,tau=tau,KEEP=keep,seed=seed,params=RT.PARAMS,base_model_sha=sha(Q/'qualified_inputs'/f'{pool}_base.npz'),base_theta=base['theta'],base_alpha=base['alpha'],sklearn_version=sklearn.__version__,context_augmented=changed,original_model_sha=ref['model_sha256'])
        (O/'models'/f'{pool}.pkl').write_bytes(pickle.dumps(bundle,protocol=4));np.savez_compressed(O/'calibration_predictions'/f'{pool}.npz',ids=cal['ids'],score=calprob,positive=cal['y']>0,parent_continue=parent,continue_=cont)
        receipts.append(dict(pool=pool,fitted=changed,new_forest_fit=changed,new_cutoff_fit=changed,KEEP=keep,tau=tau,fit_rows=len(z['ids']),augmentation_rows=int(z['is_augmented'].sum()),guard_positive=len(cal['ids']),trees=0 if keep else 64,new_trees=64 if changed else 0,nodes=int(nodes),leaves=int(leaves),min_leaf_samples=minleaf,manual_native_probability_maxerr=maxerr,fit_file_sha=sha(Q/'qualified_inputs'/f'{pool}_fit.npz'),cal_file_sha=sha(Q/'qualified_inputs'/f'{pool}_cal.npz'),wall_s=time.perf_counter()-t));save('fit_receipts.json',receipts)
    assert len(receipts)==12 and sum(r['new_trees'] for r in receipts)==576 and sum(r['trees'] for r in receipts)==704 and (time.perf_counter_ns()-start)/1e9<600;print('GRID274_NINEFIT_NINECUTOFF_DONE')
'''
score=r'''def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all12model_cutoffbundles_published=True,NNforward0=True));(O/'outer_predictions').mkdir();records=[];refs={r['pool']:r for r in read(Q/'model_references.json')}
    for file in sorted(PUBLIC.glob('*.npz')):
        pool=file.stem;parent=np.load(file);bundle=pickle.loads((O/'models'/f'{pool}.pkl').read_bytes());changed=refs[pool]['changed']
        if changed:
            raw=np.load(R/'outputs/grid249_context_bit_plan/outer_parent_public'/file.name);assert np.array_equal(parent['ids'],raw['ids']);X=raw['X'].copy();X[:,:38]=np.tanh(X[:,:38]);prob=RT.positive_score(bundle['model'],RT.as_input(X))
        else:
            old=np.load(R/'outputs/grid271_union_guard/outer_predictions'/file.name);assert np.array_equal(parent['ids'],old['ids']);prob=old['forest_score']
        cont=RT.apply_gate(parent['continue_'],prob,bundle['tau'],KEEP=bundle['KEEP']);assert not np.any(cont&~parent['continue_'])
        np.savez_compressed(O/'outer_predictions'/file.name,ids=parent['ids'],forest_score=prob,gate_margin=prob-bundle['tau'],continue_=cont,base_CONT=parent['base_CONT'],parent261_continue=parent['continue_'],parent261_logit=parent['logit']);records.append(dict(pool=pool,rows=len(prob),continues=int(cont.sum()),new_forest_forward=changed,NN_forward=False,outcome_fields_read=[]))
    assert sum(r['rows'] for r in records)==4681 and sum(r['new_forest_forward'] for r in records)==9;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'outer_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID274_SCORED_ALL12')
'''
tree=ast.parse(s);reps={'prepare':prepare,'train':train,'score':score};lines=s.splitlines(keepends=True)
for node in sorted([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in reps],key=lambda n:n.lineno,reverse=True):lines[node.lineno-1:node.end_lineno]=[reps[node.name]+'\n']
s=''.join(lines);ast.parse(s);(W/'run.py').write_text(s,encoding='utf8');print('GRID274_CODE_PARSED')
