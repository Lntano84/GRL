"""Create a source-preserving one-parameter derivative before protocol freeze."""
from pathlib import Path
import ast,hashlib,json
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent
base=R/'work/grid208_public_amount/run.py';source=base.read_text(encoding='utf8')
def replace_once(text,a,b):
    assert text.count(a)==1,a
    return text.replace(a,b)
source=source.replace('GRID208','GRID210').replace('grid208_public_amount','grid210_amount_depth').replace('grid208_fixed_base','grid210_fixed_base')
source=replace_once(source,"OLD=R/'outputs/grid206_public_event'","OLD=R/'outputs/grid208_public_amount'")
source=replace_once(source,"PRIOR=R/'outputs/grid203_tree_event'","PRIOR=R/'outputs/grid206_public_event'")
source=replace_once(source,"R/'work/grid206_public_event/run.py');P=", "R/'work/grid208_public_amount/run.py');P=")
source=replace_once(source,'max_depth=4,min_samples_leaf=10','max_depth=5,min_samples_leaf=10')
source=replace_once(source,'assert tree.tree_.max_depth<=4','assert tree.tree_.max_depth<=5')
prepare=r'''def prepare():
    assert not O.exists();O.mkdir(parents=True);P.trained();assert read(R/'outputs/grid209_lost_gain/independent_audit.json')['passed']
    save('design.json',dict(id='GRID210_AMOUNT_DEPTH',authorization='Usercontinue;pretrainingnotice exactly12fixedamountforests maxdepth4to5 only delivered.',change='Only conditionalforest max_depth4->5. Entire124input,scalers,positive198targets/weights/rows,64trees,minleaf10,allotherestimatorparams andseeds unchanged. Event208bytefrozen andNETfrozen.',motivation='GRID209 all4681paths depth4and legallysplittable mean97.93percent;lost4mean89.45percent. Structuralroomnotguarantee generalization;testoneextra level forallstates,notrescuespecificcases.',model=dict(type='ExtraTreesRegressor',params=PARAMS,seeds=[0,1,2],new_amount_forests=12,new_amount_trees=768,cap_s=600),capacity='Exactlyoneestimatorhyperparameter changed,capacityincreases;notfixedcapacity. Frozen768eventtrees+new768amounttrees,total1536active across12pairs.',labels='Exactlyold208positivefitarrays and198conditionaltargetweights. Nozeroeventmu0 labels,scaling changes,classweights ornewfeatures.',action='Samefrozenscorep * newconditionalmu > originalknowncharge buysone<=64thenreturnsbest. No fee/threshold/rank/batch/stationchanges.',evaluation='All12pairs published beforeheld. Primary208isolatesdepth;secondary206cumulativeamountinputanddepth,notpureablation.198rawreference. All4681STOProws/21402traces/all16exposedtrain dates/all3seeds. Prespecifiedsameexpectedqueryvs208and206,randomADD1;alladversekept,noallwin/5percentcriteria.',focus='Original10mainrecovery plus4old208lostprofitablepairs andallnewlosses,notonlywhether4rescued. Seed0actualnegativeandwastefractionretained.',metrics='PositiveconditionalMSEMAE,expectedgainerrors,eventAUC/Brierunchanged,actionmatch/gaincapture/query/proxyNET. Localteacherpathonly,notfullweekquality/time.',limits='No event/NETfits/forwards/newphysics/generation/final8/paper/newFRAME. No depth/leaf/treenumber/fee/thresholdsearch orautomaticextrapolation. Onefixed600s cap;preserveallpartialfailures, no extra fitting.'))
    files=[W/'run.py',W/'build_source.py',W/'source_derivation.json',O/'design.json',R/'work/grid208_public_amount/run.py',R/'work/grid208_public_amount/finalize.py',R/'work/grid203_tree_event/finalize.py',R/'outputs/grid209_lost_gain/complete.json',R/'outputs/grid209_lost_gain/independent_audit.json',R/'outputs/grid209_lost_gain/summary.json',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']
    for d in [OLD,PRIOR,LINEAR]:files += [d/n for n in ['complete.json','gate_predictions.json','oof_trajectories.json','search_metrics.json','family_summary.json','predictive_metrics.json']]
    files+=list((OLD/'models').iterdir())+list((OLD/'fit_features').glob('*.npz'))+list((FIT/'fit_features').glob('*.npz'))+list(LAB.glob('*.npz'))+list(CACHE.glob('held_f*.npz'));save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID210_PLANNED')
'''
qualify=r'''def qualify():
    verify();assert not (O/'preflight.json').exists();meta,_,_,raw=raw_inputs();(O/'fit_features').mkdir();records=[]
    from sklearn.ensemble import ExtraTreesRegressor
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';z=np.load(OLD/'fit_features'/f'{name}.npz');lab=np.load(LAB/f'{name}.npz');oldmodel=pickle.loads((OLD/'models'/f'{name}_amount.pkl').read_bytes());candidate=ExtraTreesRegressor(**PARAMS,random_state=s);assert not hasattr(candidate,'estimators_');oldparams=oldmodel.get_params();newparams=candidate.get_params();diff=[k for k in oldparams if oldparams[k]!=newparams[k]];assert diff==['max_depth'] and oldparams['max_depth']==4 and newparams['max_depth']==5;assert np.array_equal(z['target'],lab['target']) and np.array_equal(z['sample_weight'],lab['sample_weight']) and np.array_equal(z['metadata_ids'],lab['metadata_ids']);assert z['standard'].shape==(len(lab['target']),124) and (z['target']>0).all()
            hz=np.load(CACHE/f'held_{name}.npz');fd={meta[int(i)]['date'] for i in z['metadata_ids']};hd={meta[int(i)]['date'] for i in hz['ids']};assert len(fd)==12 and len(hd)==4 and not fd&hd;shutil.copy2(OLD/'fit_features'/f'{name}.npz',O/'fit_features'/f'{name}.npz');assert sha(O/'fit_features'/f'{name}.npz')==sha(OLD/'fit_features'/f'{name}.npz');records.append(dict(fold=f,seed=s,positive_rows=len(lab['target']),parameter_differences=diff,event_sha256=sha(OLD/'models'/f'{name}_event.pkl'),fit_array_sha256=sha(O/'fit_features'/f'{name}.npz')))
    save('preflight.json',dict(passed=True,pools=records,only_max_depth_changes_and_fitarrays_byte_identical=True,new_fits=0,new_NET_forwards=0));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_features').iterdir()),O/'preflight.json']});verify();archive('qualified');print('GRID210_QUALIFIED')
'''
lo=source.index('def prepare():');hi=source.index('def qualify():');source=source[:lo]+prepare+source[hi:]
lo=source.index('def qualify():');hi=source.index('def train():');source=source[:lo]+qualify+source[hi:]
source=replace_once(source,'"""Only add original public41 to conditional amount input; freeze GRID206 event."""','"""Only deepen the fixed conditional amount forest from four to five levels."""')
ast.parse(source);out=W/'run.py';assert not out.exists();out.write_text(source,encoding='utf8',newline='\n')
record=dict(base_source=str(base),base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),generated_source_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),scientific_change='Onlyregressormax_depth4->5;customprepare/qualificationverifyallotherparamsandfitarraysunchanged',old_source_preserved=True)
(W/'source_derivation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n');print('GRID210_SOURCE_CREATED')
