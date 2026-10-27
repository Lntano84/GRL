"""Single fixed input-block addition: hidden83 + original public41 event forest."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import datetime,hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid206_public_event';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID206_public_event';OLD=R/'outputs/grid203_tree_event';FIT=R/'outputs/grid192_gate_support';LAB=R/'outputs/grid196_gain_readout/fit_labels';CACHE=R/'outputs/grid190_extension_gate';LINEAR=R/'outputs/grid198_conditional_gain';PRIOR=R/'outputs/grid201_tree_amount'
sp=importlib.util.spec_from_file_location('grid206_fixed_base',R/'work/grid203_tree_event/run.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P);B=P.B;PARAMS=dict(P.PARAMS)
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
def raw_inputs():
    meta,lookup,cases=B.load();cc={c['sample_id']:c for c in cases};raw=np.array([cc[m['sample_id']]['states'][m['position']] for m in meta],float);assert raw.shape==(3582,41) and np.isfinite(raw).all();return meta,lookup,cases,raw
def prepare():
    assert not O.exists();O.mkdir(parents=True);P.trained();assert read(R/'outputs/grid205_public_separation/independent_audit.json')['passed']
    save('design.json',dict(id='GRID206_PUBLIC_EVENT',authorization='Usercontinue within ongoing internalmodule work; pretraining notice exactly12 unchangedshallowforests adding fulloriginal41publicsummary delivered.',change='Only event input: same hidden83 plus all original prefix41 raw public features,124total. No25diagnostic extras/selectedfeatures. Same83normalization;additional41 weightedfit-only mean/std scales<1e-8->1. GRID203conditionalamountforestbytefrozen,NETfrozen.',motivation='GRID205 raw41 and hidden83 showed different label-blind neighbor relations; descriptive evidence motivates test,not proof informationlost or guarantee performance.',fixed='Original192halfSTOPhalfCONT labels/weights/totalmass/datefolds/fee/12864/.9/rank/aliases/station unchanged. All16exposed historicaltrain dates/all3seeds. No classweights/gainweights/calibration/thresholdsearch.',model=dict(type='ExtraTreesClassifier',params=PARAMS,seeds=[0,1,2],new_forests=12,new_trees=768,fit_cap_s=600),capacity='Same64trees/depth4/leaf10 but extra candidate split dimensions change randomfeature selection and functionclass;notstrictsamecapacity. Frozen64amounttrees perpair,total1536active trees.',action='Native eventp * frozen201/203 conditionalmu > original knownnextcharge buysone<=64 then returnsbestobserved.',evaluation='All12weights published beforeheld. PrimaryGRID203 isolates public41addition;GRID198secondaryinternalreference andGRID201rawreference. All4681STOP/21402traces,allseeds/dates/adverse cases;prespecifiedsameexpectedqueryvs203and198 plusrandomADD1. No allwin/5percentgate.',metrics='NativeeventAUC/Brier/BCE(logclip1e-15metriconly),gainMSEMAE,unchangedpositiveconditionalMSE,actionmatch/gaincapture/queries/proxyNET;notfullweekquality/time.',limits='ZeroNETfits/forwards/newphysics/generation/final8/paper/newFRAME. All teacherprefixesandseeds correlated. No extra fits/retries/parametersearch/automaticexpansion.'))
    files=[W/'run.py',O/'design.json',R/'work/grid203_tree_event/run.py',R/'work/grid203_tree_event/finalize.py',R/'outputs/grid205_public_separation/independent_audit.json',R/'outputs/grid205_public_separation/summary.json',R/'outputs/grid205_public_separation/判读与固定训练方案.md',R/'outputs/grid113_relational_learning/train_metadata.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']
    for d in [OLD,LINEAR,PRIOR]:files += [d/n for n in ['complete.json','oof_trajectories.json','search_metrics.json','family_summary.json','gate_predictions.json','predictive_metrics.json']]
    files+=list((OLD/'models').iterdir())+list((FIT/'fit_features').glob('*.npz'))+list(LAB.glob('*.npz'))+list(CACHE.glob('held_f*.npz'));save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID206_PLANNED')
def qualify():
    verify();assert not (O/'preflight.json').exists();meta,_,_,raw=raw_inputs();(O/'fit_features').mkdir();records=[]
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';z=np.load(FIT/'fit_features'/f'{name}.npz');ids=z['ids'].astype(int);w=z['sample_weight'];x=raw[ids];rm=np.average(x,axis=0,weights=w);rs=np.sqrt(np.average((x-rm)**2,axis=0,weights=w));rs[rs<1e-8]=1.;standard=np.c_[z['standard'],(x-rm)/rs];assert standard.shape==(len(ids),124) and np.array_equal(standard[:,:83],z['standard']) and np.isfinite(standard.astype(np.float32)).all();lab=np.load(LAB/f'{name}.npz');assert np.array_equal(z['y']>0,lab['target']>0);hz=np.load(CACHE/f'held_{name}.npz');fd={meta[i]['date'] for i in ids};hd={meta[int(i)]['date'] for i in hz['ids']};assert len(fd)==12 and len(hd)==4 and not fd&hd
            np.savez_compressed(O/'fit_features'/f'{name}.npz',standard=standard,raw_mean=rm,raw_scale=rs,mean=z['mean'],scale=z['scale'],y=z['y'],sample_weight=w,ids=ids,original_STOP_count=z['original_STOP_count']);records.append(dict(fold=f,seed=s,rows=len(ids),features=124,positive=int(z['y'].sum()),sampleweight_sum=float(w.sum()),all41included=True,fit_only_scaling=True,amount_sha256=sha(OLD/'models'/f'{name}_amount.pkl')))
    save('preflight.json',dict(passed=True,pools=records,new_fits=0,new_NET_forwards=0));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_features').iterdir()),O/'preflight.json']});verify();archive('qualified');print('GRID206_QUALIFIED')
def train():
    verify();assert read(O/'preflight.json')['passed'] and not (O/'training_started.json').exists()
    for p,h in read(O/'qualified_freeze.json').items():assert sha(O/p)==h
    from sklearn.ensemble import ExtraTreesClassifier
    save('training_started.json',dict(heads=12,new_trees=768,cap_s=600,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));(O/'models').mkdir();records=[];t=time.perf_counter()
    for f in range(4):
        for s in range(3):
            if time.perf_counter()-t>600:raise TimeoutError('Preserve partial models; original600s cap')
            name=f'f{f}_s{s}';z=np.load(O/'fit_features'/f'{name}.npz');x,y,w=z['standard'],z['y'].astype(int),z['sample_weight'];start=time.perf_counter();clf=ExtraTreesClassifier(**PARAMS,random_state=s);clf.fit(x,y,sample_weight=w);nodes=leaves=0;worst=0.;manual=[]
            for tree in clf.estimators_:
                leaf=tree.apply(x);v=tree.tree_.value[leaf,0,:];manual.append(v[:,1]/v.sum(axis=1));assert tree.tree_.max_depth<=4;nodes+=tree.tree_.node_count;leaves+=int(np.sum(tree.tree_.children_left==-1))
                for j in np.unique(leaf):
                    mask=leaf==j;assert mask.sum()>=10;v=tree.tree_.value[j,0,:];err=abs(float(w[mask]@y[mask]/w[mask].sum())-float(v[1]/v.sum()));worst=max(worst,err);assert err<1e-9
            p=clf.predict_proba(x)[:,1];assert np.array_equal(clf.classes_,[0,1]) and clf.n_features_in_==124;assert np.allclose(p,np.mean(manual,axis=0),atol=1e-11,rtol=1e-10);assert abs(float(w@p/w.sum())-float(w@y/w.sum()))<1e-10
            (O/'models'/f'{name}_event.pkl').write_bytes(pickle.dumps(clf,protocol=4));shutil.copy2(OLD/'models'/f'{name}_amount.pkl',O/'models'/f'{name}_amount.pkl');cp=np.load(OLD/'models'/f'{name}.npz');np.savez_compressed(O/'models'/f'{name}.npz',mean=z['mean'],scale=z['scale'],raw_mean=z['raw_mean'],raw_scale=z['raw_scale'],positive_min=cp['positive_min'],positive_max=cp['positive_max']);records.append(dict(fold=f,seed=s,features=124,rows=len(y),nodes=nodes,leaves=leaves,max_leaf_probability_error=worst,fit_Brier=float(w@((p-y)**2)/w.sum()),fit_wall_s=time.perf_counter()-start,event_sha256=sha(O/'models'/f'{name}_event.pkl'),frozen_amount_sha256=sha(O/'models'/f'{name}_amount.pkl')));save('fit_receipts.json',records);print('GRID206_FIT',f,s,nodes,leaves,flush=True)
    save('fits.json',dict(passed=True,new_event_forests=12,new_event_trees=768,amount_and_NET_fits=0,total_wall_s=time.perf_counter()-t,records=records));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'fit_receipts.json']});verify();archive('trained');print('GRID206_TRAINED')
def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h
def publication():
    trained();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert head==remote;p=A/'trained/trained_freeze.json';assert subprocess.check_output(['git','show','HEAD:'+p.relative_to(REPO).as_posix()],cwd=REPO)==p.read_bytes();save('preheld_publication.json',dict(passed=True,git_sha=head,remote_main=head,all12pairs_published=True));print('GRID206_PUBLISHED')
def evaluate():
    trained();assert read(O/'preheld_publication.json')['passed'] and not (O/'evaluation_started.json').exists();save('evaluation_started.json',dict(all12pairs_published=True));meta,lookup,cases,raw=raw_inputs();task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];bases={(r['method'],r['sample_id']):r for r in read(OLD/'oof_trajectories.json')};oldpred={(r['seed'],r['sample_id']):r for r in read(OLD/'gate_predictions.json')};pred=[];tr=[];checks=[]
    for f in range(4):
        for s in range(3):
            name=f'f{f}_s{s}';cp=np.load(O/'models'/f'{name}.npz');hz=np.load(CACHE/f'held_{name}.npz');hidden=(hz['features']-cp['mean'])/cp['scale'];x=np.c_[hidden,(raw[hz['ids'].astype(int)]-cp['raw_mean'])/cp['raw_scale']];clf=pickle.loads((O/'models'/f'{name}_event.pkl').read_bytes());amount=pickle.loads((O/'models'/f'{name}_amount.pkl').read_bytes());p=clf.predict_proba(x)[:,1];mu=amount.predict(hidden);manual=[]
            for tree in clf.estimators_:v=tree.tree_.value[tree.apply(x),0,:];manual.append(v[:,1]/v.sum(axis=1))
            assert np.allclose(p,np.mean(manual,axis=0),atol=1e-11,rtol=1e-10);byidx={int(i):(bool(pp*mm>q),float(pp),float(mm),float(q)) for i,pp,mm,q in zip(hz['ids'],p,mu,hz['features'][:,82])}
            for c in [c for c in cases if bases[f'NET_seed{s}',c['sample_id']]['fold']==f]:
                base=bases[f'NET_seed{s}',c['sample_id']];j=base['baseline_stop'];eligible=j<len(c['states']);d,pp,mm,q=byidx[lookup[c['sample_id'],j]] if eligible else (False,0.,0.,0.)
                if eligible:
                    assert np.isclose(q,fee*c['queries'][j]/scale,atol=1e-8,rtol=2e-6);assert np.isclose(mm,oldpred[s,c['sample_id']]['conditional_normalized_gain'],atol=1e-10,rtol=1e-10);pred.append(dict(fold=f,seed=s,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=j,gate_continue=d,event_probability=pp,conditional_normalized_gain=mm,predicted_normalized_gain=pp*mm,known_charge=q,predicted_net=pp*mm-q,actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j]),actual_next_NET=float((c['gains'][j]-fee*c['queries'][j])/scale),actual_next_change=bool(c['chosen'][j]!=c['chosen'][j+1])))
                for method in ['NET','ADD1','GATE']:
                    add=bool(eligible and (method=='ADD1' or method=='GATE' and d));v=B.H.trajectory(c,np.arange(len(c['states']))<j+int(add),f'{method}_seed{s}',fee,scale);v.update(fold=f,baseline_stop=j,baseline_gain=base['gain'],baseline_queries=base['queries'],eligible=eligible,append=add,action_match=v['chosen_pool_id']==c['chosen'][-1]);assert base['queries']<=v['queries']<=int(c['queries'].sum()) and v['queries']-base['queries']<=64 and base['gain']<=v['gain'];assert not base['action_match'] or v['action_match']
                    if method in ['NET','ADD1']:
                        for k in ['gain','queries','chosen_pool_id','actions']:assert v[k]==bases[f'{method}_seed{s}',c['sample_id']][k]
                    tr.append(v)
            checks.append(dict(fold=f,seed=s,held_rows=len(p),native_probabilities_match=True,frozen_amount_bytes=sha(O/'models'/f'{name}_amount.pkl')))
    metrics={f'{m}_seed{s}':B.summarize([r for r in tr if r['method']==f'{m}_seed{s}']) for m in ['NET','ADD1','GATE'] for s in range(3)};family={m:{k:float(np.mean([metrics[f'{m}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{m}_seed0']['macro']} for m in ['NET','ADD1','GATE']};counts=dict(eligible=len(pred),append=sum(r['gate_continue'] for r in pred),append_changes=sum(r['gate_continue'] and r['actual_next_change'] for r in pred),append_no_change=sum(r['gate_continue'] and not r['actual_next_change'] for r in pred),append_positive_NET=sum(r['gate_continue'] and r['actual_next_NET']>0 for r in pred));assert len(pred)==4681 and len(tr)==21402
    for n,v in [('gate_predictions.json',pred),('oof_trajectories.json',tr),('search_metrics.json',metrics),('family_summary.json',family),('gate_counts.json',counts),('held_native_checks.json',checks)]:save(n,v)
    trained();save('review.json',dict(passed=True,all4681_native_probabilities_and_frozen_amounts=True,all21402trajectories_NET_ADD1_reproduced=True,all12weights_beforeheld=True,new_event_forests=12,amount_and_NET_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('evaluated');print('GRID206_EVALUATED',json.dumps(dict(family=family,counts=counts)))
if __name__=='__main__':globals()[sys.argv[1]]()
