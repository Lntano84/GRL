"""Single preregistered parent-coherent residual and empirical positive protection."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid247_canonical_residual';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID247_canonical_residual';Q=R/'outputs/grid246_canonical_plan';PUBLIC=R/'outputs/grid242_parent_prior_inputs';PLAN=Q;KEEP=[j for j in range(42) if j not in [14,18,19]];BASE=R/'outputs/grid208_public_amount';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def module(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
LOSS=module('grid247_loss219',R/'work/grid219_prior_objective/run.py');STEP=module('grid247_step221',R/'work/grid221_step_plan/run.py');objective=LOSS.objective
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
    for folder in {'trained':['models'],'scored':['outer_predictions']}.get(phase,[]):shutil.copytree(O/folder,d/folder)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def prepare():
    assert not (O/'source_freeze.json').exists() and not (O/'training_started.json').exists() and read(Q/'complete.json')['passed'];O.mkdir(exist_ok=True)
    save('design.json',dict(id='GRID247',representation='ONLYqualified246indices;38public+b12=39input40params,semanticthreecolumnsremoved,explicitunitL2functionalpriorchange',preregistered_candidate=read(PLAN/'design.json'),fits='Exactly11linear40paramfits+11original221scalarfits,1KEEP. Single600s totalfit/scalarwindow,noretry/extension. Thetafit8only,scalarprotectunion12originalSTOPpositives.',
        objective='MeanweightedlogisticNLL+0.5L2all40inclintercept,offsetparentb12,zero40init. OriginalL-BFGS-Bmaxiter2000/gtol1e-6/ftol1e-12/maxls50. Allsuccessandraw40gradinf<.001 beforescore.',
        score='Original221floatingexpression: endpoint=b12+linear,score=b12+alpha*(endpoint-b12). Onlyfilteroriginal208CONT,atmostsameonebatch. Sameparentpriorallroles,nosecondsupportfallback/inner8headdeployment.',
        barriers='Codeinputsfirstpublish;all12weights+stepspublishedbeforeouterpolicyscores;all12outerpredictionspublishedbeforeoutcomejoin.',
        comparison='Primarymatched243andcommonGATE;230historicalonly. ONLYcanonical38public+b12/40params vs24343params;sametargets/rows/weights/scalers/parentprior/protection. ExplicitunitL2functionalnormchange,notstrictsamecapacity. Alloldexposed16dates3seeds/counterexamples.',
        boundaries='NoNN/headfits,extraheaddeploymentinference,newphysics/final8/paper/FRAME. Originalfee/labels/controller/global128/64/stagnation2/.9/order/aliases/stationfixed;onlineoverheadnotmeasured.'))
    paths=[W/'run.py',O/'design.json',PLAN/'design.json',PLAN/'qualification.json',Q/'complete.json',Q/'qualified_freeze.json',BASE/'complete.json',BASE/'oof_trajectories.json',BASE/'gate_predictions.json',BASE/'search_metrics.json',META,R/'work/grid219_prior_objective/run.py',R/'work/grid221_step_plan/run.py',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']
    for folder in ['fit_inputs','calibration_inputs']:paths+=list((Q/folder).glob('*.npz'))
    paths+=list((PUBLIC/'outer_parent_public').glob('*.npz'))+[PUBLIC/'generated_freeze.json',Q/'qualification.json',Q/'qualified_freeze.json']
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('qualified');print('GRID247_QUALIFIED')
def train():
    verify();assert not (O/'training_started.json').exists();from scipy.optimize import minimize
    save('training_started.json',dict(linear_fits=11,scalar_fits=11,constantKEEP=1,cap_s=600));start=time.perf_counter();(O/'models').mkdir();records=[]
    for file in sorted((Q/'fit_inputs').glob('*.npz')):
        assert time.perf_counter()-start<600;pool=file.stem;z=np.load(file);c=np.load(Q/'calibration_inputs'/file.name);fallback=bool(z['fallback_KEEP'][0]);theta=np.zeros(40);initial=objective(theta,z['X'],z['y'],z['weight'],z['b'])[0];t=time.perf_counter()
        if fallback:success=True;nit=0;message='predeclaredKEEP';value=initial;g=np.zeros(40);alpha=0.;bound=1.
        else:
            result=minimize(objective,theta,args=(z['X'],z['y'],z['weight'],z['b']),method='L-BFGS-B',jac=True,options=dict(maxiter=2000,gtol=1e-6,ftol=1e-12,maxls=50));theta=result.x.copy();success=bool(result.success);nit=int(result.nit);message=str(result.message);value,g=objective(theta,z['X'],z['y'],z['weight'],z['b']);assert value<=initial+1e-8 and abs(value-result.fun)<1e-8;end=c['b']+np.sum(c['X']*theta[:-1],axis=1)+theta[-1];alpha,bound=STEP.choose_alpha(c['b'],end,c['y']>0);assert np.all((c['b']+alpha*(end-c['b']))[c['y']>0]>=0)
        np.savez_compressed(O/'models'/file.name,theta=theta,alpha=np.array([alpha]),fallback_KEEP=np.array([fallback]));records.append(dict(pool=pool,fitted=not fallback,parameters=40,fit8_rows=len(z['y']),fit8_positive=int(z['y'].sum()),union12_STOP_positive=int(c['y'].sum()),initial_loss=initial,final_loss=value,gradient_inf=float(abs(g).max()),solver_success=success,iterations=nit,message=message,alpha=alpha,analytic_bound=bound,fit_and_scalar_wall_s=time.perf_counter()-t));save('fit_receipts.json',records)
    elapsed=time.perf_counter()-start;assert elapsed<600 and len(records)==12 and sum(r['fitted'] for r in records)==11;ok=all(r['solver_success'] and r['gradient_inf']<1e-3 for r in records);save('fits.json',dict(passed=ok,new_linear_fits=11,new_scalar_fits=11,constantKEEP=1,new_NNfits=0,wall_s=elapsed,all_models_numerically_qualified=ok));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fit_receipts.json',O/'fits.json']});verify();archive('trained');print('GRID247_TRAINED',json.dumps(read(O/'fits.json')))
def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
    assert read(O/'fits.json')['all_models_numerically_qualified']
def publication():
    trained();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'trained/trained_freeze.json').relative_to(REPO).as_posix())==(O/'trained_freeze.json').read_bytes();save('pre_score_publication.json',dict(passed=True,git_sha=head,all12weights_steps_published=True));print('GRID247_MODELS_PUBLISHED')
def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all12weights_steps_published=True));(O/'outer_predictions').mkdir();records=[]
    for file in sorted((PUBLIC/'outer_parent_public').glob('*.npz')):
        z=np.load(file);p=np.load(O/'models'/file.name);t=p['theta'];a=float(p['alpha'][0]);end=z['b']+np.sum(z['X'][:,KEEP]*t[:-1],axis=1)+t[-1];logit=z['b']+a*(end-z['b']);cont=z['base_CONT']&(logit>=0);assert np.array_equal(z['base_CONT'],z['b']>0) and not np.any(cont&~z['base_CONT']);np.savez_compressed(O/'outer_predictions'/file.name,ids=z['ids'],logit=logit,continue_=cont,base_CONT=z['base_CONT']);records.append(dict(pool=file.stem,rows=len(logit),continues=int(cont.sum()),outcome_fields_read=[]))
    assert sum(r['rows'] for r in records)==4681;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'outer_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID247_SCORED_ALL12')
def scored():
    trained()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p
def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes();save('pre_join_publication.json',dict(passed=True,git_sha=head,all12predictions_published=True));print('GRID247_SCORES_PUBLISHED')
def join():
    scored();assert read(O/'pre_join_publication.json')['passed'] and not (O/'join_started.json').exists();save('join_started.json',dict(all12predictions_published=True));summarize=module('grid247_saved_summary',R/'work/grid227_context_newton/join.py').summarize;meta=read(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};parents=[r for r in read(BASE/'oof_trajectories.json') if r['method'].startswith('GATE_')];task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];scores={}
    for file in (O/'outer_predictions').glob('*.npz'):
        f,s=[int(v[1:]) for v in file.stem.split('_')];z=np.load(file)
        for i,cont in zip(z['ids'],z['continue_']):scores[f,s,int(i)]=bool(cont)
    tr=list(parents);pred=[]
    for seed in range(3):
        for old in [r for r in parents if r['method']==f'GATE_seed{seed}']:
            c=cases[old['sample_id']];f=old['fold'];j=old['baseline_stop'];eligible=j<len(c['states']);append=scores[f,seed,lookup[c['sample_id'],j]] if eligible else False;assert not append or old['append'];end=j+int(append);new=dict(old);new.update(method=f'CANON_seed{seed}',actions=[1]*end+([0] if end<len(c['states']) else []),batches=end,gain=float(c['gains'][:end].sum()),queries=int(c['queries'][:end].sum()),chosen_pool_id=c['chosen'][end],append=append);new['net_utility']=(new['gain']-fee*new['queries'])/scale;new['action_match']=new['chosen_pool_id']==c['chosen'][-1];tr.append(new)
            if eligible:pred.append(dict(fold=f,seed=seed,sample_id=c['sample_id'],position=j,date=c['date'],call=c['physical_state_group'],base_continue=old['append'],continue_=append,actual_next_NET=float((c['gains'][j]-fee*c['queries'][j])/scale),actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j])))
    assert len(tr)==14268 and len(pred)==4681;metrics={};ref=read(BASE/'search_metrics.json')
    for arm in ['GATE','CANON']:
        for seed in range(3):
            name=f'{arm}_seed{seed}';rr=[r for r in tr if r['method']==name];assert len(rr)==2378;metrics[name]=summarize(rr)
            if arm=='GATE':
                for k,v in metrics[name]['macro'].items():assert abs(v-ref[name]['macro'][k])<1e-12
    family={arm:{k:float(np.mean([metrics[f'{arm}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{arm}_seed0']['macro']} for arm in ['GATE','CANON']};paired=[]
    for seed in range(3):
        old,new=metrics[f'GATE_seed{seed}'],metrics[f'CANON_seed{seed}'];paired.append(dict(seed=seed,macro_delta={k:new['macro'][k]-old['macro'][k] for k in old['macro']},per_date=[dict(date=x['date'],NET_delta=y['mean_net_utility']-x['mean_net_utility'],query_delta=y['mean_queries']-x['mean_queries'],gain_delta=y['mean_gain']-x['mean_gain']) for x,y in zip(old['per_date'],new['per_date'])]))
    counts=dict(original208_appends=sum(r['base_continue'] for r in pred),appends=sum(r['continue_'] for r in pred),profitable_retained=sum(r['continue_'] and r['actual_next_NET']>0 for r in pred),profitable_removed=sum(r['base_continue'] and not r['continue_'] and r['actual_next_NET']>0 for r in pred),unprofitable_removed=sum(r['base_continue'] and not r['continue_'] and r['actual_next_NET']<=0 for r in pred))
    for n,v in [('oof_trajectories.json',tr),('gate_predictions.json',pred),('search_metrics.json',metrics),('family_summary.json',family),('paired_summary.json',paired),('gate_counts.json',counts)]:save(n,v)
    scored();save('review.json',dict(passed=True,all14268savedsequences_checked=True,all4681publishedscores_joined=True,commonGATEreproduced=True,all3seeds16oldexposeddates_retained=True,no_new_physics=True,final8_absent=True));archive('evaluated');print('GRID247_EVALUATED',json.dumps(dict(family=family,counts=counts,paired=[dict(seed=r['seed'],NET=r['macro_delta']['mean_net_utility'],queries=r['macro_delta']['mean_queries'],gain=r['macro_delta']['mean_gain']) for r in paired])))
if __name__=='__main__':globals()[sys.argv[1]]()
