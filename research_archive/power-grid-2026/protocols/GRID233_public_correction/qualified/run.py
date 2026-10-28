"""Fixed public42 residual: 11 convex fits and 11 joint empirical scalar steps."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid233_public_correction';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID233_public_correction'
Q=R/'outputs/grid232_public_correction_plan';P=R/'outputs/grid220_full_prior';S=R/'outputs/grid230_training_budget';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
PLAN=module('grid233_qualified232',R/'work/grid232_public_correction_plan/run.py');objective=PLAN.LOSS.objective
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    for p,h in read(Q/'source_freeze.json').items():assert sha(R/p)==h,p
    for p,h in read(Q/'qualified_freeze.json').items():assert sha(Q/p)==h,p
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    for folder in {'trained':['models'],'scored':['held_predictions']}.get(phase,[]):shutil.copytree(O/folder,d/folder)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def prepare():
    assert not O.exists() and read(Q/'qualification.json')['passed'];O.mkdir()
    save('design.json',dict(id='GRID233',qualified_plan=read(Q/'design.json'),fits='Exactly11linear43paramfits+11jointscalarcalibrations=22;oneconstantKEEP. Single600s totalwindow,noextension,retryoralternatefit.',
        optimizer='OriginalL-BFGS-Bmaxiter2000,gtol1e-6,ftol1e-12,maxls50,zero43init. Allsuccessandrawactive43gradinf<.001 beforeheldscoring. Allendpoints retained.',
        scalar='232joint_step:one maximalalpha satisfying originalSTOPpositiveprotection ANDsameparent222trainingcancellationquerymassbudget. Notchangingfee orheldbudgetmatching.',
        arm='PUBLIC42 offsetb+public41/b linearcorrection. Strong208prior unchanged,first83residualcoefficients absent. Same0.5STOP/0.5auxiliaryrisk,fullunitL2all43inclintercept. Capacityrestrictiondisclosed.',
        barriers='Sourceinputsfirstpublish;all12modelscalarfilespublishbeforeheldscore;all12predictionspublishbeforeoutcomejoin.',
        controls='230BUDGETprimary;222STOP_ONLY,208GATE reused. All3seeds16exposeddates,allcounterexamples. Notnewblindtest,newFRAME,equalheldqueryclaim orfullweekquality. No final8/physics/paper/dependencyinstall.'))
    paths=[W/'run.py',O/'design.json',R/'work/grid232_public_correction_plan/run.py',Q/'design.json',Q/'qualification.json',Q/'source_freeze.json',Q/'qualified_freeze.json',S/'complete.json',S/'independent_audit.json',S/'gate_predictions.json',S/'oof_trajectories.json',S/'search_metrics.json',S/'family_summary.json',META,R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']+list((Q/'qualified_inputs').glob('*.npz'))+list((P/'public_held_inputs').glob('*.npz'))
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('qualified');print('GRID233_QUALIFIED')
def train():
    verify();assert not (O/'training_started.json').exists();from scipy.optimize import minimize
    save('training_started.json',dict(linear_fits=11,joint_scalar_fits=11,constantKEEP=1,total_cap_s=600));start=time.perf_counter();(O/'models').mkdir();records=[]
    for f in range(4):
        for seed in range(3):
            assert time.perf_counter()-start<600,'Original600s total window exceeded; do not extend or repeat completed pools.'
            pool=f'f{f}_s{seed}';z=np.load(Q/'qualified_inputs'/f'{pool}.npz');fallback=bool(z['fallback_KEEP'][0]);theta=np.zeros(43);initial=objective(theta,z['X'],z['y'],z['weight'],z['b'])[0];t=time.perf_counter()
            if fallback:success=True;nit=0;message='constantKEEP';final=initial;gradient=np.zeros(43);a=0.;cal=dict(constantKEEP=True)
            else:
                result=minimize(objective,theta,args=(z['X'],z['y'],z['weight'],z['b']),method='L-BFGS-B',jac=True,options=dict(maxiter=2000,gtol=1e-6,ftol=1e-12,maxls=50));theta=result.x.copy();success=bool(result.success);nit=int(result.nit);message=str(result.message);final,gradient=objective(theta,z['X'],z['y'],z['weight'],z['b']);assert abs(final-result.fun)<1e-8 and final<=initial+1e-8
                d=z['calibration_X']@theta[:-1]+theta[-1];a,cal=PLAN.joint_step(z['calibration_b'],d,z['calibration_y']>0,z['query_mass'],float(z['parent_budget'][0]))
            np.savez_compressed(O/'models'/f'{pool}.npz',theta=theta,alpha=np.array([a]),fallback_KEEP=np.array([fallback]));records.append(dict(pool=pool,fitted=not fallback,parameters=43,rows=len(z['y']),positive=int(z['y'].sum()),initial_loss=initial,final_loss=final,gradient_inf=float(abs(gradient).max()),solver_success=success,iterations=nit,message=message,alpha=a,joint_calibration=cal,fit_and_scalar_wall_s=time.perf_counter()-t));save('fit_receipts.json',records)
    elapsed=time.perf_counter()-start;assert elapsed<600 and len(records)==12 and sum(r['fitted'] for r in records)==11;qualified=all(r['solver_success'] and r['gradient_inf']<1e-3 for r in records)
    save('fits.json',dict(passed=qualified,new_linear_fits=11,new_joint_scalar_fits=11,constantKEEP=1,new_NNfits=0,total_fit_calibration_wall_s=elapsed,all_effective_models_qualified=qualified));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fit_receipts.json',O/'fits.json']});verify();archive('trained');print('GRID233_TRAINED',json.dumps(read(O/'fits.json')))
def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
    assert read(O/'fits.json')['all_effective_models_qualified'],'Preserve failed qualification and stop before held scores.'
def publication():
    trained();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'trained/trained_freeze.json').relative_to(REPO).as_posix())==(O/'trained_freeze.json').read_bytes();save('pre_score_publication.json',dict(passed=True,git_sha=head,all12models_steps_published=True));print('GRID233_MODELS_PUBLISHED')
def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all12models_steps_published=True));(O/'held_predictions').mkdir();records=[]
    for f in range(4):
        for seed in range(3):
            pool=f'f{f}_s{seed}';h=np.load(P/'public_held_inputs'/f'{pool}.npz');p=np.load(O/'models'/f'{pool}.npz');t=p['theta'];a=float(p['alpha'][0]);x=h['X'][:,83:125];logit=h['b']+a*(np.sum(x*t[:-1],axis=1)+t[-1]);assert np.isfinite(logit).all();np.savez_compressed(O/'held_predictions'/f'{pool}.npz',ids=h['ids'],logit=logit,keep=logit>=0,base_CONT=h['base_CONT']);records.append(dict(pool=pool,rows=len(logit),public_only=True))
    assert sum(r['rows'] for r in records)==4681;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'held_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID233_SCORED_ALL12')
def scored():
    trained()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p
def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes();save('pre_join_publication.json',dict(passed=True,git_sha=head,all12predictions_published=True));print('GRID233_PREDICTIONS_PUBLISHED')
def join():
    scored();assert read(O/'pre_join_publication.json')['passed'] and not (O/'join_started.json').exists();save('join_started.json',dict(all12predictions_published=True));summarize=module('grid233_saved_summary',R/'work/grid227_context_newton/join.py').summarize
    meta=read(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];parent=[r for r in read(S/'oof_trajectories.json') if not r['method'].startswith('MIXED_')];tr=list(parent);pred=[];changes=[];scores={}
    for file in (O/'held_predictions').glob('*.npz'):
        f,s=[int(v[1:]) for v in file.stem.split('_')];z=np.load(file)
        for i,logit,keep in zip(z['ids'],z['logit'],z['keep']):scores[f,s,int(i)]=(float(logit),bool(keep))
    pg={(r['seed'],r['sample_id']):r for r in read(S/'gate_predictions.json')}
    for seed in range(3):
        for old in [r for r in parent if r['method']==f'BUDGET_seed{seed}']:
            c=cases[old['sample_id']];f=old['fold'];j=old['baseline_stop'];eligible=j<len(c['states']);g=pg.get((seed,c['sample_id']));logit,keep=scores[f,seed,lookup[c['sample_id'],j]] if eligible else (1.,True);append=bool(g['base_continue'] and keep) if eligible else False;end=j+int(append)
            new=dict(old);new.update(method=f'PUBLIC_seed{seed}',actions=[1]*end+([0] if end<len(c['states']) else []),batches=end,gain=float(c['gains'][:end].sum()),queries=int(c['queries'][:end].sum()),chosen_pool_id=c['chosen'][end],append=append);new['net_utility']=(new['gain']-fee*new['queries'])/scale;new['action_match']=new['chosen_pool_id']==c['chosen'][-1];tr.append(new)
            if eligible:
                row=dict(g);row.update(logit=logit,keep=keep,public_continue=append,parent_budget_continue=old['append'],NET_delta_vsBUDGET=new['net_utility']-old['net_utility'],query_delta_vsBUDGET=new['queries']-old['queries'],gain_delta_vsBUDGET=new['gain']-old['gain']);pred.append(row)
                if append!=old['append']:changes.append(dict(seed=seed,fold=f,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=j,parent_append=old['append'],public_append=append,actual_next_NET=g['actual_next_NET'],NET_delta=new['net_utility']-old['net_utility'],query_delta=new['queries']-old['queries'],gain_delta=new['gain']-old['gain'],action_match_delta=int(new['action_match'])-int(old['action_match'])))
    assert len(tr)==28536 and len(pred)==4681;metrics={};reference=read(S/'search_metrics.json')
    for arm in ['GATE','STOP_ONLY','BUDGET','PUBLIC']:
        for seed in range(3):
            name=f'{arm}_seed{seed}';rr=[r for r in tr if r['method']==name];assert len(rr)==2378;metrics[name]=summarize(rr)
            if arm!='PUBLIC':
                for k,v in metrics[name]['macro'].items():assert abs(v-reference[name]['macro'][k])<1e-12
    family={a:{k:float(np.mean([metrics[f'{a}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{a}_seed0']['macro']} for a in ['GATE','STOP_ONLY','BUDGET','PUBLIC']};paired=[]
    for base in ['GATE','STOP_ONLY','BUDGET']:
        for seed in range(3):
            old,new=metrics[f'{base}_seed{seed}'],metrics[f'PUBLIC_seed{seed}'];paired.append(dict(base=base,seed=seed,macro_delta={k:new['macro'][k]-old['macro'][k] for k in old['macro']},per_date=[dict(date=x['date'],NET_delta=y['mean_net_utility']-x['mean_net_utility'],query_delta=y['mean_queries']-x['mean_queries']) for x,y in zip(old['per_date'],new['per_date'])]))
    counts=dict(original208_appends=sum(r['base_continue'] for r in pred),appends=sum(r['public_continue'] for r in pred),retained_profitable=sum(r['public_continue'] and r['actual_next_NET']>0 for r in pred),removed_profitable=sum(r['base_continue'] and not r['public_continue'] and r['actual_next_NET']>0 for r in pred),removed_unprofitable=sum(r['base_continue'] and not r['public_continue'] and r['actual_next_NET']<=0 for r in pred))
    for n,v in [('oof_trajectories.json',tr),('gate_predictions.json',pred),('changed_choices.json',changes),('search_metrics.json',metrics),('family_summary.json',family),('paired_summary.json',paired),('gate_counts.json',counts)]:save(n,v)
    scored();save('review.json',dict(passed=True,all3seeds16dates_retained=True,all_controls_reproduced=True,no_new_physics=True,final8_absent=True));archive('evaluated');print('GRID233_EVALUATED',json.dumps(dict(family=family,counts=counts,changes=len(changes),paired=[dict(base=r['base'],seed=r['seed'],NET=r['macro_delta']['mean_net_utility'],queries=r['macro_delta']['mean_queries'],gain=r['macro_delta']['mean_gain']) for r in paired])))
if __name__=='__main__':globals()[sys.argv[1]]()
