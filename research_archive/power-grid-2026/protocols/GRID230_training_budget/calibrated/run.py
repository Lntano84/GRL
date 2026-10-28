"""One fixed own-training query-mass matched scalar control; parent endpoints frozen."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid230_training_budget';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID230_training_budget'
P=R/'outputs/grid220_full_prior';T=R/'outputs/grid222_protected_step';S=R/'outputs/grid227_context_newton';Q=R/'outputs/grid229_training_budget_plan/revision02';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def module(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
PLAN=module('grid230_qualified229',R/'work/grid229_training_budget_plan/run.py')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    for p,h in read(Q/'source_freeze.json').items():assert sha(R/p)==h,p
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    for folder in {'calibrated':['models'],'scored':['held_predictions']}.get(phase,[]):shutil.copytree(O/folder,d/folder)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def prepare():
    assert not O.exists() and read(Q/'qualification.json')['passed'];O.mkdir()
    save('design.json',dict(id='GRID230',qualified_plan=read(Q/'design.json'),fits='Exactly11 deterministic scalar query-mass calibrations+1constantKEEP;zero endpoint/NNfits. One600s totalcalibrationwindow,neverextend.',
        data='Only229revision02 own training known_query_mass and220originalSTOP fit arrays;227theta fixed. Budget parent222 same training structuralmass*knownqueries. Newalpha<=227alpha,protectoriginal90STOPpositives.',
        barriers='Sourcepublishedbeforecalibration;all12effective theta+scalar files publishedbeforeheldscores;all12predictionfilespublishedbeforeoutcomejoin.',
        evaluation='All3seeds16oldexposeddates. PrimaryBUDGETvs227MIXED;222STOP_ONLY and208GATE reused. Samepredicted gain/fee/label/state/batch. Reportmorequeriesifneeded;trainingcancellationbudget doesNOTmean heldequalqueries. No final8,newphysics,paper,FRAME.'))
    paths=[W/'run.py',O/'design.json',R/'work/grid229_training_budget_plan/run.py',Q/'design.json',Q/'qualification.json',Q/'source_freeze.json',Q/'known_query_mass.json',META,S/'complete.json',S/'trained_freeze.json',S/'scored_freeze.json',S/'gate_predictions.json',S/'oof_trajectories.json',S/'search_metrics.json',S/'family_summary.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']
    for root,folder in [(P,'fit_inputs'),(P,'public_held_inputs'),(P,'models'),(T,'models'),(S,'models')]:paths+=list((root/folder).glob('*.npz'))
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('qualified');print('GRID230_QUALIFIED')
def calibrate():
    verify();assert not (O/'calibration_started.json').exists();save('calibration_started.json',dict(scalars=11,constantKEEP=1,cap_s=600));start=time.perf_counter();(O/'models').mkdir();known=read(Q/'known_query_mass.json');records=[]
    for f in range(4):
        for seed in range(3):
            assert time.perf_counter()-start<600
            pool=f'f{f}_s{seed}';z=np.load(P/'fit_inputs'/f'{pool}.npz');p=np.load(S/'models'/f'{pool}.npz');theta=p['theta'];cap=float(p['alpha'][0]);fallback=bool(p['fallback_KEEP'][0]);old=np.load(P/'models'/f'{pool}_ANCHORED.npz')['theta'];oa=float(np.load(T/'models'/f'{pool}_ANCHORED.npz')['alpha'][0]);mass=np.array(known[pool]['query_mass']);assert known[pool]['ids']==z['ids'].tolist();d=z['X']@theta[:-1]+theta[-1];od=z['X']@old[:-1]+old[-1];budget=PLAN.saved_mass(z['b'],od,oa,mass)
            if fallback:a=0.;rec=dict(active=False,constantKEEP=True)
            else:a,rec=PLAN.match_alpha(z['b'],d,cap,mass,budget)
            selected=PLAN.saved_mass(z['b'],d,a,mass);assert selected<=budget and a<=cap and np.all((z['b']+a*d)[z['y']>0]>=0)
            np.savez_compressed(O/'models'/f'{pool}.npz',theta=theta,alpha=np.array([a]),fallback_KEEP=p['fallback_KEEP']);assert np.array_equal(np.load(O/'models'/f'{pool}.npz')['theta'],theta)
            records.append(dict(pool=pool,fitted=not fallback,parent_alpha=cap,alpha=a,parent_STOP_budget=budget,parent_MIXED_mass=PLAN.saved_mass(z['b'],d,cap,mass),selected_mass=selected,positive=int((z['y']>0).sum()),theta_unchanged=True,details=rec))
    elapsed=time.perf_counter()-start;assert elapsed<600 and sum(r['fitted'] for r in records)==11
    save('calibration_receipts.json',records);save('fits.json',dict(passed=True,scalar_calibrations=11,constantKEEP=1,new_endpoint_fits=0,new_NN_fits=0,wall_s=elapsed,changed_pools=[r['pool'] for r in records if r['alpha']!=r['parent_alpha']]))
    save('calibrated_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'calibration_receipts.json',O/'fits.json']});verify();archive('calibrated');print('GRID230_CALIBRATED',json.dumps(read(O/'fits.json')))
def calibrated():
    verify()
    for p,h in read(O/'calibrated_freeze.json').items():assert sha(O/p)==h,p
def publication():
    calibrated();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'calibrated/calibrated_freeze.json').relative_to(REPO).as_posix())==(O/'calibrated_freeze.json').read_bytes();save('pre_score_publication.json',dict(passed=True,git_sha=head,all12effective_models_steps_published=True));print('GRID230_CALIBRATIONS_PUBLISHED')
def score():
    calibrated();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all12models_steps_published=True));(O/'held_predictions').mkdir();records=[]
    for f in range(4):
        for seed in range(3):
            pool=f'f{f}_s{seed}';h=np.load(P/'public_held_inputs'/f'{pool}.npz');p=np.load(O/'models'/f'{pool}.npz');t=p['theta'];a=float(p['alpha'][0]);logit=h['b']+a*(np.sum(h['X']*t[:-1],axis=1)+t[-1]);keep=logit>=0;parent=np.load(S/'held_predictions'/f'{pool}.npz');assert np.array_equal(h['ids'],parent['ids']);assert not np.any(h['base_CONT']&parent['keep']&~keep)
            np.savez_compressed(O/'held_predictions'/f'{pool}.npz',ids=h['ids'],logit=logit,keep=keep,base_CONT=h['base_CONT']);records.append(dict(pool=pool,rows=len(logit),public_only=True))
    assert sum(r['rows'] for r in records)==4681;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'held_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID230_SCORED_ALL12')
def scored():
    calibrated()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p
def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes();save('pre_join_publication.json',dict(passed=True,git_sha=head,all12predictions_published=True));print('GRID230_PREDICTIONS_PUBLISHED')
def join():
    scored();assert read(O/'pre_join_publication.json')['passed'] and not (O/'join_started.json').exists();save('join_started.json',dict(all12predictions_published=True));summarizer=module('grid230_saved_summary',R/'work/grid227_context_newton/join.py').summarize
    meta=read(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];parent=read(S/'oof_trajectories.json');tr=list(parent);pred=[];changes=[];scores={}
    for file in (O/'held_predictions').glob('*.npz'):
        f,s=[int(v[1:]) for v in file.stem.split('_')];z=np.load(file)
        for i,score,keep in zip(z['ids'],z['logit'],z['keep']):scores[f,s,int(i)]=(float(score),bool(keep))
    parent_gate={(r['seed'],r['sample_id']):r for r in read(S/'gate_predictions.json')}
    for seed in range(3):
        for old in [r for r in parent if r['method']==f'MIXED_seed{seed}']:
            c=cases[old['sample_id']];f=old['fold'];j=old['baseline_stop'];eligible=j<len(c['states']);g=parent_gate.get((seed,c['sample_id']));logit,keep=scores[f,seed,lookup[c['sample_id'],j]] if eligible else (1.,True);append=bool(g['base_continue'] and keep) if eligible else False;end=j+int(append)
            assert not old['append'] or append
            new=dict(old);new.update(method=f'BUDGET_seed{seed}',actions=[1]*end+([0] if end<len(c['states']) else []),batches=end,gain=float(c['gains'][:end].sum()),queries=int(c['queries'][:end].sum()),chosen_pool_id=c['chosen'][end],append=append);new['net_utility']=(new['gain']-fee*new['queries'])/scale;new['action_match']=new['chosen_pool_id']==c['chosen'][-1];tr.append(new)
            if eligible:
                row=dict(g);row.update(logit=logit,keep=keep,budget_continue=append,parent_mixed_continue=old['append'],NET_delta_vsMIXED=new['net_utility']-old['net_utility'],query_delta_vsMIXED=new['queries']-old['queries'],gain_delta_vsMIXED=new['gain']-old['gain']);pred.append(row)
                if append!=old['append']:changes.append(dict(seed=seed,fold=f,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=j,actual_next_NET=g['actual_next_NET'],NET_delta=new['net_utility']-old['net_utility'],query_delta=new['queries']-old['queries'],gain_delta=new['gain']-old['gain'],action_match_delta=int(new['action_match'])-int(old['action_match'])))
    assert len(tr)==28536 and len(pred)==4681;metrics={}
    for arm in ['GATE','STOP_ONLY','MIXED','BUDGET']:
        for seed in range(3):
            name=f'{arm}_seed{seed}';rr=[r for r in tr if r['method']==name];assert len(rr)==2378;metrics[name]=summarizer(rr)
            if arm!='BUDGET':
                for k,v in metrics[name]['macro'].items():assert abs(v-read(S/'search_metrics.json')[name]['macro'][k])<1e-12
    family={arm:{k:float(np.mean([metrics[f'{arm}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{arm}_seed0']['macro']} for arm in ['GATE','STOP_ONLY','MIXED','BUDGET']};paired=[]
    for base in ['GATE','STOP_ONLY','MIXED']:
        for seed in range(3):
            old,new=metrics[f'{base}_seed{seed}'],metrics[f'BUDGET_seed{seed}'];paired.append(dict(base=base,seed=seed,macro_delta={k:new['macro'][k]-old['macro'][k] for k in old['macro']},per_date=[dict(date=x['date'],NET_delta=y['mean_net_utility']-x['mean_net_utility'],query_delta=y['mean_queries']-x['mean_queries']) for x,y in zip(old['per_date'],new['per_date'])]))
    counts=dict(original208_appends=sum(r['base_continue'] for r in pred),appends=sum(r['budget_continue'] for r in pred),retained_profitable=sum(r['budget_continue'] and r['actual_next_NET']>0 for r in pred),removed_profitable=sum(r['base_continue'] and not r['budget_continue'] and r['actual_next_NET']>0 for r in pred),removed_unprofitable=sum(r['base_continue'] and not r['budget_continue'] and r['actual_next_NET']<=0 for r in pred))
    for n,v in [('oof_trajectories.json',tr),('gate_predictions.json',pred),('changed_choices.json',changes),('search_metrics.json',metrics),('family_summary.json',family),('paired_summary.json',paired),('gate_counts.json',counts)]:save(n,v)
    scored();save('review.json',dict(passed=True,all3seeds16dates_retained=True,all_controls_reproduced=True,all_models_directions_unchanged=True,new_physics=0,final8_absent=True));archive('evaluated');print('GRID230_EVALUATED',json.dumps(dict(family=family,counts=counts,changes=len(changes),paired=[r for r in paired if r['base']=='MIXED'])))
if __name__=='__main__':globals()[sys.argv[1]]()
