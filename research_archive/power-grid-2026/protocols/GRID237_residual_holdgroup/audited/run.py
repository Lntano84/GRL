"""Fixed residual-only eight-date fits, four-date raw scores, then error labels."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid237_residual_holdgroup';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID237_residual_holdgroup';Q=R/'outputs/grid236_shared_prior_inputs';PLAN=R/'outputs/grid235_calibration_ancestry';META=R/'outputs/grid113_relational_learning/train_metadata.json'
sp=importlib.util.spec_from_file_location('grid237_frozen219',R/'work/grid219_prior_objective/run.py');LOSS=importlib.util.module_from_spec(sp);sp.loader.exec_module(LOSS);objective=LOSS.objective
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    for name in ['generated_freeze.json','qualified_fit_freeze.json']:
        for p,h in read(Q/name).items():assert sha(Q/p)==h,p
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    for folder in {'trained':['models'],'scored':['cal_predictions']}.get(phase,[]):shutil.copytree(O/folder,d/folder)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def prepare():
    assert not O.exists() and read(Q/'complete.json')['passed'];O.mkdir()
    save('design.json',dict(id='GRID237',scope='Fixedconditionalresidualdiagnostic:11public42linear43parammodels+1predeclaredKEEP,0scalars. One600s totalmodelwindow,noextension/refit. Scoreinternal4datesonly,notoldouter4.',
        objective='236newsharedprior fit8labels ONLY,meanweightedlogisticNLL+0.5L2all43inclintercept,offsetb,zero43init. OriginalL-BFGS-Bmaxiter2000/gtol1e-6/ftol1e-12/maxls50. Allsuccessandrawgradinf<.001 before4datescores.',
        scores='Rawuncalibratedb+Xtheta,threshold0tiesKEEP;predeclaredf0_s1constantKEEP.Noalpha/querybudget/selectorfit,newpolicyorcompetitioncomparison.',
        barriers='Sourceinputsfirstpublished;all12weights beforecal4scores;all12cal4scorespublishedbeforeANYinner0cal4targetarraysparsed.Trainworkermayreadonlyfit_inputs labelarrays.',
        ancestry='Residualdirectfittingexcludescal4,butbaseNET/STOPpopulation remainsouter12. Conditionalfixedupstreamdiagnosis,notfullynestedindependentOOF/blindtest. Newprior/scalers/selectionandfit8differfrom233;notpurecalibrationablation.',
        bounds='Totalnewdiagnosticfits11linear,0scalar/forest/NN/physics.All3seeds/allfixedinner0groups/counterexamples. No final8,newouter4scoring,paper,FRAME,parameterorcall-specificselection.'))
    sources=read(Q/'future_cal_target_sources.json');paths=[W/'run.py',O/'design.json',R/'work/grid219_prior_objective/run.py',Q/'complete.json',Q/'review.json',Q/'generated_freeze.json',Q/'qualified_fit_freeze.json',Q/'future_cal_target_sources.json',Q/'fit_qualification.json',PLAN/'fixed_probe_pools.json',PLAN/'qualification.json',META]+list((Q/'fit_inputs').glob('*.npz'))+list((Q/'cal_public').glob('*.npz'))+[R/r['cal4_target_source'] for r in sources]
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('access_scope.json',dict(fit_labels='236fit_inputs only',cal4_target_hashes_only_before_prediction_publication=True,new_predictions_or_fits=0));verify();archive('qualified');print('GRID237_QUALIFIED')
def train():
    verify();assert not (O/'training_started.json').exists();from scipy.optimize import minimize
    save('training_started.json',dict(new_linear_fits=11,constantKEEP=1,scalar_fits=0,cap_s=600));start=time.perf_counter();(O/'models').mkdir();records=[];access=[]
    for file in sorted((Q/'fit_inputs').glob('*.npz')):
        assert time.perf_counter()-start<600;z=np.load(file);fallback=bool(z['fallback_KEEP'][0]);theta=np.zeros(43);initial=objective(theta,z['X'],z['y'],z['weight'],z['b'])[0];t=time.perf_counter();access.append(dict(path=file.relative_to(R).as_posix(),fields=['X','b','y','weight','fallback_KEEP','domain_STOP']))
        if fallback:success=True;nit=0;message='predeclaredconstantKEEP';value=initial;g=np.zeros(43)
        else:
            res=minimize(objective,theta,args=(z['X'],z['y'],z['weight'],z['b']),method='L-BFGS-B',jac=True,options=dict(maxiter=2000,gtol=1e-6,ftol=1e-12,maxls=50));theta=res.x.copy();success=bool(res.success);nit=int(res.nit);message=str(res.message);value,g=objective(theta,z['X'],z['y'],z['weight'],z['b']);assert value<=initial+1e-8 and abs(value-res.fun)<1e-8
        np.savez_compressed(O/'models'/file.name,theta=theta,fallback_KEEP=np.array([fallback]));records.append(dict(pool=file.stem,fitted=not fallback,parameters=43,rows=len(z['y']),positive=int(z['y'].sum()),initial_loss=initial,final_loss=value,gradient_inf=float(abs(g).max()),solver_success=success,iterations=nit,message=message,fit_wall_s=time.perf_counter()-t));save('fit_receipts.json',records)
    elapsed=time.perf_counter()-start;assert elapsed<600 and len(records)==12 and sum(r['fitted'] for r in records)==11;success=all(r['solver_success'] and r['gradient_inf']<1e-3 for r in records);save('train_access.json',access);save('fits.json',dict(passed=success,new_linear_fits=11,constantKEEP=1,new_scalar_fits=0,new_NNfits=0,wall_s=elapsed,all_models_numerically_qualified=success));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fit_receipts.json',O/'fits.json',O/'train_access.json']});verify();archive('trained');print('GRID237_TRAINED',json.dumps(read(O/'fits.json')))
def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
    assert read(O/'fits.json')['all_models_numerically_qualified']
def publication():
    trained();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'trained/trained_freeze.json').relative_to(REPO).as_posix())==(O/'trained_freeze.json').read_bytes();save('pre_score_publication.json',dict(passed=True,git_sha=head,all12weights_published=True));print('GRID237_MODELS_PUBLISHED')
def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all12weights_published=True,cal4_targets_forbidden=True));(O/'cal_predictions').mkdir();records=[]
    for file in sorted((Q/'cal_public').glob('*.npz')):
        z=np.load(file);p=np.load(O/'models'/file.name);t=p['theta'];fallback=bool(p['fallback_KEEP'][0]);logit=np.ones(len(z['ids'])) if fallback else z['b']+np.sum(z['X']*t[:-1],axis=1)+t[-1];assert np.isfinite(logit).all();np.savez_compressed(O/'cal_predictions'/file.name,ids=z['ids'],logit=logit,keep=logit>=0,original_STOP=z['original_STOP']);records.append(dict(pool=file.stem,rows=len(logit),target_fields_read=[]))
    assert len(records)==12 and sum(r['rows'] for r in records)==3176;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'cal_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID237_SCORED_ALL12')
def scored():
    trained()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p
def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes();save('pre_join_publication.json',dict(passed=True,git_sha=head,all12scores_published=True));print('GRID237_SCORES_PUBLISHED')
def error_counts(ids,keep,net,mask,meta):
    pos=net>0;miss=mask&pos&~keep;cancel=mask&~pos&~keep
    return dict(rows=int(mask.sum()),positive=int((mask&pos).sum()),positive_kept=int((mask&pos&keep).sum()),positive_cancelled=int(miss.sum()),nonpositive_cancelled=int(cancel.sum()),nonpositive_kept=int((mask&~pos&keep).sum()),missed_calls=sorted({meta[int(ids[k])]['physical_state_group'] for k in np.flatnonzero(miss)}))
def join():
    scored();assert read(O/'pre_join_publication.json')['passed'] and not (O/'target_join_started.json').exists();save('target_join_started.json',dict(all12cal4scores_published=True));meta=read(META);pools={r['pool']:r for r in read(PLAN/'fixed_probe_pools.json')};sources=read(Q/'future_cal_target_sources.json');rows=[];counts=[];access=[]
    for r in sources:
        pool=r['pool'];role=pools[pool];public=np.load(Q/'cal_public'/f'{pool}.npz');pred=np.load(O/'cal_predictions'/f'{pool}.npz');target=np.load(R/r['cal4_target_source']);ids=target['meta_ids'].astype(int);assert set(meta[int(i)]['date'] for i in ids)<=set(role['calibration_dates']) and not set(meta[int(i)]['date'] for i in ids)&set(role['fit_dates']);net_all=target['actual_NET'];index={int(i):k for k,i in enumerate(ids)};net=np.array([net_all[index[int(i)]] for i in pred['ids']]);assert np.array_equal(pred['ids'],public['ids']);mask=pred['original_STOP'];keep=pred['keep'];counts.append(dict(pool=pool,STOP=error_counts(pred['ids'],keep,net,mask,meta),CONT=error_counts(pred['ids'],keep,net,~mask,meta)));access.append(dict(path=r['cal4_target_source'],fields=['meta_ids','actual_NET']))
        for i,s,k,n,stop in zip(pred['ids'],pred['logit'],keep,net,mask):rows.append(dict(pool=pool,index=int(i),sample_id=meta[int(i)]['sample_id'],position=meta[int(i)]['position'],date=meta[int(i)]['date'],call=meta[int(i)]['physical_state_group'],original_STOP=bool(stop),logit=float(s),keep=bool(k),actual_NET=float(n),positive=bool(n>0)))
    assert len(rows)==3176;summary={}
    for domain in ['STOP','CONT']:
        summary[domain]={k:sum(r[domain][k] for r in counts) for k in ['rows','positive','positive_kept','positive_cancelled','nonpositive_cancelled','nonpositive_kept']};summary[domain]['unique_missed_calls']=len({call for r in counts for call in r[domain]['missed_calls']})
    save('cal4_target_access.json',access);save('cal4_error_rows.json',rows);save('cal4_error_counts.json',counts);save('summary.json',summary);scored();save('review.json',dict(passed=True,all3176publishedscores_joined=True,fit8_and_cal4_direct_residual_dates_disjoint=True,all12models_scores_publication_barriers_checked=True,cal4_labels_first_parsed_after_scores_published=True,conditional_frozen_upstream_scope_only=True,new_scalar_fits=0,no_new_policy_evaluation=True));archive('evaluated');print('GRID237_ERRORS_COLLECTED',json.dumps(summary))
if __name__=='__main__':globals()[sys.argv[1]]()
