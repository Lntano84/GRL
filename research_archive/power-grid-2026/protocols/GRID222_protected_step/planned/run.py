"""Training-positive empirical step calibration of frozen220 linear endpoints."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid222_protected_step';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID222_protected_step';PARENT=R/'outputs/grid220_full_prior';BASE=R/'outputs/grid208_public_amount';META=R/'outputs/grid113_relational_learning/train_metadata.json';ARMS=['DIRECT','ANCHORED']
sp=importlib.util.spec_from_file_location('frozen221_rule',R/'work/grid221_step_plan/run.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args])
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    for name in ['trained_freeze.json','scored_freeze.json']:
        for p,h in read(PARENT/name).items():assert sha(PARENT/p)==h,p
    for p,h in read(R/'outputs/grid221_step_plan/source_freeze.json').items():assert sha(R/p)==h,p

def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    for folder in {'calibrated':['models'],'scored':['held_predictions']}.get(phase,[]):shutil.copytree(O/folder,dest/folder)
    (dest/'archive_manifest.json').write_text(json.dumps({p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')

def prepare():
    assert not O.exists() and read(PARENT/'independent_audit.json')['passed'] and read(R/'outputs/grid221_step_plan/synthetic_qualification.json')['passed'];O.mkdir();save('design.json',dict(id='GRID222_PROTECTED_STEP',authorization='User autonomouscontinueandtraininguntilstop2026-10-28;precalibrationnoticefitpositiveconstraintdelivered.',source='Bothfrozen220arms/11ownpools+2fallbackKEEP,allparametersunchanged. No new main model fit,teacher orNNforward.',rule='Fromoriginalpublicb>0 tosourceendpoints,score=b+a*(s-b). Fit-onlyNET>0 constraints choose maximal a in[0,1] withall fittedpositive scores>=0. a=min(1,min b/(b-s) overpositive s<b),onlyULProunddown for feasibility. f0s1KEEPa0.',change='Explicitempiricalpositive-protectionconstraintadded toone-dimensionalcalibration. Not equivalent unconstrainedlogisticobjective,notcostlabelorqueryfeechange. Allpositives usedregardlessgain;noheldlabelconstraint/temperaturegrid/penaltygrid.',fits='Exactly22 closedform scalar calibrations,2KEEPconstants,original90fitpositive perarm. No optimizer/no new main fits. All24alphafilespublishedbeforescore,all24predictionfilespublishedbeforeoutcomejoin.',evaluation='Ownsource220a1 andoriginal208a0 both fixedreferences;threepreservedseeds/16exposed dates. SameNETprefixandonlycanceloriginal208CONT,returnbestobserved. Entirehistoricaldataexposed,developmentnotblindtest. Empiricalfitconstraint doesnotguaranteeheldorfullweekquality.',scope='No newfeatures/dates/fees/NN/physics/final8/paper/FRAME. No extrapolation fromcorrelatedprefixcount,alladverseresults retained.'))
    paths=[W/'run.py',O/'design.json',R/'work/grid221_step_plan/run.py',R/'outputs/grid221_step_plan/design.json',R/'outputs/grid221_step_plan/source_freeze.json',R/'outputs/grid221_step_plan/synthetic_qualification.json',PARENT/'complete.json',PARENT/'independent_audit.json',PARENT/'trained_freeze.json',PARENT/'scored_freeze.json',PARENT/'pre_score_publication.json',PARENT/'pre_join_publication.json',PARENT/'family_summary.json',PARENT/'search_metrics.json',PARENT/'gate_predictions.json',META,R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',BASE/'gate_predictions.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json']+list((PARENT/'fit_inputs').glob('*.npz'))+list((PARENT/'public_held_inputs').glob('*.npz'))+list((PARENT/'models').glob('*.npz'));save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('planned');print('GRID222_PLANNED')

def calibrate():
    verify();assert not (O/'calibration_started.json').exists();save('calibration_started.json',dict(closed_form_fits=22,constants=2,held_files_parsed=False));(O/'models').mkdir();records=[];access=[]
    for f in range(4):
        for seed in range(3):
            pool=f'f{f}_s{seed}';file=PARENT/'fit_inputs'/f'{pool}.npz';z=np.load(file);access.append(dict(path=file.relative_to(R).as_posix(),kind='own_fit_inputs'))
            for arm in ARMS:
                cpfile=PARENT/'models'/f'{pool}_{arm}.npz';cp=np.load(cpfile);access.append(dict(path=cpfile.relative_to(R).as_posix(),kind='frozen_parent_model'));constant=bool(cp['fallback_KEEP'][0]);theta=cp['theta'];b=z['b'];end=z['X']@theta[:-1]+theta[-1]+(b if arm=='ANCHORED' else 0);y=z['y']>0;a,bound=P.choose_alpha(b,end,y,fallback=constant);score=b+a*(end-b);assert np.all(score[y]>=0);assert 0<=a<=1
                ratios=b[y&(end<b)]/(b[y&(end<b)]-end[y&(end<b)]);tight=[]
                if len(ratios):
                    minbound=float(ratios.min());sel=y&(end<b);ix=np.flatnonzero(sel);tight=[int(z['ids'][ix[k]]) for k in np.flatnonzero(np.isclose(ratios,minbound,atol=1e-12,rtol=1e-12))]
                np.savez_compressed(O/'models'/f'{pool}_{arm}.npz',alpha=np.array([a]),fallback_KEEP=np.array([constant]),source_theta_sha256=np.array([sha(cpfile)]));records.append(dict(pool=pool,arm=arm,calibrated=not constant,alpha=a,analytical_bound=bound,fit_rows=len(y),fit_positive_rows=int(y.sum()),fit_positive_kept=int((score[y]>=0).sum()),limiting_fit_ids=tight,parent_model_sha256=sha(cpfile)));save('calibration_receipts.json',records)
    assert len(records)==24 and sum(r['calibrated'] for r in records)==22 and sum(r['fit_positive_rows'] for r in records)==180;save('fit_access.json',access);save('fits.json',dict(passed=True,closed_form_scalar_fits=22,constant_KEEP=2,new_main_model_fits=0,new_NN_fits=0,all_fit_positive_constraints_satisfied=True));save('calibrated_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'calibration_receipts.json',O/'fit_access.json',O/'fits.json']});verify();archive('calibrated');print('GRID222_CALIBRATED_ALL24')

def calibrated():
    verify()
    for p,h in read(O/'calibrated_freeze.json').items():assert sha(O/p)==h,p

def publication():
    calibrated();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'calibrated/calibrated_freeze.json').relative_to(REPO).as_posix())==(O/'calibrated_freeze.json').read_bytes();save('pre_score_publication.json',dict(passed=True,git_sha=head,all24scalarfiles_published=True));print('GRID222_CALIBRATION_PUBLISHED')

def score():
    calibrated();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all24alphas_published=True));(O/'held_predictions').mkdir();records=[]
    for f in range(4):
        for seed in range(3):
            pool=f'f{f}_s{seed}';h=np.load(PARENT/'public_held_inputs'/f'{pool}.npz')
            for arm in ARMS:
                model=np.load(O/'models'/f'{pool}_{arm}.npz');a=float(model['alpha'][0]);old=np.load(PARENT/'held_predictions'/f'{pool}_{arm}.npz');assert np.array_equal(old['ids'],h['ids']);logit=h['b']+a*(old['logit']-h['b']);keep=logit>=0;np.savez_compressed(O/'held_predictions'/f'{pool}_{arm}.npz',ids=h['ids'],logit=logit,weighted_KEEP_score=np.exp(-np.logaddexp(0,-logit)),keep=keep,base_CONT=h['base_CONT']);records.append(dict(pool=pool,arm=arm,alpha=a,rows=len(logit),new_continue=int((h['base_CONT']&keep).sum()),outcomes_parsed=False))
    assert len(records)==24 and sum(r['rows'] for r in records)==9362;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'held_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID222_SCORED_ALL24')

def scored():
    calibrated()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p

def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes();save('pre_join_publication.json',dict(passed=True,git_sha=head,all24predictionfiles_published=True));print('GRID222_SCORES_PUBLISHED')
if __name__=='__main__':globals()[sys.argv[1]]()
