"""Matched direct versus baseline-offset linear cancellation, fixed convex fits."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import collections,datetime,hashlib,json,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid220_full_prior';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID220_anchored_logit';G=R/'outputs/grid216_cancellation_gate';S=R/'outputs/grid214_inner_head_generation';BASE=R/'outputs/grid208_public_amount';META=R/'outputs/grid113_relational_learning/train_metadata.json';ARMS=['DIRECT','ANCHORED']
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args])
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    for root,names in [(G,['trained_freeze.json','scored_freeze.json']),(S,['trained_freeze.json','scored_freeze.json','labelled_freeze.json'])]:
        for name in names:
            for p,h in read(root/name).items():assert sha(root/p)==h,p

def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    for folder in {'qualified':['fit_inputs','public_held_inputs'],'trained':['models'],'scored':['held_predictions']}.get(phase,[]):shutil.copytree(O/folder,d/folder)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def objective(theta,x,y,w,offset):
    z=x@theta[:-1]+theta[-1]+offset;p=np.exp(-np.logaddexp(0,-z));e=w*(p-y)/w.sum();return float(w@(np.logaddexp(0,z)-y*z)/w.sum()+.5*(theta@theta)),np.r_[x.T@e,e.sum()]+theta

def prepare():
    assert not O.exists();assert read(R/'outputs/grid217_prior_diagnosis/independent_audit.json')['passed'];assert read(R/'outputs/grid219_prior_objective/qualification.json')['passed'];O.mkdir();(O/'fit_inputs').mkdir();(O/'public_held_inputs').mkdir()
    save('design.json',dict(id='GRID220_ANCHORED_LOGIT',authorization='User explicitcontinuousautonomyincludingtraining2026-10-28;pretrainingnotice exactly22linear fits and2KEEPconstants delivered.',question='Does fixed208prior-strength offset help cancellation versus matched direct readout? This is only a regularization-center comparison within same linear function class.',data='Exactly215/216 baseCONT-at-originalNETSTOP crosshead rows,3466rows90positive;outer4x3seeds unchanged. f0_s1 botharmsKEEP(no fit). No additionalrows/seeds/labels/fees.',input='Same216public124 features with unchanged208scalers + boundedprior b=(expectedgain-charge)/(expectedgain+charge) unscaled:125inputs126parameters inclintercept. Fitprior from214inner8date heads;heldprior from published20812date model. No labels/IDs/dates/innerfold as input.',arms='DIRECT logit=Xw+c;ANCHORED logit=b+Xw+c. Samezero126init,all126parametersincludinginterceptL2penalized. Both initiallyKEEP on already-positive baseCONT domain;anchoring supplies margin and changes L2center, not uniqueinitialbehavior/newfunctionclass.',objective='WeightedMEAN216costweightedlogisticNLL(keep=originalNET>0)+.5||all126theta||^2 inclintercept. UnitL2fixed,notoptimal. No classbalance/clipping/temperature/newfee. Nativecostweighted output not calibrated payoff probability.',change_vs218='Normalize weighted loss AND penalizeintercept:cross218notonefactor. NewmatchedDIRECTvsANCHOREDisolatespriorcenter;weights/inputs/solver identical. Wholeweightscaleor duplicatedsampleinvariance per219.',solver=dict(method='L-BFGS-B',maxiter=2000,gtol=1e-6,ftol=1e-12,maxls=50,init='allzero',serial_fits=22,constant_KEEP=2,total_cap_s=600,success='all22solver_success and finalmaxabsgrad<1e-3 before scoring. Preserve every endpoint;no retry/extraiterations/extension.'),readout='retain original208CONT iff linearlogit>=0;tiesKEEP,no rescue. FrozenNETprefix and atmost originalonebatch;unchanged128/64/stagnation2/.9/order/station/aliases.',barriers='Configs and fitinputs published beforetraining. All22models+2constants published before anynewheldlogits. All24predfiles published before outcomejoin. Heldpublicinputs exported fromalreadyseen208results;exportparseslabel-containing source but emits publiccolumns only. Do not claim unseenheldlabels orblindtest.',evaluation='PrimaryANCHOREDvsDIRECT(all16exposed dates/all3seeds);secondary208 forquality/querytradeoff.216forest is historicalcontext,notcapacity-matched ablation. Reportallcost/gain/queries/actionmatch/perdate/seed/reversecases. Noallwinor5percentnewgate/no thresholdsearch.',limits='BaseNETandSTOPselector fittedouter12inclinner4:head-onlycrossgeneration. 8date to12date entrance andprior shift remain. NoNNfits/newphysics/final8/paper/newFRAME/scale. Initialequality not learnedquality guarantee.'))
    meta=read(META);lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};priors={}
    for b in read(S/'bundles.json'):
        z=np.load(S/'labelled_samples'/f"{b['bundle']}.npz");gain=z['event_p']*z['mu4'];margin=(gain-z['public_charge'])/(gain+z['public_charge'])
        for i,v in zip(z['meta_ids'],margin):priors[b['outer_fold'],b['seed'],int(i)]=float(v)
    # Explicit public export from an already exposed source. No private fields are emitted.
    public_by_pool=collections.defaultdict(dict)
    for p in read(BASE/'gate_predictions.json'):
        pool=f"f{p['fold']}_s{p['seed']}";i=lookup[p['sample_id'],p['position']];g,q=p['predicted_normalized_gain'],p['known_charge'];assert q>0 and g>=0;b=(g-q)/(g+q);assert p['gate_continue']==(g>q);public_by_pool[pool][i]=(b,bool(p['gate_continue']))
    records=[];gradmax=0.
    for f in range(4):
        for s in range(3):
            pool=f'f{f}_s{s}';z=np.load(G/'fit_inputs'/f'{pool}.npz');b=np.array([priors[f,s,int(i)] for i in z['ids']]);assert (b>0).all() and (b<=1).all();x=np.c_[z['X'],b];assert x.shape==(len(b),125) and np.isfinite(x).all();fallback=bool(z['fallback_keep'][0]);assert fallback==(pool=='f0_s1');np.savez_compressed(O/'fit_inputs'/f'{pool}.npz',X=x,b=b,y=z['y'],weight=z['weight'],ids=z['ids'],actual_NET=z['actual_NET'],structural_mass=z['structural_mass'],fallback_keep=z['fallback_keep'])
            h=np.load(R/'outputs/grid190_extension_gate'/f'held_{pool}.npz');cp=np.load(BASE/'models'/f'{pool}.npz');raw41=np.load(S/'public_raw41.npz')['values'];hx=np.c_[(h['features']-cp['mean'])/cp['scale'],(raw41[h['ids'].astype(int)]-cp['raw_mean'])/cp['raw_scale']];hb=np.array([public_by_pool[pool][int(i)][0] for i in h['ids']]);base=np.array([public_by_pool[pool][int(i)][1] for i in h['ids']]);np.savez_compressed(O/'public_held_inputs'/f'{pool}.npz',X=np.c_[hx,hb],b=hb,ids=h['ids'],base_CONT=base)
            fd={meta[int(i)]['date'] for i in z['ids']};hd={meta[int(i)]['date'] for i in h['ids']};assert not fd&hd
            for arm in ARMS:
                off=b if arm=='ANCHORED' else np.zeros(len(b));theta=np.linspace(-.01,.01,126);loss,grad=objective(theta,x,z['y'],z['weight'],off)
                for k in np.unique(np.r_[np.linspace(0,125,12,dtype=int),124,125]):
                    e=np.zeros(126);e[k]=1e-6;estimate=(objective(theta+e,x,z['y'],z['weight'],off)[0]-objective(theta-e,x,z['y'],z['weight'],off)[0])/2e-6;err=abs(estimate-grad[k])/max(1,abs(grad[k]));gradmax=max(gradmax,err);assert err<1e-5,(pool,arm,k,err)
                assert np.array_equal(off+np.zeros(len(b))>=0,np.ones(len(b),bool))
            # Same function class; anchor is a one-unit shift of the last coefficient.
            shifted=theta.copy();shifted[124]+=1;assert np.allclose(x@theta[:-1]+theta[-1]+b,x@shifted[:-1]+shifted[-1],atol=1e-12,rtol=1e-12)
            records.append(dict(pool=pool,rows=len(b),positive=int(z['y'].sum()),fallback_keep=fallback,held_feature_rows=len(hb),fit_dates=sorted(fd),held_dates=sorted(hd),parameters=126))
    save('qualification.json',dict(passed=True,pools=records,finite_difference_max_relative_error=gradmax,all24_zero_corrections_retain_baseCONT=True,same_linear_functionclass_reparameterization_proved=True,public_export_parsed_existing_private_source=True,only_public_held_columns_emitted=True,training_calls=0));paths=[W/'run.py',O/'design.json',O/'qualification.json',G/'fit_qualification.json',BASE/'gate_predictions.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json',BASE/'family_summary.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',META]+list((O/'fit_inputs').glob('*.npz'))+list((O/'public_held_inputs').glob('*.npz'))+[R/'outputs/grid217_prior_diagnosis'/n for n in ['complete.json','independent_audit.json','zero_correction_proof.json']];paths += [R/'outputs/grid219_prior_objective'/n for n in ['qualification.json','design.json','source_freeze.json']];save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('qualified');print('GRID220_QUALIFIED',gradmax)

def train():
    verify();assert not (O/'training_started.json').exists();from scipy.optimize import minimize
    save('training_started.json',dict(fits=22,constants=2,cap_s=600,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));(O/'models').mkdir();start=time.perf_counter();receipts=[]
    for f in range(4):
        for s in range(3):
            pool=f'f{f}_s{s}';z=np.load(O/'fit_inputs'/f'{pool}.npz');x,y,w,b=z['X'],z['y'],z['weight'],z['b'];fallback=bool(z['fallback_keep'][0])
            for arm in ARMS:
                assert time.perf_counter()-start<600,'Original600s cap reached; no automatic retry/extension.';t=time.perf_counter();theta=np.zeros(126);off=b if arm=='ANCHORED' else np.zeros(len(b));initial=objective(theta,x,y,w,off)[0]
                if fallback:success=True;nit=0;message='KEEP constant: no positive support';final,grad=initial,np.zeros(126)
                else:
                    result=minimize(objective,theta,args=(x,y,w,off),method='L-BFGS-B',jac=True,options=dict(maxiter=2000,gtol=1e-6,ftol=1e-12,maxls=50));theta=result.x.copy();success=bool(result.success);nit=int(result.nit);message=str(result.message);final,grad=objective(theta,x,y,w,off);assert abs(final-result.fun)<1e-8;assert final<=initial+1e-8
                np.savez_compressed(O/'models'/f'{pool}_{arm}.npz',theta=theta,fallback_KEEP=np.array([fallback]),anchored=np.array([arm=='ANCHORED']));receipts.append(dict(pool=pool,arm=arm,fitted=not fallback,parameters=126,rows=len(y),positive_rows=int(y.sum()),initial_loss=initial,final_loss=final,gradient_inf=float(abs(grad).max()),solver_success=success,iterations=nit,message=message,fit_wall_s=time.perf_counter()-t));save('fit_receipts.json',receipts)
    assert len(receipts)==24 and sum(r['fitted'] for r in receipts)==22;save('fits.json',dict(passed=True,new_linear_fits=22,constant_KEEP=2,new_NN_fits=0,total_fit_and_check_wall_s=time.perf_counter()-start,all_solvers_success=all(r['solver_success'] and r['gradient_inf']<1e-3 for r in receipts),records=receipts));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fit_receipts.json',O/'fits.json']});verify();archive('trained');print('GRID220_TRAINED',read(O/'fits.json')['all_solvers_success'])

def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p

def publication():
    trained();assert read(O/'fits.json')['all_solvers_success'];head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'trained/trained_freeze.json').relative_to(REPO).as_posix())==(O/'trained_freeze.json').read_bytes();save('pre_score_publication.json',dict(passed=True,git_sha=head,all22weights_and2constants_published=True));print('GRID220_PUBLISHED')

def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(allweights_published=True));(O/'held_predictions').mkdir();receipts=[]
    for f in range(4):
        for s in range(3):
            pool=f'f{f}_s{s}';h=np.load(O/'public_held_inputs'/f'{pool}.npz')
            for arm in ARMS:
                cp=np.load(O/'models'/f'{pool}_{arm}.npz');theta=cp['theta'];fallback=bool(cp['fallback_KEEP'][0]);logit=np.ones(len(h['ids'])) if fallback else h['X']@theta[:-1]+theta[-1]+(h['b'] if arm=='ANCHORED' else 0);prob=np.exp(-np.logaddexp(0.,-logit));keep=logit>=0;assert np.isfinite(logit).all();np.savez_compressed(O/'held_predictions'/f'{pool}_{arm}.npz',ids=h['ids'],logit=logit,weighted_KEEP_score=prob,keep=keep,base_CONT=h['base_CONT']);receipts.append(dict(pool=pool,arm=arm,rows=len(logit),new_continue=int((h['base_CONT']&keep).sum()),private_outcomes_parsed_in_score=False))
    assert len(receipts)==24 and sum(r['rows'] for r in receipts)==9362;save('score_receipts.json',receipts);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'held_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID220_SCORED_ALL24')

def scored():
    trained()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p

def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes();save('pre_join_publication.json',dict(passed=True,git_sha=head,all24predictionfiles_published=True));print('GRID220_SCORES_PUBLISHED')
if __name__=='__main__':globals()[sys.argv[1]]()
