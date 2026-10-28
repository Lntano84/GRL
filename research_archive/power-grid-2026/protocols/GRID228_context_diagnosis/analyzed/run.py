"""Read-only diagnosis of sealed mixed-support decisions; no fitting or policy changes."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import collections, hashlib, importlib.util, json, shutil, sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid228_context_diagnosis'
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID228_context_diagnosis'
S=R/'outputs/grid227_context_newton';P=R/'outputs/grid220_full_prior';T=R/'outputs/grid222_protected_step'
Q=R/'outputs/grid224_context_support_plan/revision02';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def module(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8')
def prepare():
    assert not O.exists() and read(S/'complete.json')['passed'];O.mkdir()
    save('design.json',dict(id='GRID228',scope='Read-only sealed47 changes and entire610 original208CONT cohort. No fitting,calibration,newNNforward,newpolicy,newphysics.',
        questions='Which input blocks change corrections? Are two repeated savings,Nov16 restorations,and Jan5 loss numerically separated from decision boundary? All47 changes retained; labels for description only.',
        numeric='Full unitL2 objective is1-stronglyconvex. Exact-arithmetic parameter error <= Euclidean norm of objective gradient. Protected cancellation iff endpoint<0 AND (b_positive*x_held-b_held*x_positive)dot(theta)<0 for EVERY fitpositive. Apply Cauchy bounds to these linear inequalities. Floating gradient is diagnostic,NOTintervalcertificate. Inconclusive bound does not imply solver caused error. Original ULP-down scalar may retain additional ties;record consistency.',
        inputs='Fixed125 columns:83latent/publicderived block,41rawpublicsummary,onepriorb;intercept separately. Compare ownfold STOP andauxCONT scales,margin contributions and held withinfit ranges;descriptive notOODproof. No held-informed threshold/step/penalty selection.',
        followup='Only after evidence,qualify a prospective residual with public41+b inputs,keeping strong learnedprior. This stage does not fit or evaluate that proposed restriction. Existing208/222/227 fixed.',
        boundaries='All3seeds16exposed dates. Seed/prefix repetitions correlated;proxyNET notfullweekquality orlatency. Final8,physics,NN,paper,FRAME untouched.'))
    paths=[W/'run.py',O/'design.json',META,S/'complete.json',S/'independent_audit.json',S/'changed_choices.json',S/'gate_predictions.json',S/'search_metrics.json',S/'effect_decomposition.json',P/'qualification.json',R/'work/grid219_prior_objective/run.py',R/'work/grid221_step_plan/run.py']
    for root,folder in [(P,'fit_inputs'),(P,'public_held_inputs'),(P,'models'),(T,'models'),(T,'held_predictions'),(S,'models'),(S,'held_predictions'),(Q,'qualified_inputs')]:paths+=list((root/folder).glob('*.npz'))
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('qualified');print('GRID228_QUALIFIED',len(paths))
def robustness(x,b,theta,fitx,fitb,positive,gradnorm,fallback):
    if fallback:return dict(status='constant_KEEP',algebra_cancel=False,error_radius=None)
    aug=np.r_[x,1.];a=np.c_[fitx[positive],np.ones(int(positive.sum()))];bp=fitb[positive]
    coeff=np.vstack([aug,bp[:,None]*aug-b*a]);constant=np.r_[b,np.zeros(len(bp))]
    val=coeff@theta+constant;radius=np.linalg.norm(coeff,axis=1)*gradnorm
    cancel=bool(np.all(val<0));stable_cancel=bool(np.all(val+radius<0));stable_keep=bool(np.any(val-radius>=0))
    assert not(stable_cancel and stable_keep)
    dist=np.divide(np.abs(val),np.linalg.norm(coeff,axis=1),out=np.full(len(val),np.inf),where=np.linalg.norm(coeff,axis=1)>0)
    return dict(status='robust_cancel_diagnostic' if stable_cancel else 'robust_keep_diagnostic' if stable_keep else 'inconclusive_diagnostic',algebra_cancel=cancel,error_radius=gradnorm,min_parameter_distance_to_any_inequality=float(np.min(dist)),endpoint=float(val[0]),endpoint_error_bound=float(radius[0]),endpoint_only_separated=bool(abs(val[0])>radius[0]))
def blocks(x):
    return {name:dict(max_abs=float(np.abs(x[:,sl]).max()),median_row_max_abs=float(np.median(np.max(np.abs(x[:,sl]),axis=1)))) for name,sl in [('latent83',slice(0,83)),('public41',slice(83,124)),('prior1',slice(124,125))]}
def analyze():
    verify();assert not (O/'rows.json').exists();loss=module('grid228_loss',R/'work/grid219_prior_objective/run.py');step=module('grid228_step',R/'work/grid221_step_plan/run.py')
    metadata=read(META);lookup={(v['sample_id'],v['position']):i for i,v in enumerate(metadata)}
    changes=read(S/'changed_choices.json');changeidx={(r['seed'],r['sample_id'],r['position']):r for r in changes}
    preds=read(S/'gate_predictions.json');cohort=[r for r in preds if r['base_continue']];assert len(cohort)==610
    oldpred={(r['arm'],r['seed'],r['sample_id'],r['position']):r for r in read(T/'gate_predictions.json')}
    rows=[];pools=[];limits=[];synthetic=[];rng=np.random.default_rng(228)
    # Fixed synthetic equivalence checks, no optimization or data-driven decision choices.
    for trial in range(100):
        x=rng.normal(size=(12,5));b=rng.uniform(.01,1,12);theta=rng.normal(size=6);pos=np.arange(12)%3==0;end=b+x@theta[:-1]+theta[-1];alpha,_=step.choose_alpha(b,end,pos)
        hx=rng.normal(size=5);hb=float(rng.uniform(.01,1));actual=hb+alpha*(hx@theta[:-1]+theta[-1]);r=robustness(hx,hb,theta,x,b,pos,0.,False);assert r['algebra_cancel']==(actual<0)
        synthetic.append(dict(trial=trial,algebra_matches=True))
    for fold in range(4):
        for seed in range(3):
            pool=f'f{fold}_s{seed}';z=np.load(P/'fit_inputs'/f'{pool}.npz');q=np.load(Q/'qualified_inputs'/f'{pool}.npz');h=np.load(P/'public_held_inputs'/f'{pool}.npz')
            assert np.array_equal(z['X'],q['calibration_X']) and np.array_equal(z['ids'],q['calibration_ids'])
            positive=z['y']>0;fallback=pool=='f0_s1';index={int(i):k for k,i in enumerate(h['ids'])}
            ot=np.load(P/'models'/f'{pool}_ANCHORED.npz')['theta'];oa=float(np.load(T/'models'/f'{pool}_ANCHORED.npz')['alpha'][0]);cp=np.load(S/'models'/f'{pool}.npz');nt=cp['theta'];na=float(cp['alpha'][0])
            models={}
            for name,t,alpha,fit in [('STOP_ONLY',ot,oa,z),('MIXED',nt,na,q)]:
                value,g=loss.objective(t,fit['X'],fit['y'],fit['weight'],fit['b']);gn=float(np.linalg.norm(g));end=z['b']+z['X']@t[:-1]+t[-1];a,bound=step.choose_alpha(z['b'],end,positive,fallback);assert abs(alpha-a)<1e-14
                ratios=np.full(len(end),np.inf);mask=positive&(end<z['b']);ratios[mask]=z['b'][mask]/(z['b'][mask]-end[mask]);lim=np.flatnonzero(np.isclose(ratios,bound,rtol=1e-10,atol=1e-12)) if bound<1 else []
                for k in lim:
                    mi=int(z['ids'][k]);limits.append(dict(pool=pool,arm=name,index=mi,call=metadata[mi]['physical_state_group'],position=metadata[mi]['position'],b=float(z['b'][k]),endpoint=float(end[k]),actual_NET=float(z['actual_NET'][k]),alpha=alpha))
                models[name]=dict(theta=t,alpha=alpha,gradient_norm=gn)
            pi=dict(pool=pool,fallback=fallback,STOP_rows=len(z['X']),CONT_rows=int((~q['domain_STOP']).sum()),STOP_positive=int(positive.sum()),STOP_scales=blocks(z['X']),CONT_scales=blocks(q['X'][~q['domain_STOP']]) if np.any(~q['domain_STOP']) else None,held_scales=blocks(h['X']),grad_norms={k:v['gradient_norm'] for k,v in models.items()});pools.append(pi)
            for r in cohort:
                if r['fold']!=fold or r['seed']!=seed:continue
                mi=lookup[r['sample_id'],r['position']];k=index[mi];x=h['X'][k];b=float(h['b'][k]);assert b>0
                entry=dict(pool=pool,index=mi,seed=seed,sample_id=r['sample_id'],position=r['position'],call=r['physical_state_group'],date=r['date'],actual_NET=r['actual_next_NET'],actual_gain=r['actual_next_gain'],queries=r['actual_next_queries'],b=b,changed=(seed,r['sample_id'],r['position']) in changeidx,arms={})
                for name,targ in models.items():
                    t=targ['theta'];alpha=targ['alpha'];components=dict(latent83=float(x[:83]@t[:83]),public41=float(x[83:124]@t[83:124]),prior_coefficient=float(x[124]*t[124]),intercept=float(t[-1]));delta=sum(components.values());score=b+alpha*delta
                    saved=r['logit'] if name=='MIXED' else oldpred['ANCHORED',seed,r['sample_id'],r['position']]['logit'];assert abs(score-saved)<1e-8,(pool,name,score,saved)
                    robust=robustness(x,b,t,z['X'],z['b'],positive,targ['gradient_norm'],fallback)
                    # Scores are sealed. Algebra is diagnostic, allowing ULP-level protected ties.
                    if robust['algebra_cancel']!=(score<0):assert abs(score)<1e-10
                    entry['arms'][name]=dict(alpha=alpha,endpoint=b+delta,score=score,keep=score>=0,raw_components=components,protected_components={k:alpha*v for k,v in components.items()},numeric=robust)
                entry['public_support']={}
                for name,sl in [('latent83',slice(0,83)),('public41',slice(83,124))]:
                    fit=q['X'][:,sl];v=x[sl];outside=(v<fit.min(axis=0))|(v>fit.max(axis=0));entry['public_support'][name]=dict(max_abs=float(abs(v).max()),outside_mixed_fit_column_ranges=int(outside.sum()),columns=len(v))
                rows.append(entry)
    assert len(rows)==610 and sum(r['changed'] for r in rows)==47
    counts={}
    for arm in ['STOP_ONLY','MIXED']:
        counts[arm]={}
        for group,rr in [('all',rows),('changed',[r for r in rows if r['changed']]),('changed_profitable',[r for r in rows if r['changed'] and r['actual_NET']>0])]:
            counts[arm][group]=dict(collections.Counter(r['arms'][arm]['numeric']['status'] for r in rr))
    focus=[r for r in rows if r['changed'] and (r['call'] in ['2037-01-05_101|call001','2037-11-16_101|call001'] or r['sample_id'] in ['077c228d0254d5b751b98a62','539b1f1919b80a68cb2264cb'])]
    changed=[r for r in rows if r['changed']];shift=[]
    for r in changed:
        old,new=r['arms']['STOP_ONLY'],r['arms']['MIXED'];cs={k:new['protected_components'][k]-old['protected_components'][k] for k in old['protected_components']};assert abs(sum(cs.values())-(new['score']-old['score']))<1e-9
        shift.append(dict(seed=r['seed'],call=r['call'],sample_id=r['sample_id'],position=r['position'],profitable=r['actual_NET']>0,score_delta=new['score']-old['score'],component_deltas=cs,largest_abs_component=max(cs,key=lambda k:abs(cs[k]))))
    summary=dict(cohort=610,changed=47,focus=len(focus),numeric_status=counts,changed_largest_block=dict(collections.Counter(r['largest_abs_component'] for r in shift)),new_fits=0,new_calibrations=0,new_NN_forwards=0,new_policy_evaluations=0,source_models_scores_unchanged=True,diagnostic_not_interval_certificate=True)
    for n,v in [('rows.json',rows),('pool_scales.json',pools),('limiting_fit_rows.json',limits),('focus_cases.json',focus),('component_shifts.json',shift),('synthetic_checks.json',synthetic),('summary.json',summary)]:save(n,v)
    verify();save('analysis_complete.json',dict(passed=True,cohort=610,changes=47,synthetic_algebra_checks=100,no_new_fitting_or_policy=True));archive('analyzed');print('GRID228_ANALYZED',json.dumps(summary))
if __name__=='__main__':globals()[sys.argv[1]]()
