"""Prospective constant-amount comparator for fixed-event conditional gain heads."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid199_constant_amount';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID199_constant_amount';PARENT=R/'outputs/grid198_conditional_gain';OLD=R/'outputs/grid190_extension_gate';EVENT=R/'outputs/grid193_gate_precondition';DIRECT=R/'outputs/grid196_gain_readout'
sp=importlib.util.spec_from_file_location('grid199_frozen_search_helpers',R/'work/grid198_conditional_gain/run.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M);B=M.B
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    if (O/'models').exists():shutil.copytree(O/'models',d/'models')
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    B.H.sealed()
def prepare():
    assert not O.exists();O.mkdir(parents=True);B.H.sealed();assert read(PARENT/'complete.json')['passed']
    save('design.json',dict(id='GRID199_FIXED_EVENT_CONSTANT_POSITIVE_AMOUNT',authorization='User持续继续;notify supplied: freeze198 andtest probabilitytimesfitpositivemean withoutnewoptimization.',
        question='Does learnedconditionalamount198 add value beyond sameeventprobability times a single fittedpositiveconditionalmean?',
        estimator='Reuse193frozen12eventheads. Perpoolconstant=sum(198positiveconditionalweights*target)/sumweights,exactstatisticalmeanalreadyusedasdiagnostic198. No newoptimizerfits;12training-derivedconstantsrecomputed/serialized,not claimzeroestimation. 84frozeventparameters+1constant=85readoutparameters vs198168.',
        fixed='Same192mixturepopulationpositivecondition/83features/normalization/labels/fee/ranking/NET/global12864/.9. No STOP-onlyorCONT-only alternative mean,threshold/scale/calibrationsearch.',
        action='p(frozenevent|x)*constant_normalized_gain >knownnextfee buysone<=64thenreturnsbest. No .5cutoff,neverheldlabelsforconstant. No automaticensembleorreplacement.',
        publication='All12derivedconstantmodels frozenandpublishedbefore newpolicyheldscoring. Original198weightsandallresults retained.',
        comparisons='NET,ADD1,198learnedconditionalamount primary;190and196rawreferences. Prespecified sameexpectedquery comparisons vs198and190 withcost-onlyindependentthinningperdate/seed;ADD1randomreference. All16dates/all3seeds includingadverse.',
        metrics='Actionmatch,gaincapture,queries,proxyNET,predictivegainMSEMAE,conditionalpositiveMSEvs198diagnosticidentity,zero-gainqueries. Sameexpectedbudget analyticnotwalltime.',
        limits='Post198outcome hypothesis,but newbaselineconfigurationfixedbefore its heldscoring;notblindtest,nopurecapacitymatchedclaim. Localrewardnotfullweekquality. 12constants arefittedstatistics,zero newoptimizationfits/NNforwards/physics/data/final8/paper.'))
    files=[W/'run.py',R/'work/grid198_conditional_gain/run.py',O/'design.json',PARENT/'complete.json',PARENT/'predictive_metrics.json',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid113_relational_learning/train_metadata.json']
    for d in [PARENT,OLD,DIRECT]:files += [d/n for n in ['oof_trajectories.json','search_metrics.json','family_summary.json']]
    files += list((PARENT/'fit_labels').glob('*.npz'))+list((EVENT/'models').glob('*.npz'))+list(OLD.glob('held_f*.npz'))+[R/p for p in read(OLD/'code_freeze.json')];save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID199_PLANNED')
def qualify():
    verify();assert not (O/'qualification.json').exists();(O/'models').mkdir();records=[]
    for f in range(4):
        for s in range(3):
            lab=np.load(PARENT/'fit_labels'/f'f{f}_s{s}.npz');cp=np.load(EVENT/'models'/f'f{f}_s{s}.npz');y,w=lab['target'],lab['sample_weight'];assert (y>0).all();constant=float(w@y/w.sum());manual=float(sum(float(a)*float(b) for a,b in zip(w,y))/sum(float(a) for a in w));assert abs(constant-manual)<1e-12;np.savez_compressed(O/'models'/f'f{f}_s{s}.npz',conditional_constant=np.array([constant]),event_coef=cp['coef'][0],event_intercept=cp['intercept'],mean=cp['mean'],scale=cp['scale']);records.append(dict(fold=f,seed=s,positive_rows=len(y),normalized_constant=constant,active_readout_parameters=85,frozen_event_parameters=84,training_derived_scalars=1))
    save('qualification.json',dict(passed=True,new_optimization_fits=0,training_derived_constants=12,all_fit_only_positive_weights_exactly198=True,all_event_models_unchanged=True,pools=records));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'qualification.json']});verify();archive('qualified');print('GRID199_QUALIFIED')
def qualified():
    verify()
    for p,h in read(O/'qualified_freeze.json').items():assert sha(O/p)==h
def publication():
    qualified();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert head==remote;p=A/'qualified/qualified_freeze.json';assert subprocess.check_output(['git','show','HEAD:'+p.relative_to(REPO).as_posix()],cwd=REPO)==p.read_bytes();save('preheld_publication.json',dict(passed=True,git_sha=head,remote_main=head,all12constants_published_before_held=True));print('GRID199_PUBLISHED')
def evaluate():
    qualified();assert read(O/'preheld_publication.json')['passed'] and not (O/'evaluation_started.json').exists();save('evaluation_started.json',dict(all12constants_published=True));meta,lookup,cases=B.load();task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];oldtr=read(OLD/'oof_trajectories.json');bases={(r['method'],r['sample_id']):r for r in oldtr};pred=[];tr=[]
    for f in range(4):
        for s in range(3):
            cp=np.load(O/'models'/f'f{f}_s{s}.npz');z=np.load(OLD/f'held_f{f}_s{s}.npz');x=(z['features']-cp['mean'])/cp['scale'];logit=x@cp['event_coef']+cp['event_intercept'][0];prob=np.exp(-np.logaddexp(0.,-logit));gain=prob*cp['conditional_constant'][0];charge=z['features'][:,82];byidx={int(i):(bool(g>c),float(p),float(g),float(c)) for i,p,g,c in zip(z['ids'],prob,gain,charge)};cc=[c for c in cases if bases[f'NET_seed{s}',c['sample_id']]['fold']==f]
            for c in cc:
                base=bases[f'NET_seed{s}',c['sample_id']];j=base['baseline_stop'];eligible=j<len(c['states']);d,p,g,q=byidx[lookup[c['sample_id'],j]] if eligible else (False,0.,0.,0.)
                if eligible:
                    assert np.isclose(q,fee*c['queries'][j]/scale,atol=1e-8,rtol=2e-6);pred.append(dict(fold=f,seed=s,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=j,gate_continue=d,event_probability=p,conditional_normalized_gain=float(cp['conditional_constant'][0]),predicted_normalized_gain=g,known_charge=q,predicted_net=g-q,actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j]),actual_next_NET=float((c['gains'][j]-fee*c['queries'][j])/scale),actual_next_change=bool(c['chosen'][j]!=c['chosen'][j+1])))
                for method in ['NET','ADD1','GATE']:
                    add=bool(eligible and (method=='ADD1' or method=='GATE' and d));end=j+int(add);name=f'{method}_seed{s}';v=B.H.trajectory(c,np.arange(len(c['states']))<end,name,fee,scale);v.update(fold=f,baseline_stop=j,baseline_gain=base['gain'],baseline_queries=base['queries'],eligible=eligible,append=add,action_match=v['chosen_pool_id']==c['chosen'][-1]);assert base['queries']<=v['queries']<=int(c['queries'].sum()) and v['queries']-base['queries']<=64;assert base['gain']<=v['gain'] and (not base['action_match'] or v['action_match'])
                    if method in ['NET','ADD1']:
                        for k in ['gain','queries','chosen_pool_id','actions']:assert v[k]==bases[name,c['sample_id']][k]
                    tr.append(v)
    metrics={f'{method}_seed{s}':B.summarize([r for r in tr if r['method']==f'{method}_seed{s}']) for method in ['NET','ADD1','GATE'] for s in range(3)};families={method:{k:float(np.mean([metrics[f'{method}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{method}_seed0']['macro']} for method in ['NET','ADD1','GATE']};counts=dict(eligible=len(pred),append=sum(p['gate_continue'] for p in pred),append_changes=sum(p['gate_continue'] and p['actual_next_change'] for p in pred),append_no_change=sum(p['gate_continue'] and not p['actual_next_change'] for p in pred),append_positive_NET=sum(p['gate_continue'] and p['actual_next_NET']>0 for p in pred));assert len(pred)==4681 and len(tr)==21402
    save('gate_predictions.json',pred);save('oof_trajectories.json',tr);save('search_metrics.json',metrics);save('family_summary.json',families);save('gate_counts.json',counts);qualified();save('review.json',dict(passed=True,all21402trajectory_bounds_and_NET_ADD1_reproduced=True,all4681constantproducts_and_costs_checked=True,all12fit_only_constants_published_before_held=True,new_optimization_fits=0,training_derived_constants=12,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('evaluated');print('GRID199_EVALUATED',json.dumps(dict(families=families,counts=counts),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
