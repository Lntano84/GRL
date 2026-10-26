"""Twelve conditional-positive gain heads with fixed GRID193 event probabilities."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import datetime,hashlib,importlib.util,json,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid198_conditional_gain';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID198_conditional_gain';FIT=R/'outputs/grid192_gate_support';EVENT=R/'outputs/grid193_gate_precondition';DIRECT=R/'outputs/grid196_gain_readout';OLD=R/'outputs/grid190_extension_gate'
sp=importlib.util.spec_from_file_location('grid198_fixed_algebra',R/'work/grid196_gain_readout/run.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M);B=M.B
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    for name in ['fit_labels','models']:
        if (O/name).exists():shutil.copytree(O/name,d/name)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    B.H.sealed()
def score(x,cp):
    z=(x-cp['mean'])/cp['scale'];logit=z@cp['event_coef']+cp['event_intercept'][0];p=np.exp(-np.logaddexp(0.,-logit));raw=z@cp['coef']+cp['intercept'][0];mu=np.maximum(raw,0);gain=p*mu
    return p,mu,raw,gain,logit
def prepare():
    assert not O.exists();O.mkdir(parents=True);B.H.sealed();assert read(R/'outputs/grid197_gain_opportunity_audit/complete.json')['passed']
    save('design.json',dict(id='GRID198_FIXED_EVENT_CONDITIONAL_GAIN',authorization='User继续 following197 sparsegainfinding;pretrainingnotice fixed12conditionalheads,existingevent/NETfrozen,allseedsretained.',
        hypothesis='Large reward opportunities retained by196 but zero-gainqueries remainhigh evenfitSTOP. Separate event incidence from strictlypositive rewardamount insteadof spreading a single linearmean overzeros.',
        population='Exactly GRID192 halfSTOP/halfactualpreSTOPCONT originalfitarrays andweights,83features/originalSTOPstandardizer/gain_scale. Eventmodel=all12existingGRID193 logits trained on thatsamepopulation. Conditionalhead uses only observedgain>0 fitrows,not heldoutcomes.',
        labels='Conditionaltarget=onebatchgain/gain_scale onlypositive. Conditionalweights=originalsampleweight restrictedpositive,then renormalizedtooriginalNSTOPtotal. Thisis originalweightedpopulation conditionalonpositiveevent,not calluniformconditionalreweighting. Zeros not assignedfictitious magnitude targets.',
        objective='Samealpha1unpenalizedintercept weightedridge onconditionalrows,closedformSVD fromzero. 84newparameters;frozenexistingeventhead84 =>168readoutparameters active,not samecapacity as196. Originalneuralnetwork neverchanges. Conditionaldistribution differs;not claimed statisticalregularizationequivalent.',
        inference='p=sigmoid(frozen193logit);mu=max(0,newconditionalridge);expectedgain=p*mu;buyone<=64batch iffexpectedgain>knownfee*nextqueries/gain_scale. No event .5cutoff,noprobabilitycalibration,no multiplyingunconditional196prediction again.',
        fits='Exactly12fixedconditionalheads,fouroldfoldsxthreeNETseeds,alpha1,600s,no search/retry/additionalfit. Existing19311originalgradientcertified/1limitedendpointallretained,nobestseedselection. Publishall12pairedweightsbeforeheldscoring.',
        comparisons='NET,ADD1 plus saved190STOPeventgate,193samepopulationeventgate,196directgain. All16exposedteacherdateOOF/all3seeds. Predefinedsameexpectedquerycomparison separatelyvs190and196 usingcostonlyindependentthinningperdate/seed;randomADD1reference. No onlinehardbudget/timeclaim.',
        metrics='Actionmatch/gaincapture/query/proxyNET;weightedMSE/MAE ofexpectedgain,conditionalMAE/MSE onlypositives,eventBrier/BCE allrows. Falsezeroqueries andallcounterexamples retained. No allwinor5percentgate.',
        qualification='Positiveeventequalsrankchange,weightedmean identity Eg=Ppositive*E[g|positive] onactualfitlabels. Conditionalweights/rowIDs/dateisolation/eventnormalizerschecked. Syntheticclosedform/rankdeficiencyandp0p1product checks beforeactualfit.',
        limitations='Event probability is estimated onmixture andnot calibrated deploymentSTOP probability. Algebraicidentitydoesnotguarantee learnedaccuracy/quality. Localrewardonly,not fullweek/crossgrid/time/test.',
        prohibited=['eventorNETretraining','changingdata/fees/ranking/stations/global12864/.9','alphathresholdcalibrationsearch/extra seeds','newphysics/data/final8/paper','posthocensembleorautomaticrollouts']))
    files=[W/'run.py',R/'work/grid196_gain_readout/run.py',O/'design.json',R/'outputs/grid197_gain_opportunity_audit/complete.json',R/'outputs/grid197_gain_opportunity_audit/summary.json',DIRECT/'fits.json',DIRECT/'complete.json',EVENT/'fits.json',EVENT/'complete.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid113_relational_learning/train_metadata.json']
    for d in [OLD,EVENT,DIRECT]:files += [d/n for n in ['oof_trajectories.json','search_metrics.json','family_summary.json']]
    files += list((FIT/'fit_features').glob('*.npz'))+list((DIRECT/'fit_labels').glob('*.npz'))+list((EVENT/'models').glob('*.npz'))+list(OLD.glob('held_f*.npz'))+[R/p for p in read(OLD/'code_freeze.json')]
    save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID198_PLANNED')
def qualify():
    verify();assert not (O/'preflight.json').exists();rng=np.random.default_rng(198);x=rng.normal(size=(35,9));x[:,8]=x[:,1];y=np.abs(rng.normal(size=35));w=rng.uniform(.2,2,35);beta,bias,_=M.solve(x,y,w);aug=np.vstack([np.sqrt(w)[:,None]*np.c_[x,np.ones(35)],np.c_[np.eye(9),np.zeros(9)]]);target=np.r_[np.sqrt(w)*y,np.zeros(9)];ref=np.linalg.lstsq(aug,target,rcond=None)[0];assert np.allclose(np.r_[beta,bias],ref,atol=1e-12,rtol=1e-12);assert np.array_equal(np.array([0.,1.])*np.array([2.,2.]),[0.,2.])
    meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');oldtr=read(OLD/'oof_trajectories.json');(O/'fit_labels').mkdir();records=[]
    for f in range(4):
        for s in range(3):
            z=np.load(FIT/'fit_features'/f'f{f}_s{s}.npz');lab=np.load(DIRECT/'fit_labels'/f'f{f}_s{s}.npz');cp=np.load(EVENT/'models'/f'f{f}_s{s}.npz');gain=lab['target'];pos=gain>0;assert np.array_equal(pos,z['y']>0);assert pos.any() and (~pos).any();assert np.array_equal(cp['mean'],z['mean']) and np.array_equal(cp['scale'],z['scale']);mass=z['sample_weight'].sum();wm=z['sample_weight'][pos];conditional=wm*mass/wm.sum();eventmass=wm.sum()/mass;identity=abs(float(z['sample_weight']@gain/mass)-eventmass*float(wm@gain[pos]/wm.sum()));assert identity<1e-12 and abs(conditional.sum()-mass)<1e-8
            fitdates={meta[int(i)]['date'] for i in z['ids']};helddates={v['date'] for v in oldtr if v['fold']==f and v['method']==f'NET_seed{s}'};assert len(fitdates)==12 and len(helddates)==4 and not fitdates&helddates
            np.savez_compressed(O/'fit_labels'/f'f{f}_s{s}.npz',indices=np.flatnonzero(pos),target=gain[pos],sample_weight=conditional,original_total=np.array([mass]),event_mass=np.array([eventmass]),metadata_ids=z['ids'][pos]);records.append(dict(fold=f,seed=s,original_rows=len(gain),conditional_rows=int(pos.sum()),conditional_calls=len({meta[int(i)]['physical_state_group'] for i in z['ids'][pos]}),conditional_dates=len({meta[int(i)]['date'] for i in z['ids'][pos]}),event_mass=float(eventmass),mean_identity_error=identity,event_weight_sha256=sha(EVENT/'models'/f'f{f}_s{s}.npz')))
    save('preflight.json',dict(passed=True,all12positiveonlytargets_weights_and_date_splits_checked=True,all12empirical_mean_identities=True,synthetic_SVD_and_product_checks=True,pools=records,new_real_fits=0,new_NET_forwards=0));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_labels').iterdir()),O/'preflight.json']});verify();archive('qualified');print('GRID198_QUALIFIED')
def train():
    verify();assert read(O/'preflight.json')['passed'];assert not (O/'training_started.json').exists()
    for p,h in read(O/'qualified_freeze.json').items():assert sha(O/p)==h
    save('training_started.json',dict(new_conditional_fits=12,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cap_s=600));(O/'models').mkdir();records=[];start=time.perf_counter()
    for f in range(4):
        for s in range(3):
            if time.perf_counter()-start>600:raise TimeoutError('Original600s cap;noautomaticextension')
            z=np.load(FIT/'fit_features'/f'f{f}_s{s}.npz');lab=np.load(O/'fit_labels'/f'f{f}_s{s}.npz');event=np.load(EVENT/'models'/f'f{f}_s{s}.npz');x=z['standard'][lab['indices']];y=lab['target'];w=lab['sample_weight'];t=time.perf_counter();beta,bias,singular=M.solve(x,y,w);value=x@beta+bias;err=value-y;grad=np.r_[x.T@(w*err)+beta,w@err]/w.sum();denom=1+max(float(abs(x.T@(w*value)).max()),float(abs(x.T@(w*y)).max()))/w.sum();rel=float(abs(grad).max())/denom;assert rel<1e-8 and np.isfinite(beta).all() and np.isfinite(bias)
            path=O/'models'/f'f{f}_s{s}.npz';np.savez_compressed(path,coef=beta,intercept=np.array([bias]),event_coef=event['coef'][0],event_intercept=event['intercept'],mean=z['mean'],scale=z['scale'],singular=singular,alpha=np.array([1.]));records.append(dict(fold=f,seed=s,conditional_rows=len(y),objective=float((w@(err*err)+beta@beta)/(2*w.sum())),original_gradient_infinity=float(abs(grad).max()),relative_stationarity=rel,new_parameters=84,frozen_event_parameters=84,total_readout_parameters=168,fit_wall_s=time.perf_counter()-t,paired_weight_sha256=sha(path),source_event_weight_sha256=sha(EVENT/'models'/f'f{f}_s{s}.npz')));save('fit_receipts.json',records);print('FIT_CONDITIONAL',f,s,len(y),rel,flush=True)
    save('fits.json',dict(passed=True,new_conditional_fits=12,event_fits=0,NET_fits=0,total_wall_s=time.perf_counter()-start,records=records));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'fit_receipts.json']});verify();archive('trained');print('GRID198_ALL12_TRAINED')
def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h
def publication():
    trained();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert head==remote;p=A/'trained/trained_freeze.json';assert subprocess.check_output(['git','show','HEAD:'+p.relative_to(REPO).as_posix()],cwd=REPO)==p.read_bytes();save('preheld_publication.json',dict(passed=True,git_sha=head,remote_main=head,all12pairedweights_published=True));print('GRID198_PUBLISHED')
def evaluate():
    trained();assert read(O/'preheld_publication.json')['passed'] and not (O/'evaluation_started.json').exists();save('evaluation_started.json',dict(all12pairedweights_published=True));meta,lookup,cases=B.load();task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];oldtr=read(OLD/'oof_trajectories.json');bases={(r['method'],r['sample_id']):r for r in oldtr};tr=[];predictions=[]
    for f in range(4):
        for s in range(3):
            cp=np.load(O/'models'/f'f{f}_s{s}.npz');z=np.load(OLD/f'held_f{f}_s{s}.npz');p,mu,raw,gain,logit=score(z['features'],cp);charge=z['features'][:,82];decision=gain>charge;byidx={int(idx):(bool(d),float(a),float(b),float(c),float(g),float(h),float(l)) for idx,d,a,b,c,g,h,l in zip(z['ids'],decision,p,mu,raw,gain,charge,logit)}
            cc=[c for c in cases if bases[f'NET_seed{s}',c['sample_id']]['fold']==f]
            for c in cc:
                base=bases[f'NET_seed{s}',c['sample_id']];pos=base['baseline_stop'];eligible=pos<len(c['states']);key=lookup[c['sample_id'],pos] if eligible else None;d,prob,mu1,raw1,gain1,cost1,logit1=byidx[key] if eligible else (False,0.,0.,0.,0.,0.,0.)
                if eligible:
                    assert np.isclose(cost1,fee*c['queries'][pos]/scale,atol=1e-8,rtol=2e-6);assert gain1==prob*mu1 and 0<=prob<=1 and mu1>=0
                    predictions.append(dict(fold=f,seed=s,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=pos,gate_continue=d,event_probability=prob,event_logit=logit1,conditional_normalized_gain=mu1,unclipped_conditional_gain=raw1,predicted_normalized_gain=gain1,known_charge=cost1,predicted_net=gain1-cost1,actual_next_gain=float(c['gains'][pos]),actual_next_queries=int(c['queries'][pos]),actual_next_NET=float((c['gains'][pos]-fee*c['queries'][pos])/scale),actual_next_change=bool(c['chosen'][pos]!=c['chosen'][pos+1])))
                for method in ['NET','ADD1','GATE']:
                    add=bool(eligible and (method=='ADD1' or method=='GATE' and d));end=pos+int(add);name=f'{method}_seed{s}';v=B.H.trajectory(c,np.arange(len(c['states']))<end,name,fee,scale);v.update(fold=f,baseline_stop=pos,baseline_gain=base['gain'],baseline_queries=base['queries'],eligible=eligible,append=add,action_match=v['chosen_pool_id']==c['chosen'][-1]);assert base['queries']<=v['queries']<=int(c['queries'].sum()) and v['queries']-base['queries']<=64;assert base['gain']<=v['gain'];assert not base['action_match'] or v['action_match']
                    if method in ['NET','ADD1']:
                        for k in ['gain','queries','chosen_pool_id','actions']:assert v[k]==bases[name,c['sample_id']][k]
                    tr.append(v)
    metrics={f'{method}_seed{s}':B.summarize([r for r in tr if r['method']==f'{method}_seed{s}']) for method in ['NET','ADD1','GATE'] for s in range(3)};families={method:{k:float(np.mean([metrics[f'{method}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{method}_seed0']['macro']} for method in ['NET','ADD1','GATE']};counts=dict(eligible=len(predictions),append=sum(p['gate_continue'] for p in predictions),append_changes=sum(p['gate_continue'] and p['actual_next_change'] for p in predictions),append_no_change=sum(p['gate_continue'] and not p['actual_next_change'] for p in predictions),append_positive_NET=sum(p['gate_continue'] and p['actual_next_NET']>0 for p in predictions));assert len(tr)==21402 and len(predictions)==4681
    save('oof_trajectories.json',tr);save('gate_predictions.json',predictions);save('search_metrics.json',metrics);save('family_summary.json',families);save('gate_counts.json',counts);trained();save('review.json',dict(passed=True,all21402trajectories_and_4681products_checked=True,all_NET_ADD1_controls_reproduced=True,all12pairedweights_before_held=True,all168readout_parameters_disclosed=True,new_conditional_fits=12,event_and_NET_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('evaluated');print('GRID198_EVALUATED',json.dumps(dict(families=families,counts=counts),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
