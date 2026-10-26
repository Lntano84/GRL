"""Fixed twelve one-batch gain readouts on frozen GRID192 features, no new NET."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,datetime,hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid196_gain_readout';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID196_gain_readout';SRC=R/'outputs/grid192_gate_support';OLD=R/'outputs/grid190_extension_gate'
sp=importlib.util.spec_from_file_location('grid196_frozen_trajectory_tools',R/'work/grid190_extension_gate/run.py');B=importlib.util.module_from_spec(sp);sp.loader.exec_module(B)
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
def solve(x,y,w,alpha=1.):
    mass=w.sum();mean=w@x/mass;ym=float(w@y/mass);xc=x-mean;yc=y-ym
    u,s,vt=np.linalg.svd(np.sqrt(w)[:,None]*xc,full_matrices=False)
    beta=vt.T@((s/(s*s+alpha))*(u.T@(np.sqrt(w)*yc)));bias=ym-float(mean@beta)
    return beta,bias,s
def prepare():
    assert not O.exists();O.mkdir(parents=True);B.H.sealed()
    save('design.json',dict(id='GRID196_FIXED_GAIN_READOUT',authorization='User持续继续, autonomous bounded learning authorized. Pretrainingnotice:12fixedlinear gainreadouts;no capacity/physics expansion.',
        hypothesis='Existing binarychange head doesnot model rewardamount. Test whether same frozen representation and covered fitpopulation support conditionalmean nextbatch gain.',
        data='Exactly GRID192 halfSTOP/halfactualpreSTOPCONT arrays, all12foldseedpairs,83features,originalSTOPnormalizer and totalweights unchanged. No addedrows or outcome-dependent rowweights. OriginalNET never updated.',
        objective='Normalized onebatchgain target g/gain_scale. Minimize (sum w*(Z beta+b-y)^2 + alpha*||beta||^2)/(2*sumw),alpha=1,interceptunpenalized. Weightedcentered SVD closedform,84parameters. Different readoutobjective fromBCE,not regularization-equivalent nor isolatedlabel-only ablation.',
        fits='12closedform ridge fits,one per old4foldx3NETseed,alpha1 fixed,no sweep/retry/endpointselection,600sec cap. SVD stable originalobjective withnoiterationcap. All12publishedbefore scoringheld.',
        decision='OriginalNET unchanged untilfirstSTOP. Predict normalizedgain=max(0,Zbeta+b),subtractpublic knownnextcharge=fee*nextqueries/gain_scale. Strictpositive buysone<=64batch thenreturnsbest. BaselineNET protected,not guaranteeGRID190 gate orfullweek quality. No .5 probabilitythreshold;not probability.',
        comparisons='UnchangedNET,unconditionalADD1,existingGRID190 STOPgate andGRID193 same-mixturebinary gate. All16 historicaldates/all3seeds. Sameprespecified cost-only uniformthinning to minimumextraquerybudget perdate/seed vs190 and separatelyvs193;ADD1randomreference. Analyticexpectation only.',
        prior='GRID120 utilityweightedBCE didnot beatoldHuber;GRID123auxiliary/GRID125gainheads notblanket evidencefornewgainfit. Here frozenrelationrepresentation/boundedpostSTOPaction/convexmeanloss;no claimnovel regression algorithm.',
        qualification='Syntheticclosedformvsaugmentedleast-squares check before realfits. Everytarget reconstructedfromsavedonebatchlabels,weightsmatch;all12fit/helddatesdisjoint. No hiddenfutureinputs.',
        limits='Exposedteacherpath OOFdevelopment;notnewclosedloopweek/crossgrid/timing/test. Localrewardproxy distinct fromfullweekquality;preservealloutcomes,notrequireallwin/5percent.',
        prohibited=['extraalpha/threshold/seedsearch','newNETforwards/training','newdata/physics/final8/paper','automaticensembleor rollout']))
    files=[W/'run.py',O/'design.json',R/'outputs/grid195_gate_label_audit/publication_audit.json',R/'outputs/grid195_gate_label_audit/canonical_pool_statistics.json',R/'outputs/grid195_gate_label_audit/complete.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid113_relational_learning/train_metadata.json',OLD/'oof_trajectories.json',OLD/'search_metrics.json',OLD/'family_summary.json',OLD/'complete.json',R/'outputs/grid193_gate_precondition/oof_trajectories.json',R/'outputs/grid193_gate_precondition/search_metrics.json',R/'outputs/grid193_gate_precondition/family_summary.json',R/'outputs/grid193_gate_precondition/complete.json',R/'outputs/grid120_decision_objective/decision.json']+list((SRC/'fit_features').glob('*.npz'))+list(OLD.glob('held_f*.npz'))+[R/p for p in read(OLD/'code_freeze.json')]
    save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID196_PLANNED')
def qualify():
    verify();assert not (O/'preflight.json').exists();rng=np.random.default_rng(196);x=rng.normal(size=(31,8));x[:,7]=x[:,3];y=rng.normal(size=31);w=rng.uniform(.1,2,size=31);beta,bias,_=solve(x,y,w);aug=np.vstack([np.sqrt(w)[:,None]*np.c_[x,np.ones(31)],np.c_[np.eye(8),np.zeros(8)]]);target=np.r_[np.sqrt(w)*y,np.zeros(8)];direct=np.linalg.lstsq(aug,target,rcond=None)[0];assert np.allclose(np.r_[beta,bias],direct,atol=1e-12,rtol=1e-12)
    meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');_,_,cases=B.load();cm={c['sample_id']:c for c in cases};task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];(O/'fit_labels').mkdir();pools=[]
    oldtr=read(OLD/'oof_trajectories.json')
    for f in range(4):
        for s in range(3):
            z=np.load(SRC/'fit_features'/f'f{f}_s{s}.npz');g=[];q=[]
            for idx in z['ids']:
                m=meta[int(idx)];c=cm[m['sample_id']];j=m['position'];g.append(float(c['gains'][j]));q.append(int(c['queries'][j]));assert c['date']==m['date']
            g=np.array(g);q=np.array(q);assert np.array_equal(g>0,z['y']>0);assert abs(z['sample_weight'].sum()-int(z['original_STOP_count']))<1e-8
            fitdates={meta[int(i)]['date'] for i in z['ids']};helddates={v['date'] for v in oldtr if v['fold']==f and v['method']==f'NET_seed{s}'};assert len(fitdates)==12 and len(helddates)==4 and not fitdates&helddates
            np.savez_compressed(O/'fit_labels'/f'f{f}_s{s}.npz',gain=g,target=g/scale,queries=q,known_charge=fee*q/scale)
            pools.append(dict(fold=f,seed=s,rows=len(g),positive=int((g>0).sum()),fitdates=sorted(fitdates),helddates=sorted(helddates),source_sha256=sha(SRC/'fit_features'/f'f{f}_s{s}.npz')))
    save('preflight.json',dict(passed=True,synthetic_rankdeficient_SVD_matches_augmented_lstsq=True,all12_targets_and_sampling_weights_checked=True,pools=pools,new_real_fits=0,new_NET_forwards=0));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_labels').iterdir()),O/'preflight.json']});verify();archive('qualified');print('GRID196_QUALIFIED')
def train():
    verify();assert read(O/'preflight.json')['passed'];assert not (O/'training_started.json').exists()
    for p,h in read(O/'qualified_freeze.json').items():assert sha(O/p)==h
    save('training_started.json',dict(new_real_fits=12,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cap_s=600));(O/'models').mkdir();records=[];start=time.perf_counter()
    for f in range(4):
        for s in range(3):
            if time.perf_counter()-start>600:raise TimeoutError('Original600s cap, preservepartialoutputs')
            z=np.load(SRC/'fit_features'/f'f{f}_s{s}.npz');target=np.load(O/'fit_labels'/f'f{f}_s{s}.npz')['target'];x=z['standard'];w=z['sample_weight'];t=time.perf_counter();beta,bias,singular=solve(x,target,w);pred=x@beta+bias;err=pred-target;grad=(x.T@(w*err)+beta)/w.sum();gb=float(w@err/w.sum());denom=1+max(float(abs(x.T@(w*pred)).max()),float(abs(x.T@(w*target)).max()))/w.sum();relative=max(float(abs(grad).max()),abs(gb))/denom;assert relative<1e-8 and np.isfinite(beta).all() and np.isfinite(bias)
            name=f'f{f}_s{s}';np.savez_compressed(O/'models'/f'{name}.npz',coef=beta,intercept=np.array([bias]),mean=z['mean'],scale=z['scale'],singular=singular,alpha=np.array([1.]))
            records.append(dict(fold=f,seed=s,rows=len(target),objective=float((w@(err*err)+beta@beta)/(2*w.sum())),original_gradient_infinity=max(float(abs(grad).max()),abs(gb)),relative_stationarity=relative,effective_degrees=float(np.sum(singular**2/(singular**2+1)))+1,fit_wall_s=time.perf_counter()-t,weight_sha256=sha(O/'models'/f'{name}.npz')));save('fit_receipts.json',records);print('FIT_GAIN_READOUT',f,s,relative,flush=True)
    save('fits.json',dict(passed=True,new_fits=12,all_from_closedform_fixedalpha=True,total_wall_s=time.perf_counter()-start,records=records));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'fit_receipts.json']});verify();archive('trained');print('GRID196_ALL12_TRAINED')
def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h
def publication():
    trained();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert head==remote
    p=A/'trained/trained_freeze.json';assert subprocess.check_output(['git','show','HEAD:'+p.relative_to(REPO).as_posix()],cwd=REPO)==p.read_bytes();save('preheld_publication.json',dict(passed=True,git_sha=head,remote_main=head,all12weights_published=True));print('GRID196_PUBLISHED')
def evaluate():
    trained();assert read(O/'preheld_publication.json')['passed'];assert not (O/'evaluation_started.json').exists();save('evaluation_started.json',dict(all12weights_published=True));meta,lookup,cases=B.load();task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];oldtr=read(OLD/'oof_trajectories.json');bases={(r['method'],r['sample_id']):r for r in oldtr};tr=[];predictions=[]
    for f in range(4):
        for s in range(3):
            cp=np.load(O/'models'/f'f{f}_s{s}.npz');z=np.load(OLD/f'held_f{f}_s{s}.npz');x=z['features'];standard=(x-cp['mean'])/cp['scale'];raw=standard@cp['coef']+cp['intercept'][0];gain=np.maximum(raw,0);charge=x[:,82];assert not (x[:,81]>x[:,80]).any();decisions=gain>charge;byidx={int(i):(bool(d),float(g),float(r),float(c)) for i,d,g,r,c in zip(z['ids'],decisions,gain,raw,charge)}
            cc=[c for c in cases if bases[f'NET_seed{s}',c['sample_id']]['fold']==f]
            for c in cc:
                base=bases[f'NET_seed{s}',c['sample_id']];p=base['baseline_stop'];eligible=p<len(c['states']);key=lookup[c['sample_id'],p] if eligible else None;d,g,r,charge1=byidx[key] if eligible else (False,0.,0.,0.)
                if eligible:
                    assert np.isclose(charge1,fee*c['queries'][p]/scale,atol=1e-8,rtol=2e-6)
                    predictions.append(dict(fold=f,seed=s,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=p,gate_continue=d,predicted_normalized_gain=g,unclipped_normalized_gain=r,known_charge=charge1,predicted_net=g-charge1,actual_next_gain=float(c['gains'][p]),actual_next_queries=int(c['queries'][p]),actual_next_NET=float((c['gains'][p]-fee*c['queries'][p])/scale),actual_next_change=c['chosen'][p]!=c['chosen'][p+1]))
                for method in ['NET','ADD1','GATE']:
                    add=bool(eligible and (method=='ADD1' or method=='GATE' and d));end=p+int(add);name=f'{method}_seed{s}';v=B.H.trajectory(c,np.arange(len(c['states']))<end,name,fee,scale);v.update(fold=f,baseline_stop=p,baseline_gain=base['gain'],baseline_queries=base['queries'],eligible=eligible,append=add,action_match=v['chosen_pool_id']==c['chosen'][-1]);assert base['queries']<=v['queries']<=int(c['queries'].sum()) and v['queries']-base['queries']<=64;assert base['gain']<=v['gain'];assert not base['action_match'] or v['action_match']
                    if method=='NET':
                        for k in ['gain','queries','chosen_pool_id','actions']:assert v[k]==base[k]
                    if method=='ADD1':
                        for k in ['gain','queries','chosen_pool_id','actions']:assert v[k]==bases[name,c['sample_id']][k]
                    tr.append(v)
    metrics={f'{method}_seed{s}':B.summarize([r for r in tr if r['method']==f'{method}_seed{s}']) for method in ['NET','ADD1','GATE'] for s in range(3)}
    families={method:{k:float(np.mean([metrics[f'{method}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{method}_seed0']['macro']} for method in ['NET','ADD1','GATE']}
    counts=dict(eligible=len(predictions),append=sum(p['gate_continue'] for p in predictions),append_changes=sum(p['gate_continue'] and p['actual_next_change'] for p in predictions),append_no_change=sum(p['gate_continue'] and not p['actual_next_change'] for p in predictions),append_positive_NET=sum(p['gate_continue'] and p['actual_next_NET']>0 for p in predictions))
    assert len(tr)==21402 and len(predictions)==4681
    save('oof_trajectories.json',tr);save('gate_predictions.json',predictions);save('search_metrics.json',metrics);save('family_summary.json',families);save('gate_counts.json',counts);trained();save('review.json',dict(passed=True,all21402trajectory_checks=True,all7134NET_and_ADD1_reproduced=True,all4681publiccharges_and_decisions_checked=True,all12before_held=True,new_fits=12,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('evaluated');print('GRID196_EVALUATED',json.dumps(dict(families=families,counts=counts),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
