"""Single preregistered parent-coherent residual and empirical positive protection."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid265_guarded_bias';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID265_guarded_bias';Q=R/'outputs/grid264_guarded_bias_plan';PUBLIC=R/'outputs/grid261_bounded_basis/outer_predictions';PLAN=Q;BASE=R/'outputs/grid208_public_amount';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def module(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
N=R/'outputs/grid261_bounded_basis';RUNTIME=R/'work/grid264_guarded_bias_plan/runtime264.py';RT=module('grid265_qualified_runtime',RUNTIME)
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
    assert not O.exists() and read(Q/'decision.json')['ready_for_single265_scalar_component'];head=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
    ar=REPO/'research_archive/power-grid-2026/protocols/GRID264_guarded_bias_plan/qualified'
    for rel,h in read(Q/'qualified_freeze.json').items():assert hashlib.sha256(git('show',head+':'+(ar/rel).relative_to(REPO).as_posix())).hexdigest()==h
    assert git('show',head+':'+(ar/'runtime264.py').relative_to(REPO).as_posix())==RUNTIME.read_bytes() and sha(RUNTIME)==read(Q/'runtime_freeze.json')['work/grid264_guarded_bias_plan/runtime264.py']
    oldar=REPO/'research_archive/power-grid-2026/protocols/GRID261_bounded_basis/scored/outer_predictions'
    for f in PUBLIC.glob('*.npz'):assert git('show',head+':'+(oldar/f.name).relative_to(REPO).as_posix())==f.read_bytes()
    assert len(list(PUBLIC.glob('*.npz')))==12;O.mkdir();save('pre_input_publication.json',dict(passed=True,git_sha=head,qualified24inputs_runtime_and12publicscores_published=True))
    save('design.json',dict(id='GRID265',candidate='ONEfixedmaximaldownwardscorebias onfrozen261;11scalar+1KEEP/one600swindow/0thetaNNfits/noextension/refit/grid.',objective='Maxdelta>=0 subjectto old261score-delta>=0 onALLoriginalown12selectedSTOPpositive. Closedformminpositivescore;tiescontinue. Explicit threshold/operatingpoint/empiricaltargetchange,not originalNLL orunchangedthreshold.',fit='Only264qualifiedcalandfrozenmodels;500positive/active464/KEEP36. f0_s1or emptyguarddelta0keep261;invalidnegativeguardfail. Runtime264unchanged.',score='ALL12compositeweights/scalars publishedfirst;byte-reuse261publishedids/logit/base_CONT then subtractdelta,monotonesubsetof261CONT. ALL12predictionspublishedbeforeoldoutcomejoin. No NNheadforward.',controls='Primaryfrozen261/shared208GATE/250lowquery/230historicalreuse,nocontrolrefits. All3seeds16olddevelopmentdates/adversecases.',count='Distinctnewhypothesis+11;branch62->73. 6oldcontinuations8285updates/NN604.703sactual/678.156cap separatelyretained.',limits='Own12empiricalguardnot independentorouterquality guarantee,parentown12/internalcal4ancestry/exposedcorrelationremain. Queryproxy notwholeweekoractualtime. Final8/physics/grid/paper/FRAME/install untouched;fee/labels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.',engineering='Preparationencodingname typo utf8-sig before sourcegeneration/freeze;0fits/score/output. Error preserved andcorrectedto utf-8-sig,protocolunchanged.'))
    paths=[W/'run.py',W/'preparation_error01.json',O/'design.json',O/'pre_input_publication.json',RUNTIME,Q/'decision.json',Q/'qualification.json',Q/'qualified_freeze.json',Q/'runtime_freeze.json',*list((Q/'qualified_inputs').glob('*.npz')),*list(PUBLIC.glob('*.npz')),META,R/'work/grid227_context_newton/join.py',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json',N/'oof_trajectories.json',N/'gate_predictions.json',N/'family_summary.json',N/'search_metrics.json',N/'gate_counts.json']
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('source');print('GRID265_SOURCE_FROZEN')

def train():
    verify();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head and git('show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes()
    assert not (O/'training_started.json').exists();save('training_started.json',dict(newscalar11=True,cap_s600=True,source_published=head));start=time.perf_counter();(O/'models').mkdir();receipts=[];qual={r['pool']:r for r in read(Q/'qualification.json')['pools']}
    for file in sorted((Q/'qualified_inputs').glob('*_cal.npz')):
        assert time.perf_counter()-start<600;pool=file.name[:-8];cal=np.load(file);m=np.load(Q/'qualified_inputs'/f'{pool}_model.npz');score=RT.base_score(cal['X'],cal['b'],m['theta'],float(m['alpha'][0]));pos=cal['y']>0;assert hashlib.sha256(score[pos].tobytes()).hexdigest()==qual[pool]['oldscore_vector_hash'];t=time.perf_counter();delta,keep=RT.guarded_shift(score,pos,KEEP=bool(m['fallback_KEEP'][0]));assert np.all((score-delta)[pos]>=0)
        if not keep:assert np.any((score-float(np.nextafter(delta,np.inf)))[pos]<0)
        np.savez_compressed(O/'models'/f'{pool}.npz',theta=m['theta'],alpha=m['alpha'],fallback_KEEP=m['fallback_KEEP'],delta=np.array([delta]),scalar_KEEP=np.array([keep]));receipts.append(dict(pool=pool,fitted=not keep,delta=delta,protected_positive=int(pos.sum()),all_guarded_nonnegative=True,originaltheta_alpha_unchanged=True,scalar_wall_s=time.perf_counter()-t));save('fit_receipts.json',receipts)
    elapsed=time.perf_counter()-start;assert elapsed<600 and len(receipts)==12 and sum(r['fitted'] for r in receipts)==11 and sum(r['protected_positive'] for r in receipts)==500;save('fits.json',dict(passed=True,new_scalar_fits=11,constantKEEP=1,newtheta_NNfits=0,wall_s=elapsed,all500_empiricalpositives_protected=True,branchcumulative73=True));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fit_receipts.json',O/'fits.json']});archive('trained');print('GRID265_TRAINED',json.dumps(read(O/'fits.json')))

def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
    assert read(O/'fits.json')['passed']
def publication():
    trained();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'trained/trained_freeze.json').relative_to(REPO).as_posix())==(O/'trained_freeze.json').read_bytes()
    for rel,h in read(O/'trained_freeze.json').items():assert hashlib.sha256(git('show',head+':'+(A/'trained'/rel).relative_to(REPO).as_posix())).hexdigest()==h
    save('pre_score_publication.json',dict(passed=True,git_sha=head,all12weights_steps_published=True));print('GRID265_MODELS_PUBLISHED')
def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all12scalars_published=True));(O/'outer_predictions').mkdir();records=[]
    for file in sorted(PUBLIC.glob('*.npz')):
        z=np.load(file);m=np.load(O/'models'/file.name);delta=float(m['delta'][0]);logit,cont=RT.apply_shift(z['logit'],z['base_CONT'],delta);assert not np.any(cont&~z['continue_']);np.savez_compressed(O/'outer_predictions'/file.name,ids=z['ids'],logit=logit,continue_=cont,base_CONT=z['base_CONT'],source261_logit=z['logit'],source261_continue=z['continue_']);records.append(dict(pool=file.stem,rows=len(logit),continues=int(cont.sum()),outcome_fields_read=[]))
    assert sum(r['rows'] for r in records)==4681;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'outer_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID265_SCORED_ALL12')

def scored():
    trained()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p
def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes()
    for rel,h in read(O/'scored_freeze.json').items():assert hashlib.sha256(git('show',head+':'+(A/'scored'/rel).relative_to(REPO).as_posix())).hexdigest()==h
    save('pre_join_publication.json',dict(passed=True,git_sha=head,all12predictions_published=True));print('GRID265_SCORES_PUBLISHED')
def join():
    scored();assert read(O/'pre_join_publication.json')['passed'] and not (O/'join_started.json').exists();save('join_started.json',dict(all12predictions_published=True));summarize=module('grid265_saved_summary',R/'work/grid227_context_newton/join.py').summarize;meta=read(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};parents=[r for r in read(BASE/'oof_trajectories.json') if r['method'].startswith('GATE_')];task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];scores={}
    for file in (O/'outer_predictions').glob('*.npz'):
        f,s=[int(v[1:]) for v in file.stem.split('_')];z=np.load(file)
        for i,cont in zip(z['ids'],z['continue_']):scores[f,s,int(i)]=bool(cont)
    tr=list(parents);pred=[]
    for seed in range(3):
        for old in [r for r in parents if r['method']==f'GATE_seed{seed}']:
            c=cases[old['sample_id']];f=old['fold'];j=old['baseline_stop'];eligible=j<len(c['states']);append=scores[f,seed,lookup[c['sample_id'],j]] if eligible else False;assert not append or old['append'];end=j+int(append);new=dict(old);new.update(method=f'SHIFT_seed{seed}',actions=[1]*end+([0] if end<len(c['states']) else []),batches=end,gain=float(c['gains'][:end].sum()),queries=int(c['queries'][:end].sum()),chosen_pool_id=c['chosen'][end],append=append);new['net_utility']=(new['gain']-fee*new['queries'])/scale;new['action_match']=new['chosen_pool_id']==c['chosen'][-1];tr.append(new)
            if eligible:pred.append(dict(fold=f,seed=seed,sample_id=c['sample_id'],position=j,date=c['date'],call=c['physical_state_group'],base_continue=old['append'],continue_=append,actual_next_NET=float((c['gains'][j]-fee*c['queries'][j])/scale),actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j])))
    assert len(tr)==14268 and len(pred)==4681;metrics={};ref=read(BASE/'search_metrics.json')
    for arm in ['GATE','SHIFT']:
        for seed in range(3):
            name=f'{arm}_seed{seed}';rr=[r for r in tr if r['method']==name];assert len(rr)==2378;metrics[name]=summarize(rr)
            if arm=='GATE':
                for k,v in metrics[name]['macro'].items():assert abs(v-ref[name]['macro'][k])<1e-12
    family={arm:{k:float(np.mean([metrics[f'{arm}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{arm}_seed0']['macro']} for arm in ['GATE','SHIFT']};paired=[]
    for seed in range(3):
        old,new=metrics[f'GATE_seed{seed}'],metrics[f'SHIFT_seed{seed}'];paired.append(dict(seed=seed,macro_delta={k:new['macro'][k]-old['macro'][k] for k in old['macro']},per_date=[dict(date=x['date'],NET_delta=y['mean_net_utility']-x['mean_net_utility'],query_delta=y['mean_queries']-x['mean_queries'],gain_delta=y['mean_gain']-x['mean_gain']) for x,y in zip(old['per_date'],new['per_date'])]))
    counts=dict(original208_appends=sum(r['base_continue'] for r in pred),appends=sum(r['continue_'] for r in pred),profitable_retained=sum(r['continue_'] and r['actual_next_NET']>0 for r in pred),profitable_removed=sum(r['base_continue'] and not r['continue_'] and r['actual_next_NET']>0 for r in pred),unprofitable_removed=sum(r['base_continue'] and not r['continue_'] and r['actual_next_NET']<=0 for r in pred))
    for n,v in [('oof_trajectories.json',tr),('gate_predictions.json',pred),('search_metrics.json',metrics),('family_summary.json',family),('paired_summary.json',paired),('gate_counts.json',counts)]:save(n,v)
    scored();save('review.json',dict(passed=True,all14268savedsequences_checked=True,all4681publishedscores_joined=True,commonGATEreproduced=True,all3seeds16oldexposeddates_retained=True,no_new_physics=True,final8_absent=True));archive('evaluated');print('GRID265_EVALUATED',json.dumps(dict(family=family,counts=counts,paired=[dict(seed=r['seed'],NET=r['macro_delta']['mean_net_utility'],queries=r['macro_delta']['mean_queries'],gain=r['macro_delta']['mean_gain']) for r in paired])))
if __name__=='__main__':globals()[sys.argv[1]]()
