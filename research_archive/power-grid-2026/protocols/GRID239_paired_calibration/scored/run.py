"""Preregistered matched FIT8/CAL4 scalar comparison on frozen residuals."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid239_paired_calibration';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID239_paired_calibration'
PLAN=R/'outputs/grid238_calibration_support';S=R/'outputs/grid237_residual_holdgroup';Q=R/'outputs/grid236_shared_prior_inputs';ROLE=R/'outputs/grid235_calibration_ancestry/fixed_probe_pools.json';H=R/'outputs/grid214_inner_head_generation';BASE=R/'outputs/grid208_public_amount';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def module(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
STEP=module('grid239_original221',R/'work/grid221_step_plan/run.py');ARMS=['FIT8','CAL4']
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    for p,h in read(PLAN/'source_freeze.json').items():assert sha(R/p)==h,p
    for p,h in read(S/'trained_freeze.json').items():assert sha(S/p)==h,p
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    for folder in {'calibrated':['models'],'scored':['public_outer','outer_predictions']}.get(phase,[]):shutil.copytree(O/folder,d/folder)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def prepare():
    assert not O.exists() and read(PLAN/'qualification.json')['passed'];O.mkdir();save('design.json',dict(id='GRID239',protocol=read(PLAN/'future239_design.json'),implementation='OnlytwoarmsFIT8/CAL4;22scalarfits+2KEEP,one600s scalarwindow. Reuse237theta43 exactly. No newmodel/NN/forestfit. Newouter4offlinecomparisonexplicitlyseparatefromcompleted237.'))
    paths=[W/'run.py',O/'design.json',PLAN/'future239_design.json',PLAN/'qualification.json',PLAN/'source_freeze.json',S/'complete.json',S/'trained_freeze.json',S/'cal4_error_rows.json',Q/'future_cal_target_sources.json',ROLE,META,H/'public_raw41.npz',BASE/'gate_predictions.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'work/grid221_step_plan/run.py']+list((Q/'fit_inputs').glob('*.npz'))+list((Q/'cal_public').glob('*.npz'))+list((S/'models').glob('*.npz'))
    for r in read(ROLE):paths += [R/r['scaler_file'],R/'outputs/grid190_extension_gate'/f"held_{r['pool']}.npz",R/'outputs/grid220_full_prior/public_held_inputs'/f"{r['pool']}.npz"]+[R/p for p in r['head_models']]
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in set(paths)});verify();archive('qualified');print('GRID239_QUALIFIED')
def calibration_data(pool,arm):
    if arm=='FIT8':
        z=np.load(Q/'fit_inputs'/f'{pool}.npz');return z['X'],z['b'],(z['y']>0)&z['domain_STOP'],z['ids']
    z=np.load(Q/'cal_public'/f'{pool}.npz');rows=[r for r in read(S/'cal4_error_rows.json') if r['pool']==pool];lookup={r['index']:r for r in rows};return z['X'],z['b'],np.array([lookup[int(i)]['positive'] for i in z['ids']])&z['original_STOP'],z['ids']
def calibrate():
    verify();assert not (O/'calibration_started.json').exists();save('calibration_started.json',dict(scalar_fits=22,KEEPconstants=2,cap_s=600,new_models=0));start=time.perf_counter();(O/'models').mkdir();records=[]
    for r in read(ROLE):
        pool=r['pool'];p=np.load(S/'models'/f'{pool}.npz');theta=p['theta'];fallback=bool(p['fallback_KEEP'][0])
        for arm in ARMS:
            assert time.perf_counter()-start<600;x,b,pos,ids=calibration_data(pool,arm);endpoint=b+np.sum(x*theta[:-1],axis=1)+theta[-1]
            if fallback:a=0.;bound=1.
            else:assert pos.any();a,bound=STEP.choose_alpha(b,endpoint,pos)
            assert np.all((b+a*(endpoint-b))[pos]>=0);np.savez_compressed(O/'models'/f'{pool}_{arm}.npz',theta=theta,alpha=np.array([a]),fallback_KEEP=np.array([fallback]));records.append(dict(pool=pool,arm=arm,calibrated=not fallback,alpha=a,analytic_bound=bound,positive_rows=int(pos.sum()),positive_ids=ids[pos].tolist(),all_source_positives_preserved=True,parent_theta_unchanged=True))
    elapsed=time.perf_counter()-start;assert elapsed<600 and len(records)==24 and sum(r['calibrated'] for r in records)==22;save('calibration_receipts.json',records);save('fits.json',dict(passed=True,new_scalar_fits=22,new_model_fits=0,KEEPconstants=2,wall_s=elapsed));save('calibrated_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'calibration_receipts.json',O/'fits.json']});verify();archive('calibrated');print('GRID239_CALIBRATED',json.dumps(read(O/'fits.json')))
def calibrated():
    verify()
    for p,h in read(O/'calibrated_freeze.json').items():assert sha(O/p)==h,p
def publication():
    calibrated();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'calibrated/calibrated_freeze.json').relative_to(REPO).as_posix())==(O/'calibrated_freeze.json').read_bytes();save('pre_score_publication.json',dict(passed=True,git_sha=head,all24effective_models_steps_published=True));print('GRID239_STEPS_PUBLISHED')
def score():
    calibrated();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all24models_steps_published=True,outer_targets_forbidden=True));raw=np.load(H/'public_raw41.npz')['values'];meta=read(META);(O/'public_outer').mkdir();(O/'outer_predictions').mkdir();records=[]
    for r in read(ROLE):
        pool=r['pool'];h=np.load(R/'outputs/grid190_extension_gate'/f'held_{pool}.npz');base=np.load(R/'outputs/grid220_full_prior/public_held_inputs'/f'{pool}.npz');q=np.load(R/r['scaler_file']);ids=h['ids'];assert np.array_equal(ids,base['ids']) and {meta[int(i)]['date'] for i in ids}<=set(r['outer_dates']);x124=np.c_[(h['features']-q['mean83'])/q['scale83'],(raw[ids.astype(int)]-q['mean41'])/q['scale41']];cp=pickle.loads((R/r['head_models'][0]).read_bytes());mu=pickle.loads((R/r['head_models'][1]).read_bytes());p=cp.predict_proba(x124)[:,1];amount=mu.predict(x124);charge=h['features'][:,82];assert (charge>0).all() and (amount>0).all();b=(p*amount-charge)/(p*amount+charge);x=np.c_[x124[:,83:],b];support=b>0;passthrough=base['base_CONT']&~support
        np.savez_compressed(O/'public_outer'/f'{pool}.npz',ids=ids,X=x,b=b,event_p=p,mu4=amount,charge=charge,base_CONT=base['base_CONT'],inner_support=support,passthrough=passthrough)
        for arm in ARMS:
            cp=np.load(O/'models'/f'{pool}_{arm}.npz');t=cp['theta'];a=float(cp['alpha'][0]);logit=b+a*(np.sum(x*t[:-1],axis=1)+t[-1]);cont=base['base_CONT']&(~support|(logit>=0));assert not np.any(cont&~base['base_CONT']);np.savez_compressed(O/'outer_predictions'/f'{pool}_{arm}.npz',ids=ids,logit=logit,continue_=cont,base_CONT=base['base_CONT'],inner_support=support,passthrough=passthrough);records.append(dict(pool=pool,arm=arm,rows=len(ids),baseCONT=int(base['base_CONT'].sum()),supportCONT=int((base['base_CONT']&support).sum()),passthroughCONT=int(passthrough.sum()),continues=int(cont.sum()),outcome_fields_read=[]))
    assert len(records)==24 and sum(r['rows'] for r in records)==9362;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'public_outer').iterdir()),*list((O/'outer_predictions').iterdir()),O/'score_receipts.json']});calibrated();archive('scored');print('GRID239_SCORED_ALL24')
def scored():
    calibrated()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p
def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes();save('pre_join_publication.json',dict(passed=True,git_sha=head,all24outer_predictions_published=True));print('GRID239_OUTER_SCORES_PUBLISHED')
def join():
    scored();assert read(O/'pre_join_publication.json')['passed'] and not (O/'join_started.json').exists();save('join_started.json',dict(all24outer_predictions_published=True));summarize=module('grid239_saved_summary',R/'work/grid227_context_newton/join.py').summarize;meta=read(META);lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());byid={c['sample_id']:c for c in cases};parents=[r for r in read(BASE/'oof_trajectories.json') if r['method'].startswith('GATE_')];task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];scores={}
    for file in (O/'outer_predictions').glob('*.npz'):
        f,s,arm=file.stem.split('_');z=np.load(file)
        for i,cont,passthrough in zip(z['ids'],z['continue_'],z['passthrough']):scores[int(f[1:]),int(s[1:]),arm,int(i)]=(bool(cont),bool(passthrough))
    tr=list(parents);pred=[]
    for seed in range(3):
        for old in [r for r in parents if r['method']==f'GATE_seed{seed}']:
            c=byid[old['sample_id']];j=old['baseline_stop'];f=old['fold'];eligible=j<len(c['states']);actual=float((c['gains'][j]-fee*c['queries'][j])/scale) if eligible else None
            for arm in ARMS:
                append,passthrough=scores[f,seed,arm,lookup[c['sample_id'],j]] if eligible else (False,False);assert not append or old['append'];end=j+int(append);new=dict(old);new.update(method=f'{arm}_seed{seed}',actions=[1]*end+([0] if end<len(c['states']) else []),batches=end,gain=float(c['gains'][:end].sum()),queries=int(c['queries'][:end].sum()),chosen_pool_id=c['chosen'][end],append=append);new['net_utility']=(new['gain']-fee*new['queries'])/scale;new['action_match']=new['chosen_pool_id']==c['chosen'][-1];tr.append(new)
                if eligible:pred.append(dict(fold=f,seed=seed,arm=arm,sample_id=c['sample_id'],position=j,date=c['date'],call=c['physical_state_group'],base_continue=old['append'],continue_=append,passthrough=passthrough,actual_next_NET=actual,actual_next_gain=float(c['gains'][j]),actual_next_queries=int(c['queries'][j])))
    assert len(tr)==21402 and len(pred)==9362;metrics={};ref=read(BASE/'search_metrics.json')
    for arm in ['GATE',*ARMS]:
        for seed in range(3):
            name=f'{arm}_seed{seed}';rr=[r for r in tr if r['method']==name];assert len(rr)==2378;metrics[name]=summarize(rr)
            if arm=='GATE':
                for k,v in metrics[name]['macro'].items():assert abs(v-ref[name]['macro'][k])<1e-12
    family={arm:{k:float(np.mean([metrics[f'{arm}_seed{s}']['macro'][k] for s in range(3)])) for k in metrics[f'{arm}_seed0']['macro']} for arm in ['GATE',*ARMS]};paired=[]
    for seed in range(3):
        old,new=metrics[f'FIT8_seed{seed}'],metrics[f'CAL4_seed{seed}'];paired.append(dict(seed=seed,macro_delta={k:new['macro'][k]-old['macro'][k] for k in old['macro']},per_date=[dict(date=x['date'],NET_delta=y['mean_net_utility']-x['mean_net_utility'],query_delta=y['mean_queries']-x['mean_queries'],gain_delta=y['mean_gain']-x['mean_gain']) for x,y in zip(old['per_date'],new['per_date'])]))
    counts={arm:dict(base_appends=sum(r['base_continue'] for r in pred if r['arm']==arm),appends=sum(r['continue_'] for r in pred if r['arm']==arm),profitable_retained=sum(r['continue_'] and r['actual_next_NET']>0 for r in pred if r['arm']==arm),profitable_removed=sum(r['base_continue'] and not r['continue_'] and r['actual_next_NET']>0 for r in pred if r['arm']==arm),unprofitable_removed=sum(r['base_continue'] and not r['continue_'] and r['actual_next_NET']<=0 for r in pred if r['arm']==arm),passthrough=sum(r['passthrough'] for r in pred if r['arm']==arm)) for arm in ARMS}
    for n,v in [('oof_trajectories.json',tr),('gate_predictions.json',pred),('search_metrics.json',metrics),('family_summary.json',family),('paired_summary.json',paired),('gate_counts.json',counts)]:save(n,v)
    scored();save('review.json',dict(passed=True,all21402saved_trajectories_checked=True,all9362published_predictions_joined=True,originalGATEcontrols_reproduced=True,all3seeds16exposeddates=True,no_new_physics=True,final8_absent=True));archive('evaluated');print('GRID239_EVALUATED',json.dumps(dict(family=family,counts=counts,paired=[dict(seed=r['seed'],NET=r['macro_delta']['mean_net_utility'],queries=r['macro_delta']['mean_queries'],gain=r['macro_delta']['mean_gain']) for r in paired])))
if __name__=='__main__':globals()[sys.argv[1]]()
