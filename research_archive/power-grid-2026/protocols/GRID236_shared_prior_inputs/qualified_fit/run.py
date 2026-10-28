"""Reuse fixed eight-date heads to export same-source priors without outcomes."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,pickle,shutil,subprocess,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid236_shared_prior_inputs';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID236_shared_prior_inputs'
PLAN=R/'outputs/grid235_calibration_ancestry';H=R/'outputs/grid214_inner_head_generation';G=R/'outputs/grid213_inner_head_plan';FULL=R/'outputs/grid192_gate_support';META=R/'outputs/grid113_relational_learning/train_metadata.json';ACCESS=[]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def fields(path,allowed):
    z=np.load(path);out={k:z[k].copy() for k in allowed};ACCESS.append(dict(path=path.relative_to(R).as_posix(),fields=list(allowed)));return out
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    for p,h in read(PLAN/'source_freeze.json').items():assert sha(R/p)==h,p
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    for folder in {'generated':['public_full','fit_public','cal_public'],'qualified_fit':['fit_inputs']}.get(phase,[]):shutil.copytree(O/folder,d/folder)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def prepare():
    assert not O.exists() and read(PLAN/'qualification.json')['passed'];O.mkdir()
    save('design.json',dict(id='GRID236',scope='Publicregenerationthenfit8-onlytargetqualification. Zero model/scalarfits,NNforwards,physicalqueries. Conditionalupstreamouter12frozen.',
        heads='Reuse24publishedinner0event/amount_depth4forests and213inner8scalers;onepairperpoolforALLown12dates. Public42=inner8scaledraw41+b;charge=raw83col82. No newforestfitting.',
        roles='Fixed235inner0fit8/cal4,neverchoosebasedonsupport. OriginalSTOPmaskfrom192prefixcountvalidated213fitmask. Freshb>0selectsfit/calrecords;doNOTreuse224eligibility.',
        barriers='Codeinputsfirstpublish;verify24existingweightsat214pre_scorepublication. Allpublicpredictionspublishbeforeanytargetparse. ThenloadONLYold214inner1/2labelledsamples whose scoredateswithinfit8. Neverparseinner0cal4targetarrayshere.237cal4residualscoresmustpublishbeforecal4targetjoin.',
        fitweights='For each selectedfit8domain independentlynormalize(original_source_mass*absNET),mix0.5STOP+0.5CONT. Onlyfit8targets. f0_s1orzeroSTOPpositiveoremptydomain=>fixedKEEP,nofoldchange.',
        subsequent='Atmost11new43parameterresiduals,0scalarfits,one600s modelwindow. No newouter4scoring/policycomparison. Conditionalresidualerrorcollection,not independentOOF/blind/fullpipelinevalidation. No final8/physics/paper/FRAME.'))
    pools=read(PLAN/'fixed_probe_pools.json');paths=[W/'run.py',O/'design.json',PLAN/'fixed_probe_pools.json',PLAN/'qualification.json',PLAN/'source_freeze.json',PLAN/'count_correction.json',META,H/'public_raw41.npz',H/'bundles.json',H/'trained_freeze.json',H/'pre_score_publication.json']
    pub=read(H/'pre_score_publication.json');assert pub['all108weights_published'];checked=[]
    for r in pools:
        for model in r['head_models']:
            p=R/model;blob=git('show',pub['git_sha']+':research_archive/power-grid-2026/protocols/GRID214_inner_head_generation/trained/models/'+p.name);assert hashlib.sha256(blob).hexdigest()==sha(p);checked.append(dict(path=p.relative_to(R).as_posix(),sha256=sha(p)));paths.append(p)
        paths += [R/r['scaler_file'],FULL/'fit_features'/f"{r['pool']}.npz",H/'predictions'/f"{r['bundle']}.npz"]
    assert len(checked)==24;save('existing_head_publication.json',dict(passed=True,git_sha=pub['git_sha'],weights=checked));save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('qualified');print('GRID236_QUALIFIED',len(checked))
def generate():
    verify();assert not (O/'generation_started.json').exists();save('generation_started.json',dict(new_fits=0,reused_heads=24,public_only=True));pools=read(PLAN/'fixed_probe_pools.json');meta=read(META);raw=fields(H/'public_raw41.npz',['values'])['values'];records=[]
    for folder in ['public_full','fit_public','cal_public']:(O/folder).mkdir()
    for r in pools:
        pool,bundle=r['pool'],r['bundle'];src=fields(FULL/'fit_features'/f'{pool}.npz',['x','ids','original_STOP_count']);q=fields(R/r['scaler_file'],['mean83','scale83','mean41','scale41','fit_positions','score_positions','fit_meta_ids','score_meta_ids','fit_STOP_mask','score_X_sha256']);ids=src['ids'].astype(int);stop=np.arange(len(ids))<int(src['original_STOP_count']);assert np.array_equal(stop[q['fit_positions']],q['fit_STOP_mask']);assert np.array_equal(ids[q['fit_positions']],q['fit_meta_ids']) and np.array_equal(ids[q['score_positions']],q['score_meta_ids'])
        dates=[meta[int(i)]['date'] for i in ids];fit=np.array([d in r['fit_dates'] for d in dates]);cal=np.array([d in r['calibration_dates'] for d in dates]);assert np.all(fit^cal) and not set(dates)&set(r['outer_dates']);x124=np.c_[(src['x']-q['mean83'])/q['scale83'],(raw[ids]-q['mean41'])/q['scale41']];assert hashlib.sha256(x124[q['score_positions']].tobytes()).hexdigest()==str(q['score_X_sha256'][0]);charge=src['x'][:,82];assert (charge>0).all()
        event_path,amount_path=[R/p for p in r['head_models']];before=[sha(event_path),sha(amount_path)];event=pickle.loads(event_path.read_bytes());amount=pickle.loads(amount_path.read_bytes());assert np.array_equal(event.classes_,[0,1]);p=event.predict_proba(x124)[:,1];mu=amount.predict(x124);assert (mu>0).all() and np.isfinite(p).all();expected=fields(H/'predictions'/f'{bundle}.npz',['meta_ids','event_p','mu4','public_charge']);order=q['score_positions'];assert np.array_equal(ids[order],expected['meta_ids']) and np.allclose(p[order],expected['event_p'],atol=1e-12,rtol=1e-12) and np.allclose(mu[order],expected['mu4'],atol=1e-12,rtol=1e-12) and np.array_equal(charge[order],expected['public_charge']);assert before==[sha(event_path),sha(amount_path)]
        b=(p*mu-charge)/(p*mu+charge);head_CONT=b>0;x42=np.c_[x124[:,83:],b];assert x42.shape==(len(ids),42) and np.isfinite(x42).all();np.savez_compressed(O/'public_full'/f'{pool}.npz',ids=ids,X=x42,b=b,event_p=p,mu4=mu,charge=charge,head_CONT=head_CONT,fit_role=fit,cal_role=cal,original_STOP=stop)
        for folder,mask in [('fit_public',fit&head_CONT),('cal_public',cal&head_CONT)]:np.savez_compressed(O/folder/f'{pool}.npz',ids=ids[mask],X=x42[mask],b=b[mask],charge=charge[mask],original_STOP=stop[mask])
        records.append(dict(pool=pool,bundle=bundle,source_rows=len(ids),fit8_selected=int((fit&head_CONT).sum()),fit8_STOP=int((fit&head_CONT&stop).sum()),fit8_CONT=int((fit&head_CONT&~stop).sum()),cal4_selected=int((cal&head_CONT).sum()),cal4_STOP=int((cal&head_CONT&stop).sum()),cal4_CONT=int((cal&head_CONT&~stop).sum()),oldcalpublicprediction_reproduced=True,headweights_unchanged=True))
    forbidden={'y','actual_NET','gain_normalized','event_y','original_source_mass'};assert not any(forbidden&set(a['fields']) for a in ACCESS);save('public_access.json',ACCESS);save('generation_receipts.json',records);save('generated_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'public_full').iterdir()),*list((O/'fit_public').iterdir()),*list((O/'cal_public').iterdir()),O/'generation_receipts.json',O/'public_access.json']});verify();archive('generated');print('GRID236_PUBLIC_GENERATED',json.dumps(dict(pools=12,rows=sum(r['source_rows'] for r in records),cal_rows=sum(r['cal4_selected'] for r in records),new_fits=0)))
def generated():
    verify()
    for p,h in read(O/'generated_freeze.json').items():assert sha(O/p)==h,p
def publication():
    generated();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'generated/generated_freeze.json').relative_to(REPO).as_posix())==(O/'generated_freeze.json').read_bytes();save('pre_fit_label_publication.json',dict(passed=True,git_sha=head,all36publicfiles_published=True));print('GRID236_PUBLIC_PUBLISHED')
def qualify_fit():
    generated();assert read(O/'pre_fit_label_publication.json')['passed'] and not (O/'fit_label_join_started.json').exists();save('fit_label_join_started.json',dict(public_predictions_published=True,cal4_target_parse_forbidden=True));pools=read(PLAN/'fixed_probe_pools.json');bundles=read(H/'bundles.json');records=[];(O/'fit_inputs').mkdir();future=[]
    for r in pools:
        pool=r['pool'];f,s=[int(v[1:]) for v in pool.split('_')];targets={}
        for record in bundles:
            if record['outer_fold']!=f or record['seed']!=s or record['inner_fold']==0:continue
            assert set(record['score_dates'])<=set(r['fit_dates']) and not set(record['score_dates'])&set(r['calibration_dates']);z=fields(H/'labelled_samples'/f"{record['bundle']}.npz",['meta_ids','actual_NET','original_source_mass','original_STOP_mask'])
            for i,net,mass,stop in zip(z['meta_ids'],z['actual_NET'],z['original_source_mass'],z['original_STOP_mask']):assert int(i) not in targets;targets[int(i)]=(float(net),float(mass),bool(stop))
        z=np.load(O/'fit_public'/f'{pool}.npz');ids=z['ids'].astype(int);v=[targets[i] for i in ids];net=np.array([t[0] for t in v]);mass=np.array([t[1] for t in v]);stop=z['original_STOP'];assert np.array_equal(stop,np.array([t[2] for t in v]));pos=net>0;fallback=pool=='f0_s1' or not np.any(stop&pos) or not stop.any() or not (~stop).any();weight=np.zeros(len(ids));risk=[]
        for mask in [stop,~stop]:
            raw=mass[mask]*abs(net[mask]);total=float(raw.sum());risk.append(total)
            if total>0:weight[mask]=.5*raw/total
        if not fallback:assert np.all(weight>0) and abs(weight[stop].sum()-.5)<1e-12 and abs(weight[~stop].sum()-.5)<1e-12;weight*=int(stop.sum())
        np.savez_compressed(O/'fit_inputs'/f'{pool}.npz',X=z['X'],b=z['b'],ids=ids,y=pos.astype(int),weight=weight,actual_NET=net,domain_STOP=stop,fallback_KEEP=np.array([fallback]));records.append(dict(pool=pool,fit_rows=len(ids),STOP_rows=int(stop.sum()),STOP_positive=int((stop&pos).sum()),CONT_rows=int((~stop).sum()),CONT_positive=int((~stop&pos).sum()),fallback_KEEP=fallback,new_linear_fit_planned=not fallback,fit_only_domain_risk_totals=risk,cal4_target_rows_read=0));future.append(dict(pool=pool,cal4_target_source=(H/'labelled_samples'/f"{r['bundle']}.npz").relative_to(R).as_posix(),may_parse_only_after_237_public_scores_published=True))
    assert sum(r['new_linear_fit_planned'] for r in records)<=11;assert not any('f0_i0_' in a['path'] and 'actual_NET' in a['fields'] for a in ACCESS);save('fit_label_access.json',ACCESS);save('fit_qualification.json',dict(passed=True,pools=records,new_linear_fits_planned=sum(r['new_linear_fit_planned'] for r in records),new_scalar_fits_planned=0,cal4_target_parses=0,only_fixedfit8_targets_used=True));save('future_cal_target_sources.json',future);save('qualified_fit_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_inputs').iterdir()),O/'fit_qualification.json',O/'future_cal_target_sources.json',O/'fit_label_access.json']});generated();archive('qualified_fit');print('GRID236_FIT_QUALIFIED',json.dumps(read(O/'fit_qualification.json')))
if __name__=='__main__':globals()[sys.argv[1]]()
