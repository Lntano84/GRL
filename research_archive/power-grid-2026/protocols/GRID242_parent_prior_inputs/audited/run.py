"""Public parent-prior alignment, then separate fit8 and union12 constraints."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,pickle,shutil,subprocess,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid242_parent_prior_inputs';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID242_parent_prior_inputs'
PLAN=R/'outputs/grid241_parent_prior_plan';H=R/'outputs/grid214_inner_head_generation';FULL=R/'outputs/grid192_gate_support';META=R/'outputs/grid113_relational_learning/train_metadata.json';ACCESS=[]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def fields(p,names):
    z=np.load(p);out={n:z[n].copy() for n in names};ACCESS.append(dict(path=p.relative_to(R).as_posix(),fields=list(names)));return out
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
    for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
    for p,h in read(PLAN/'source_freeze.json').items():assert sha(R/p)==h,p
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    for folder in {'generated':['public_full','fit_public','calibration_public','outer_parent_public'],'qualified_fit':['fit_inputs','calibration_inputs']}.get(phase,[]):shutil.copytree(O/folder,d/folder)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def prepare():
    assert not O.exists() and read(PLAN/'qualification.json')['passed'];O.mkdir()
    save('design.json',dict(id='GRID242',candidate=read(PLAN/'design.json'),scope='Publicparentpriorregenerationandfitqualificationonly,0fits/NN/physics;outerparentreferencejustreproducesalready-published208prior,no newcandidatepolicy.',
        public='208parent124scalers and24frozenheads produceb12 forfullown12. Residualpublic41still213inner8scaler+b12. Freshb12>0eligibility;fit8training,outer12selectedoriginalSTOPunioncalibration.',
        barriers='Sourcefirstpublish;allparentpublicexportsandouter208priorreproductionpublishbeforetargetjoin. Thenonlyfit8forrisk;fit8+cal4uniontargetsseparateempiricalconstraintfile. No newalpha untilfuture243.',
        fallback='f0_s1orfit8STOPpositive0oremptydomain=>KEEP,nofoldchange. Future<=11linear+11scalar/600s. No newdatasets/final8/paper/FRAME.'))
    paths=[W/'run.py',O/'design.json',PLAN/'design.json',PLAN/'qualification.json',PLAN/'qualified_pools.json',PLAN/'source_freeze.json',META,H/'public_raw41.npz',H/'bundles.json']
    for r in read(PLAN/'qualified_pools.json'):paths += [R/r['parent_scaler'],R/r['residual_public41_scaler'],FULL/'fit_features'/f"{r['pool']}.npz",R/'outputs/grid190_extension_gate'/f"held_{r['pool']}.npz",R/'outputs/grid220_full_prior/public_held_inputs'/f"{r['pool']}.npz"]+[R/p for p in r['parent_heads']]
    save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in set(paths)});verify();archive('qualified');print('GRID242_QUALIFIED')
def generate():
    verify();assert not (O/'generation_started.json').exists();save('generation_started.json',dict(new_fits=0,reused_parentheads=24,public_only=True));raw=fields(H/'public_raw41.npz',['values'])['values'];meta=read(META);records=[]
    for folder in ['public_full','fit_public','calibration_public','outer_parent_public']:(O/folder).mkdir()
    for r in read(PLAN/'qualified_pools.json'):
        pool=r['pool'];full=fields(FULL/'fit_features'/f'{pool}.npz',['x','ids','original_STOP_count']);p=fields(R/r['parent_scaler'],['mean','scale','raw_mean','raw_scale']);q=fields(R/r['residual_public41_scaler'],['mean41','scale41','fit_positions','fit_STOP_mask']);ids=full['ids'].astype(int);stop=np.arange(len(ids))<int(full['original_STOP_count']);assert np.array_equal(stop[q['fit_positions']],q['fit_STOP_mask']);fit=np.array([meta[int(i)]['date'] in r['fit_dates'] for i in ids]);cal=np.array([meta[int(i)]['date'] in r['cal4_dates'] for i in ids]);assert np.all(fit^cal)
        x124=np.c_[(full['x']-p['mean'])/p['scale'],(raw[ids]-p['raw_mean'])/p['raw_scale']];epath,apath=[R/p for p in r['parent_heads']];before=[sha(epath),sha(apath)];event=pickle.loads(epath.read_bytes());amount=pickle.loads(apath.read_bytes());eventp=event.predict_proba(x124)[:,1];mu=amount.predict(x124);charge=full['x'][:,82];assert (charge>0).all() and (mu>0).all();b=(eventp*mu-charge)/(eventp*mu+charge);cont=b>0;x42=np.c_[(raw[ids]-q['mean41'])/q['scale41'],b];assert np.isfinite(x42).all()
        np.savez_compressed(O/'public_full'/f'{pool}.npz',ids=ids,X=x42,b=b,event_p=eventp,mu=mu,charge=charge,head_CONT=cont,fit_role=fit,cal_role=cal,original_STOP=stop)
        for folder,mask in [('fit_public',fit&cont),('calibration_public',stop&cont)]:np.savez_compressed(O/folder/f'{pool}.npz',ids=ids[mask],X=x42[mask],b=b[mask],original_STOP=stop[mask],fit_role=fit[mask])
        h=fields(R/'outputs/grid190_extension_gate'/f'held_{pool}.npz',['ids','features']);old=fields(R/'outputs/grid220_full_prior/public_held_inputs'/f'{pool}.npz',['ids','b','base_CONT']);assert np.array_equal(h['ids'],old['ids']) and {meta[int(i)]['date'] for i in h['ids']}<=set(r['outer_dates']);hx=np.c_[(h['features']-p['mean'])/p['scale'],(raw[h['ids'].astype(int)]-p['raw_mean'])/p['raw_scale']];hg=event.predict_proba(hx)[:,1]*amount.predict(hx);hc=h['features'][:,82];hb=(hg-hc)/(hg+hc);err=float(abs(hb-old['b']).max());assert np.allclose(hb,old['b'],atol=1e-12,rtol=1e-12) and np.array_equal(hb>0,old['base_CONT']);assert before==[sha(epath),sha(apath)]
        # Reuse the published parent b bit-for-bit for the future correction input.
        outerx=np.c_[(raw[h['ids'].astype(int)]-q['mean41'])/q['scale41'],old['b']];np.savez_compressed(O/'outer_parent_public'/f'{pool}.npz',ids=h['ids'],X=outerx,b=old['b'],base_CONT=old['base_CONT']);records.append(dict(pool=pool,source_rows=len(ids),fit8_selected=int((fit&cont).sum()),fit8_STOP=int((fit&cont&stop).sum()),fit8_CONT=int((fit&cont&~stop).sum()),union12_STOP=int((stop&cont).sum()),outer_parent_rows=len(h['ids']),old208_parent_b_max_abs_error=err,old208_entry_reproduced=True,parent_weights_unchanged=True))
    assert not any({'y','actual_NET','original_source_mass','event_y'}&set(r['fields']) for r in ACCESS);save('public_access.json',ACCESS);save('generation_receipts.json',records);save('generated_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'public_full').iterdir()),*list((O/'fit_public').iterdir()),*list((O/'calibration_public').iterdir()),*list((O/'outer_parent_public').iterdir()),O/'generation_receipts.json',O/'public_access.json']});verify();archive('generated');print('GRID242_PUBLIC_GENERATED',json.dumps(dict(pools=12,rows=sum(r['source_rows'] for r in records),max_parent_error=max(r['old208_parent_b_max_abs_error'] for r in records))))
def generated():
    verify()
    for p,h in read(O/'generated_freeze.json').items():assert sha(O/p)==h,p
def publication():
    generated();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'generated/generated_freeze.json').relative_to(REPO).as_posix())==(O/'generated_freeze.json').read_bytes();save('pre_target_publication.json',dict(passed=True,git_sha=head,all48publicexports_published=True));print('GRID242_PUBLIC_PUBLISHED')
def qualify_fit():
    generated();assert read(O/'pre_target_publication.json')['passed'] and not (O/'target_join_started.json').exists();save('target_join_started.json',dict(public_exports_published=True));bundles=read(H/'bundles.json');records=[]
    for folder in ['fit_inputs','calibration_inputs']:(O/folder).mkdir()
    for r in read(PLAN/'qualified_pools.json'):
        pool=r['pool'];f,s=[int(v[1:]) for v in pool.split('_')];targets={}
        for source in bundles:
            if source['outer_fold']!=f or source['seed']!=s:continue
            assert set(source['score_dates'])<=set(r['fit_dates'])|set(r['cal4_dates']);z=fields(H/'labelled_samples'/f"{source['bundle']}.npz",['meta_ids','actual_NET','original_source_mass','original_STOP_mask'])
            for i,n,m,stop in zip(z['meta_ids'],z['actual_NET'],z['original_source_mass'],z['original_STOP_mask']):assert int(i) not in targets;targets[int(i)]=(float(n),float(m),bool(stop))
        z=np.load(O/'fit_public'/f'{pool}.npz');v=[targets[int(i)] for i in z['ids']];net=np.array([t[0] for t in v]);mass=np.array([t[1] for t in v]);stop=z['original_STOP'];assert np.array_equal(stop,np.array([t[2] for t in v]));pos=net>0;fallback=pool=='f0_s1' or not np.any(pos&stop) or not stop.any() or not (~stop).any();weight=np.zeros(len(net))
        for mask in [stop,~stop]:
            risk=mass[mask]*abs(net[mask]);total=float(risk.sum())
            if total>0:weight[mask]=.5*risk/total
        if not fallback:assert (weight>0).all() and abs(weight[stop].sum()-.5)<1e-12 and abs(weight[~stop].sum()-.5)<1e-12;weight*=int(stop.sum())
        np.savez_compressed(O/'fit_inputs'/f'{pool}.npz',X=z['X'],b=z['b'],ids=z['ids'],y=pos.astype(int),weight=weight,actual_NET=net,domain_STOP=stop,fallback_KEEP=np.array([fallback]));c=np.load(O/'calibration_public'/f'{pool}.npz');cn=np.array([targets[int(i)][0] for i in c['ids']]);assert all(targets[int(i)][2] for i in c['ids']);np.savez_compressed(O/'calibration_inputs'/f'{pool}.npz',X=c['X'],b=c['b'],ids=c['ids'],y=(cn>0).astype(int),actual_NET=cn,fit_role=c['fit_role']);assert fallback or np.any(cn>0)
        records.append(dict(pool=pool,fit8_rows=len(net),fit8_STOP=int(stop.sum()),fit8_STOP_positive=int((stop&pos).sum()),fit8_CONT=int((~stop).sum()),fit8_CONT_positive=int((~stop&pos).sum()),union12_STOP=len(cn),union12_STOP_positive=int((cn>0).sum()),fallback_KEEP=fallback,new_linear_fit_planned=not fallback,new_scalar_fit_planned=not fallback))
    active=sum(r['new_linear_fit_planned'] for r in records);assert active<=11;save('target_access.json',ACCESS);save('fit_qualification.json',dict(passed=True,pools=records,linear_fits_planned=active,scalar_fits_planned=active,total_fits_planned=2*active,fit8_only_risk_normalization=True,union12_empirical_STOP_constraints_explicit=True));save('qualified_fit_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_inputs').iterdir()),*list((O/'calibration_inputs').iterdir()),O/'fit_qualification.json',O/'target_access.json']});generated();archive('qualified_fit');print('GRID242_FIT_QUALIFIED',json.dumps(read(O/'fit_qualification.json')))
if __name__=='__main__':globals()[sys.argv[1]]()
