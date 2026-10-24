"""One frozen fit for a single learned exit at128; saved feedback only."""
from pathlib import Path
import collections, datetime, gzip, hashlib, importlib.util, json, os, pickle, shutil, sys, time
sys.path.insert(0, 'D:/Codex/GRL研究/work/grid144_class_mass')
import support as S
import numpy as np
B = S.B
R = B.R
W = Path(__file__).parent
O = R/'outputs/grid146_single_entry'
REPO = B.REPO
A = REPO/'research_archive/power-grid-2026/protocols/GRID146_single_entry'
PREV = S.O

def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save(n,x): (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def gzsave(n,x):
    with gzip.open(O/n,'wt',encoding='utf8',newline='\n') as f:json.dump(x,f,ensure_ascii=False)
def rowfile(p): return S.rowfile(p)
def entry_rows(rows): return [r for r in rows if r['meta']['visible_count']==128]
def scoremap(rows,p): return {(r['meta']['call_id'],128):float(v) for r,v in zip(rows,p)}
def prior_verify():
    S.verify_trained()
    S.diagnostic_api().verify()
    p=R/'outputs/grid145_risk_feasibility'
    assert read(p/'complete.json')['passed'] and read(p/'final_verification.json')['passed']
    for phase in (REPO/'research_archive/power-grid-2026/protocols/GRID145_risk_feasibility').iterdir():
        for n,h in read(phase/'archive_manifest.json').items():assert sha(phase/n)==h
def verify():
    prior_verify()
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
def verify_trained():
    verify()
    for n,h in read(O/'trained_freeze.json').items():assert sha(O/n)==h,n
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def weights(rows):
    counts=collections.Counter(r['meta']['date'] for r in rows)
    assert len(counts)==12
    return np.asarray([len(rows)/(len(counts)*counts[r['meta']['date']]) for r in rows],float)

def replay(ref,tau,sm):
    """One gate only. Continue executes original global loop with zero later gates."""
    c=B.call_at(ref['source']);order=c['order'];m=len(order)
    n=min(128,m);initial=B.P.best(c['outcomes'][:n]);safe=initial is not None and initial['rho']<.9
    b=initial;stale=0;visited=[];kind='exhaustion';stopped=None
    while n<m:
        forced=(not safe and b is not None and b['rho']<.9) or (b is not None and n>=256 and stale>=2)
        if forced:kind='original_forced_stop';break
        if n==128 and b is not None:
            p=sm[ref['call_id'],128]
            visited.append(dict(visible_count=128,score=p,stop=p>tau))
            if p>tau:kind='learned_early_stop';stopped=p;break
        old=b;n=min(n+64,m);b=B.P.best(c['outcomes'][:n]);stale=0 if B.P.improves(old,b) else stale+1
    extra=[]
    if b is not None and n<m and (not safe or B.P.bid(b)!=B.P.bid(initial)):
        station=B.subids()[b['pool_id']]
        ids={int(rep) for alias,rep in c['aliases'].items() if B.subids()[int(alias)]==station}
        extra=[r for r in c['outcomes'][n:] if r['pool_id'] in ids]
        b=B.P.best(c['outcomes'][:n]+extra)
    paid=order[:n]+[r['pool_id'] for r in extra];assert len(paid)==len(set(paid))
    q=len(paid);original=ref['global_n']+ref['local_queries'];assert len(visited)<=1
    return dict(call_id=ref['call_id'],partition=ref['partition'],date=ref['date'],scenario=ref['scenario'],
        global_queries=n,local_queries=len(extra),queries=q,original_queries=original,selected=B.P.bid(b),
        original_selected=ref['frame_choice'],action_match=B.P.bid(b)==ref['frame_choice'],stop_kind=kind,
        stopped_score=stopped,model_decisions=visited,model_decision_count=len(visited),
        bad_learned_stop=kind=='learned_early_stop' and (B.P.bid(b)!=ref['frame_choice'] or q>=original))

def shadow_check(ref,tau,sm):
    """Independent old loop, disabling later exits arithmetically. No model prediction."""
    c=B.call_at(ref['source'])
    shadow={(ref['call_id'],n):(sm.get((ref['call_id'],128),-1.) if n==128 else -1.) for n in range(128,len(c['order']),64)}
    a=replay(ref,tau,sm);b=B.replay(ref,'MODEL',tau,shadow)
    assert all(a[k]==b[k] for k in ['selected','global_queries','local_queries','queries','action_match','bad_learned_stop','stop_kind'])
    assert a['model_decisions']==[v for v in b['model_decisions'] if v['visible_count']==128]
    return a

def calibrate(refs,sm):
    tau=.5;history=[];initial=None
    for _ in range(3):
        rr=[replay(r,tau,sm) for r in refs]
        if initial is None:initial=rr
        bad=[r for r in rr if r['bad_learned_stop']]
        history.append(dict(threshold=tau,summary=B.summarize(rr),bad_stops=[dict(call_id=r['call_id'],score=r['stopped_score'],selected=r['selected'],original=r['original_selected']) for r in bad]))
        if not bad:return tau,history,initial,rr
        nxt=max(r['stopped_score'] for r in bad);assert nxt>tau;tau=nxt
    raise AssertionError('Single-entry empirical calibration should need at most two passes;never refit')

def metrics(rows,p,tau):
    from sklearn.metrics import average_precision_score,roc_auc_score
    y=np.asarray([r['target']['stop_label'] for r in rows],bool);p=np.asarray(p);stop=p>tau
    return dict(prefixes=len(rows),calls=len({r['meta']['call_id'] for r in rows}),dates=len({r['meta']['date'] for r in rows}),
        continue_count=int((~y).sum()),stop_count=int(y.sum()),continue_prevalence=float((~y).mean()),
        continue_ap=float(average_precision_score(~y,1-p)) if (~y).any() else None,
        auc=float(roc_auc_score(y,p)) if y.any() and (~y).any() else None,
        continue_recall=float(np.mean(~stop[~y])) if (~y).any() else None,
        valid_stop_recall=float(np.mean(stop[y])) if y.any() else None,
        wrong_stop_proposals=int(np.sum(stop&~y)),valid_stop_proposals=int(np.sum(stop&y)),threshold=tau)

def prepare():
    assert not O.exists();O.mkdir(parents=True);prior_verify()
    d=read(B.O/'design.json');fit=entry_rows(rowfile(B.O/'fit_rows.json.gz'));cal=entry_rows(rowfile(B.O/'calibration_rows.json.gz'))
    assert len(fit)==153 and sum(not r['target']['stop_label'] for r in fit)==35 and len(cal)==66
    assert {r['meta']['date'] for r in fit}==set(d['fit_dates'])
    w=weights(fit);assert abs(w.sum()-153)<1e-10
    design=dict(id='GRID146_SINGLE_ENTRY_128',utc=stamp(),authorization='User continues after GRID145 proposed one entry;notify immediately before the sole fit.',
        task='Exactly one learned decision at the original128 checkpoint if eligible. STOP executes unchanged completion;CONTINUE delegates to originalFRAME remainder with no later learned stops.',
        fit_dates=d['fit_dates'],calibration_dates=d['calibration_dates'],validation_dates=d['validation_dates'],
        fit_prefixes=len(fit),fit_continue=35,calibration_prefixes=len(cal),features=d['features'],model_parameters=d['model_parameters'],
        fit_weights='From eligible fit128 rows only,w_i=N/(D*n_date). Equal total date mass;one row per call. No class balancing;no coefficient search. Same date-equal principle asRF140,recomputed after population change.',
        preprocessing='Original18 features +18missing indicators;each median re-estimated on153fit128 rows only,as in original fit-only procedure. Not claim same numerical medians.',
        changed='Learning exit structure and fitting population. Distinguish with cachedRF140 scores at128 under the same one-entry gate and independently pre-validation calibrated cutoff. Not a one-factor parameter intervention.',
        fixed='Original labels,reference action,float32reward/tie/order/aliases,p128,batch64,patience2,rho.9,conditional completion. No model at later checkpoints.',
        calibration='Same empirical zero-joint-failure rule from0.5,strict> threshold,raise to highest bad first-entry score and recheck full calls. Not formalCRC/LTT or future-zero-risk proof.',
        diagnostic_control='OLD140_ENTRY128_FIXED_0.5 and OLD140_ENTRY128_CALIBRATED reuse old scores/model;no refit. Both fixed before new validation scoring. No alternate cutoff selected.',
        primary='RF146_ENTRY128_CALIBRATED',secondary='RF146_ENTRY128_FIXED_0.5 retained regardless of outcome',
        fit_limit=1,fit_wall_cap_s=120,
        screen='All62dev searches preserve original actions,no non-saving learned exits;date-macro query reduction strictly exceedsCAP256,each date queries no more thanCAP256 and at least one strictly fewer. No automatic adoption/physics/scale even if passed.',
        scope='Already exposed same-grid FULL-teacher states;not fresh independent validation,actualFRAME coverage,weekly economics or measured control time.',
        prohibited=['second fit','parameter/classweight/entry/threshold grid','new label or risk tolerance','validation-selected policy','new simulation/scenario','final8tests','RL/NN','paper edit','automatic scale'])
    save('design.json',design);gzsave('fit_rows.json.gz',fit);gzsave('calibration_rows.json.gz',cal)
    save('planned_fit_weights.json',[dict(sample_id=r['meta']['sample_id'],date=r['meta']['date'],call_id=r['meta']['call_id'],weight=float(v)) for r,v in zip(fit,w)])
    allrows=rowfile(B.O/'fit_rows.json.gz')+rowfile(B.O/'calibration_rows.json.gz')
    assert S.public_checks(fit+cal)==219
    dummy={(r['meta']['call_id'],128):.75 for r in fit+cal}
    checks=0
    for ref in B.refs('train',set(d['fit_dates']+d['calibration_dates'])):
        for tau in [.5,1.]:
            a=shadow_check(ref,tau,dummy);checks+=1
            if tau==1.:assert a['action_match'] and a['queries']==a['original_queries']
    save('preflight.json',dict(passed=True,public_suffix_checks=219,shadow_search_checks=checks,fit_only_dates=True,features=36,
        fit_rows=153,calibration_rows=66,validation_not_scored=True,one_entry_fallback_equals_frame=True))
    sources=[O/n for n in ['design.json','fit_rows.json.gz','calibration_rows.json.gz','planned_fit_weights.json']]
    sources += [B.O/n for n in ['design.json','fit_rows.json.gz','calibration_rows.json.gz','train_scores.npz','validation_scores.npz','model_export.json','model.pkl','cutoff.json']]
    sources += [PREV/n for n in ['all_search_replays.json','summary.json','trained_freeze.json']]
    sources += [R/'outputs/grid145_risk_feasibility'/n for n in ['单入口学习方案.md','risk_support_by_depth.json','review.json','decision.json','final_verification.json']]
    sources += [R/n for n in ['work/grid140_global_compression/common.py','work/grid144_class_mass/support.py']]
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sources})
    save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')})
    verify();save('state.json',dict(stage='frozen_before_one_fit'));archive('frozen')
    print(json.dumps(dict(frozen=True,preflight=read(O/'preflight.json'),fit_date_mass=153/12,fit_stop_mass=float(w[[r['target']['stop_label'] for r in fit]].sum()),fit_continue_mass=float(w[[not r['target']['stop_label'] for r in fit]].sum())),ensure_ascii=False))

def fit():
    verify();assert not (O/'fit_started.json').exists()
    d=read(O/'design.json');rows=rowfile(O/'fit_rows.json.gz');cal=rowfile(O/'calibration_rows.json.gz');names=d['features']
    raw=np.asarray([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in names] for r in rows])
    medians=[float(np.median(raw[np.isfinite(raw[:,j]),j])) if np.isfinite(raw[:,j]).any() else 0. for j in range(len(names))]
    x=B.transform(rows,names,medians);xc=B.transform(cal,names,medians);w=weights(rows)
    assert np.array_equal(w,np.asarray([r['weight'] for r in read(O/'planned_fit_weights.json')]))
    save('preprocessing.json',dict(features=names,medians=medians,append_missing_flags=True,dimensions=36,fit_rows=153))
    save('fit_sample_weights.json',read(O/'planned_fit_weights.json'))
    from sklearn.ensemble import RandomForestClassifier
    import sklearn
    est=RandomForestClassifier(**d['model_parameters']);y=np.asarray([r['target']['stop_label'] for r in rows],int)
    save('fit_started.json',dict(utc=stamp(),fits_permitted=1));t=time.perf_counter();est.fit(x,y,sample_weight=w);elapsed=time.perf_counter()-t
    with (O/'model.pkl').open('wb') as f:pickle.dump(est,f,protocol=5)
    save('model_export.json',B.export_model(est))
    save('fit_receipt.json',dict(new_fits=1,old_fits=0,rows=153,classes=dict(collections.Counter(int(z) for z in y)),elapsed_s=elapsed,
        trees=64,input_dimensions=36,numpy=np.__version__,sklearn=sklearn.__version__,validation_scored=False))
    assert elapsed<=d['fit_wall_cap_s']
    pf=est.predict_proba(x)[:,list(est.classes_).index(1)];pc=est.predict_proba(xc)[:,list(est.classes_).index(1)]
    assert max(abs(B.exported_scores(read(O/'model_export.json'),x)-pf))<1e-10
    assert max(abs(B.exported_scores(read(O/'model_export.json'),xc)-pc))<1e-10
    np.savez_compressed(O/'train_scores.npz',fit=pf,calibration=pc)
    oldfit=rowfile(B.O/'fit_rows.json.gz');oldcal=rowfile(B.O/'calibration_rows.json.gz');oldz=np.load(B.O/'train_scores.npz')
    oldmap={(r['meta']['call_id'],128):float(v) for r,v in zip(oldfit+oldcal,np.concatenate([oldz['fit'],oldz['calibration']])) if r['meta']['visible_count']==128}
    refs=B.refs('train',set(d['calibration_dates']))
    tau,h,initial,final=calibrate(refs,scoremap(cal,pc))
    oldtau,oldh,oldinitial,oldfinal=calibrate(refs,oldmap)
    save('calibration_history.json',h);save('old_entry_calibration_history.json',oldh)
    save('calibration_final_calls.json',final);save('calibration_fixed_calls.json',initial)
    save('old_entry_calibration_final_calls.json',oldfinal);save('old_entry_calibration_fixed_calls.json',oldinitial)
    save('cutoff.json',dict(primary_threshold=tau,old140_entry_control_threshold=oldtau,fixed_threshold=.5,strict_greater=True,
        calibration_dates=d['calibration_dates'],empirical_only=True,validation_scored=False))
    save('fit_calibration_metrics.json',[dict(partition=part,reading=kind,**metrics(rr,p,threshold)) for part,rr,p in [('fit',rows,pf),('calibration',cal,pc)] for kind,threshold in [('fixed',.5),('calibrated',tau)]])
    frozen=['design.json','preprocessing.json','fit_sample_weights.json','fit_receipt.json','model.pkl','model_export.json','train_scores.npz',
        'calibration_history.json','old_entry_calibration_history.json','calibration_final_calls.json','calibration_fixed_calls.json',
        'old_entry_calibration_final_calls.json','old_entry_calibration_fixed_calls.json','cutoff.json','fit_calibration_metrics.json']
    save('trained_freeze.json',{n:sha(O/n) for n in frozen});verify_trained();save('state.json',dict(stage='trained_before_validation'));archive('trained')
    print(json.dumps(dict(fit=read(O/'fit_receipt.json'),cutoff=read(O/'cutoff.json'),calibration= h[-1]['summary'],old_entry_calibration=oldh[-1]['summary']),ensure_ascii=False))

def publication():
    import subprocess
    verify_trained()
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0]
    assert local==remote
    rel=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix()
    archived=subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=REPO)
    assert hashlib.sha256(archived).hexdigest()==sha(O/'trained_freeze.json')
    save('prevalidation_publication.json',dict(passed=True,local_sha=local,remote_sha=remote,trained_freeze_sha256=sha(O/'trained_freeze.json'),utc=stamp()))
    print('PREVALIDATION_PUBLISHED',local)

def validate():
    verify_trained();pub=read(O/'prevalidation_publication.json');assert pub['passed'] and pub['trained_freeze_sha256']==sha(O/'trained_freeze.json')
    assert not (O/'validation_started.json').exists()
    save('validation_started.json',dict(utc=stamp(),published_sha=pub['local_sha'],fits_so_far=1))
    d=read(O/'design.json');pre=read(O/'preprocessing.json');cut=read(O/'cutoff.json')
    valall=B.eligible_data('validation');val=entry_rows(valall);assert S.public_checks(val)==len(val)
    gzsave('validation_rows.json.gz',val)
    with (O/'model.pkl').open('rb') as f:est=pickle.load(f)
    xv=B.transform(val,pre['features'],pre['medians']);pv=est.predict_proba(xv)[:,list(est.classes_).index(1)]
    assert max(abs(B.exported_scores(read(O/'model_export.json'),xv)-pv))<1e-10
    np.savez_compressed(O/'validation_scores.npz',scores=pv)
    fitrows=rowfile(O/'fit_rows.json.gz');cal=rowfile(O/'calibration_rows.json.gz');z=np.load(O/'train_scores.npz')
    sm=scoremap(fitrows+cal+val,np.concatenate([z['fit'],z['calibration'],pv]))
    oldrows=rowfile(B.O/'fit_rows.json.gz')+rowfile(B.O/'calibration_rows.json.gz')+valall
    oz=np.load(B.O/'train_scores.npz');ov=np.load(B.O/'validation_scores.npz')['scores']
    oldsm={(r['meta']['call_id'],128):float(p) for r,p in zip(oldrows,np.concatenate([oz['fit'],oz['calibration'],ov])) if r['meta']['visible_count']==128}
    prior=read(PREV/'all_search_replays.json');old={(r['evaluation_partition'],r['method'],r['call_id']):r for r in prior}
    existing=list(dict.fromkeys(r['method'] for r in prior));assert len(existing)==11
    newmethods=['OLD140_ENTRY128_FIXED_0.5','OLD140_ENTRY128_CALIBRATED','RF146_ENTRY128_FIXED_0.5','RF146_ENTRY128_CALIBRATED']
    records=[];summaries=[]
    for part,refs in [('fit',B.refs('train',set(d['fit_dates']))),('calibration',B.refs('train',set(d['calibration_dates']))),('validation',B.refs('validation'))]:
        for name in existing+newmethods:
            rr=[]
            for ref in refs:
                if name in existing:r=dict(old[part,name,ref['call_id']])
                else:
                    oldcontrol=name.startswith('OLD140');threshold=.5 if name.endswith('0.5') else cut['old140_entry_control_threshold' if oldcontrol else 'primary_threshold']
                    r=replay(ref,threshold,oldsm if oldcontrol else sm)
                rr.append(dict(r,evaluation_partition=part,method=name))
            records+=rr;summaries.append(dict(method=name,evaluation_partition=part,**B.summarize(rr),joint_failures=sum(r['bad_learned_stop'] for r in rr)))
    save('all_search_replays.json',records);save('summary.json',summaries)
    save('validation_metrics.json',[dict(reading=kind,**metrics(val,pv,t)) for kind,t in [('fixed',.5),('calibrated',cut['primary_threshold'])]])
    get=lambda name:next(x for x in summaries if x['evaluation_partition']=='validation' and x['method']==name)
    a=get(d['primary']);cap=get('CAP256');old=get('RF140_CALIBRATED');structure=get('OLD140_ENTRY128_CALIBRATED')
    bd={r['date']:r for r in cap['by_date']};dd=[dict(date=r['date'],action_differences=r['action_differences'],queries=r['queries'],cap256_queries=bd[r['date']]['queries']) for r in a['by_date']]
    passed=a['action_differences']==0 and a['joint_failures']==0 and a['date_macro_query_reduction']>cap['date_macro_query_reduction'] and all(r['queries']<=r['cap256_queries'] for r in dd) and any(r['queries']<r['cap256_queries'] for r in dd)
    save('decision.json',dict(primary=d['primary'],passes_development_screen=passed,ready_for_frame_coverage_proposal=passed,ready_to_scale=False,adopted=False,
        action_differences=a['action_differences'],joint_failures=a['joint_failures'],query_reduction_macro=a['date_macro_query_reduction'],
        cap256_macro=cap['date_macro_query_reduction'],old_rf140_macro=old['date_macro_query_reduction'],structure_only_macro=structure['date_macro_query_reduction'],
        per_date=dd,new_fits=1,old_fits=0,new_physical=0,final8_absent=True,scope=d['scope']))
    verify_trained();save('state.json',dict(stage='validated_pending_audit'))
    print(json.dumps(dict(decision=read(O/'decision.json'),validation_metrics=read(O/'validation_metrics.json')),ensure_ascii=False))

def manual_ap(rows,pp):
    y=[not r['target']['stop_label'] for r in rows];n=sum(y)
    if not n:return None
    order=sorted(range(len(rows)),key=lambda i:pp[i]);tp=0;seen=0;ap=0.;j=0
    while j<len(order):
        end=j+1
        while end<len(order) and pp[order[end]]==pp[order[j]]:end+=1
        add=sum(y[order[k]] for k in range(j,end));tp+=add;seen+=end-j;ap+=(tp/seen)*(add/n);j=end
    return ap

def audit():
    verify_trained();assert read(O/'state.json')['stage']=='validated_pending_audit';assert not (O/'review.json').exists()
    d=read(O/'design.json');cut=read(O/'cutoff.json');pre=read(O/'preprocessing.json')
    fitrows=rowfile(O/'fit_rows.json.gz');cal=rowfile(O/'calibration_rows.json.gz');val=rowfile(O/'validation_rows.json.gz');rows=fitrows+cal+val
    assert fitrows==entry_rows(rowfile(B.O/'fit_rows.json.gz')) and cal==entry_rows(rowfile(B.O/'calibration_rows.json.gz')) and val==entry_rows(B.eligible_data('validation'))
    assert len({r['meta']['call_id'] for r in rows})==len(rows)
    w=np.asarray([r['weight'] for r in read(O/'fit_sample_weights.json')]);assert np.array_equal(w,weights(fitrows))
    for date in d['fit_dates']:assert abs(sum(v for r,v in zip(fitrows,w) if r['meta']['date']==date)-153/12)<1e-10
    assert d['features']==read(B.O/'design.json')['features'] and d['model_parameters']==read(B.O/'design.json')['model_parameters']
    raw=np.asarray([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in d['features']] for r in fitrows])
    mm=[float(np.median(raw[np.isfinite(raw[:,j]),j])) if np.isfinite(raw[:,j]).any() else 0. for j in range(18)]
    assert mm==pre['medians']
    with (O/'model.pkl').open('rb') as f:est=pickle.load(f)
    assert all(est.get_params()[k]==v for k,v in d['model_parameters'].items())
    z=np.load(O/'train_scores.npz');pv=np.load(O/'validation_scores.npz')['scores'];pp=np.concatenate([z['fit'],z['calibration'],pv])
    manual=B.exported_scores(read(O/'model_export.json'),B.transform(rows,d['features'],mm));assert max(abs(manual-pp))<1e-10
    suffix=S.public_checks(rows);assert suffix==len(rows)
    sm=scoremap(rows,pp)
    oldrows=rowfile(B.O/'fit_rows.json.gz')+rowfile(B.O/'calibration_rows.json.gz')+B.eligible_data('validation')
    oz=np.load(B.O/'train_scores.npz');ov=np.load(B.O/'validation_scores.npz')['scores']
    oldsm={(r['meta']['call_id'],128):float(p) for r,p in zip(oldrows,np.concatenate([oz['fit'],oz['calibration'],ov])) if r['meta']['visible_count']==128}
    for name,src in [('calibration_history.json',sm),('old_entry_calibration_history.json',oldsm)]:
        expected_tau,history,initial,final=calibrate(B.refs('train',set(d['calibration_dates'])),src)
        assert history==read(O/name)
        assert expected_tau==cut['old140_entry_control_threshold' if name.startswith('old') else 'primary_threshold']
    records=read(O/'all_search_replays.json');assert len(records)==15*305
    prev={(r['evaluation_partition'],r['method'],r['call_id']):r for r in read(PREV/'all_search_replays.json')}
    refmap={r['call_id']:r for r in B.refs()};checks=collections.Counter();errors=[]
    for r in records:
        ref=refmap[r['call_id']]
        if r['method'].startswith(('OLD140_ENTRY128','RF146_ENTRY128')):
            isold=r['method'].startswith('OLD140');threshold=.5 if r['method'].endswith('0.5') else cut['old140_entry_control_threshold' if isold else 'primary_threshold']
            actual=shadow_check(ref,threshold,oldsm if isold else sm)
            assert all(r[k]==v for k,v in actual.items());checks['single_entry_shadow_searches']+=1
            assert all(x['visible_count']==128 for x in r['model_decisions']) and len(r['model_decisions'])<=1
            if r['bad_learned_stop']:errors.append(r)
        else:
            prior=prev[r['evaluation_partition'],r['method'],r['call_id']]
            assert all(r[k]==v for k,v in prior.items());checks['old_reference_rows_unchanged']+=1
        c=B.call_at(ref['source']);n=r['global_queries'];b=B.P.best(c['outcomes'][:n]);initial=B.P.best(c['outcomes'][:min(128,len(c['order']))]);extra=[]
        if b is not None and n<len(c['order']) and (initial is None or initial['rho']>=.9 or B.P.bid(b)!=B.P.bid(initial)):
            station=B.subids()[b['pool_id']];ids={int(rep) for alias,rep in c['aliases'].items() if B.subids()[int(alias)]==station};extra=[v for v in c['outcomes'][n:] if v['pool_id'] in ids]
        chosen=B.P.best(c['outcomes'][:n]+extra)
        assert B.P.bid(chosen)==r['selected'] and n+len(extra)==r['queries'] and len(extra)==r['local_queries'];checks['complete_searches']+=1
    # Gate loss changes only at this sole score; check both sides and full fallback.
    for src in [sm,oldsm]:
        for ref in refmap.values():
            p=src.get((ref['call_id'],128))
            if p is not None:
                before=replay(ref,float(np.nextafter(p,-np.inf)),src);after=replay(ref,p,src)
                assert not after['bad_learned_stop'] and after['action_match'] and after['queries']==after['original_queries']
                assert len(before['model_decisions'])==1 and len(after['model_decisions'])==1
                checks['single_entry_monotone_score_boundaries']+=1
    for part,rr,p in [('fit',fitrows,z['fit']),('calibration',cal,z['calibration']),('validation',val,pv)]:
        m=metrics(rr,p,cut['primary_threshold']);assert abs(manual_ap(rr,p)-m['continue_ap'])<1e-12
    save('all_single_entry_failures.json',errors)
    save('review.json',dict(passed=True,new_fits=read(O/'fit_receipt.json')['new_fits'],old_fits=0,
        source_labels_rows_features_same=True,original_parameters_same=True,fit_only_date_equal_weights=True,missing_values_fit_only=True,
        public_suffix_checks=suffix,manual_score_checks=len(rows),max_score_error=float(max(abs(manual-pp))),
        single_gate_and_fallback_verified=True,empirical_calibration_reexecuted=True,checks=dict(checks),
        frozen_model_and_cutoffs_published_before_validation=True,final8_absent=True,no_new_physical=True))
    verify_trained();save('state.json',dict(stage='audited_pending_report'))
    print(json.dumps(read(O/'review.json'),ensure_ascii=False))

if __name__=='__main__':
    {'prepare':prepare,'fit':fit,'publication':publication,'validate':validate,'audit':audit}[sys.argv[1]]()
