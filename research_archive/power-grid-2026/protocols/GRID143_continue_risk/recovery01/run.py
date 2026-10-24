"""Fixed-model rare-continue ranking and calibration-date robustness diagnosis."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,math,shutil,sys
sys.path.insert(0,'D:/Codex/GRL研究/work/grid142_batch_representation')
import base as G
import numpy as np
R=G.R;W=Path(__file__).parent;O=R/'outputs/grid143_continue_risk';REPO=G.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID143_continue_risk'
read=G.read;sha=G.sha
MODELS={'RF140':G.PREV,'RF142':G.O}
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rowfile(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def verify():
    G.verify_trained()
    for name in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/name).items():assert sha(R/p)==h,p
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def payload():
    fit=rowfile(G.PREV/'fit_rows.json.gz');cal=rowfile(G.PREV/'calibration_rows.json.gz');val=G.B.eligible_data('validation');rows=fit+cal+val
    parts=['fit']*len(fit)+['calibration']*len(cal)+['validation']*len(val);scores={}
    for model,root in MODELS.items():
        z=np.load(root/'train_scores.npz');scores[model]=np.concatenate([z['fit'],z['calibration'],np.load(root/'validation_scores.npz')['scores']]);assert len(scores[model])==len(rows)==726
    return rows,parts,scores
def prepare():
    assert not O.exists();O.mkdir(parents=True);G.verify_trained();assert read(G.O/'publication_verification.json')['passed']
    save('design.json',dict(id='GRID143_CONTINUE_RANK_AND_CALIBRATION_DATE_DIAGNOSIS',utc=stamp(),authorization='User continues after GRID142 failed representation-only intervention;diagnose before fitting.',
        models=['FixedRF140','FixedRF142'],population='All726 already exposed fit/calibration/development validation prefixes,including all122continue labels. No new scores.',
        ranking='Continue-positive AP of1-stop_score,ROC,prevalence and both class recall at each original frozen cutoff. Per-partition and per-date,all cases. Original-query-depth/prefix-safe matched pairwise ranking;no learned matcher.',
        separation='Count true-stop scores strictly above every continue score in each partition as hindsight rank-separation statistic only. Do not replay or adopt a validation-derived threshold.',
        calibration_robustness='For each fixed model,omit each of the original4calibration dates in turn. Use exact original monotone whole-call calibration on other3dates starting0.5. Evaluate only omitted calibration date. Save all8folds;no new validation replay,model selection or adopted cutoff. Donor score maps exclude omitted date.',
        continuation_path='For all122negative prefixes,using saved complete feedback only,compute whether one further original64batch plus unchanged completion restores originalFRAME action. Trace first restoring original checkpoint. Separate next-batch incumbent improvement from exact final-action restoration;future values are labels only.',
        weighting='Reconstruct existing fit sample weights and continue-class effective mass. Calculate one possible equal-class-mass loss normalization from fit only as a mathematical planning quantity,not a new model.',
        limits='Already exposed development records. Date sensitivity is not independent cross-validation or a safety certificate. Prefixes/replicas not independent physical scenarios.',
        prohibited=['fit','predict','new physics/scenarios','final8tests','validation-selected threshold or model','threshold grid','new labels or cost margin','RL/NN','paper edit','automatic scale']))
    sources=[R/'outputs/grid141_feedback_diagnosis/negative_label_summary.json',R/'work/grid142_batch_representation/base.py',R/'work/grid140_global_compression/common.py',G.O/'publication_verification.json']
    for root in MODELS.values():sources += [root/n for n in ['design.json','cutoff.json','train_scores.npz','validation_scores.npz','fit_rows.json.gz','calibration_rows.json.gz','fit_sample_weights.json','summary.json','all_search_replays.json','calibration_history.json','review.json','trained_freeze.json']]
    sources.append(O/'design.json');save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sources});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});verify();save('state.json',dict(stage='frozen'));archive('frozen');print('GRID143_FROZEN')
def rank_metrics(rows,scores,tau):
    from sklearn.metrics import average_precision_score,roc_auc_score
    y=np.asarray([r['target']['stop_label'] for r in rows],bool);p=np.asarray(scores,float);negative=~y;stop=p>tau
    positive_scores=p[y];negative_scores=p[negative]
    return dict(prefixes=len(rows),calls=len({r['meta']['call_id'] for r in rows}),dates=len({r['meta']['date'] for r in rows}),continue_prefixes=int(negative.sum()),stop_prefixes=int(y.sum()),
        continue_prevalence=float(negative.mean()),continue_ap=float(average_precision_score(negative,1-p)) if negative.any() else None,
        auc=float(roc_auc_score(y,p)) if y.any() and negative.any() else None,
        frozen_cutoff=float(tau),continue_recall=float(np.mean(~stop[negative])) if negative.any() else None,valid_stop_recall=float(np.mean(stop[y])) if y.any() else None,
        wrong_stop_proposals=int(sum(stop&negative)),true_stop_proposals=int(sum(stop&y)),all_continue_proposals=int(sum(~stop)),
        hindsight_stop_scores_above_all_continue=int(sum(positive_scores>max(negative_scores))) if len(negative_scores) else None,
        hindsight_scope='Ranking count uses known labels;not a new cutoff or executed policy.')
def calibrate(callrefs,scoremap):
    tau=.5;history=[]
    for iteration in range(200):
        rr=[G.replay(r,'MODEL',tau,scoremap) for r in callrefs];bad=[r for r in rr if r['bad_learned_stop']]
        history.append(dict(iteration=iteration,threshold=tau,bad_calls=[dict(call_id=r['call_id'],score=r['stopped_score']) for r in bad],summary=G.summarize(rr)))
        if not bad:return tau,history,rr
        nxt=max(r['stopped_score'] for r in bad);assert nxt>tau;tau=nxt
    raise AssertionError('Original monotone calibration failed to terminate;no model fit permitted')
def hypothetical(c,n):
    order=c['order'];best=G.P.best(c['outcomes'][:n]);initial=G.P.best(c['outcomes'][:128]);extra=[]
    if best is not None and n<len(order) and (initial is None or initial['rho']>=.9 or G.P.bid(best)!=G.P.bid(initial)):
        station=G.B.subids()[best['pool_id']];ids={int(rep) for alias,rep in c['aliases'].items() if G.B.subids()[int(alias)]==station};extra=[v for v in c['outcomes'][n:] if v['pool_id'] in ids];best=G.P.best(c['outcomes'][:n]+extra)
    return dict(selected=G.P.bid(best),queries=n+len(extra),local_queries=len(extra))
def run():
    verify();assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(utc=stamp(),fits=0,predict_calls=0,physical=0));rows,parts,scores=payload();D=read(G.PREV/'design.json');metrics=[];date_metrics=[];pairwise=[];negative_rankings=[]
    for model,pp in scores.items():
        tau=read(MODELS[model]/'cutoff.json')['primary_threshold']
        for part in ['fit','calibration','validation']:
            ii=[i for i,p in enumerate(parts) if p==part];rr=[rows[i] for i in ii];pv=pp[ii];metrics.append(dict(model=model,partition=part,**rank_metrics(rr,pv,tau)))
            for date in sorted({r['meta']['date'] for r in rr}):
                jj=[i for i in ii if rows[i]['meta']['date']==date];date_metrics.append(dict(model=model,partition=part,date=date,**rank_metrics([rows[i] for i in jj],pp[jj],tau)))
            strata=collections.defaultdict(list)
            for i in ii:strata[rows[i]['meta']['visible_count'],rows[i]['features']['prefix_safe']].append(i)
            for (n,safe),jj in sorted(strata.items()):
                pos=[float(pp[i]) for i in jj if rows[i]['target']['stop_label']];neg=[float(pp[i]) for i in jj if not rows[i]['target']['stop_label']]
                correct=sum(p>q for p in pos for q in neg);ties=sum(p==q for p in pos for q in neg);count=len(pos)*len(neg)
                pairwise.append(dict(model=model,partition=part,visible_count=n,prefix_safe=safe,stop_prefixes=len(pos),continue_prefixes=len(neg),pairs=count,correct=correct,ties=ties,auc=(correct+.5*ties)/count if count else None))
            positive=[i for i in ii if rows[i]['target']['stop_label']]
            for i in ii:
                if rows[i]['target']['stop_label']:continue
                negative_rankings.append(dict(model=model,partition=part,sample_id=rows[i]['meta']['sample_id'],call_id=rows[i]['meta']['call_id'],date=rows[i]['meta']['date'],visible_count=rows[i]['meta']['visible_count'],stop_score=float(pp[i]),stop_scores_above=sum(pp[j]>pp[i] for j in positive),stop_scores_below=sum(pp[j]<pp[i] for j in positive),stop_scores_tied=sum(pp[j]==pp[i] for j in positive),wrong_at_frozen_cutoff=bool(pp[i]>tau)))
    save('ranking_metrics.json',metrics);save('ranking_per_date.json',date_metrics);save('same_depth_branch_pairwise_ranking.json',pairwise);save('all_continue_score_positions.json',negative_rankings)
    folds=[];histories=[];heldout=[]
    for model,pp in scores.items():
        for omitted in D['calibration_dates']:
            donor_dates=set(D['calibration_dates'])-{omitted};donor_ids=[i for i,p in enumerate(parts) if p=='calibration' and rows[i]['meta']['date'] in donor_dates];test_ids=[i for i,p in enumerate(parts) if p=='calibration' and rows[i]['meta']['date']==omitted]
            donor_map={(rows[i]['meta']['call_id'],rows[i]['meta']['visible_count']):float(pp[i]) for i in donor_ids};test_map={(rows[i]['meta']['call_id'],rows[i]['meta']['visible_count']):float(pp[i]) for i in test_ids};assert not set(donor_map)&set(test_map)
            donor_refs=G.refs('train',donor_dates);test_refs=G.refs('train',{omitted});tau,h,donor_rr=calibrate(donor_refs,donor_map);rr=[G.replay(r,'MODEL',tau,test_map) for r in test_refs]
            fold=dict(model=model,omitted_calibration_date=omitted,donor_dates=sorted(donor_dates),donor_prefixes=len(donor_ids),heldout_prefixes=len(test_ids),diagnostic_threshold=tau,iterations=len(h),donor_summary=G.summarize(donor_rr),heldout_summary=G.summarize(rr),not_an_adopted_threshold=True)
            folds.append(fold);histories.append(dict(model=model,omitted_calibration_date=omitted,history=h));heldout += [dict(r,model=model,omitted_calibration_date=omitted) for r in rr]
    save('all_calibration_date_folds.json',folds);save('all_calibration_date_histories.json',histories);save('all_omitted_calibration_call_replays.json',heldout)
    refsmap={r['call_id']:r for r in G.refs()};continuations=[]
    for i,r in enumerate(rows):
        if r['target']['stop_label']:continue
        n=r['meta']['visible_count'];ref=refsmap[r['meta']['call_id']];c=G.B.call_at(r['meta']['source']);before=hypothetical(c,n);assert before['selected']==r['target']['after_stop_choice_label'] and before['queries']==r['target']['stop_total_queries_known_now']
        nextn=min(n+64,ref['global_n']);after=hypothetical(c,nextn);checkpoints=list(range(n+64,ref['global_n']+1,64))
        if ref['global_n'] not in checkpoints:checkpoints.append(ref['global_n'])
        first=next(v for v in checkpoints if hypothetical(c,v)['selected']==ref['frame_choice']);a=G.P.best(c['outcomes'][:n]);b=G.P.best(c['outcomes'][:nextn])
        continuations.append(dict(sample_id=r['meta']['sample_id'],partition=parts[i],call_id=r['meta']['call_id'],date=r['meta']['date'],visible_count=n,current_stop_choice=before['selected'],original_frame_choice=ref['frame_choice'],next_global_queries=nextn,next_batch_improves_incumbent=G.P.improves(a,b),next_then_completion_restores_action=after['selected']==ref['frame_choice'],next_then_completion_choice=after['selected'],next_then_completion_queries=after['queries'],first_restoring_global_queries=first,first_restore_batches=math.ceil((first-n)/64),future_label_only=True))
    save('all_continue_future_paths.json',continuations);path_summary=[]
    for part in ['fit','calibration','validation']:
        rr=[r for r in continuations if r['partition']==part];path_summary.append(dict(partition=part,prefixes=len(rr),calls=len({r['call_id'] for r in rr}),dates=len({r['date'] for r in rr}),next_incumbent_improves=sum(r['next_batch_improves_incumbent'] for r in rr),one_batch_restores=sum(r['next_then_completion_restores_action'] for r in rr),needs_more_than_one_batch=sum(not r['next_then_completion_restores_action'] for r in rr),restore_batch_counts=dict(collections.Counter(r['first_restore_batches'] for r in rr))))
    save('continue_path_summary.json',path_summary)
    fitrows=rows[:386];weights=np.asarray([v['weight'] for v in read(G.PREV/'fit_sample_weights.json')]);y=np.asarray([r['target']['stop_label'] for r in fitrows],bool);tot=float(weights.sum());ws=float(weights[y].sum());wc=float(weights[~y].sum());factors={False:tot/(2*wc),True:tot/(2*ws)};balanced=weights*np.asarray([factors[bool(v)] for v in y]);assert abs(balanced[y].sum()-balanced[~y].sum())<1e-10
    weighting=dict(fit_prefixes=386,raw_stop_count=int(y.sum()),raw_continue_count=int((~y).sum()),base_stop_weight=ws,base_continue_weight=wc,base_continue_weight_fraction=wc/tot,
        planning_only_equal_mass_stop_multiplier=factors[True],planning_only_equal_mass_continue_multiplier=factors[False],continue_to_stop_multiplier_ratio=factors[False]/factors[True],no_fit=True,no_weights_changed=True,
        caveat='Prospective class-loss normalization. Existing within-class relative weights preserved,but overall date/call totals would change;not a physical cost ratio or new operational label.')
    save('fit_class_mass_diagnosis.json',weighting)
    foldsummary=[]
    for model in scores:
        ff=[f for f in folds if f['model']==model];rr=[r for r in heldout if r['model']==model];foldsummary.append(dict(model=model,folds=len(ff),folds_with_heldout_action_differences=sum(f['heldout_summary']['action_differences']>0 for f in ff),all_omitted_calibration_calls=G.summarize(rr)))
    save('calibration_sensitivity_summary.json',foldsummary);verify();save('state.json',dict(stage='analysed_pending_audit'));print(json.dumps(dict(ranking=metrics,paths=path_summary,calibration=foldsummary,weighting=weighting),ensure_ascii=False))
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
