"""Fixed-forest cross-date diagnosis; no fitting, new policy or threshold replay."""
from pathlib import Path
import collections,datetime,hashlib,importlib.util,json,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid147_generalization_diagnosis'
spec=importlib.util.spec_from_file_location('grid147_previous_single_entry',R/'work/grid146_single_entry/run.py')
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
B=G.B;REPO=G.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID147_generalization_diagnosis'
read=G.read;sha=G.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify():
    G.verify_trained()
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def prepare():
    assert not O.exists();O.mkdir();G.verify_trained();assert read(G.O/'final_verification.json')['passed']
    save('design.json',dict(id='GRID147_FIXED_FOREST_GENERALIZATION_DIAGNOSIS',utc=stamp(),
        authorization='User says continue investigation after GRID146 failed;read-only analysis,no further training.',
        models=['frozenRF146_single_entry','frozenRF140_original'],
        population='All275 entry128 rows:153fit,66cal,56exposeddev. ForRF140 bootstrap support use original386fit rows,then compare the same153entry rows.',
        questions=['How much fitted ranking survives row-out-of-bag aggregation of existing trees?','Are high-score errors unsupported extrapolation or mixed with real training STOP states?','How do date/branch support and visible-feature distributions differ?'],
        out_of_bag='Manual votes only from existing trees whose bootstrap did not draw the row. No model.predict or refit. Also count trees omitting the whole calendar date. OOB is row exclusion,not date-held-out evidence;preprocessing/weights were fit globally.',
        leaves='Manually traverse export,float32 inputs;verify native apply routing;reconstruct each leaf class mass from fixed bootstrap multiplicities and fit weights. Save actual/OOB date/label supports.',
        matching='For ALL275entry rows,nearest fitSTOP and fitCONT within same initial-safe branch,exclude same calendar date. Fit153median/IQR,range fallback,active nonconstant transformed channels;deterministic sample-id ties. Not a learned classifier or selection rule.',
        distributions='All18public features;fit-derived quartile bins for varying raw numeric features,original safe branch/date counts,fit-range breaches. No posthoc feature selection or p-value claims.',
        cases='All primary calibrated validation stops,all max-score calibrationCONT ties,and all fitCONT above frozen cutoff. Same fixed states inspected under both forests.',
        collisions='Exact original-feature equality and full64leaf-signature collisions across275. Mixed signatures show representation pooling,not raw-input indistinguishability.',
        limits='Existing development exposure,correlated date replicas. Counts/pairs/tree votes not independent experiments;no new risk guarantee or proof task impossible.',
        prohibited=['fit','model.predict','threshold/policy replay','new adopted score or threshold','parameter/feature/label change','new physics or scenarios','final8tests','paper edit','automatic scale']))
    sources=[G.O/n for n in ['design.json','fit_rows.json.gz','calibration_rows.json.gz','validation_rows.json.gz','model.pkl','model_export.json','preprocessing.json','fit_sample_weights.json','train_scores.npz','validation_scores.npz','cutoff.json','calibration_history.json','all_search_replays.json','post_diagnosis.json','review.json','final_verification.json']]
    sources += [B.O/n for n in ['fit_rows.json.gz','calibration_rows.json.gz','model.pkl','model_export.json','preprocessing.json','fit_sample_weights.json','train_scores.npz','validation_scores.npz','trained_freeze.json']]
    sources += [R/'work/grid146_single_entry/run.py',R/'work/grid140_global_compression/common.py',O/'design.json']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sources})
    save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});verify();save('state.json',dict(stage='frozen_read_only'));archive('frozen');print('GRID147_FROZEN')

def leaf(tree,x,trace=False):
    node=0;path=[]
    while tree['left'][node]!=-1:
        f=tree['feature'][node];v=float(x[f]);cut=tree['threshold'][node];left=v<=cut
        if trace:path.append(dict(node=node,feature_index=f,value=v,threshold=cut,left=left))
        node=tree['left'][node] if left else tree['right'][node]
    return (node,path) if trace else node
def stats(v):return dict(min=float(min(v)),median=float(np.median(v)),max=float(max(v))) if len(v) else None
def ranks(rows,p):
    from sklearn.metrics import average_precision_score,roc_auc_score
    y=np.asarray([not r['target']['stop_label'] for r in rows],bool);p=np.asarray(p,float)
    return dict(rows=len(rows),continue_count=int(y.sum()),continue_prevalence=float(y.mean()),
        continue_ap=float(average_precision_score(y,1-p)) if y.any() else None,
        auc=float(roc_auc_score(~y,p)) if y.any() and (~y).any() else None)
def manual_auc(rows,p):
    pos=[float(s) for r,s in zip(rows,p) if r['target']['stop_label']]
    neg=[float(s) for r,s in zip(rows,p) if not r['target']['stop_label']]
    return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg)) if pos and neg else None

def run():
    verify();assert not (O/'started.json').exists();save('started.json',dict(utc=stamp(),fits=0,physical=0))
    fit=G.rowfile(G.O/'fit_rows.json.gz');cal=G.rowfile(G.O/'calibration_rows.json.gz');val=G.rowfile(G.O/'validation_rows.json.gz')
    entry=fit+cal+val;parts=['fit']*153+['calibration']*66+['validation']*56
    assert len(entry)==275
    key=lambda r:r['meta']['sample_id']
    entry_idx={key(r):i for i,r in enumerate(entry)}
    cut=read(G.O/'cutoff.json')['primary_threshold'];nz=np.load(G.O/'train_scores.npz');npv=np.load(G.O/'validation_scores.npz')['scores'];newp=np.concatenate([nz['fit'],nz['calibration'],npv])
    caseids={key(r) for r,p,part in zip(entry,newp,parts) if part=='fit' and not r['target']['stop_label'] and p>cut}
    calmax=max(p for r,p in zip(cal,nz['calibration']) if not r['target']['stop_label'])
    caseids|={key(r) for r,p in zip(cal,nz['calibration']) if not r['target']['stop_label'] and p==calmax}
    stops=[r for r in read(G.O/'all_search_replays.json') if r['evaluation_partition']=='validation' and r['method']=='RF146_ENTRY128_CALIBRATED' and r['stop_kind']=='learned_early_stop']
    caseids|={key(r) for r in val if r['meta']['call_id'] in {v['call_id'] for v in stops}}
    save('prespecified_case_ids.json',sorted(caseids))
    leaves=[];cases=[];oob_rows=[];rankrows=[];collisions=[];route_checks=0;maxleaf=0.;maxscore=0.;split_usage=[]
    for model,root in [('RF146',G.O),('RF140',B.O)]:
        train=G.rowfile(root/'fit_rows.json.gz');n=len(train);rr=train+cal+val
        pre=read(root/'preprocessing.json');x=B.transform(rr,pre['features'],pre['medians']).astype(np.float32)
        with (root/'model.pkl').open('rb') as f:est=pickle.load(f)
        exported=read(root/'model_export.json');assert list(est.classes_)==exported['classes']==[0,1]
        routes=np.asarray([[leaf(t,v) for t in exported['trees']] for v in x],int)
        assert np.array_equal(routes,est.apply(x));route_checks+=int(routes.size)
        zz=np.load(root/'train_scores.npz')
        originalcal=cal if model=='RF146' else G.rowfile(B.O/'calibration_rows.json.gz')
        originalval=val if model=='RF146' else B.eligible_data('validation')
        oldv=np.load(root/'validation_scores.npz')['scores']
        cm={key(r):float(p) for r,p in zip(originalcal,zz['calibration'])};vm={key(r):float(p) for r,p in zip(originalval,oldv)}
        expected=np.concatenate([zz['fit'],[cm[key(r)] for r in cal],[vm[key(r)] for r in val]])
        votes=np.asarray([[t['value'][routes[i,ti]][1]/sum(t['value'][routes[i,ti]]) for ti,t in enumerate(exported['trees'])] for i in range(len(rr))])
        pp=votes.mean(axis=1);maxscore=max(maxscore,float(max(abs(pp-expected))));assert max(abs(pp-expected))<1e-10
        y=np.asarray([r['target']['stop_label'] for r in train],bool);weights=np.asarray([r['weight'] for r in read(root/'fit_sample_weights.json')])
        counts=np.asarray([np.bincount(z,minlength=n) for z in est.estimators_samples_],int)
        supports={}
        for ti,t in enumerate(exported['trees']):
            for node in np.unique(routes[:n,ti]):
                inleaf=routes[:n,ti]==node;ib=np.flatnonzero(inleaf&(counts[ti]>0));oob=np.flatnonzero(inleaf&(counts[ti]==0))
                ww=weights[ib]*counts[ti,ib];v=float(ww[y[ib]].sum()/ww.sum());stored=t['value'][node][1]/sum(t['value'][node])
                maxleaf=max(maxleaf,abs(v-stored));assert abs(v-stored)<1e-10
                z=dict(model=model,tree=ti,node=int(node),stop_fraction=v,unique_inbag_rows=len(ib),inbag_dates=sorted({train[j]['meta']['date'] for j in ib}),
                    inbag_continue_rows=int((~y[ib]).sum()),inbag_continue_dates=sorted({train[j]['meta']['date'] for j in ib if not y[j]}),
                    oob_rows=len(oob),oob_continue_rows=int((~y[oob]).sum()),oob_dates=sorted({train[j]['meta']['date'] for j in oob}))
                supports[ti,int(node)]=z;leaves.append(z)
        train_dates=np.asarray([r['meta']['date'] for r in train]);same_date_ib={d:np.any(counts[:,train_dates==d]>0,axis=1) for d in set(train_dates)}
        train_entry_indices=[i for i,r in enumerate(train) if r['meta']['visible_count']==128];oobp=[]
        for i in train_entry_indices:
            mask=counts[:,i]==0;assert mask.any()
            p=float(votes[i,mask].mean());oobp.append(p)
            # An independent index-list implementation audits the voting mask.
            indices=[ti for ti,z in enumerate(est.estimators_samples_) if i not in set(z)]
            assert indices==list(np.flatnonzero(mask)) and abs(p-sum(votes[i,j] for j in indices)/len(indices))<1e-12
            oob_rows.append(dict(model=model,sample_id=key(train[i]),date=train[i]['meta']['date'],stop_label=bool(y[i]),fitted_score=float(pp[i]),
                row_oob_score=p,row_oob_trees=int(mask.sum()),whole_date_oob_trees=int((~same_date_ib[train[i]['meta']['date']]).sum())))
        same=[train[i] for i in train_entry_indices]
        assert [key(r) for r in same]==[key(r) for r in fit]
        for reading,p in [('fitted_all_trees',pp[train_entry_indices]),('row_oob_diagnostic',np.asarray(oobp))]:
            m=ranks(same,p);assert abs(m['auc']-manual_auc(same,p))<1e-12
            rankrows.append(dict(model=model,partition='fit128',reading=reading,**m))
        for part,subset,p in [('calibration',cal,pp[n:n+66]),('validation',val,pp[n+66:])]:
            m=ranks(subset,p);assert abs(m['auc']-manual_auc(subset,p))<1e-12
            rankrows.append(dict(model=model,partition=part,reading='frozen_all_trees',**m))
        entry_routes=np.concatenate([routes[train_entry_indices],routes[n:]],axis=0)
        entry_p=np.concatenate([pp[train_entry_indices],pp[n:]])
        groups=collections.defaultdict(list)
        for i,v in enumerate(entry_routes):groups[tuple(v)].append(i)
        mixed=[]
        for signature,ii in groups.items():
            if len({entry[i]['target']['stop_label'] for i in ii})>1:
                mixed.append(dict(rows=[dict(sample_id=key(entry[i]),partition=parts[i],date=entry[i]['meta']['date'],stop_label=entry[i]['target']['stop_label']) for i in ii],score=float(entry_p[ii[0]])))
        collisions.append(dict(model=model,signature_groups=len(groups),mixed_groups=len(mixed),mixed_rows=sum(len(g['rows']) for g in mixed),groups=mixed))
        usage=collections.Counter(pre['features'][f] if f<18 else pre['features'][f-18]+'__missing' for t in exported['trees'] for f,left in zip(t['feature'],t['left']) if left!=-1)
        split_usage.append(dict(model=model,split_node_counts=dict(usage),not_causal_importance=True))
        posmap={key(r):i for i,r in enumerate(rr)}
        for identity in sorted(caseids):
            i=posmap[identity];r=rr[i];ss=[supports[ti,int(node)] for ti,node in enumerate(routes[i])]
            trees_detail=[]
            for ti,t in enumerate(exported['trees']):
                node,path=leaf(t,x[i],True)
                for step in path:
                    f=step['feature_index'];step['feature']=pre['features'][f] if f<18 else pre['features'][f-18]+'__missing'
                trees_detail.append(dict(**supports[ti,node],path=path))
            case=dict(model=model,sample_id=identity,call_id=r['meta']['call_id'],date=r['meta']['date'],partition=parts[entry_idx[identity]],stop_label=r['target']['stop_label'],score=float(pp[i]),
                pure_stop_trees=sum(s['stop_fraction']>=1-1e-12 for s in ss),leaf_inbag_dates=stats([len(s['inbag_dates']) for s in ss]),
                leaf_inbag_continue_rows=stats([s['inbag_continue_rows'] for s in ss]),public_features=r['features'],trees=trees_detail)
            cases.append(case)
    save('bootstrap_leaf_supports.json',leaves);save('case_leaf_paths.json',cases);save('row_out_of_bag_scores_diagnostic.json',oob_rows)
    save('ranking_comparison.json',rankrows);save('forest_signature_collisions.json',collisions);save('tree_split_usage.json',split_usage)
    # Fit-only matching geometry. No neighbor predictions or policy decisions are made.
    pre=read(G.O/'preprocessing.json');names=pre['features'];xx=B.transform(entry,names,pre['medians'])
    fitx=xx[:153];scales=np.ones(36)
    for j in range(18):
        scale=float(np.quantile(fitx[:,j],.75)-np.quantile(fitx[:,j],.25));scales[j]=scale or float(np.ptp(fitx[:,j])) or 1.
    active=np.flatnonzero(np.ptp(fitx,axis=0)>0);norm=xx[:,active]/scales[active]
    save('matching_geometry.json',dict(fit_only=True,active_channels=[names[j] if j<18 else names[j-18]+'__missing' for j in active],
        full_scales=scales.tolist(),medians=pre['medians'],same_initial_safe_branch=True,exclude_same_calendar_date=True))
    pairs=[];outliers=[]
    for i,r in enumerate(entry):
        neighbors={}
        for label,labelname in [(True,'STOP'),(False,'CONTINUE')]:
            jj=[j for j,s in enumerate(fit) if bool(s['target']['stop_label'])==label and s['meta']['date']!=r['meta']['date'] and s['features']['prefix_safe']==r['features']['prefix_safe']]
            if not jj:neighbors[labelname]=None;continue
            distance={j:float(np.sqrt(np.mean((norm[i]-norm[j])**2))) for j in jj}
            j=min(jj,key=lambda j:(distance[j],key(fit[j])))
            neighbors[labelname]=dict(sample_id=key(fit[j]),call_id=fit[j]['meta']['call_id'],date=fit[j]['meta']['date'],distance=distance[j],
                score=float(newp[j]),features=fit[j]['features'])
        pairs.append(dict(sample_id=key(r),call_id=r['meta']['call_id'],partition=parts[i],date=r['meta']['date'],stop_label=r['target']['stop_label'],score=float(newp[i]),
                          features=r['features'],neighbors=neighbors))
        outside=[names[j] if j<18 else names[j-18]+'__missing' for j in range(36) if xx[i,j]<fitx[:,j].min() or xx[i,j]>fitx[:,j].max()]
        outliers.append(dict(sample_id=key(r),partition=parts[i],date=r['meta']['date'],stop_label=r['target']['stop_label'],outside_fit_range=outside))
    save('all_crossdate_fit_neighbors.json',pairs);save('fit_range_checks.json',outliers)
    matchsummary=[]
    for part in ['fit','calibration','validation']:
        q=[p for p in pairs if p['partition']==part and not p['stop_label']];both=[p for p in q if all(p['neighbors'].values())]
        matchsummary.append(dict(partition=part,continue_rows=len(q),matched_both_labels=len(both),closer_to_fit_stop=sum(p['neighbors']['STOP']['distance']<p['neighbors']['CONTINUE']['distance'] for p in both)))
    save('neighbor_summary.json',matchsummary)
    ranges=[]
    for j,name in enumerate(names):
        if np.ptp(fitx[:,j])==0:continue
        cuts=sorted(set(float(v) for v in np.quantile(fitx[:,j],[.25,.5,.75])))
        for part in ['fit','calibration','validation']:
            for bi in range(len(cuts)+1):
                ii=[i for i,r in enumerate(entry) if parts[i]==part and np.searchsorted(cuts,xx[i,j],side='right')==bi]
                ranges.append(dict(feature=name,fit_quantile_cuts=cuts,partition=part,bin=bi,rows=len(ii),continue_rows=sum(not entry[i]['target']['stop_label'] for i in ii),dates=len({entry[i]['meta']['date'] for i in ii})))
    save('all_fit_quantile_support_bins.json',ranges)
    bydate=[]
    for part in ['fit','calibration','validation']:
        for date in sorted({r['meta']['date'] for r,p in zip(entry,parts) if p==part}):
            for branch in [False,True]:
                ii=[i for i,r in enumerate(entry) if parts[i]==part and r['meta']['date']==date and r['features']['prefix_safe']==branch]
                if not ii:continue
                bydate.append(dict(partition=part,date=date,prefix_safe=branch,**ranks([entry[i] for i in ii],newp[ii])))
    save('all_date_branch_support.json',bydate)
    rawgroups=collections.defaultdict(list)
    for i,r in enumerate(entry):rawgroups[json.dumps(r['features'],sort_keys=True)].append(i)
    rawmixed=[ii for ii in rawgroups.values() if len({entry[i]['target']['stop_label'] for i in ii})>1]
    critical=[dict(case=c,neighbors=next(p for p in pairs if p['sample_id']==c['sample_id']),range=next(p for p in outliers if p['sample_id']==c['sample_id'])) for c in cases if c['model']=='RF146']
    # Keep the compact summary separate from64-tree case paths.
    for c in critical:c['case']={k:v for k,v in c['case'].items() if k!='trees'}
    save('case_comparison_summary.json',critical)
    bymodel=[]
    for name in ['RF140','RF146']:
        oo=[v for v in oob_rows if v['model']==name]
        bymodel.append(dict(model=name,row_oob_tree_counts=stats([v['row_oob_trees'] for v in oo]),
            rows_with_zero_whole_date_oob_trees=sum(v['whole_date_oob_trees']==0 for v in oo),rows=len(oo)))
    save('summary.json',dict(entry_rows=275,continue_rows=76,raw_feature_mixed_groups=len(rawmixed),raw_feature_distinct_groups=len(rawgroups),
        signature_mixed_groups={x['model']:x['mixed_groups'] for x in collisions},oob_support=bymodel,ranking=rankrows,neighbors=matchsummary,
        no_new_fit=True,no_new_policy=True,no_physical=True,final8_absent=True))
    verify();save('review.json',dict(passed=True,new_fits=0,native_predict_calls=0,new_physical=0,
        native_apply_route_checks=route_checks,leaf_weight_reconstructions=len(leaves),max_leaf_error=maxleaf,max_original_score_error=maxscore,
        manually_audited_oob_rows=len(oob_rows),independently_audited_auc_groups=len(rankrows),fit_only_geometry=True,
        cases_prespecified=True,old_sources_models_scores_cutoffs_labels_unchanged=True,no_threshold_or_policy_replay=True))
    save('state.json',dict(stage='analysed_pending_report'))
    print(json.dumps(read(O/'summary.json'),ensure_ascii=False))

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
