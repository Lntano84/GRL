"""Audit existing fit/calibration scores for expensive zero-gain completions."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import math
import sys
import numpy as np

R=Path('D:/Codex/GRL研究');O=R/'outputs/grid131_expensive_coverage';P=R/'outputs/grid129_baseline_risk';S=R/'outputs/grid130_calibrated_gate'
sys.path.insert(0,str(R/'work/grid107_direct_value'))
from common107 import sealed

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def auc_ap(y,score):
    from sklearn.metrics import roc_auc_score,average_precision_score
    y=np.asarray(y,bool);score=np.asarray(score,float)
    if not y.any() or y.all():return dict(auc=None,ap=float(y.mean()),positive_prevalence=float(y.mean()))
    a=score[y,None];b=score[None,~y]
    manual=float(np.mean((a>b)+.5*(a==b)));auc=float(roc_auc_score(y,score));assert abs(manual-auc)<1e-12
    npos=int(y.sum());oldtp=0;ap=0.
    for v in sorted(set(score),reverse=True):
        hit=score>=v;tp=int(y[hit].sum());ap+=(tp-oldtp)/npos*tp/int(hit.sum());oldtp=tp
    lib=float(average_precision_score(y,score));assert abs(ap-lib)<1e-12
    return dict(auc=auc,ap=lib,positive_prevalence=float(y.mean()))

def describe(ids,rows,scores,known_q,positive,lam):
    ids=np.asarray(ids,int);zero=ids[~positive[ids]];pos=ids[positive[ids]];paid=int(known_q[ids].sum());zeroq=int(known_q[zero].sum())
    above=zero[scores[zero]>float(scores[pos].max())] if len(pos) else np.array([],int)
    dd={}
    for d in sorted({rows[i]['date'] for i in ids}):
        ii=np.array([i for i in ids if rows[i]['date']==d]);zz=ii[~positive[ii]];den=int(known_q[ii].sum());skip=ii[scores[ii]>lam]
        dd[d]=dict(prefixes=len(ii),positive=int(positive[ii].sum()),zero=len(zz),
            full_queries=den,zero_queries=int(known_q[zz].sum()),zero_query_fraction=int(known_q[zz].sum())/den,
            gate_saved_query_fraction=int(known_q[skip].sum())/den,
            model=auc_ap(positive[ii],1-scores[ii]),count_only=auc_ap(positive[ii],known_q[ii]))
    valids=[v['model']['auc'] for v in dd.values() if v['model']['auc'] is not None]
    validq=[v['count_only']['auc'] for v in dd.values() if v['count_only']['auc'] is not None]
    return dict(prefixes=len(ids),calls=len({rows[i]['call'] for i in ids}),dates=len(dd),positive=len(pos),zero=len(zero),full_queries=paid,
        zero_queries=zeroq,zero_query_fraction=zeroq/paid,
        zero_calls=len({rows[i]['call'] for i in zero}),positive_calls=len({rows[i]['call'] for i in pos}),
        fixed_gate_skipped=int((scores[ids]>lam).sum()),fixed_gate_skipped_zero=int((scores[zero]>lam).sum()),
        zero_score_median=float(np.median(scores[zero])) if len(zero) else None,positive_score_median=float(np.median(scores[pos])) if len(pos) else None,
        max_positive_score=float(scores[pos].max()) if len(pos) else None,
        within_group_separation_diagnostic=dict(zero_scores_above_all_positive_scores=len(above),their_queries=int(known_q[above].sum()),
            fraction_of_zero_query_mass=int(known_q[above].sum())/zeroq if zeroq else None,
            note='Descriptive class-overlap boundary,not a selected or deployed alternative gate. No validation evaluation.'),
        pooled_model=auc_ap(positive[ids],1-scores[ids]),pooled_count_only=auc_ap(positive[ids],known_q[ids]),
        date_macro_auc_model=float(np.mean(valids)) if valids else None,date_macro_auc_count_only=float(np.mean(validq)) if validq else None,
        dates_with_two_classes=len(valids),per_date=dd)

def main():
    assert not O.exists();O.mkdir(parents=True);sealed()
    paths=[Path(__file__),P/'train_label_ledger.jsonl.gz',P/'train_public_inputs.npz',S/'train_calibration_scores.npz',
        S/'fits.json',S/'model.joblib',R/'outputs/grid101_gain_learning/design.json',S/'design.json']
    freeze={p.relative_to(R).as_posix():sha(p) for p in paths}
    save('plan.json',dict(id='GRID131_EXPENSIVE_COVERAGE',scope='Existing12fit/4calibration dates only,no new validation access/scoring.',
        expensive='Known remaining candidates>64,defined by original local batch size;not selected from outcome percentiles.',
        questions=['How much zero-gain paid-query mass remains in multi-batch tasks?','Does one rare cheap positive determine the global gate?','How much do expensive positive/zero score ranges overlap?','Does the37-feature LR improve on known count ranking inside expensive tasks?', 'Which exact linear contributions differ across cohorts?'],
        restrictions=['no fits','no changed gate or validation results','no group-specific deployment','no threshold search','no new data/physics/test','no causal or statistical significance claim']))
    with gzip.open(paths[1],'rt',encoding='utf8') as f:rows=[json.loads(line) for line in f]
    with np.load(paths[2]) as z:X=z['X'][:,:37].astype(float);ids=z['sample_ids']
    with np.load(paths[3]) as z:score=z['score'];fit=z['fitmask'];cal=z['calmask']
    assert len(rows)==len(X)==len(score)==2378 and all(str(sid)==r['sample_id'] for sid,r in zip(ids,rows))
    D=read(paths[4]);lam=D['calibrated_lambda'];q=np.array([r['known_saved_queries_if_skip'] for r in rows]);positive=np.array([r['full_baseline_gain']>0 for r in rows])
    assert np.array_equal(fit,np.array([r['date'] in D['fit_dates'] for r in rows])) and np.array_equal(cal,~fit)
    assert sha(S/'model.joblib')==D['model_sha256']
    summaries={}
    for part,mask in [('fit',fit),('calibration',cal)]:
        summaries[part]={}
        for label,cond in [('all',np.ones(len(q),bool)),('one_batch',q<=64),('multi_batch',q>64),('batch2',(q>64)&(q<=128)),('batch3',(q>128)&(q<=192)),('batch4',q>192)]:
            ii=np.flatnonzero(mask&cond)
            if len(ii):summaries[part][label]=describe(ii,rows,score,q,positive,lam)
        assert summaries[part]['one_batch']['full_queries']+summaries[part]['multi_batch']['full_queries']==summaries[part]['all']['full_queries']
        assert summaries[part]['one_batch']['zero_queries']+summaries[part]['multi_batch']['zero_queries']==summaries[part]['all']['zero_queries']
    anchors=[dict(rows[i],score=float(score[i])) for i in np.flatnonzero(cal&positive) if score[i]==lam]
    assert anchors and not ((score>lam)&positive&cal).any()
    # Exact linear decomposition of already saved scores;no new scored states or changed coefficients.
    import joblib
    model=joblib.load(S/'model.joblib');scaler=model['scaler'];coef=model['classifier'].coef_[0]
    zz=(X-scaler.mean_)/scaler.scale_;contrib=zz*coef;logits=contrib.sum(1)+model['classifier'].intercept_[0]
    error=float(abs(1/(1+np.exp(logits))-score).max());assert error<1e-12
    names=read(paths[6])['feature_names'];feature_rows=[]
    for j,name in enumerate(names):
        row=dict(feature=name,coefficient=float(coef[j]),groups={})
        for part,m in [('fit',fit),('calibration',cal)]:
            for label,c in [('cheap_zero',(q<=64)&~positive),('expensive_zero',(q>64)&~positive),('expensive_positive',(q>64)&positive)]:
                mm=m&c;row['groups'][part+'_'+label]=dict(n=int(mm.sum()),raw_mean=float(X[mm,j].mean()),raw_quantiles=np.quantile(X[mm,j],[.25,.5,.75]).tolist(),mean_logit_contribution=float(contrib[mm,j].mean()))
        row['calibration_expensive_zero_minus_cheap_zero_logit']=row['groups']['calibration_expensive_zero']['mean_logit_contribution']-row['groups']['calibration_cheap_zero']['mean_logit_contribution']
        row['fit_positive_minus_zero_expensive_logit']=row['groups']['fit_expensive_positive']['mean_logit_contribution']-row['groups']['fit_expensive_zero']['mean_logit_contribution']
        row['calibration_positive_minus_zero_expensive_logit']=row['groups']['calibration_expensive_positive']['mean_logit_contribution']-row['groups']['calibration_expensive_zero']['mean_logit_contribution']
        feature_rows.append(row)
    expected=float(logits[cal&(q>64)&~positive].mean()-logits[cal&(q<=64)&~positive].mean())
    assert abs(sum(v['calibration_expensive_zero_minus_cheap_zero_logit'] for v in feature_rows)-expected)<1e-12
    save('cohorts.json',summaries);save('global_calibration_anchor.json',anchors);save('all_feature_contributions.json',feature_rows)
    save('source_hashes.json',freeze);save('review.json',dict(passed=True,original_gate_unchanged=True,source_hashes_unchanged=(freeze=={p.relative_to(R).as_posix():sha(p) for p in paths}),
        no_new_validation_evaluation=True,all_train_calibration_rows_retained=True,query_mass_conserved=True,
        auc_pairwise_and_ap_tie_group_recomputed=True,all37_feature_contributions_retained=True,linear_decomposition_conserved=True,
        reconstructed_saved_scores=len(score),new_scored_states=0,independent_numpy_error=error,
        fits=0,physical_or_prediction_calls=0,new_data=0,final_tests_absent=True))
    sealed();print(json.dumps({'anchor':anchors,'calibration_multi':summaries['calibration']['multi_batch'],'fit_multi':summaries['fit']['multi_batch']},ensure_ascii=False))

if __name__=='__main__':main()
