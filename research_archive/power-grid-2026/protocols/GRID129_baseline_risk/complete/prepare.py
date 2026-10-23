"""Train-only baseline-relative loss ledger and risk-control qualification; no fits."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import math
import pickle
import sys
import numpy as np

R=Path('D:/Codex/GRL研究');O=R/'outputs/grid129_baseline_risk'
sys.path.insert(0,str(R/'work/grid107_direct_value'))
from common107 import sealed

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def crc_select(losses,levels,alpha):
    """Synthetic bounded monotone-loss qualification only,not applied to learned scores."""
    a=np.asarray(losses,float);assert a.ndim==2 and a.shape[1]==len(levels)
    assert (a>=0).all() and (a<=1).all() and (np.diff(a,axis=1)<=0).all()
    n=len(a);ok=np.flatnonzero((a.sum(0)+1)/(n+1)<=alpha)
    return levels[int(ok[0])] if len(ok) else 'BASELINE_FALLBACK'

def main():
    assert not O.exists();O.mkdir(parents=True);sealed()
    paths=[Path(__file__),R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',
        R/'outputs/grid101_gain_learning/GRID101_report.md',R/'outputs/grid121_date_transfer/design.json',
        R/'outputs/grid128_learning_rethink/horizon_audit.json',R/'work/grid101_gain_learning/features.py']
    freeze={p.relative_to(R).as_posix():sha(p) for p in paths}
    cases=[c for c in pickle.loads(paths[1].read_bytes()) if c['partition']=='train']
    assert len(cases)==2378 and len({c['date'] for c in cases})==16
    ledger=[];checks=0
    for c in cases:
        gains=[float(x) for x in c['gains']];queries=[int(x) for x in c['queries']]
        g=math.fsum(gains);q=sum(queries);same=c['chosen'][0]==c['chosen'][-1]
        assert g>=0 and q>0 and len(c['states'][0])==41
        assert c['states'][0][37]==0 and c['states'][0][38]==0 and c['states'][0][40]==1
        assert abs(math.fsum(float(v) for v in c['gains'])-float(c['gains'].sum()))<1e-12
        # Do not replace a tail-loss label by next-batch loss,especially across zero-gain barriers.
        for k in range(len(gains)+1):
            assert abs(math.fsum(gains[:k])+math.fsum(gains[k:])-g)<1e-12;checks+=1
        ledger.append(dict(sample_id=c['sample_id'],date=c['date'],scenario=c['scenario'],call=c['physical_state_group'],
            full_baseline_gain=g,known_saved_queries_if_skip=q,action_changed_if_skip=not same,
            zero_full_gain=(g==0),first_batch_gain=gains[0],tail_gain_after_first=math.fsum(gains[1:]),
            batches=len(gains),baseline_action=c['chosen'][-1],skip_action=c['chosen'][0]))
    assert all(v['zero_full_gain']==(not v['action_changed_if_skip']) for v in ledger)
    dates={}
    for d in sorted({v['date'] for v in ledger}):
        rs=[v for v in ledger if v['date']==d];Q=sum(v['known_saved_queries_if_skip'] for v in rs);G=math.fsum(v['full_baseline_gain'] for v in rs)
        assert G>=0
        dates[d]=dict(prefixes=len(rs),calls=len({v['call'] for v in rs}),total_full_gain=G,total_queries=Q,
            zero_gain_prefixes=sum(v['zero_full_gain'] for v in rs),
            zero_gain_oracle_query_saving=sum(v['known_saved_queries_if_skip'] for v in rs if v['zero_full_gain'])/Q,
            first_zero_later_positive=sum(v['first_batch_gain']==0 and v['tail_gain_after_first']>0 for v in rs))
    # Check candidate loss definitions on a deterministic nested gate,not an empirical gate search.
    # The ordering is artificial and label-free;no score or threshold is evaluated on held-out data.
    idx=sorted(range(len(ledger)),key=lambda i:hashlib.sha256(ledger[i]['sample_id'].encode()).hexdigest())
    ranks={i:rank for rank,i in enumerate(idx)}
    old={d:(1.,1.) for d in dates};monotone_checks=0
    for keep_after in [0,len(idx)//4,len(idx)//2,3*len(idx)//4,len(idx)]:
        # Increasing keep_after removes SKIP interventions and falls back more often.
        for d in dates:
            ids=[i for i,v in enumerate(ledger) if v['date']==d];skip=[i for i in ids if ranks[i]>=keep_after]
            den=dates[d]['total_full_gain'];loss=math.fsum(ledger[i]['full_baseline_gain'] for i in skip)/den if den else 0.
            mismatch=sum(ledger[i]['action_changed_if_skip'] for i in skip)/len(ids)
            assert 0<=loss<=old[d][0]+1e-12 and 0<=mismatch<=old[d][1]+1e-12
            old[d]=(loss,mismatch);monotone_checks+=1
    assert all(x==(0.,0.) for x in old.values())
    # The CRC correction is a property of this calibration rule,not a lower bound for every method.
    correction={str(n):1/(n+1) for n in [4,8,12,16]}
    assert crc_select(np.zeros((4,2)),[.5,.9],.05)=='BASELINE_FALLBACK'
    assert crc_select(np.zeros((19,2)),[.5,.9],.05)==.5
    save('plan.json',dict(id='GRID129_BASELINE_RISK',scope='Train-only preparation and mathematical qualification; no model or threshold selection.',
        baseline='Original FRAME full remaining locked-station completion,shared global policy unchanged.',
        intervention='First version: one decision before the local stage. SKIP returns the current observed best; otherwise run the unchanged full local baseline. No per-batch learned decisions.',
        labels='Entire remaining completion gain and action disagreement,kept separately from public inputs. GRID101 already predicted full-completion gain; this label is not claimed new.',
        loss='For each date, skipped full-baseline gains / sum full-baseline gains (zero if denominator zero); separate skipped action disagreements / all prefixes. Both in[0,1] and monotone as a frozen gate becomes more conservative.',
        efficiency='Saved known local queries relative to baseline; no actual-time or whole-week claim.',
        gate_concept='Frozen pre-query no-improvement score s(x);SKIP iff s(x)>lambda. Larger lambda reverts more often to baseline. Score alone is not calibrated reliability.',
        calibration='Fit and calibration calendar dates must be disjoint;all same-date replicas/calls/prefixes stay together. Existing four-date folds may define development splits,not independent final tests.',
        no_guarantee='Date exchangeability/distribution assumptions unverified;CRC controls expected chosen local loss under assumptions,not each-date loss,physical safety or whole-week cost. Risk alpha not selected here.',
        previous_work='GRID101 full-gain LR/RF ranking;GRID120 weighted classification failed. New question is baseline-relative risk and calibrated intervention,not sign reversal or fresh loss replacement.',
        prohibited=['fitting','model_forward','new_validation_evaluation','threshold_or_alpha_selection','new_scenarios','physical_calls','final_test','paper_or_figures']))
    with gzip.open(O/'train_label_ledger.jsonl.gz','wt',encoding='utf8') as f:
        for row in ledger:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    np.savez_compressed(O/'train_public_inputs.npz',X=np.stack([c['states'][0] for c in cases]),known_queries=np.array([v['known_saved_queries_if_skip'] for v in ledger]),sample_ids=np.array([c['sample_id'] for c in cases]))
    summary=dict(prefixes=len(ledger),dates=len(dates),calls=len({v['call'] for v in ledger}),
        zero_full_gain_prefixes=sum(v['zero_full_gain'] for v in ledger),zero_gain_action_change=sum(v['zero_full_gain'] and v['action_changed_if_skip'] for v in ledger),
        first_batch_zero_later_gain=sum(v['first_batch_gain']==0 and v['tail_gain_after_first']>0 for v in ledger),
        zero_loss_oracle_query_saving_date_macro=float(np.mean([x['zero_gain_oracle_query_saving'] for x in dates.values()])),per_date=dates)
    save('train_summary.json',summary)
    save('calibration_feasibility.json',dict(rule='(sum_i L_i(lambda)+B)/(n+1)<=alpha,B=1;else original-baseline fallback.',
        correction_for_calibration_date_counts=correction,
        illustrations_only_not_selected_risk=[dict(alpha=.05,necessary_zero_empirical_loss_n=19),dict(alpha=.01,necessary_zero_empirical_loss_n=99)],
        statement='These are arithmetic necessary sizes for non-fallback acceptance by this standard CRC correction with B=1 and nonnegative empirical loss;not sufficient guarantees or universal sample-complexity bounds. Independent fit data would still be required.',
        limitations=['No exchangeability established for seasonal dates.','Original4validation dates repeatedly exposed,not a new independent calibration/test certificate.','16 available training dates cannot all be calibration dates for a model fitted on those same16dates.','Current three neural seeds are not a calibration distribution.','Loss here is local gain/mismatch,not the existing0.5% whole-week-cost tolerance.']))
    assert freeze=={p.relative_to(R).as_posix():sha(p) for p in paths};sealed();save('source_hashes.json',freeze)
    save('review.json',dict(passed=True,baseline_label_identity_checks=checks,monotone_group_loss_checks=monotone_checks,
        train_cases_all_retained=True,public_inputs_separate_from_labels=True,new_model_fits=0,new_model_forwards=0,
        new_validation_scoring=0,new_data=0,physical_or_prediction_calls=0,final_tests_absent=True,source_unchanged=True,
        note='No empirical gate selected. Label-free hash ordering only checks loss monotonicity,not predictive usefulness. Oracle zero-loss capacity is privileged hindsight.'))
    print(json.dumps({k:v for k,v in summary.items() if k!='per_date'},ensure_ascii=False))

if __name__=='__main__':main()
