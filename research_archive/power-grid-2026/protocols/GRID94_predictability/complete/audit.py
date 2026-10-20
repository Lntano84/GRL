"""Independent split, output-label and confusion-count checks for the frozen screen."""
from collections import Counter
import math
from data import ROOT,W,OUT,read,save,sha


def probability(d,x):
    if d['type']=='CONSTANT':return d['probability']
    if d['type']=='LOGISTIC':
        z=d['intercept']+sum((v-m)/s*b for v,m,s,b in zip(x,d['mean'],d['scale'],d['coefficient']))
        return 1/(1+math.exp(-max(-700,min(700,z))))
    x=[__import__('struct').unpack('f',__import__('struct').pack('f',v))[0] for v in x]
    node=0
    while d['children_left'][node]>=0:
        node=d['children_left'][node] if x[d['feature'][node]]<=d['threshold'][node] else d['children_right'][node]
    values=d['value'][node][0];return values[d['classes'].index(1)]/sum(values)


def main():
    freeze=read(OUT/'code_freeze.json');assert all(sha(ROOT/p)==h for p,h in freeze['files'].items())
    rows=read(OUT/'dataset.json');lookup={r['row_id']:r for r in rows};d=read(W/'design.json')
    features=d['features'];pred=read(OUT/'predictions.json');fits=read(OUT/'fits.json');metrics=read(OUT/'metrics.json')
    checks=Counter()
    assert len(lookup)==len(rows)==109 and len(features)==15
    assert not any(k in features for k in ['row_id','step','week','source','label_skip_same_choice','completion_queries','reward_gain','before_choice','after_choice'])
    model_predictions={(r['evaluation'],r['fold'],r['model'],r['row_id']):r for r in pred if r['model'] in ('LOGISTIC','TREE2')}
    for f in fits:
        train=[lookup[i] for i in f['train_rows']];test=[lookup[i] for i in f['test_rows']]
        assert not ({r['calendar_date'] for r in train}&{r['calendar_date'] for r in test})
        assert not ({r['state_hash'] for r in train}&{r['state_hash'] for r in test})
        if f['evaluation']=='GROUPED_OOF':
            assert {r['row_id'] for r in train}=={r['row_id'] for r in rows if r['calendar_date']!=f['fold']}
            assert {r['row_id'] for r in test}=={r['row_id'] for r in rows if r['calendar_date']==f['fold']}
        else:
            assert all(r['cohort']=='old13' for r in train) and all(r['cohort'] in ('G1','G2') for r in test)
        checks['split_and_state_isolation']+=1
        for r in test:
            p=model_predictions[(f['evaluation'],f['fold'],f['model'],r['row_id'])]
            assert abs(p['probability']-probability(f['details'],[r[k] for k in features]))<1e-12
            checks['independent_probabilities']+=1
    for p in pred:
        r=lookup[p['row_id']]
        assert all(p[k]==r[k] for k in r)
        assert r['label_skip_same_choice']==int(r['before_choice']==r['after_choice'])
        checks['prediction_provenance']+=1
    for m in metrics:
        pp=[p for p in pred if p['evaluation']==m['evaluation'] and p['model']==m['model'] and (m['cohort']=='ALL' or p['cohort']==m['cohort'])]
        chosen=[p for p in pp if p['probability']>=m['threshold']]
        wrong=[p for p in chosen if not p['label_skip_same_choice']]
        assert len(pp)==m['calls'] and len(chosen)==m['skipped_calls'] and len(wrong)==m['missed_choice_changes']
        assert sum(p['completion_queries'] for p in chosen)==m['nominal_queries_saved']
        assert {p['row_id'] for p in wrong}==set(m['missed_rows'])
        checks['metric_count_reconciliations']+=1
    for model in ['LOGISTIC','TREE2','ALWAYS_COMPLETE','ALWAYS_SKIP','TRAIN_PRIOR']:
        for evaluation in ['GROUPED_OOF','OLD_TO_GENERATED']:
            rr=[p['row_id'] for p in pred if p['model']==model and p['evaluation']==evaluation]
            expect=[r['row_id'] for r in rows if evaluation=='GROUPED_OOF' or r['cohort']!='old13']
            assert len(rr)==len(set(rr)) and set(rr)==set(expect)
            checks['prediction_coverage']+=1
    save(OUT/'review.json',dict(passed=True,checks=dict(checks),source_hashes_unchanged=True,
        model_fits=len(fits),new_forecasts=0,new_closed_loop_runs=0,
        interpretation='Classical classifier fits are authorized here; no controller/meta/RL weights were changed. Economic harmfulness remains unlabelled.'))
    print(dict(checks))


if __name__=='__main__':main()
