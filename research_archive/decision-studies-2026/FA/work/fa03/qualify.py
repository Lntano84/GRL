"""FA03 dataset gate only. Do not fit models or reinterpret memout runtimes."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib,json,math
import arff
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'work/fa03/source/QBF-2011'
OUT=ROOT/'outputs/fa03'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name):return arff.load((SOURCE/(name+'.arff')).open())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    design=json.loads((ROOT/'outputs/fa03_design/FA03_design_freeze.json').read_text())
    assert digest(ROOT/'outputs/fa03_design/FA03_next_step.md')==design['plan_sha256']
    runs=load('algorithm_runs')['data'];cv=load('cv')['data']
    features=load('feature_values');statuses=load('feature_runstatus')['data']
    ids=sorted({r[0] for r in runs});algos=sorted({r[2] for r in runs})
    records={(r[0],r[2]):r for r in runs};counts=Counter(r[4] for r in runs)
    checks=dict(repetition_one=all(r[1]==1 for r in runs),
        unique_complete_matrix=len(records)==len(runs)==len(ids)*len(algos),
        expected_shape=len(ids)==1368 and len(algos)==5,
        finite_nonnegative=all(r[3] is not None and math.isfinite(r[3]) and r[3]>=0 for r in runs),
        successful_runtime_within_native_cutoff=all(r[3]<=3600 for r in runs if r[4]=='ok'),
        timeout_at_native_cutoff=all(r[3]==3600 for r in runs if r[4]=='timeout'),
        cv_ids_match=len(cv)==len(ids) and len({r[0] for r in cv})==len(cv) and {r[0] for r in cv}==set(ids),
        frozen_ids_match=ids==design['ids'],
        feature_ids_unique_match=len(features['data'])==len(ids) and len({r[0] for r in features['data']})==len(ids) and {r[0] for r in features['data']}==set(ids),
        feature_status_ids_unique_match=len(statuses)==len(ids) and len({r[0] for r in statuses})==len(ids) and {r[0] for r in statuses}==set(ids),
        inherited_description_matches=digest(SOURCE/'description.txt')==digest(ROOT/'work/fa03_design/source/DATASETS/QBF-2011/description.txt'),
        inherited_cv_matches=digest(SOURCE/'cv.arff')==digest(ROOT/'work/fa03_design/source/DATASETS/QBF-2011/cv.arff'))
    fold={r[0]:int(r[2]) for r in cv}
    checks['test_fold_matches']=[i for i,x in enumerate(ids) if fold[x]==1]==design['test']
    pool=np.array([i for i,x in enumerate(ids) if fold[x]!=1]);shuffled=np.random.default_rng(20261001).permutation(pool)
    nval=math.ceil(len(pool)*.1)
    checks['train_validation_match']=np.sort(shuffled[:nval]).tolist()==design['validation'] and np.sort(shuffled[nval:]).tolist()==design['train']
    X=np.array([[float(v) if v is not None else np.nan for v in r[2:]] for r in features['data']])
    checks['feature_no_infinity']=not bool(np.isinf(X).any())
    checks['46_features']=X.shape==(1368,46)
    # Independent text-tail parsing, not the ARFF parser's status output.
    text=(SOURCE/'algorithm_runs.arff').read_text();body=text.split('@DATA',1)[1] if '@DATA' in text else text.split('@data',1)[1]
    tail=Counter();memout_times=Counter()
    for line in body.splitlines():
        line=line.strip()
        if not line or line.startswith('%'):continue
        _,runtime,status=line.rsplit(',',2);status=status.strip().strip("'\"")
        tail[status]+=1
        if status=='memout':memout_times[float(runtime)]+=1
    checks['independent_status_counts_match']=tail==counts
    checks['independent_memout_values_match']=memout_times==Counter(r[3] for r in runs if r[4]=='memout')
    supported={'ok','timeout'};unsupported=sorted(set(counts)-supported)
    checks['feedback_status_supported']=not unsupported
    # This is a cost/feedback gate, not a rejection of ASLib's numerical format.
    # The source paper, section 4/page 5, states that unsuccessful and memory-out
    # runs were assigned the timeout value; it does not report failure event times.
    provenance=ROOT/'work/fa03/provenance'
    evidence=json.loads((provenance/'manifest.json').read_text())
    for item in evidence:
        assert digest(provenance/item['file'])==item['sha256']
    gate='PASSED' if all(checks.values()) else 'NOT_QUALIFIED'
    split_counts={name:dict(Counter(records[(ids[i],a)][4] for i in design[name] for a in algos)) for name in ['train','validation','test']}
    file_hashes={str(p.relative_to(ROOT)):digest(p) for p in SOURCE.iterdir() if p.is_file()}
    file_hashes[str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
    result=dict(stage='FA03',qualification=gate,checked_utc=datetime.now(timezone.utc).isoformat(),
        checks=checks,unsupported_statuses=unsupported,status_counts=dict(counts),
        native_cutoff_s=3600,planned_cap_s=600,planned_budget_s=249300,
        memout_runtime_histogram=dict(memout_times),memout_fraction=counts['memout']/len(runs),
        split_status_counts=split_counts,feature_shape=list(X.shape),feature_missing_values=int(np.isnan(X).sum()),
        feature_status_counts=dict(Counter(v for r in statuses for v in r[2:])),
        feature_cost_file_available=(SOURCE/'feature_costs.arff').exists(),
        feature_cost_handling='unknown; not zero; no end-to-end speed claim',
        actual_trajectories=0,model_fits=0,test_scores_computed=0,file_hashes=file_hashes,
        numerical_data_qualification='PASSED' if all(v for k,v in checks.items() if k!='feedback_status_supported') else 'FAILED',
        cost_feedback_qualification='NOT_QUALIFIED',
        provenance=evidence,
        provenance_finding=dict(source='kotthoff_evaluation_2012.pdf',section='4',printed_page=5,
            pdf_page=5,url='https://www.cs.uwyo.edu/~larsko/papers/kotthoff_evaluation_2012.pdf',
            finding='Source paper assigns timeout runtime to memory-out or unsolved runs; recorded 3600 is not an established measurement of the memory-failure event time.'),
        reason='1720 memout rows record 3600 seconds. The data author documents assignment of timeout values to unsuccessful/memory-out runs. Actual failure event times are not provided in the acquired tables; using these assigned values as execution costs or 600-second censored feedback would introduce an unfrozen assumption.',
        scope='Numerical dataset checks passed; frozen actual-cost/feedback replay not qualified; strategy/research-value comparison not performed.')
    (OUT/'FA03_qualification.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    (OUT/'RUN_STATE.json').write_text(json.dumps(dict(status='stopped_at_qualification',trajectories_started=0,
        completed=0,target=15,models_fitted=0,active_jobs=False),indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ['qualification','status_counts','unsupported_statuses','memout_runtime_histogram','feature_shape','feature_missing_values','feature_status_counts','split_status_counts']},indent=2))

if __name__=='__main__':main()
