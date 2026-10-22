"""Build and seal a first-development simple-model experiment, before fitting."""
from pathlib import Path
import copy, hashlib, json, platform, shutil, sys
import numpy as np
import sklearn
from features import project, NAMES

ROOT=Path(__file__).resolve().parents[2]; W=Path(__file__).parent
SOURCE=ROOT/'outputs/grid100_data_foundation'; OUT=ROOT/'outputs/grid101_gain_learning'
ARCHIVE=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID101_gain_learning'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main():
    assert not OUT.exists(),'Preserve existing experiments'
    assert read(SOURCE/'bulk_complete.json')['passed'] and read(SOURCE/'bulk_review.json')['passed']
    meta=read(SOURCE/'bulk_dataset_manifest.json')['samples']
    assert len(meta)==2968
    OUT.mkdir(); rows=[]; matrix=[]; targets=[]; hashes={}; groups={}; checks=0
    for m in meta:
        role=m['partition'];assert role in ['train','validation']
        groups.setdefault(m['calendar_group'],set()).add(role)
        dataset=ROOT/m['dataset_path']; xp=dataset/m['input_file']; yp=dataset/m['target_file']
        x=read(xp)
        assert not set(x)&{'local_reward_gain','after_id','choice_changes','final_survival','whole_week_cost'}
        v=project(x)  # Targets not supplied to projection.
        poisoned=copy.deepcopy(x);poisoned.update(local_reward_gain=999999.,after_id=-999,whole_week_cost=-1.,date='2099-01-01',scenario='POISON')
        assert np.array_equal(v,project(poisoned))
        relabeled=copy.deepcopy(x);relabeled['public_subids']=[100001+17*int(s) for s in x['public_subids']]
        assert np.array_equal(v,project(relabeled)), 'Station identifiers entered scalar inputs'
        assert abs(v[1]-len(x['visible'])/len(x['public_order']))<1e-15
        assert abs(np.expm1(v[3])-x['known_completion_queries'])<1e-9
        y=read(yp);assert y['sample_id']==m['sample_id'] and y['local_reward_gain']>=0
        rows.append({k:m[k] for k in ['sample_id','partition','calendar_group','scenario','physical_state_group']})
        matrix.append(v);targets.append(y['local_reward_gain'])
        hashes.update({p.relative_to(ROOT).as_posix():sha(p) for p in [xp,yp]});checks+=1
    assert all(len(v)==1 for v in groups.values())
    assert sum(r['partition']=='train' for r in rows)==2378 and sum(r['partition']=='validation' for r in rows)==590
    design=dict(id='GRID101_GAIN_LEARNING',scope='One exploratory train/validation round, not final test or deployed control.',
        target='Continuous nonnegative immediate best-admissible float32 reward gain; classifier target is gain>0 exactly.',
        feature_names=NAMES,feature_scope='37 numeric prefix feedback/public-structure summaries; no full raw observation, calendar, IDs as scalars or future feedback.',
        split='Existing GRID10032 train/8 validation scenarios,16/4 date groups. All prefixes/replicates of a date retain role; final8 test scenarios unmaterialized.',
        weights='Each training calendar group has equal total weight, normalized to mean1. Main validation summaries macro over four date groups; pooled also shown.',
        constants=['zero_gain','training_date_weighted_mean_gain_and_positive_prior'],
        logistic=dict(C=1.,solver='lbfgs',max_iter=2000,random_state=20261022,class_weight=None),
        forest=dict(n_estimators=64,max_depth=4,min_samples_leaf=20,max_features=1.,random_state=20261022,n_jobs=1),
        fitting='Exactly1 scaler/logistic fit and1 regressor fit on train only. No hyperparameter search, val-selected threshold or refit. Fixed classifier threshold0.5.',
        evaluation=['positive-gain AUC and AP','classifier Brier/recall/balanced accuracy at0.5','regression MAE/RMSE, positive-only MAE',
                    'within-date top20% row-count selection: positive recall, reward-gain capture and known query fraction; offline diagnostic only'],
        top_fraction=.2,ranking_ties='Expected average over uniform choices within equal-score boundary block; no favorable sample-ID tie choice.',
        limitations='FULL-path local targets, correlated prefixes; gain>0 is not whole-week value. No physical queries, deployed omissions, actual latency/speedup or RL.',
        next='Report all outcomes, then user arranges further models/RL. Training/validation development can be tuned in future distinct records; final test remains sealed.')
    save(OUT/'design.json',design);save(OUT/'rows.json',rows)
    np.savez_compressed(OUT/'data.npz',X=np.stack(matrix),gain=np.asarray(targets,dtype=np.float64))
    save(OUT/'preflight.json',dict(passed=True,samples=checks,target_poison_checks=checks,station_relabel_checks=checks,
        independent_count_checks=checks*2,date_groups=len(groups),new_physical_calls=0,model_fits=0,final_test_materialized=False))
    save(OUT/'environment.json',dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,executable=sys.executable))
    for p in [SOURCE/'bulk_dataset_manifest.json',SOURCE/'design.json',SOURCE/'training_input_contract.json']:
        hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'data_freeze.json',hashes)
    files=list(W.glob('*.py'))+[OUT/n for n in ['design.json','rows.json','data.npz','preflight.json','environment.json']]
    save(OUT/'code_freeze.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    dest=ARCHIVE/'frozen';dest.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+list(OUT.iterdir()):
        if p.is_file():shutil.copy2(p,dest/p.name)
    print(json.dumps(dict(preflight_passed=True,samples=checks,features=len(NAMES),planned_model_fits=2,final_test_materialized=False)))


if __name__=='__main__':main()
