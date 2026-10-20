from common import *
import importlib.metadata as metadata,platform,shutil
def main():
    assert not (OUT/'freeze.json').exists(),'Do not overwrite an existing frozen study'
    clean=ROOT/'outputs/frame_label_cleanup_20261020'
    for n,h in read(clean/'artifact_manifest.json').items():assert sha(clean/n)==h,n
    original=ROOT/'outputs/grid96_value_learning'
    for n,h in read(clean/'cleanup_review.json')['source_sha256'].items():assert sha(original/n)==h,n
    raw=read(clean/'clean_records.json');features=read(original/'design.json')['learning']['features']
    data=[dict(**r['metadata'],features=r['pre_completion_features'],y_same_path=int(r['outcome']['physical_path_identical']),both_complete=r['outcome']['both_complete'],original_outcome=r['outcome']) for r in raw]
    assert len(data)==56 and sum(r['y_same_path'] for r in data)==39 and len(features)==15
    assert len({r['case_id'] for r in data})==56
    folds=read(clean/'06_既有日期留组索引.json')['folds'];lookup={r['case_id']:r for r in data}
    assert len(folds)==18
    for f in folds:
        a=set(f['train_ids']);b=set(f['test_ids']);assert not a&b and a|b==set(lookup)
        if f['scheme']=='calendar_loo':assert not {lookup[i]['calendar_date'] for i in a}&{lookup[i]['calendar_date'] for i in b}
        else:assert all(lookup[i]['cohort']=='old13' for i in a) and all(lookup[i]['cohort'] in ['G1','G2'] for i in b)
    d=dict(id='GRID97_EFFECT_LEARNING',utc=utc(),target='y=1: audited identical entire physical path after omitting one completion; y=0: physical trajectory changes. Not an economic value label.',
        user_scope='Compare logistic, shallow forest and local pretrained TabPFNv2 offline. No physical rollouts or deployed policy.',n=56,features=features,folds=folds,
        weighting='All three methods receive each training state once, equal row weight, no class oversampling. This differs from GRID96 date-weighted three-class task. Report date-macro accuracy and balanced accuracy too.',
        models=dict(LOGISTIC=dict(C=1.,solver='lbfgs',max_iter=2000,random_state=20261020),FOREST64=dict(n_estimators=64,max_depth=4,min_samples_leaf=3,max_features='sqrt',random_state=20261020,n_jobs=1),TABPFN_V2=dict(n_estimators=4,device='cpu',random_state=20261020,n_jobs=1,fit_mode='fit_with_cache')),
        thresholds=[.5,.95],threshold_rule='Fixed in advance. >= threshold proposes omission. A changed-path false omission is not necessarily economically harmful.',
        primary='53 non-pilot calendar-LOO predictions; all 56 also reported, three known pilots separate. Old13-to-G1/G2 transfer secondary; same calendar G1/G2 withheld together.',
        timing=dict(cpu_threads=1,fit_context_repeats=1,first_batch_predict=1,warm_batch_repeats=3,single_row_repeats_per_case=1,audit_context_reconstruction=1,
            convention='Host perf_counter; feature scaling included in prediction. Imports/downloads excluded and context/model load counted in fit. No latency superiority claim from three timing repetitions. All single-row calls scored and compared to batch; online gate cost uses single-row timing.'),
        checkpoint=dict(repo='Prior-Labs/TabPFN-v2-clf',revision='f851f2a3c941544733b712d8c0f96dfae9b28862',file='tabpfn-v2-classifier-v2_default.ckpt',sha256=sha(W/'assets/tabpfn-v2-classifier-v2_default.ckpt'),license='Prior Labs License 1.1, Apache2 derivative plus attribution; local inference only; no uploaded data'),
        forbidden=['extra cases','threshold/model/seed selection from outcomes','gradient training or fine tuning of TabPFN','new physical simulations','closed-loop savings claim from offline scores','forcing MIXED into good/bad'],
        limitations=['Already exposed development states, no new independent test','9 SAME_PATH states have both branches end early','Only 17 CHANGED_PATH; only one KEEP, a previously known pilot','Path identity differs from utility; changed trajectory can benefit or hurt','No independent AC re-solve','Seed and timing repetitions are not independent scenarios'])
    save('dataset.json',data);save('design.json',d)
    versions={}
    for n in ['tabpfn','scikit-learn','torch','numpy','scipy','pandas','einops','huggingface-hub','pydantic','pydantic-settings','tabpfn-common-utils','joblib']:versions[n]=metadata.version(n)
    save('environment.json',dict(python=platform.python_version(),executable=os.sys.executable,versions=versions,platform=platform.platform(),telemetry_disabled=True,offline_weight=True,
        synthetic_preflight='24 independent Gaussian synthetic rows, 15 features; 20 train/4 test binary classification passed before current data predictions. No protocol test or prediction outcome seen.',old_runtime_untouched=True))
    paths=list(W.glob('*.py'))+list((W/'assets').glob('*'))+[OUT/n for n in ['dataset.json','design.json','environment.json']]
    paths+=[clean/n for n in read(clean/'artifact_manifest.json')]+[clean/'artifact_manifest.json']
    paths+=[original/n for n in read(clean/'cleanup_review.json')['source_sha256']]
    package=W/'.venv/Lib/site-packages'
    paths+=list((package/'tabpfn').rglob('*.py'))+list((package/'tabpfn_common_utils').rglob('*.py'))
    save('freeze.json',dict(utc=utc(),sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},source_manifest_verified=True))
    save('pipeline_state.json',dict(stage='frozen_before_current_predictions',utc=utc(),expected_model_contexts=54,expected_baseline_contexts=54,new_physical_replays=0))
    print('Frozen',len(paths),'items; predictions not started')
if __name__=='__main__':main()
