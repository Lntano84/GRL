from common import *
import shutil,platform,importlib.metadata as metadata
def main():
    assert not (OUT/'freeze.json').exists()
    old=ROOT/'outputs/grid97_effect_learning';enriched=ROOT/'outputs/grid98_feature_enrichment'
    paths=[]
    for stage in [old,enriched]:
        for n,h in read(stage/'freeze.json')['sha256'].items():assert sha(ROOT/n)==h,n
        assert read(stage/'review.json')['passed']
        complete=read(stage/'complete.json')
        for n,h in complete['archive_sha256'].items():assert sha(Path(complete['archive'])/n)==h,n
    data=read(old/'dataset.json');features={r['case_id']:r for r in read(enriched/'features.json')};prior=read(old/'design.json')
    for r in data:
        f=features[r['case_id']];assert f['original15']==r['features']
        r['features']=f['all35'];assert len(r['features'])==35
    assert len(data)==56 and sum(r['y_same_path'] for r in data)==39
    assert read(enriched/'original_folds_unchanged.json')==prior['folds']
    d=dict(stage='GRID99_ENRICHED_FEATURE_LEARNING',utc=utc(),authorization='用户：好的，训练一次吧。Fixed original15 vs enriched35 comparison, three existing model families once per frozen date fold.',
        feature_sets=dict(ORIGINAL15=prior['features'],ENRICHED35=list(data[0]['features'])),models=prior['models'],folds=prior['folds'],thresholds=prior['thresholds'],timing=prior['timing'],
        checkpoint_path='work/grid97_effect_learning/assets/tabpfn-v2-classifier-v2_default.ckpt',checkpoint=prior['checkpoint'],
        preprocessing='Same StandardScaler training-fold mean/std for logistic, raw input forest, internal train-context TabPFNv2 preprocessing. Keep every column of each predefined set including constants/duplicates. Equal row weight, no class reweighting or feature selection.',
        evaluation='53 nonpilot calendar-LOO primary; all56, three pilots, old13-to-generated22 nonpilots, cohorts, same-path incomplete and dates also reported. Same dates G1/G2 held out together.',
        target=prior['target'],sample_count=56,original15_repeat='Must reproduce GRID97 held-out predictions within 1e-5 for TabPFN, 1e-10 for LR/RF; reuse previous three baseline predictions with source hashes. Timing comes from this study only.',
        fixed_execution='18 date folds x 2 feature sets x 3 model families =108 fitted/prepared contexts; each once. Audit reconstructs36 TabPFN contexts only for verification, no gradient updates or selection. Three old baseline families reused, not refitted.',
        reporting='Paired AUC/BA/Brier differences and fixed0.5/0.95 omission/error counts. Descriptive development comparison, no significance claim, no selection of threshold/feature subset/model based on outputs.',
        limitations=prior['limitations']+['Features designed after seeing GRID97 negative results; not an independent confirmation','More inputs on56 states may overfit; constants/duplicates deliberately not searched away','Changed physical path is not necessarily economic damage','Feature extraction cost excluded from estimator timing; GRID98 pure arithmetic mean13.142ms is not total online cost'],
        forbidden=['new cases or labels','new physical forecasts/replays','hyperparameter/seed/threshold sweep','gradient training or finetuning of TabPFN','feature subset search','learned gate deployment','sum single-omission gains into policy savings'])
    save('dataset.json',data);save('design.json',d)
    save('environment.json',dict(python=platform.python_version(),executable=os.sys.executable,platform=platform.platform(),versions={n:metadata.version(n) for n in ['tabpfn','scikit-learn','torch','numpy','scipy','pandas','joblib']},threads=1,telemetry_disabled=True,new_installations=0))
    paths+=list(W.glob('*.py'))+[OUT/n for n in ['dataset.json','design.json','environment.json']]
    paths+=[old/n for n in ['dataset.json','design.json','freeze.json','review.json','predictions.json','fits.json','results.json','readout_verification.json']]
    paths+=[enriched/n for n in ['features.json','review.json','freeze.json','provenance.json','labels_and_groups_unchanged.json','original_folds_unchanged.json','summary.json','design.json','duplicate_columns_no_labels.json']]
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    for stage in [old,enriched]:hashes.update(read(stage/'freeze.json')['sha256'])
    save('freeze.json',dict(utc=utc(),sha256=hashes,frozen_before_current_fits=True))
    save('pipeline_state.json',dict(stage='frozen',expected_contexts=108,completed_contexts=0))
    archive=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID99_feature_learning/frozen';archive.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+[OUT/n for n in ['dataset.json','design.json','environment.json','freeze.json']]:shutil.copy2(p,archive/p.name)
    print('FROZEN',len(hashes),'items, 108 contexts; current model fitting not started')
if __name__=='__main__':main()
