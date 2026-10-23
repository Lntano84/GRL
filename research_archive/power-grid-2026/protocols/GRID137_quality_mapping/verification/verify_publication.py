"""Validate publication, explicit missing values and all retained file hashes."""
from run import *

def main():
    verify();rows=read(O/'quality_training_evidence_v2.json');old=read(O/'quality_evidence_ledger.json')
    assert len(rows)==len(old)==56 and len({r['case_id'] for r in rows})==56
    for a,b in zip(old,rows):
        assert all(b[k]==v for k,v in a.items() if k!='nominal_cost_supervision_known')
        assert b['nominal_cost_comparison_known']==bool(b['full_complete'] and b['skip_complete'])
        if b['nominal_cost_comparison_known']:
            assert b['single_omission_quality_pass']==(b['skip_total_cost']-b['full_cost']<=.005*b['full_cost'])
        elif b['single_omission_quality_pass'] is not False:assert b['single_omission_quality_pass'] is None
    preds=read(R/'outputs/grid96_value_learning/predictions.json')
    assert len({(p['scheme'],p['model'],p['case_id']) for p in preds})==len(preds)
    summary=read(O/'training_feasibility.json')
    assert summary['qualified_identical']+summary['qualified_action_changed']==summary['qualified_cases']==45
    assert sum(summary['qualified_query_outcomes'].values())==45
    fold=[r for r in summary['nonpilot_date_folds'] if r['test_fail']]
    assert len(fold)==1 and fold[0]['train_fail']==0 and fold[0]['test_fail']==1
    n=0
    for phase in ['frozen','complete','publication']:
        root=A/phase
        for p,h in read(root/'archive_manifest.json').items():assert sha(root/p)==h;n+=1
    c=read(O/'collaboration_receipt.json');collab=R/c['directory']
    for p,h in c['hashes'].items():assert sha(collab/p)==h and sha(O/p)==h
    save('final_verification.json',dict(passed=True,archive_files_verified=n,collaboration_files_verified=len(c['hashes']),
        frozen_source_files_verified=len(read(O/'source_hashes.json')),case_ids_unique=True,saved_predictions_unique=True,
        schema_semantics_verified=True,cost_threshold_independent_difference_check=True,original_outputs_retained=True,
        fits=0,forwards=0,physical=0,final_tests_absent=True))
    root=A/'verification';root.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*[O/x for x in ['analysis.log','publication.log','final_verification.json']]]:shutil.copy2(p,root/p.name)
    (root/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in root.iterdir()},indent=2)+'\n',encoding='utf8')
    print('GRID137_FINAL_VERIFIED',n,len(c['hashes']))

if __name__=='__main__':main()
