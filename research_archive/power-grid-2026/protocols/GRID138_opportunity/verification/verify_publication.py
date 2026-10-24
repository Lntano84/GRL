"""Independent saved arithmetic and publication manifest checks."""
from run import *
from scope_resume01 import v

def main():
    v();assert read(O/'publication_receipt.json')['passed'];assert not (O/'final_verification.json').exists()
    r=read(O/'timing_runs.json');assert len(r)==48 and len({(x['week'],x['repeat'],x['method']) for x in r})==48
    for x in r:
        assert abs((x['optimizer_s']+x['all_forecast_s']+x['search_overhead_s']+x['other_s'])-x['controller_s'])<1e-7
        assert x['local_after64_s']<=x['local_s'] and x['unchanged_choice_local_s']<=x['local_s']+1e-7
    h=read(O/'historical_timing_runs.json');assert len(h)==25 and sum(x['calls'] for x in h)==179
    for x in h:
        assert 0<=x['local_share_lower']<=x['local_share_upper']<=1
        # Raw local-row time independently brackets pure simulate time after subtracting all extra processing.
        assert abs(x['local_share_upper']*x['controller_s']-x['local_outer_s'])<1e-8
        assert abs(x['local_share_lower']*x['controller_s']-max(0,x['local_outer_s']-x['timer_extra_s']))<1e-8
    led=read(O/'public_feature_outcome_ledger.json');assert len(led)==56
    changed=[x for x in led if not x['physical_path_identical']];assert len(changed)==17
    assert sum(x['cost_direction']=='higher' for x in changed)==9 and sum(x['cost_direction']=='lower' for x in changed)==7
    assert sum(x['cost_direction']=='not_comparable' for x in changed)==1
    dates={x['case_id']:x['date'] for x in led};pilots={x['case_id']:x['pilot_known'] for x in led}
    n=read(O/'date_excluded_feature_neighbors.json');assert len(n)==17
    for x in n:
        for y in x['all_distances']:assert dates[y['case_id']]!=x['date'] and not pilots[y['case_id']]
        assert x['nearest3']==x['all_distances'][:3]
    import scipy.stats
    completed=[x for x in changed if x['cost_delta'] is not None]
    correlations=read(O/'all_feature_descriptive_correlations.json')['rows'];assert len(correlations)==15
    for x in correlations:
        if x['spearman'] is None:continue
        actual=float(scipy.stats.spearmanr([z['features'][x['feature']] for z in completed],[z['cost_delta'] for z in completed]).statistic)
        assert abs(actual-x['spearman'])<1e-12
    files=0
    for root in A.iterdir():
        if not root.is_dir():continue
        for p,hsh in read(root/'archive_manifest.json').items():assert sha(root/p)==hsh;files+=1
    c=read(O/'collaboration_receipt.json')
    for p,hsh in c['hashes'].items():assert sha(R/c['directory']/p)==hsh and sha(O/p)==hsh
    assert sha(W/'scope_resume02.py')==read(O/'scope_count_correction.json')['script_sha256']
    save('final_verification.json',dict(passed=True,archive_files_verified=files,collaboration_files_verified=len(c['hashes']),
        raw_timing_arithmetic_and_intervals_checked=True,all_neighbors_date_excluded=True,all_nonconstant_correlations_checked_with_scipy=True,
        frozen_sources_unchanged=True,engineering_errors_preserved=True,new_fits=0,new_forwards=0,new_physical=0,final_tests_absent=True))
    dest=A/'verification';dest.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*[O/n for n in ['publication.log','publication_verification.json','final_verification.json']]]:shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8');v();print('GRID138_FINAL_VERIFIED',files)
if __name__=='__main__':main()
