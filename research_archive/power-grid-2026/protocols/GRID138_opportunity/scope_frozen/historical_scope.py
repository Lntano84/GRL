"""Extend component coverage to all pre-existing25 scenario references."""
from run import *

def scope_verify():
    verify()
    for n in ['scope_source_hashes.json','scope_code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
def prepare():
    assert not (O/'scope_plan.json').exists();assert read(O/'review.json')['passed']
    save('scope_plan.json',dict(purpose='After four-date decomposition,avoid generalizing it to all historical scenarios;cover every25 original GRID91 FRAME references without selection.',
        scope='Descriptive ratios within each original run,not cross-batch speed comparisons or extra independent scenarios.',
        no_new_physical=True,no_new_fits=True,primary_four_date_result_retained=True))
    paths=[O/'scope_plan.json',R/'outputs/grid91_mechanism/design.json',R/'outputs/grid91_mechanism/review.json']
    for m in read(R/'outputs/grid91_mechanism/design.json')['references']:
        p=R/m['frame'];paths += [p/'summary.json',*sorted(p.glob('call[0-9][0-9][0-9].json'))]
    save('scope_source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sorted(set(paths))})
    save('scope_code_freeze.json',{Path(__file__).relative_to(R).as_posix():sha(Path(__file__))});scope_verify()
    d=A/'scope_frozen';d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*[O/n for n in ['scope_plan.json','scope_source_hashes.json','scope_code_freeze.json']]]:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8');print('GRID138_SCOPE_FROZEN')
def run_scope():
    scope_verify();assert not (O/'historical_timing_runs.json').exists();rows=[]
    for m in read(R/'outputs/grid91_mechanism/design.json')['references']:
        p=R/m['frame'];s=read(p/'summary.json');cc=[read(x) for x in sorted(p.glob('call[0-9][0-9][0-9].json'))];assert s['week']==m['week'] and len(cc)==s['n1_calls']
        assert sum(c['public_queries'] for c in cc)==s['n1_forecasts'];local=tail=same=0.0;localq=0
        for c in cc:
            ll=[r for r in c['outcomes'] if r['phase']=='local_completion'];assert len(ll)==c['local_queries']==c['completion_budget']
            l=math.fsum(r['forecast_s'] for r in ll);local+=l;tail+=math.fsum(r['forecast_s'] for r in ll[64:]);localq+=len(ll)
            assert abs(math.fsum(r['forecast_s'] for r in c['outcomes'])-c['forecast_s'])<1e-7
            if c['pre_extra_choice']==c['chosen_pool_id']:same+=l
        assert local<=s['n1_forecast_s']+1e-7
        rows.append(dict(cohort=m['cohort'],week=m['week'],frame_reference=m['frame'],complete=s['complete'],steps=s['steps'],controller_s=s['controller_s'],local_s=local,local_queries=localq,
            local_share=local/s['controller_s'],after64_share=tail/s['controller_s'],unchanged_choice_share=same/s['controller_s'],calls=len(cc)))
    save('historical_timing_runs.json',rows);groups=[]
    for name in ['old13','G1','G2','all25']:
        rr=rows if name=='all25' else [r for r in rows if r['cohort']==name]
        complete=[r for r in rr if r['complete']]
        groups.append(dict(cohort=name,runs=len(rr),complete=len(complete),calls=sum(r['calls'] for r in rr),
            date_entry_macro={k:statistics.mean(r[k] for r in rr) for k in ['local_share','after64_share','unchanged_choice_share']},
            complete_entry_macro={k:statistics.mean(r[k] for r in complete) for k in ['local_share','after64_share','unchanged_choice_share']},
            local_share_range=[min(r['local_share'] for r in rr),max(r['local_share'] for r in rr)]))
    save('historical_timing_summary.json',dict(groups=groups,passed=True,existing_calendar_repetitions_not_independent=True,
        limit='Original one timing reference per scenario;within-run component ratios only. Not a pooled speed trial or attainable closed-loop upper bound.',new_fits=0,new_physical=0))
    scope_verify();d=A/'scope_complete';d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*[O/n for n in ['historical_timing_runs.json','historical_timing_summary.json']]]:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8');print(json.dumps(groups,ensure_ascii=False))
if __name__=='__main__':{'prepare':prepare,'run':run_scope}[sys.argv[1]]()
