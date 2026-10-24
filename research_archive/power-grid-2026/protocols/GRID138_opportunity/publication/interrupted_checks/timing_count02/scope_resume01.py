"""Preserve failed strict-timer check;bound differing historical timer semantics."""
from historical_scope import scope_verify
from run import *

def v():
    scope_verify()
    for p,h in read(O/'scope_recovery_hashes.json').items():assert sha(R/p)==h,p
def prepare():
    scope_verify();assert not (O/'scope_recovery_plan.json').exists()
    src=R/'work/grid89_review/adapter.py';code=src.read_text(encoding='utf8')
    assert 'forecast_s+=time.perf_counter()-t' in code and 'strict_admissible=valid,forecast_s=time.perf_counter()-t' in code
    d=O/'interrupted_checks/timing_semantics01';d.mkdir(parents=True,exist_ok=False)
    for p in [W/'historical_scope.py',O/'scope.log']:shutil.copy2(p,d/p.name)
    save('scope_recovery_plan.json',dict(reason='Strict equality of outer candidate-row timer and inner obs.simulate timer failed in historical GRID89 adapter;44G2calls,not policy or trajectory error.',
        source_evidence='GRID89adapter accumulates forecast_s immediately after simulate;row forecast_s sampled after flag/rho/reward processing. Inner sum and outer sum differ legitimately.',
        method='For each run D=sum(outer candidate timers)-sum(inner call forecast timers). Each nonnegative postprocessing share unknown by phase. Local pure prediction seconds lie in[max(0,outerLocal-D),outerLocal]. Report intervals;do not modify history or allocate overhead proportionally.',
        first_check_original_preserved=True,new_fits=0,new_replays=0,original_primary4_unchanged=True))
    paths=[Path(__file__),src,O/'scope_recovery_plan.json',d/'historical_scope.py',d/'scope.log']
    save('scope_recovery_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});v()
    d=A/'scope_recovery_frozen';d.mkdir(parents=True,exist_ok=False)
    for p in paths+[O/'scope_recovery_hashes.json']:
        name=p.relative_to(O).as_posix() if p.is_relative_to(O) else p.name
        target=d/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8');print('SCOPE_TIMER_RECOVERY_FROZEN')
def run_scope():
    v();assert not (O/'historical_timing_runs.json').exists();rows=[];different=[]
    for m in read(R/'outputs/grid91_mechanism/design.json')['references']:
        p=R/m['frame'];s=read(p/'summary.json');cc=[read(x) for x in sorted(p.glob('call[0-9][0-9][0-9].json'))];assert s['week']==m['week'] and len(cc)==s['n1_calls']
        assert sum(c['public_queries'] for c in cc)==s['n1_forecasts'];local=tail=same=over=0.0;localq=0
        for j,c in enumerate(cc):
            ll=[r for r in c['outcomes'] if r['phase']=='local_completion'];assert len(ll)==c['local_queries']==c['completion_budget']
            l=math.fsum(r['forecast_s'] for r in ll);local+=l;tail+=math.fsum(r['forecast_s'] for r in ll[64:]);localq+=len(ll)
            delta=math.fsum(r['forecast_s'] for r in c['outcomes'])-c['forecast_s'];assert delta>=-1e-7
            if delta>1e-7:
                assert m['cohort']=='G2';different.append(dict(cohort=m['cohort'],week=m['week'],index=j,outer_minus_inner_s=delta))
            over+=max(0,delta)
            if c['pre_extra_choice']==c['chosen_pool_id']:same+=l
        assert abs(math.fsum(c['forecast_s'] for c in cc)-s['n1_forecast_s'])<1e-7
        assert local-over<=s['n1_forecast_s']+1e-7
        rows.append(dict(cohort=m['cohort'],week=m['week'],frame_reference=m['frame'],complete=s['complete'],steps=s['steps'],controller_s=s['controller_s'],local_outer_s=local,local_queries=localq,
            local_share_lower=max(0,local-over)/s['controller_s'],local_share_upper=local/s['controller_s'],
            after64_share_lower=max(0,tail-over)/s['controller_s'],after64_share_upper=tail/s['controller_s'],
            unchanged_choice_share_lower=max(0,same-over)/s['controller_s'],unchanged_choice_share_upper=same/s['controller_s'],timer_extra_s=over,calls=len(cc)))
    assert len(rows)==25 and len(different)==44
    save('historical_timing_runs.json',rows);groups=[]
    fields=['local_share_lower','local_share_upper','after64_share_lower','after64_share_upper','unchanged_choice_share_lower','unchanged_choice_share_upper']
    for name in ['old13','G1','G2','all25']:
        rr=rows if name=='all25' else [r for r in rows if r['cohort']==name];complete=[r for r in rr if r['complete']]
        groups.append(dict(cohort=name,runs=len(rr),complete=len(complete),calls=sum(r['calls'] for r in rr),
            date_entry_macro={k:statistics.mean(r[k] for r in rr) for k in fields},complete_entry_macro={k:statistics.mean(r[k] for r in complete) for k in fields},
            local_share_range=[min(r['local_share_lower'] for r in rr),max(r['local_share_upper'] for r in rr)]))
    save('historical_timing_summary.json',dict(groups=groups,passed=True,existing_calendar_repetitions_not_independent=True,timer_mismatch_calls=different,
        limit='25 original one-timing references,within-run pure forecast component intervals. Not pooled speed trial or attainable closed-loop upper bound. Primary4 modern runs have equal inner/outer timers.',new_fits=0,new_physical=0))
    v();d=A/'scope_complete';d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*[O/n for n in ['historical_timing_runs.json','historical_timing_summary.json']]]:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8');print(json.dumps(groups,ensure_ascii=False))
if __name__=='__main__':{'prepare':prepare,'run':run_scope}[sys.argv[1]]()
