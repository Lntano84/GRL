"""Account for added station completion in future-informed query compression."""
from run import *
def main():
    verify();assert read(O/'review.json')['passed'];assert not (O/'equivalent_completion_opportunity.json').exists()
    save('completion_check_plan.json',dict(purpose='Do not overstate earlier-global-stop savings:include all same-station queries newly needed after earlier stop.',
        construction='Future-informed earliest128/64 prefix matching originalFRAME precompletion incumbent;apply unchanged conditional station-completion eligibility from that prefix;verify final selected pool id exactly equals originalFRAME.',
        mask='Primary eligibility forbids stopping while no admissible incumbent. Original no-action exhaustive calls remain exhaustive. Unmasked opportunity listed separately,not a recommendation.',
        limits='Saved fixed FULL-teacher states only;not a learned gate,actual timing,wholeweek result or proof learnability. No new online rule executed.',
        all_calls=305,new_fits=0,new_physical=0))
    frozen=A/'completion_frozen';frozen.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),O/'completion_check_plan.json']:shutil.copy2(p,frozen/p.name)
    (frozen/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in frozen.iterdir()},indent=2)+'\n',encoding='utf8')
    subs=read(R/'outputs/grid28/offline.json')['public_subids'];records=[]
    for r in read(O/'call_ledger.json'):
        c=read(R/r['source']);n=r['earliest_guard_prefix'];current=best(c['outcomes'][:n]);prefix=best(c['outcomes'][:min(128,len(c['order']))]);m=len(c['order']);local=[]
        assert bid(current)==r['global_choice']
        if current is not None and n<m and (not r['prefix_safe'] or bid(current)!=bid(prefix)):
            station=subs[current['pool_id']];ids={int(rep) for alias,rep in c['aliases'].items() if subs[int(alias)]==station};local=[i for i in c['order'][n:] if i in ids]
        truth={x['pool_id']:x for x in c['outcomes']};picked=best(c['outcomes'][:n]+[truth[i] for i in local])
        assert bid(picked)==r['frame_choice'];before=r['global_n']+r['local_queries'];after=n+len(local);assert after<=before
        masked_after=after if current is not None else before
        records.append(dict(call_id=r['call_id'],partition=r['partition'],date=r['date'],scenario=r['scenario'],full_available=r['full_available'],original_queries=before,earlier_global_n=n,
            earlier_local_queries=len(local),original_local_queries=r['local_queries'],extra_local_queries=len(local)-r['local_queries'],unmasked_queries=after,
            masked_queries=masked_after,masked_queries_saved=before-masked_after,exact_original_frame_action=True,stop_allowed_current_incumbent=current is not None))
    assert len(records)==305
    summary=[]
    for partition in ['train','validation']:
        rr=[r for r in records if r['partition']==partition];scenarios=[]
        for scenario in sorted({r['scenario'] for r in rr}):
            ss=[r for r in rr if r['scenario']==scenario];scenarios.append(dict(date=ss[0]['date'],ratio=sum(r['masked_queries_saved'] for r in ss)/sum(r['original_queries'] for r in ss)))
        dates=sorted({r['date'] for r in scenarios});macro=statistics.mean(statistics.mean(r['ratio'] for r in scenarios if r['date']==d) for d in dates)
        summary.append(dict(partition=partition,calls=len(rr),eligible_incumbent_calls=sum(r['stop_allowed_current_incumbent'] for r in rr),
            original_frame_queries=sum(r['original_queries'] for r in rr),masked_hindsight_queries=sum(r['masked_queries'] for r in rr),masked_queries_saved=sum(r['masked_queries_saved'] for r in rr),
            date_macro_hindsight_reduction=macro,additional_completion_queries=sum(max(0,r['extra_local_queries']) for r in rr),
            exact_original_frame_actions=sum(r['exact_original_frame_action'] for r in rr),not_a_deployable_model_result=True))
    save('equivalent_completion_opportunity.json',dict(passed=True,summary=summary,calls=records))
    verify();d=A/'completion_complete';d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),O/'equivalent_completion_opportunity.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
