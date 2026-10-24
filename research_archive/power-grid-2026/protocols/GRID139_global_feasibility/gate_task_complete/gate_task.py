"""Separate proposed compression-gate labels from next-batch reward targets."""
from run import *
def gz(n):
    with gzip.open(O/n,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]
def main():
    verify();assert read(O/'equivalent_completion_opportunity.json')['passed'];assert not (O/'compression_task_summary.json').exists()
    save('compression_task_plan.json',dict(status='Training-task draft only,no model fit or adoption.',
        action='At originalFRAME global checkpoints after128 then64 batches,an earlier STOP dispatches usual conditional same-station completion;CONTINUE retains next original batch. Original forced stops remain and model cannot extend past them.',
        mask='No admissible incumbent => retain original search,not a learned stop. Only original visited pre-forced-stop checkpoints with incumbent eligible.',
        supervision='Compare stopping now plus same original station-completion rule to originalFRAME final pool id,using complete saved feedback. Preserve action equality and candidate query difference separately;positive stop target requires equality and strictly fewer total candidate queries.',
        information='Same previously derived revealed-prefix inputs;date/source/identity/teacher answer excluded from features. Original forced stop availability computed from revealed feedback.',
        scope='Fixed FULL-teacher physical states;no new wholeweek-cost label,same candidate pool,controller/order/admissibility. Final tests untouched. Subsequent closed-loop validation required.',
        models='No model,hyperparameters,calibration cutoff or fee selected in this diagnostic. First training requires a separate fixed configuration and notification.'))
    d=A/'gate_task_frozen';d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),O/'compression_task_plan.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
    metas=gz('metadata.jsonl.gz');inputs=gz('public_inputs.jsonl.gz');lookup={r['call_id']:r for r in read(O/'call_ledger.json')};subs=read(R/'outputs/grid28/offline.json')['public_subids'];cache={};rows=[]
    for m,x in zip(metas,inputs):
        assert m['sample_id']==x['sample_id'];f=x['features'];n=f['visible_count']
        if m['visited_by_replayed_guard_global']:
            force=((not f['prefix_safe']) and f['incumbent_rho'] is not None and f['incumbent_rho']<.9) or (f['incumbent_available'] and n>=256 and f['stale_batches']>=2)
            assert bool(force)==m['original_guard_stop']
        if not m['visited_by_replayed_guard_global'] or m['original_guard_stop'] or not f['incumbent_available']:continue
        if m['call_id'] not in cache:cache[m['call_id']]=read(R/m['source'])
        c=cache[m['call_id']];r=lookup[m['call_id']];current=best(c['outcomes'][:n]);prefix=best(c['outcomes'][:min(128,len(c['order']))]);extra=[]
        if n<len(c['order']) and (not f['prefix_safe'] or bid(current)!=bid(prefix)):
            station=subs[current['pool_id']];ids={int(rep) for alias,rep in c['aliases'].items() if subs[int(alias)]==station};extra=[i for i in c['order'][n:] if i in ids]
        truth={v['pool_id']:v for v in c['outcomes']};picked=best(c['outcomes'][:n]+[truth[i] for i in extra]);q=n+len(extra);oldq=r['global_n']+r['local_queries'];same=bid(picked)==r['frame_choice']
        rows.append(dict(sample_id=m['sample_id'],call_id=m['call_id'],partition=m['partition'],date=m['date'],scenario=m['scenario'],visible_count=n,
            preserves_original_frame_action=same,stop_total_queries_known_now=q,original_future_total_queries_label=oldq,queries_saved=oldq-q,
            stop_label=same and q<oldq,after_stop_choice_label=bid(picked),original_frame_choice_label=r['frame_choice'],future_fields_are_targets_only=True))
    with gzip.open(O/'compression_task_targets.jsonl.gz','wt',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    summary=[]
    for partition in ['train','validation']:
        rr=[r for r in rows if r['partition']==partition]
        summary.append(dict(partition=partition,eligible_prefixes=len(rr),calls=len({r['call_id'] for r in rr}),dates=len({r['date'] for r in rr}),
            positive_stop_labels=sum(r['stop_label'] for r in rr),continue_labels=sum(not r['stop_label'] for r in rr),
            action_preserved=sum(r['preserves_original_frame_action'] for r in rr),action_changes=sum(not r['preserves_original_frame_action'] for r in rr),
            preserved_but_no_query_saving=sum(r['preserves_original_frame_action'] and r['queries_saved']<=0 for r in rr),
            prefix_multiplicity_warning='Prefixes share calls/date;not independent sample count or model performance.'))
    assert [r['eligible_prefixes'] for r in summary]==[584,142]
    save('compression_task_summary.json',dict(passed=True,summary=summary,training_started=False,adopted=False,final_tests_absent=True))
    verify();d=A/'gate_task_complete';d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),O/'compression_task_targets.jsonl.gz',O/'compression_task_summary.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
