"""Independent arithmetic,group isolation and publication verification."""
from run import *
def gz(n):
    with gzip.open(O/n,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]
def main():
    verify();assert read(O/'publication_receipt.json')['passed'];assert not (O/'final_verification.json').exists()
    meta=gz('metadata.jsonl.gz');xx=gz('public_inputs.jsonl.gz');yy=gz('targets.jsonl.gz');assert len(meta)==len(xx)==len(yy)==3788
    assert len({m['sample_id'] for m in meta})==3788
    train={m['date'] for m in meta if m['partition']=='train'};val={m['date'] for m in meta if m['partition']=='validation'};assert len(train)==16 and len(val)==4 and not train&val
    lookup={m['sample_id']:(m,x,y) for m,x,y in zip(meta,xx,yy)}
    for m,x,y in lookup.values():
        assert m['sample_id']==x['sample_id']==y['sample_id'];f=x['features']
        assert f['visible_count']==m['visible_count'] and f['remaining_candidates']==y['remaining_queries']
        assert y['next_queries']==min(64,y['remaining_queries'])
        if not f['incumbent_available']:assert y['remaining_reward_gain'] is None and y['next_batch_reward_gain'] is None
        assert y['any_remaining_improves'] or not y['next_batch_improves']
        assert not any(k in f for k in ['date','scenario','sample_id','full_choice','original_frame_choice_label','original_future_total_queries_label','remaining_reward_gain'])
    task=gz('compression_task_targets.jsonl.gz');assert len(task)==726 and len({r['sample_id'] for r in task})==726
    for r in task:
        m,x,y=lookup[r['sample_id']];assert m['visited_by_replayed_guard_global'] and not m['original_guard_stop'] and x['features']['incumbent_available']
        assert r['stop_label']==(r['after_stop_choice_label']==r['original_frame_choice_label'] and r['queries_saved']>0)
        assert r['queries_saved']==r['original_future_total_queries_label']-r['stop_total_queries_known_now']
    coverage=read(O/'compression_label_coverage.json')
    assert next(r for r in coverage if r['partition']=='train' and not r['stop_label'])['dates']==15
    assert next(r for r in coverage if r['partition']=='validation' and not r['stop_label'])['dates']==4
    op=read(O/'equivalent_completion_opportunity.json')
    for r in op['calls']:
        assert r['exact_original_frame_action'] and r['unmasked_queries']<=r['original_queries']
        assert r['masked_queries_saved']==r['original_queries']-r['masked_queries']
        if not r['stop_allowed_current_incumbent']:assert r['masked_queries']==r['original_queries']
    baseline=read(O/'compression_baseline_calls.json');summary=read(O/'compression_baseline_summary.json')
    assert len(baseline)==305*4
    for s in summary:
        rr=[r for r in baseline if r['partition']==s['partition'] and r['cap']==s['cap']]
        assert sum(r['original_action_match'] for r in rr)==s['original_action_matches']
        assert sum(r['queries'] for r in rr)==s['queries']
        dates=sorted({r['date'] for r in rr});values=[]
        for d in dates:
            dd=[r for r in rr if r['date']==d];per=[]
            for sc in sorted({r['scenario'] for r in dd}):
                ss=[r for r in dd if r['scenario']==sc];per.append((sum(r['original_queries'] for r in ss)-sum(r['queries'] for r in ss))/sum(r['original_queries'] for r in ss))
            values.append(sum(per)/len(per))
        assert abs(sum(values)/len(values)-s['date_macro_query_reduction'])<1e-12
    files=0
    for root in A.iterdir():
        if root.is_dir():
            for p,h in read(root/'archive_manifest.json').items():assert sha(root/p)==h;files+=1
    for phase,script in [('progress_frozen','progress_check.py'),('completion_frozen','completion_check.py'),('gate_task_frozen','gate_task.py'),('task_baselines_frozen','task_baselines.py')]:assert sha(W/script)==sha(A/phase/script)
    c=read(O/'collaboration_receipt.json')
    for p,h in c['hashes'].items():assert sha(R/c['directory']/p)==h and sha(O/p)==h
    verify();save('final_verification.json',dict(passed=True,archive_files_verified=files,collaboration_files_verified=len(c['hashes']),global_prefixes=3788,
        source_hashes_unchanged=True,split_calendar_isolation=True,undefined_gain_not_zero=True,compression_labels_and_masks_checked=True,
        all_cheap_baseline_macro_arithmetic_checked=True,no_claim_of_wholeweek_quality=True,new_fits=0,new_model_forwards=0,new_physical=0,final_tests_absent=True))
    d=A/'verification';d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),O/'publication.log',O/'final_verification.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8');print('GRID139_FINAL_VERIFIED',files)
if __name__=='__main__':main()
