"""Read existing prefix labels and public context; no policy execution or new scores."""
import gzip,json,collections
import run as D

def main():
    D.verify();O,R=D.O,D.R;B=D.B
    assert not (O/'feedback_context.json').exists()
    sources=[R/'outputs/grid143_continue_risk/all_continue_future_paths.json',
             R/'outputs/grid141_feedback_diagnosis/public_supplemental_inputs.jsonl.gz',
             R/'outputs/grid142_batch_representation/design.json',D.G.O/'all_search_replays.json',
             O/'case_comparison_summary.json',O/'all_crossdate_fit_neighbors.json',
             R/'outputs/grid139_global_feasibility/compression_task_targets.jsonl.gz',
             R/'outputs/grid139_global_feasibility/public_inputs.jsonl.gz',
             R/'outputs/grid139_global_feasibility/metadata.jsonl.gz']
    hashes={p.relative_to(R).as_posix():D.sha(p) for p in sources}
    D.save('feedback_context_source_hashes.json',hashes)
    allrows={part:(D.G.rowfile(B.O/f'{part}_rows.json.gz') if part!='validation' else B.eligible_data('validation')) for part in ['fit','calibration','validation']}
    oldz=D.np.load(B.O/'train_scores.npz')
    oldscores=dict(fit=oldz['fit'],calibration=oldz['calibration'],validation=D.np.load(B.O/'validation_scores.npz')['scores'])
    bydepth=[];transitions=[]
    for part,rows in allrows.items():
        for n in sorted({r['meta']['visible_count'] for r in rows}):
            ii=[i for i,r in enumerate(rows) if r['meta']['visible_count']==n]
            rr=[rows[i] for i in ii];pp=oldscores[part][ii]
            result=D.ranks(rr,pp)
            assert result['auc'] is None or abs(result['auc']-D.manual_auc(rr,pp))<1e-12
            bydepth.append(dict(partition=part,depth=n,**result))
        before={r['meta']['call_id']:r for r in rows if r['meta']['visible_count']==128}
        after={r['meta']['call_id']:r for r in rows if r['meta']['visible_count']==192}
        labels=collections.Counter((bool(before[k]['target']['stop_label']),bool(after[k]['target']['stop_label'])) for k in before.keys()&after.keys())
        transitions.append(dict(partition=part,common_calls=len(before.keys()&after.keys()),
            stop_to_stop=labels[True,True],stop_to_continue=labels[True,False],continue_to_stop=labels[False,True],continue_to_continue=labels[False,False],
            scope='Same-call labels at two observed depths,not a trained transition model or a new policy.'))
    paths=D.read(sources[0]);nextbatch=[]
    for part in allrows:
        rr=[p for p in paths if p['partition']==part and p['visible_count']==128]
        nextbatch.append(dict(partition=part,continue_at128=len(rr),next_batch_incumbent_improves=sum(p['next_batch_improves_incumbent'] for p in rr),
            one_more_batch_then_original_completion_restores_action=sum(p['next_then_completion_restores_action'] for p in rr)))
    # Public supplemental fields already frozen/checked byGRID141;join only the declared cases/neighbors.
    with gzip.open(sources[1],'rt',encoding='utf8') as f:extra={x['sample_id']:x['features'] for x in map(json.loads,f)}
    cases=D.read(O/'case_comparison_summary.json');contexts=[]
    for c in cases:
        ids=[c['case']['sample_id']]+[v['sample_id'] for v in c['neighbors']['neighbors'].values() if v is not None]
        contexts.append(dict(case_sample_id=c['case']['sample_id'],case_call_id=c['case']['call_id'],
            public_supplemental_by_sample=[dict(sample_id=k,public_features=extra[k]) for k in ids]))
    call='2037-03-02_100|call009';ref=next(r for r in B.refs() if r['call_id']==call);saved=B.call_at(ref['source'])
    checkpoints=[]
    for n in range(128,ref['global_n']+1,64):
        best=B.P.best(saved['outcomes'][:n]);row=next((r for r in allrows['validation'] if r['meta']['call_id']==call and r['meta']['visible_count']==n),None)
        checkpoints.append(dict(visible_count=n,incumbent=best['pool_id'],incumbent_station=B.subids()[best['pool_id']],incumbent_rho=best['rho'],
            original_saved_stop_target=row['target'] if row else None,
            original_public_features=row['features'] if row else None))
    for p,h in hashes.items():assert D.sha(R/p)==h
    D.save('feedback_context.json',dict(fits=0,policy_executions=0,bydepth_fixed_rf140=bydepth,same_call_128_to192=transitions,
        original_continue_future_labels=nextbatch,declared_case_public_supplemental=contexts,
        error_call_original_saved_checkpoints=checkpoints,
        prior_failed_extra_features=D.read(sources[2])['added_features'],
        scope='Existing scores/public context plus future LABEL evidence only. No new fitted inputs selected.'))
    print(json.dumps(dict(nextbatch=nextbatch,transitions=transitions,error_checkpoints=checkpoints),ensure_ascii=False))

if __name__=='__main__':main()
