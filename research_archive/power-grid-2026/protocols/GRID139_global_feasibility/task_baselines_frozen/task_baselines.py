"""Strong cheap saved-state references with same completion and original forced stops."""
from run import *
def main():
    verify();assert read(O/'compression_task_summary.json')['passed'];assert not (O/'compression_baseline_summary.json').exists()
    save('compression_baseline_plan.json',dict(reason='Literal fixed prefixes without completion are not sufficient strong controls for a stop-plus-completion task.',
        references='All inherited128,128+64,256,512 query checkpoints;retain originalFRAME forced stop and no-incumbent continue mask;on earlier stop apply same conditional station completion. Retain all four,no best budget selected.',
        evaluation='Same saved FULL physical states;originalFRAME action equality and query difference,not business quality or actual speed.',new_fits=0,new_physical=0))
    d=A/'task_baselines_frozen';d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),O/'compression_baseline_plan.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
    subs=read(R/'outputs/grid28/offline.json')['public_subids'];rows=[]
    for r in read(O/'call_ledger.json'):
        c=read(R/r['source']);cp=[n for n in checkpoints(len(c['order'])) if n<=r['global_n']];assert cp[-1]==r['global_n']
        for cap in [128,192,256,512]:
            n=next((n for n in cp if n>=cap and best(c['outcomes'][:n]) is not None),r['global_n'])
            current=best(c['outcomes'][:n]);prefix=best(c['outcomes'][:min(128,len(c['order']))]);extra=[]
            if current is not None and n<len(c['order']) and (not r['prefix_safe'] or bid(current)!=bid(prefix)):
                station=subs[current['pool_id']];ids={int(rep) for alias,rep in c['aliases'].items() if subs[int(alias)]==station};extra=[i for i in c['order'][n:] if i in ids]
            truth={x['pool_id']:x for x in c['outcomes']};selected=best(c['outcomes'][:n]+[truth[i] for i in extra]);oldq=r['global_n']+r['local_queries'];q=n+len(extra)
            rows.append(dict(partition=r['partition'],date=r['date'],scenario=r['scenario'],call_id=r['call_id'],cap=cap,original_queries=oldq,queries=q,global_queries=n,local_queries=len(extra),
                original_action_match=bid(selected)==r['frame_choice'],selected=bid(selected),original=r['frame_choice']))
    summary=[]
    for partition in ['train','validation']:
        for cap in [128,192,256,512]:
            rr=[r for r in rows if r['partition']==partition and r['cap']==cap];sr=[]
            for scenario in sorted({r['scenario'] for r in rr}):
                ss=[r for r in rr if r['scenario']==scenario];sr.append(dict(date=ss[0]['date'],reduction=1-sum(r['queries'] for r in ss)/sum(r['original_queries'] for r in ss),action_match=sum(r['original_action_match'] for r in ss)/len(ss)))
            dates=sorted({r['date'] for r in sr});drows=[dict(date=d,reduction=statistics.mean(r['reduction'] for r in sr if r['date']==d),action_match=statistics.mean(r['action_match'] for r in sr if r['date']==d)) for d in dates]
            summary.append(dict(partition=partition,cap=cap,calls=len(rr),original_action_matches=sum(r['original_action_match'] for r in rr),action_differences=sum(not r['original_action_match'] for r in rr),
                original_queries=sum(r['original_queries'] for r in rr),queries=sum(r['queries'] for r in rr),date_macro_query_reduction=statistics.mean(r['reduction'] for r in drows),
                date_macro_action_match=statistics.mean(r['action_match'] for r in drows),by_date=drows))
    save('compression_baseline_calls.json',rows);save('compression_baseline_summary.json',summary)
    with gzip.open(O/'compression_task_targets.jsonl.gz','rt',encoding='utf8') as f:labels=[json.loads(s) for s in f]
    coverage=[]
    for partition in ['train','validation']:
        for label in [False,True]:
            rr=[r for r in labels if r['partition']==partition and r['stop_label']==label]
            coverage.append(dict(partition=partition,stop_label=label,prefixes=len(rr),calls=len({r['call_id'] for r in rr}),dates=len({r['date'] for r in rr}),by_date=dict(collections.Counter(r['date'] for r in rr))))
    save('compression_label_coverage.json',coverage);verify();d=A/'task_baselines_complete';d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*[O/n for n in ['compression_baseline_calls.json','compression_baseline_summary.json','compression_label_coverage.json']]]:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(summary=[{k:v for k,v in r.items() if k!='by_date'} for r in summary],coverage=coverage),ensure_ascii=False))
if __name__=='__main__':main()
