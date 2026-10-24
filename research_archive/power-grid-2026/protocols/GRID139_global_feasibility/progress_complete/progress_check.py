"""Compare inherited plateau signal at matched public query progress;no model."""
from run import *
def gz(n):
    with gzip.open(O/n,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]
def main():
    verify();assert read(O/'review.json')['passed'];assert not (O/'matched_progress.json').exists()
    save('progress_check_plan.json',dict(purpose='Check whether inherited two-stale indicator associations merely reflect query progress.',
        grouping='Within original partition,exact visible count,candidate count,prefix-safe,current-incumbent availability,current-safe. Compare stale>=2 versus other;only cells with both sides. Save all cells including unsupported cells. No new threshold,model or policy.',
        summary='Descriptive differences;report all matched cells,sign counts and overlap-weighted difference using min(side counts).No p values or causal/generalization certification.',
        extra='All prefix at original minimum256,split only by entry-safe;all305 per-call hindsight global opportunity and no-FULL-admissible contribution. No prediction-time input from labels.'))
    frozen=A/'progress_frozen';frozen.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),O/'progress_check_plan.json']:shutil.copy2(p,frozen/p.name)
    (frozen/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in frozen.iterdir()},indent=2)+'\n',encoding='utf8')
    meta=gz('metadata.jsonl.gz');xx=gz('public_inputs.jsonl.gz');yy=gz('targets.jsonl.gz');assert len(meta)==len(xx)==len(yy)==3788
    groups=collections.defaultdict(list);at256=collections.defaultdict(list)
    for m,x,y in zip(meta,xx,yy):
        assert m['sample_id']==x['sample_id']==y['sample_id'];f=x['features'];k=(m['partition'],f['visible_count'],f['candidate_count'],f['prefix_safe'],f['incumbent_available'],False if f['incumbent_rho'] is None else f['incumbent_rho']<.9)
        groups[k].append((m,f,y))
        if f['visible_count']==256 and f['incumbent_available']:at256[(m['partition'],f['prefix_safe'],f['stale_batches']>=2)].append((m,f,y))
    cells=[]
    for k,rr in sorted(groups.items()):
        sides=[]
        for stale in [False,True]:
            ss=[r for r in rr if (r[1]['stale_batches']>=2)==stale]
            sides.append(dict(two_stale=stale,n=len(ss),positive=sum(r[2]['any_remaining_improves'] for r in ss),dates=sorted({r[0]['date'] for r in ss})))
        supported=all(s['n'] for s in sides)
        cells.append(dict(partition=k[0],visible_count=k[1],candidate_count=k[2],prefix_safe=k[3],incumbent_available=k[4],current_safe=k[5],sides=sides,supported=supported,
            plateau_minus_other_positive_rate=sides[1]['positive']/sides[1]['n']-sides[0]['positive']/sides[0]['n'] if supported else None))
    summary=[]
    for partition in ['train','validation']:
        cc=[c for c in cells if c['partition']==partition and c['supported']];weights=[min(s['n'] for s in c['sides']) for c in cc];assert cc
        summary.append(dict(partition=partition,matched_cells=len(cc),negative=sum(c['plateau_minus_other_positive_rate']<0 for c in cc),equal=sum(c['plateau_minus_other_positive_rate']==0 for c in cc),positive=sum(c['plateau_minus_other_positive_rate']>0 for c in cc),
            overlap_weight=sum(weights),overlap_weighted_rate_difference=math.fsum(w*c['plateau_minus_other_positive_rate'] for w,c in zip(weights,cc))/sum(weights),
            warning='Repeated prefixes and public strata;descriptive association,not causal effect or proof a learner improves original heuristic.'))
    saved256=[]
    for k,rr in sorted(at256.items()):saved256.append(dict(partition=k[0],prefix_safe=k[1],two_stale=k[2],n=len(rr),future_positive=sum(r[2]['any_remaining_improves'] for r in rr),next_positive=sum(r[2]['next_batch_improves'] for r in rr)))
    save('matched_progress.json',dict(cells=cells,summary=summary,at_original256=saved256,passed=True))
    calls=read(O/'call_ledger.json');op=[]
    for partition in ['train','validation']:
        for available in [False,True,None]:
            rr=[r for r in calls if r['partition']==partition and (available is None or r['full_available']==available)];sr=[]
            for scenario in sorted({r['scenario'] for r in rr}):
                ss=[r for r in rr if r['scenario']==scenario];total=sum(r['global_n'] for r in ss);sr.append(dict(date=ss[0]['date'],ratio=sum(r['global_queries_hindsight_redundant'] for r in ss)/total))
            dates=sorted({r['date'] for r in sr});dm=statistics.mean(statistics.mean(r['ratio'] for r in sr if r['date']==d) for d in dates) if sr else None
            op.append(dict(partition=partition,full_available=available,calls=len(rr),dates=len(dates),global_queries=sum(r['global_n'] for r in rr),hindsight_redundant=sum(r['global_queries_hindsight_redundant'] for r in rr),date_macro_hindsight_ratio=dm,
                limit='Future-informed earliest equivalent global incumbent;excludes possibly extra local queries,model cost and closedloop effect.'))
    save('hindsight_global_opportunity.json',op);verify()
    d=A/'progress_complete';d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),O/'matched_progress.json',O/'hindsight_global_opportunity.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(summary=summary,at256=saved256,opportunity=op),ensure_ascii=False))
if __name__=='__main__':main()
