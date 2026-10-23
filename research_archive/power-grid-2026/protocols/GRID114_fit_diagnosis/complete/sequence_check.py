"""Saved-label complete train/validation sequences and hindsight prefix ceiling. No deployment or fits."""
from analyze import R,O,S,read,save,sha,np,old_predictions,pickle,collections
def main():
    assert not (O/'sequence_objective_check.json').exists()
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());D=read(S/'design.json');fee=D['target']['cost_per_query'];scale=D['target']['gain_scale'];Q=np.load(S/'state_predictions.npz');results={};bridges=[];extra=[]
    for part in ['train','validation']:
        meta=read(S/f'{part}_metadata.json');lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};old=old_predictions(meta)
        scores={**old,**{f'{a}_seed{s}':Q[f'{a}_seed{s}__{part}'] for a in ['JOINT','FACTOR'] for s in range(3)}}
        rows=[]
        for c in cases:
            if c['partition']!=part:continue
            n=(c['gains']-fee*c['queries'])/scale;cum=np.r_[0.,np.cumsum(n)];optimal=int(np.argmax(cum));myopic=0
            while myopic<len(n) and n[myopic]>0:myopic+=1
            if optimal>myopic and cum[optimal]>cum[myopic]+1e-12:
                bridges.append(dict(partition=part,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],
                    myopic_batches=myopic,hindsight_best_batches=optimal,per_batch_normalized_net=n.tolist(),missed_later_net=float(cum[optimal]-cum[myopic])))
            actions={k:(v[[lookup[c['sample_id'],j] for j in range(len(n))],1]>v[[lookup[c['sample_id'],j] for j in range(len(n))],0]) for k,v in scores.items()}
            for method in list(actions)+['BATCH1','STOP','MYOPIC_LABEL_ORACLE','HINDSIGHT_PREFIX_CEILING']:
                if method=='BATCH1':k=1
                elif method=='STOP':k=0
                elif method=='MYOPIC_LABEL_ORACLE':k=myopic
                elif method=='HINDSIGHT_PREFIX_CEILING':k=optimal
                else:
                    k=0
                    while k<len(n) and actions[method][k]:k+=1
                row=dict(method=method,date=c['date'],sample_id=c['sample_id'],physical_state_group=c['physical_state_group'],batches=k,
                    net_utility=float(cum[k]),horizon=len(n),first_stop_current_net=float(n[k]) if k<len(n) else None,
                    regret_to_hindsight=float(cum[optimal]-cum[k]))
                rows.append(row)
        summaries={}
        for method in sorted({r['method'] for r in rows}):
            rr=[r for r in rows if r['method']==method];bydate={d:[r for r in rr if r['date']==d] for d in sorted({r['date'] for r in rr})}
            perdate={d:dict(cases=len(rs),net_utility=float(np.mean([r['net_utility'] for r in rs])),regret_to_hindsight=float(np.mean([r['regret_to_hindsight'] for r in rs]))) for d,rs in bydate.items()}
            summaries[method]=dict(macro_utility=float(np.mean([r['net_utility'] for r in perdate.values()])),per_date=perdate)
        if part=='validation':
            prev=read(S/'validation_results.json')
            for method in scores:assert abs(summaries[method]['macro_utility']-prev[method]['all']['macro']['mean_net_utility'])<1e-12
        results[part]=summaries;extra.extend(rows)
    save(O/'sequence_objective_check.json',dict(passed=True,results=results,hindsight_bridge_cases=bridges,
        limits='Hindsight values use unbought teacher feedback AFTER execution, are unavailable to a deployed model, and are diagnostic upper bounds only. No new training labels accepted or policy changed.',
        source_sha256=sha(__import__('pathlib').Path(__file__)),new_fits=0,new_physical_calls=0))
    save(O/'sequence_diagnostic_rows.json',extra)
    print(__import__('json').dumps({p:{m:results[p][m]['macro_utility'] for m in results[p]} for p in results},indent=2))
if __name__=='__main__':main()
