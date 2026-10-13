"""Portable standard-library check using only bundled summaries; no simulator."""
import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--materials', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.materials.resolve()
    assert not args.out.exists(), 'Output must be new.'
    with (root/'tables/main_results_unified.csv').open(encoding='utf-8-sig',newline='') as f:
        reported = {(r['stage'],r['method']):r for r in csv.DictReader(f)}
    checked = []
    for stage in ('grid61','grid62'):
        base = root/'reproducibility/summaries'/stage
        runs = json.loads((base/'finished.json').read_text(encoding='utf-8'))['summaries']
        old = json.loads((base/'review.json').read_text(encoding='utf-8'))['summaries']
        groups = {}
        for r in runs: groups.setdefault((r['week'],r['method']),[]).append(r)
        for method in ('FULL','FIRST_SAFE_BEST','NO_LOCAL','FRAME'):
            times,queries,costs,quality,complete,n = [],[],[],0,0,0
            for (week,m), items in groups.items():
                if m!=method: continue
                n+=1;ref=groups[week,'FULL']
                assert len(items)==len(ref)==2
                byrep={r['repeat']:r for r in ref}
                same = all(r['steps']==byrep[r['repeat']]['steps'] and r['complete']==byrep[r['repeat']]['complete'] for r in items)
                survived = all(r['complete'] if byrep[r['repeat']]['complete'] else r['steps']>=byrep[r['repeat']]['steps'] for r in items)
                done=all(r['complete'] for r in items);complete+=done
                value=None
                if done and all(r['complete'] for r in ref):
                    rc=sum(r['cost'] for r in ref)
                    value=(sum(r['cost'] for r in items)-rc)/abs(rc);costs.append(value)
                quality+=bool(survived and (value is None or value<=.005))
                if same:
                    times.append(1-sum(r['controller_s'] for r in items)/sum(r['controller_s'] for r in ref))
                    rq=sum(r['n1_forecasts'] for r in ref)
                    if rq:queries.append(1-sum(r['n1_forecasts'] for r in items)/rq)
            values={'families':n,'runs':2*n,'completed_families':complete,'quality_families':quality,
                'control_comparable_families':len(times),'query_comparable_families':len(queries),
                'macro_control_reduction_full':sum(times)/len(times),
                'macro_query_reduction_full':sum(queries)/len(queries),
                'worst_complete_cost_delta_full':max(costs),
                'legacy_report_macro_control':next(x['macro_control_reduction_full'] for x in old if x['method']==method)}
            expected=reported[stage.upper(),method]
            for key,v in values.items():assert math.isclose(float(expected[key]),v,rel_tol=0,abs_tol=1e-12),(stage,method,key)
            checked.append({'stage':stage.upper(),'method':method,**values})
    args.out.mkdir(parents=True)
    (args.out/'validation.json').write_text(json.dumps({'passed':True,'rows':checked,
        'scope':'Aggregate numerical check, no simulation/fit or independent AC re-solve.'},indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'rows':len(checked)}))


if __name__=='__main__':main()
