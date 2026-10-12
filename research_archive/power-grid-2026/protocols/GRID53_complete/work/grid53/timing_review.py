"""Reconcile existing timings; do not compare different dates as model ablations."""
import csv, hashlib, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid53'
OUT.mkdir(exist_ok=True)
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def total(xs): return math.fsum(xs)
def savecsv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

records = []; files = []; checks = 0
for stage in ['grid44', 'grid51', 'grid52']:
    src = ROOT / 'outputs' / stage
    files.append(src / 'review.json')
    assert read(src / 'review.json')['passed_saved_data']
    D = read(src / 'design.json')
    for item in D['execution_order']:
        ident = f"r{item['repeat']}__{item['week']}__{item['method']}"
        folder = src / 'runs' / ident
        sp, cp = folder / 'summary.json', folder / 'components.json'
        s, c = read(sp), read(cp); files += [sp, cp]
        calls = [read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]
        files += sorted(folder.glob('call[0-9][0-9][0-9].json'))
        assert len(calls) == s['n1_calls']
        fields = ['source_s', 'forecast_s', 'prepare_s', 'prior_s', 'feature_s', 'forward_rank_s']
        sums = {k: total(r[k] for r in calls) for k in fields}
        summary_names = {'source_s':'n1_source_s', 'forecast_s':'n1_forecast_s', 'prepare_s':'n1_prepare_s',
                         'prior_s':'prior_s', 'feature_s':'feature_s', 'forward_rank_s':'forward_rank_s'}
        for k, v in sums.items():
            assert abs(v - s[summary_names[k]]) <= 1e-6, (stage, ident, k)
            checks += 1
        active = [r for r in c['calls'] if r['in_control']]
        solver = {k:total(r['wall_s'] for r in active if r['phase']==k)
                  for k in ['run_dc', 'compute_optimum_safe', 'compute_optimum_unsafe']}
        topology_nonforecast = sums['source_s'] - sums['forecast_s']
        search_other = sums['source_s'] - total(sums[k] for k in fields if k != 'source_s')
        remainder = s['controller_s'] - total(solver.values()) - s['all_forecast_s'] - topology_nonforecast
        assert topology_nonforecast >= 0 and search_other >= -1e-6 and remainder >= -1e-6
        r = {'stage':stage, 'week':s['week'], 'repeat':s['repeat'], 'method':s['method'],
             'steps':s['steps'], 'complete':s['complete'], 'cost':s['cost'], 'control_s':s['controller_s'],
             'topology_calls':s['n1_calls'], 'queries':s['n1_forecasts'], **sums,
             **solver, 'all_forecast_s':s['all_forecast_s'], 'topology_nonforecast_s':topology_nonforecast,
             'search_other_s':search_other, 'controller_other_s':remainder}
        assert abs(total([*solver.values(),s['all_forecast_s'],topology_nonforecast,remainder])-s['controller_s'])<1e-6
        records.append(r)

aggregates = []
keys = ['control_s','topology_calls','queries', 'source_s','forecast_s','prepare_s','prior_s','feature_s',
        'forward_rank_s','run_dc','compute_optimum_safe','compute_optimum_unsafe','all_forecast_s',
        'topology_nonforecast_s','search_other_s','controller_other_s']
for stage, method in sorted({(r['stage'],r['method']) for r in records}):
    group = [r for r in records if (r['stage'],r['method'])==(stage,method)]
    a = {'stage':stage,'method':method,'runs':len(group), **{k:total(r[k] for r in group) for k in keys}}
    a['prior_ms_per_topology_call'] = 1000*a['prior_s']/a['topology_calls'] if a['topology_calls'] else None
    a['forward_ms_per_topology_call'] = 1000*a['forward_rank_s']/a['topology_calls'] if a['topology_calls'] else None
    a['forecast_ms_per_query'] = 1000*a['forecast_s']/a['queries'] if a['queries'] else None
    aggregates.append(a)

paired = []
for new, old in [('COMPRESSED_GNN0','LOCAL_GNN0'),('COMPRESSED_GNN1','LOCAL_GNN1')]:
    old_rows = [r for r in records if r['stage']=='grid51' and r['method']==old]
    new_rows = [r for r in records if r['stage']=='grid51' and r['method']==new]
    assert [(r['week'],r['repeat']) for r in old_rows] == [(r['week'],r['repeat']) for r in new_rows]
    paired.append({'stage':'grid51','new':new,'old':old,'same_date_pairs':len(old_rows),
                   'pooled_control_reduction':1-total(r['control_s'] for r in new_rows)/total(r['control_s'] for r in old_rows),
                   'control_difference_s':total(r['control_s'] for r in new_rows)-total(r['control_s'] for r in old_rows),
                   **{k+'_difference':total(r[k] for r in new_rows)-total(r[k] for r in old_rows) for k in keys if k!='control_s'}})
decomp = []
for own in ['COMPRESSED_GNN0','COMPRESSED_GNN1']:
    a=next(r for r in aggregates if r['stage']=='grid52' and r['method']==own)
    for base in ['LOCAL_PPO','SOFTGNN0','SOFTGNN1']:
        b=next(r for r in aggregates if r['stage']=='grid52' and r['method']==base)
        decomp.append({'method':own,'opponent':base,
                      **{k+'_difference_s':a[k]-b[k] for k in ['control_s','run_dc','compute_optimum_safe','compute_optimum_unsafe','all_forecast_s','topology_nonforecast_s','controller_other_s','prior_s','forward_rank_s','feature_s','prepare_s','search_other_s']},
                      'queries_difference':a['queries']-b['queries'],
                      'topology_calls_difference':a['topology_calls']-b['topology_calls']})

savecsv('per_run_components.csv',records);savecsv('method_components.csv',aggregates)
savecsv('paired_old_new.csv',paired);savecsv('baseline_component_differences.csv',decomp)
manifest = [{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(files))]
result = {'passed':True,'input_timing_reconciliations':checks,'runs':len(records),'aggregates':aggregates,
          'same_batch_old_new':paired,'baseline_differences':decomp,
          'date_scope':{s:sorted({r['week'] for r in records if r['stage']==s}) for s in ['grid44','grid51','grid52']},
          'new_fits':0,'new_forecasts':0,
          'limitations':'GRID44andGRID52dates differ;their percentages are not a direct old/new model comparison. Component differences include different physical paths/call counts,not a causal attribution. Saved timers cannot identify internal prior subcomponents without a new isolated measurement.'}
(OUT/'timing_review.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
(OUT/'timing_inputs_manifest.json').write_text(json.dumps({'files':manifest},indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'runs':len(records),'reconciliations':checks,'same_batch_old_new':paired,
                  'grid52_components':[r for r in aggregates if r['stage']=='grid52'],
                  'baseline_differences':decomp},indent=2))
