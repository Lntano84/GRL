"""Descriptive analysis of GRID06 January. No environment or model imports."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/grid06'
OUT = ROOT / 'outputs/grid07'
OUT.mkdir(parents=True, exist_ok=True)
SCENE = '2035-01-08_5'
POLICIES = ['NN20', 'FALLBACK90', 'NN352']

def load(p):
    return json.loads(p.read_text(encoding='utf-8'))

def lines(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, x):
    (OUT / name).write_text(json.dumps(x, indent=2, allow_nan=False), encoding='utf-8')

assert load(SOURCE / 'GRID06_partial_audit.json')['completed_data_audit_passed']
for rel, h in load(SOURCE / 'delivery_manifest.json').items():
    assert sha(ROOT / rel) == h, rel

rows, events, action_metrics = {}, {}, {}
inputs = {}
source_lines = {}
for p in POLICIES:
    folder = SOURCE / 'runs' / f'{SCENE}__{p}'
    for name in ['steps.jsonl', 'events.jsonl', 'metadata.json', 'initial.npz', 'completion.json']:
        path = folder / name
        inputs[str(path.relative_to(ROOT))] = sha(path)
    rows[p], events[p] = lines(folder / 'steps.jsonl'), lines(folder / 'events.jsonl')
    assert len(rows[p]) == 2017 and load(folder / 'completion.json')['reached_native_horizon']
    layout = load(folder / 'metadata.json')['action_vector_layout']
    slices, pos = {}, 0
    for field in layout:
        slices[field['name']] = slice(pos, pos + field['size'])
        pos += field['size']
    metrics = []
    for r in rows[p]:
        path = folder / r['file']
        assert sha(path) == r['file_sha256']
        with np.load(path, allow_pickle=False) as z:
            a = z['action']
            red = a[slices['_redispatch']]
            extra = dict(actual_dispatch_vector=z['actual_dispatch'].astype(float).tolist()) if r['step']==221 else {}
            metrics.append(dict(step=r['step'], redispatch_action_l1=float(np.abs(red).sum(dtype=np.float64)),
                nonzero_redispatch_commands=int(np.count_nonzero(red)),
                actual_dispatch_l1=r['ledger']['redispatch_mwh']*12,
                max_rho=r['after_rho_max'],
                modified_topology_positions=np.flatnonzero(a[slices['_set_topo_vect']]).tolist(),
                topology_set_values=a[slices['_set_topo_vect']][a[slices['_set_topo_vect']] != 0].astype(int).tolist(),**extra))
    action_metrics[p] = metrics
    print('OFFLINE_VECTORS_READ',p,len(metrics),flush=True)

costs = {p: sum(r['raw_operational_cost'] for r in rows[p]) for p in POLICIES}
assert all(rows['NN20'][i]['action_hash'] == rows['FALLBACK90'][i]['action_hash']
           and rows['NN20'][i]['observation_hash'] == rows['FALLBACK90'][i]['observation_hash'] for i in range(220))
assert rows['NN20'][220]['before_observation_hash'] == rows['FALLBACK90'][220]['before_observation_hash']
assert rows['NN20'][220]['action_hash'] != rows['FALLBACK90'][220]['action_hash']
price_profiles = {p:dict(Counter(r['ledger']['marginal_cost'] for r in rows[p])) for p in POLICIES}
different_price_steps = sum(a['ledger']['marginal_cost']!=b['ledger']['marginal_cost'] for a,b in zip(rows['NN20'], rows['FALLBACK90']))

timeline = []
cum = 0.
for i in range(2017):
    n, f = rows['NN20'][i], rows['FALLBACK90'][i]
    delta = f['raw_operational_cost'] - n['raw_operational_cost']
    cum += delta
    timeline.append(dict(step=i+1, timestamp=n['timestamp'], nn20_raw_cost=n['raw_operational_cost'],
        fallback_raw_cost=f['raw_operational_cost'], cost_difference=delta, cumulative_cost_difference=cum,
        nn20_actual_dispatch_l1=action_metrics['NN20'][i]['actual_dispatch_l1'],
        fallback_actual_dispatch_l1=action_metrics['FALLBACK90'][i]['actual_dispatch_l1'],
        nn20_redispatch_command_l1=action_metrics['NN20'][i]['redispatch_action_l1'],
        fallback_redispatch_command_l1=action_metrics['FALLBACK90'][i]['redispatch_action_l1']))
assert abs(cum - (costs['FALLBACK90'] - costs['NN20'])) < 1e-7
with (OUT / 'january_cost_dispatch_timeline.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(timeline[0])); w.writeheader(); w.writerows(timeline)

first = {}
for p in POLICIES:
    es = [e for e in events[p] if e['step'] == 221]
    tm = next(e for e in es if e['kind'] == 'module' and e['module'] == 'topo_12_unsafe')
    candidate = next(e for e in es if e['kind'] == 'simulate' and e['module'] == 'topo_12_unsafe' and e['action_hash'] == tm['returned_action_hash'])
    first[p] = dict(timestamp=rows[p][220]['timestamp'], before_rho=rows[p][220]['before_rho_max'],
        topology_forecast_rho=candidate['rho_max'], topology_reward=candidate['reward'],
        final_action_hash=rows[p][220]['action_hash'], actual_after_rho=rows[p][220]['after_rho_max'],
        raw_operational_cost=rows[p][220]['raw_operational_cost'],
        action_metrics=action_metrics[p][220],
        solve_events=[e for e in es if e['kind'] == 'solve'],
        fallback_events=[e for e in es if e['kind'] == 'fallback'])
assert first['FALLBACK90']['topology_reward'] > first['NN20']['topology_reward']
assert first['FALLBACK90']['raw_operational_cost'] < first['NN20']['raw_operational_cost']
assert first['FALLBACK90']['actual_after_rho'] > first['NN20']['actual_after_rho']
save('first_common_state_comparison.json', first)

components = load(SOURCE / 'operational_cost_components_partial.json')
component_map = {r['policy']: r for r in components if r['scenario'] == SCENE}
diff = {key: component_map['FALLBACK90'][key] - component_map['NN20'][key]
        for key in ['loss_cost', 'redispatch_cost', 'curtailment_change_cost', 'storage_cost']}

partition = [(1,220),(221,288),(289,576),(577,864),(865,1152),(1153,1440),(1441,1728),(1729,2017)]
blocks = []
for a,b in partition:
    blocks.append(dict(first_step=a,last_step=b,
        cost={p:sum(r['raw_operational_cost'] for r in rows[p][a-1:b]) for p in POLICIES},
        redispatch_cost={p:sum(r['ledger']['marginal_cost']*r['ledger']['redispatch_mwh'] for r in rows[p][a-1:b]) for p in POLICIES}))

persistent = {}
for p in POLICIES:
    nonoptim = {i for i in range(1,2018)} - {e['step'] for e in events[p] if e['kind']=='module' and e['module']=='optim'}
    zero = [i for i,m in enumerate(action_metrics[p]) if m['nonzero_redispatch_commands']==0]
    has_dispatch = [i for i in zero if action_metrics[p][i]['actual_dispatch_l1']>1e-5]
    optim_calls = sum(e['kind']=='module' and e['module']=='optim' for e in events[p])
    persistent[p] = dict(optim_module_calls=optim_calls,
        steps_without_redispatch_commands=len(zero),
        steps_with_zero_command_but_nonzero_actual_dispatch=len(has_dispatch),
        zero_command_dispatch_fees=sum(rows[p][i]['ledger']['marginal_cost']*rows[p][i]['ledger']['redispatch_mwh'] for i in has_dispatch),
        dispatch_fees_when_optim_not_called=sum(rows[p][i-1]['ledger']['marginal_cost']*rows[p][i-1]['ledger']['redispatch_mwh'] for i in sorted(nonoptim)),
        rho_above_one_steps=sum(r['after_rho_max']>1 for r in rows[p]),
        highest_actual_rho=max(r['after_rho_max'] for r in rows[p]),
        solve_statuses=dict(Counter(e['problem_status'] for e in events[p] if e['kind']=='solve')))

summary = dict(scenario=SCENE,offline_only=True,model_fits=0,new_simulations=0,new_physical_steps=0,
    raw_costs=costs,relative_cost_increase=(costs['FALLBACK90']/costs['NN20']-1),
    first_action_difference_step=221,common_prior_observation=True,
    first_step_more_search_improves_topology_forecast_rho=True,
    first_step_more_search_improves_final_cost=True,
    first_step_more_search_worsens_final_rho=True,
    prefix_costs_exactly_equal=True,marginal_price_profiles=price_profiles,different_price_steps=different_price_steps,
    descriptive_fee_difference=diff,blocks=blocks,persistent_dispatch_description=persistent,
    most_cost_disadvantage_appears_after_step=576,
    limitations=['First-decision and whole-week contrasts are different estimands.',
        'This is descriptive accounting, not a counterfactual intervention isolating a single decision.',
        'Zero redispatch command does not imply zero cumulative actual redispatch.',
        'Finite user_limit/optimal_inaccurate solver returns do not certify an exact QP optimum.',
        'A rho surrogate limitation does not establish a learnable or novel RL advantage.'])
save('GRID07_offline_summary.json',summary)
save('offline_inputs_frozen.json',inputs)
save('action_metrics.json',action_metrics)
for rel,h in inputs.items():
    assert sha(ROOT/rel)==h
print(json.dumps({k:v for k,v in summary.items() if k not in ['blocks','limitations']},indent=2),flush=True)
