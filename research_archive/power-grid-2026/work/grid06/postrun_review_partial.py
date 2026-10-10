"""Separate arithmetic from saved data, explicitly scoped to the stopped prefix."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid06'

def load(p):
    return json.loads(p.read_text(encoding='utf-8'))

def lines(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

design = load(OUT / 'design_freeze.json')
summary = load(OUT / 'GRID06_partial_summary.json')
audit = load(OUT / 'GRID06_partial_audit.json')
manifest = load(OUT / 'run_manifest.json')
assert len(design['plan']) == 12 and len(manifest) == 10
assert all(m['exit'] == 0 for m in manifest[:9])
assert manifest[9]['exit'] != 0
for a, b in zip(manifest, design['plan']):
    assert (a['scenario'], a['policy']) == (b['scenario'], b['policy'])

perrun = {(r['scenario'], r['policy']): r for r in load(OUT / 'per_run_partial.json')}
assert len(perrun) == 9
meta, steps, costs, raw_rows = {}, {}, {}, {}
for job in manifest[:9]:
    key = (job['scenario'], job['policy'])
    folder = OUT / 'runs' / f'{key[0]}__{key[1]}'
    rows = lines(folder / 'steps.jsonl')
    events = lines(folder / 'events.jsonl')
    raw_rows[key] = rows
    meta[key] = load(folder / 'completion.json')
    steps[key] = len(rows)
    costs[key] = sum(r['raw_operational_cost'] for r in rows)
    assert steps[key] == perrun[key]['steps'] == meta[key]['physical_calls']
    assert costs[key] == perrun[key]['native_raw_cost_sum']
    assert meta[key]['reached_native_horizon'] == perrun[key]['completed_native_week']
    assert sum(r['simulate_calls'] for r in rows) == sum(e['kind'] == 'simulate' for e in events)
    fb = [e for e in events if e['kind'] == 'fallback']
    short = [e for e in events if e['kind'] == 'shortlist']
    topo_sims = sum(e['kind'] == 'simulate' and e['module'] == 'topo_12_unsafe' for e in events)
    if key[1] == 'FALLBACK90':
        assert all(not e['forced_test'] for e in fb)
        assert len(short) == len(fb) + sum(e['expanded'] for e in fb)
        assert topo_sims == 20 * len(fb) + 332 * sum(e['expanded'] for e in fb)
    else:
        assert not fb
        assert topo_sims == (20 if key[1] == 'NN20' else 352) * len(short)

# Use the frozen multiplicative threshold directly, not the analysis gain field.
threshold = design['quality_screen']['maximum_complete_week_relative_cost_regression']
assert threshold == .02
rejection_weeks = []
for s in design['selected_weeks'][:3]:
    f, n = (s, 'FALLBACK90'), (s, 'NN20')
    joint = meta[f]['reached_native_horizon'] and meta[n]['reached_native_horizon']
    rejection = steps[f] < steps[n] or (joint and costs[f] > (1 + threshold) * costs[n])
    if rejection:
        rejection_weeks.append(s)
    c = next(c for c in summary['contrasts'] if c['scenario'] == s and c['reference'] == 'NN20')
    assert c['jointly_complete'] == joint
    assert (c['earlier_end'] or c['cost_regression_exceeds_frozen_2pct']) == rejection
    if not joint:
        assert c['cost_reduction_relative_to_reference'] is None
assert rejection_weeks == [c['scenario'] for c in summary['confirmed_rejection_witnesses']]
assert rejection_weeks == [design['selected_weeks'][0]]
assert summary['verdict'] == 'FROZEN_REJECTION_ITEM_CONFIRMED_ON_COMPLETED_DATA'
assert summary['completed_runs'] == 9 and summary['failed_attempts'] == 1 and summary['not_started_runs'] == 2
assert summary['matrix_status'] == 'INCOMPLETE_IMPORT_WATCHDOG_STOP'
assert not summary['resource_acceptance_passed'] and not audit['matrix_complete']

failed = manifest[9]
folder = OUT / 'runs' / f"{failed['scenario']}__{failed['policy']}"
assert {p.name for p in folder.iterdir()} == {'console.log', 'watchdog_timeout.json'}
assert (folder / 'console.log').stat().st_size == 0
watch = load(folder / 'watchdog_timeout.json')
assert not watch['import_seen'] and watch['wall_s'] > design['caps']['import_s']
for job in design['plan'][10:]:
    assert not (OUT / 'runs' / f"{job['scenario']}__{job['policy']}").exists()
assert audit['parent_total_wall_s'] is None
assert sum(steps.values()) == audit['physical_steps'] == 12630
assert sum(sum(r['ledger_eligible'] for r in rows) for rows in raw_rows.values()) == 12627
assert summary['failed_week_no_strategy_outcome'] == failed['scenario']

# Reconcile the descriptive fee decomposition against raw ledgers once more.
for c in load(OUT / 'operational_cost_components_partial.json'):
    rows = raw_rows[(c['scenario'], c['policy'])]
    valid = [r for r in rows if r['ledger_eligible']]
    fields = {'loss_cost': 'losses_mwh', 'redispatch_cost': 'redispatch_mwh',
              'curtailment_change_cost': 'curtailment_delta_mwh', 'storage_cost': 'storage_throughput_mwh'}
    for name, field in fields.items():
        total = sum(r['ledger']['marginal_cost'] * r['ledger'][field] for r in valid)
        assert total == c[name]
    assert c['native_verified_total'] == sum(r['raw_operational_cost'] for r in valid)

hash_counts = {}
for name in ['execution_code_frozen.json', 'reused_assets_frozen.json', 'data_files_frozen.json']:
    frozen = load(OUT / name)
    for rel, digest in frozen.items():
        assert sha(ROOT / rel) == digest, rel
    hash_counts[name] = len(frozen)

review = dict(completed_data_arithmetic_review_passed=True, matrix_complete=False,
    resource_acceptance_passed=False, completed_runs=9, failed_attempts=1, not_started_runs=2,
    physical_steps=sum(steps.values()), raw_costs_and_event_counts_verified=True,
    confirmed_rejection_weeks=rejection_weeks, verdict=summary['verdict'],
    frozen_hashes_unchanged=hash_counts,
    scope='Separate arithmetic on the same saved observations; not an independent simulator, neural inference, QP certification, or full-matrix confirmation.')
(OUT / 'GRID06_partial_postrun_review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')

# Seal stable files only; writer logs and this manifest itself are excluded.
files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(OUT.glob('*'))
         if p.is_file() and p.suffix != '.log' and p.name != 'delivery_manifest.json'}
files.update({str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'work/grid06').glob('*.py'))})
(OUT / 'delivery_manifest.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
for rel, digest in load(OUT / 'delivery_manifest.json').items():
    assert sha(ROOT / rel) == digest, rel
print(json.dumps(review, indent=2), flush=True)
