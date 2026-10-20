"""Recheck saved feedback and one redundant default condition; no simulator imports.

Historical policies and records are read-only. qmin=0 is a logical equivalence
check for the current p=128/h=64/patience=2, not a new tuned policy.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/eenergy_mechanism_20261020'
OUT.mkdir(parents=True, exist_ok=True)


def module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


P = module('eenergy_review_policy', 'work/grid89_review/policies.py')
R = module('eenergy_frozen_guard', 'work/grid70/policies.py')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inputs = {}


def track(p):
    inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    return p


def signature(result):
    return {
        'trace': [(r['pool_id'], r['phase']) for r in result['outcomes']],
        'choice': result['chosen_pool_id'],
        'stop': result['stop_reason'],
        'prefix': result['shortlist_choice'],
        'local': result['completion_candidate_ids'],
        'events': result['events'],
    }


def run(c, subs, qmin=256, reference=False):
    truth = {r['pool_id']: r for r in c['outcomes']}
    seen = set()

    def reveal(i, phase):
        assert i in truth and i not in seen
        seen.add(i)
        return dict(truth[i], phase=phase)

    args = (c['order'], reveal, 'FRAME_GUARD', subs,
            {int(k): int(v) for k, v in c['aliases'].items()})
    if reference:
        result = R.run(*args, limit=c['rho_limit'])
    else:
        result = P.run(*args, params={'qmin': qmin}, limit=c['rho_limit'])
    assert result['public_queries'] == len(seen)
    return result


packpath = track(ROOT / 'outputs/grid85_mechanism/mechanism_feedback.json.gz')
with gzip.open(packpath, 'rt', encoding='utf-8') as f:
    pack = json.load(f)
cases = []
for source, raw in pack['records'].items():
    c = dict(raw, outcomes=[dict(zip(pack['fields'], r)) for r in raw['outcomes']])
    for r in c['outcomes']:
        r['exceptions'] = r.pop('has_exception')
    cohort = 'old13' if '__2035-' in source else 'G1'
    cases.append((source, cohort, c))

sealed = ROOT / 'outputs/grid89_review/sealed/runs'
for p in sorted(sealed.glob('r0__*__FULL/call[0-9][0-9][0-9].json')):
    c = json.loads(track(p).read_text(encoding='utf-8'))
    cases.append((p.relative_to(ROOT).as_posix(), 'G2', c))
assert cases and any(cohort == 'G2' for _, cohort, _ in cases)

rows = []
for source, cohort, c in cases:
    a = run(c, pack['subids'])
    b = run(c, pack['subids'], qmin=0)
    r = run(c, pack['subids'], reference=True)
    assert signature(a) == signature(b), source
    assert signature(a) == signature(r), source
    rows.append(dict(source=source, cohort=cohort,
                     full_feedback=len(c['order']) == len(c['outcomes']),
                     queries=a['public_queries'], choice=a['chosen_pool_id'],
                     identical_query_trace=True, identical_choice=True,
                     identical_stop=True, frozen_policy_agrees=True))

# Boundary checks also cover no incumbent, truncated last batches, safety
# transitions, plateaus, and late improvements. These do not create grid data.
boundary_count = 0
boundary_stop_labels = []
for n in [0, 1, 63, 64, 127, 128, 129, 191, 192, 193, 255, 256,
          257, 319, 320, 321, 512, 909, 1000]:
    for pattern in ['flat_unsafe', 'flat_safe', 'late_gain', 'none_admissible',
                    'safe_after_192', 'continuing_gain']:
        outcomes = []
        for i in range(n):
            reward = (float(i // 64) if pattern == 'continuing_gain'
                      else 2.0 if pattern == 'late_gain' and i == n - 1 else 1.0)
            rho = .8 if (pattern == 'flat_safe' or
                        pattern == 'safe_after_192' and i >= 192) else 1.1
            outcomes.append(dict(pool_id=i, reward=reward, rho=rho,
                                 strict_admissible=pattern != 'none_admissible'))
        c = dict(order=list(range(n)), outcomes=outcomes,
                 aliases={i: i for i in range(n)}, rho_limit=2.)
        subs = [i % 3 for i in range(n)]
        a, b = run(c, subs), run(c, subs, qmin=0)
        assert signature(a) == signature(b), (n, pattern)
        frozen = run(c, subs, reference=True)
        current_sig, frozen_sig = signature(a), signature(frozen)
        current_reason, frozen_reason = current_sig.pop('stop'), frozen_sig.pop('stop')
        assert current_sig == frozen_sig, (n, pattern, current_sig, frozen_sig)
        if current_reason != frozen_reason:
            boundary_stop_labels.append(dict(candidates=n, pattern=pattern,
                                            review_label=current_reason,
                                            frozen_label=frozen_reason,
                                            trace_and_choice_identical=True))
        boundary_count += 1

with (OUT / 'redundancy_cases.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# Recompute same-state differences from saved CSV, without pooling cohorts.
def csvread(rel):
    p = track(ROOT / rel)
    return list(csv.DictReader(p.open(encoding='utf-8-sig')))


pair_rows = csvread('outputs/grid85_mechanism/equal_budget_pairs.csv')
summary = []
for old, label in [('old13', 'old13'), ('new6', 'G1')]:
    rr = [r for r in pair_rows if r['cohort'] == old]
    gains = [float(r['local_reward_gain']) - float(r['global_reward_gain'])
             for r in rr if r['local_reward_gain'] and r['global_reward_gain']]
    summary.append(dict(cohort=label, all_full_states=len(rr),
                        reward_comparable_states=len(gains),
                        triggered_states=sum(int(r['extra_queries']) > 0 for r in rr),
                        local_reward_wins=sum(g > 1e-6 for g in gains),
                        global_reward_wins=sum(g < -1e-6 for g in gains),
                        mean_local_minus_global_reward=sum(gains)/len(gains),
                        minimum_signed_difference=min(gains),
                        maximum_signed_difference=max(gains)))
rr = csvread('outputs/grid89_review/sealed_equal_budget_pairs.csv')
gains = [float(r['local_minus_global_reward']) for r in rr
         if r['local_minus_global_reward']]
assert all(r['same_budget'] == 'True' for r in rr)
summary.append(dict(cohort='G2', all_full_states=len(rr),
                    reward_comparable_states=len(gains),
                    triggered_states=sum(int(r['extra_queries']) > 0 for r in rr),
                    local_reward_wins=sum(g > 1e-6 for g in gains),
                    global_reward_wins=sum(g < -1e-6 for g in gains),
                    mean_local_minus_global_reward=sum(gains)/len(gains),
                    minimum_signed_difference=min(gains),
                    maximum_signed_difference=max(gains)))
(OUT / 'local_effect_by_cohort.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

for rel in ['work/grid70/policies.py', 'work/grid89_review/policies.py',
            'work/grid67/policies.py', 'work/grid63/policies.py',
            'work/grid60/policies.py', 'work/grid27/policy.py']:
    track(ROOT / rel)
receipt = dict(passed=True, saved_calls=len(rows),
               cohorts={s: sum(r['cohort'] == s for r in rows)
                        for s in ['old13', 'G1', 'G2']},
               boundary_cases=boundary_count,
               boundary_reference_label_differences=boundary_stop_labels,
               conclusion='qmin=256 is redundant only for current default '
                          'p=128,h=64,patience=2 and existing stopping semantics',
               proof='Plateau stop needs at least two post-prefix batches. '
                     'Any nonfinal second batch ends at >=128+2*64=256. '
                     'A final partial batch cannot satisfy rows<len(order). '
                     'The separate first-below-safe stop is unchanged.',
               original_sources_modified=False, new_training=0,
               new_grid_predictions=0, new_closed_loop_runs=0,
               scope='Saved-feedback replay and synthetic boundary verification; '
                     'no independent AC solve and no new generalization evidence',
               inputs_sha256=inputs, script_sha256=sha(Path(__file__)))
(OUT / 'audit_receipt.json').write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in receipt.items() if k != 'inputs_sha256'},
                 ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
