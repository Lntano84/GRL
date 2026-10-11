"""Finish substations represented by observed best-reward ties, without oracle access."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('grid42_staged', ROOT / 'work/grid27/policy.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
key = old.key


def tied_complete(order, observe, policy, subids, aliases, safe=.9):
    result = old.staged(order, observe, policy, safe)
    seen = {r['pool_id'] for r in result['outcomes']}
    groups = set(); added = 0
    while len(seen) < len(order):
        valid = [r for r in result['outcomes'] if r['strict_admissible']]
        if not valid: break
        best = max(key(r)[0] for r in valid)
        wanted = {int(subids[r['pool_id']]) for r in valid if key(r)[0] == best}
        pending = wanted - groups
        if not pending: break
        groups |= pending
        targets = {int(rep) for i, rep in aliases.items() if int(subids[int(i)]) in pending}
        for i in order:
            if i in targets and i not in seen:
                row = observe(i, 'tie_completion')
                assert row['pool_id'] == i
                result['outcomes'].append(row); seen.add(i); added += 1
        # A better value discovered by completion can identify another group.
        # Only feedback already purchased enters this loop.
    valid = [r for r in result['outcomes'] if r['strict_admissible']]
    winner = max(valid, key=key)['pool_id'] if valid else None
    before = result['chosen_pool_id']
    result.update(chosen_pool_id=winner, public_queries=len(seen),
                  local_substation=None, local_queries=added,
                  tie_substations=sorted(groups),
                  stop_reason=result['stop_reason'] + '_observed_tie_complete')
    result['events'].append({'queries': len(seen), 'phase': 'tie_completion',
                            'chosen': winner, 'improved': winner != before})
    assert len(seen) == len(result['outcomes']) <= len(order)
    return result


def hand_checks():
    order = list(range(400)); aliases = {i:i for i in order}
    subs = [0] * 400; subs[399] = 1
    def feedback(i, phase):
        return {'pool_id': i, 'reward': 1., 'rho': .8,
                'strict_admissible': True, 'phase': phase}
    # Safe-prefix ties are completed, but unobserved remote groups are not.
    a = tied_complete(order, feedback, 'PATIENCE64', subs, aliases)
    assert a['public_queries'] == 399 and a['chosen_pool_id'] == 0
    # A second tied substation visible in the prefix brings its late member in.
    subs[127] = 1
    b = tied_complete(order, feedback, 'PATIENCE64', subs, aliases)
    assert b['public_queries'] == 400
    # Lowest-ID exact fp32 tie is recovered even when ranked outside prefix.
    subs = [0] * 400
    c = tied_complete(order[::-1], feedback, 'PATIENCE64', subs, aliases)
    assert c['chosen_pool_id'] == 0 and c['public_queries'] == 400
    d = tied_complete(order, lambda i,p: {**feedback(i,p), 'strict_admissible': False},
                      'PATIENCE64', subs, aliases)
    assert d['chosen_pool_id'] is None and d['public_queries'] == 400
    assert key({'pool_id': 4, 'reward': 1.+1e-9})[0] == 1.
    return 5
