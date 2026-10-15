"""One prospective repair: verify a plateau before every compressed return.

No date, step, hidden outcome, or FULL result is an online input.  The old
FRAME/FULL policies remain delegated to their immutable implementations.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sp = importlib.util.spec_from_file_location('grid69_reference', ROOT/'work/grid67/policies.py')
old = importlib.util.module_from_spec(sp); sp.loader.exec_module(old)
key = old.key
POLICIES = old.POLICIES + ('FRAME_VERIFY',)

def run(order, observe, policy, subids, aliases, safe=.9, limit=None):
    if policy != 'FRAME_VERIFY':
        return old.run(order, observe, policy, subids, aliases, safe, limit)
    assert len(order) == len(set(order))
    rows, events, best = [], [], None
    def purchase(ids, phase):
        nonlocal best
        before = None if best is None else key(best)
        for i in ids:
            r = observe(i, phase); assert r['pool_id'] == i
            rows.append(r)
            if r['strict_admissible'] and (best is None or key(r) > key(best)):
                best = r
        after = None if best is None else key(best)
        improved = after is not None and (before is None or after > before)
        events.append(dict(queries=len(rows), phase=phase,
                           chosen=None if best is None else best['pool_id'], improved=improved))
        return improved
    n = min(128, len(order)); purchase(order[:n], 'shortlist')
    prefix = None if best is None else best['pool_id']
    fallback = n < len(order)
    reason, stale = 'full_exhaustion', 0
    # rho < .9 is only a one-step observation.  It no longer bypasses the
    # common minimum coverage and feedback-based stopping check.
    for offset in range(n, len(order), 64):
        improved = purchase(order[offset:offset+64], 'verification')
        stale = 0 if improved else stale+1
        if best is not None and len(rows) >= 256 and stale >= 2 and len(rows) < len(order):
            reason = 'minimum_verified_plateau'; break
    chosen = None if best is None else best['pool_id']
    sub, local = None, []
    if chosen is not None and len(rows) < len(order):
        sub = int(subids[chosen])
        targets = {int(rep) for alias, rep in aliases.items() if int(subids[int(alias)]) == sub}
        seen = {r['pool_id'] for r in rows}
        local = [i for i in order if i not in seen and i in targets]
        purchase(local, 'local_completion'); reason += '_local_complete'
    return dict(outcomes=rows, chosen_pool_id=None if best is None else best['pool_id'],
                shortlist_choice=prefix, fallback=fallback, stop_reason=reason,
                events=events, public_queries=len(rows), local_substation=sub,
                local_queries=len(local), completion_budget=len(local), global_extra_queries=0,
                pre_extra_choice=chosen, completion_candidate_ids=local, extra_selected_ids=local,
                initial_prefix=128, minimum_verification=256, stale_batches=2)
