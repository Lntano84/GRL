"""Pure online eligibility rules for BC07; no outcome or target input."""
import hashlib


HASH_SEED = 'BC07-request-hash-v1-seed-20260930'


def request_hash64(request_sequence):
    payload = f'{HASH_SEED}:{int(request_sequence)}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big', signed=False)


def decide(method, *, parent_accept, score, threshold, candidate_source,
           block_gap_seconds, request_hash, is_evaluation, check_only, calibration):
    """Return (effective decision, supplemented, reason) from current state only."""
    if check_only:
        return bool(parent_accept), False, 'check_only_parent_unchanged'
    if bool(parent_accept):
        return True, False, 'parent_accept'
    if not is_evaluation:
        return False, False, 'warmup_parent_unchanged'
    if candidate_source != 'demand':
        return False, False, 'not_demand_candidate'

    # parent_accept=False implies score <= the live parent threshold. Retain an
    # explicit guard so a future caller cannot turn this into a new threshold.
    if float(score) > float(threshold):
        return False, False, 'parent_reject_guard_failed'

    if method == 'HIGH-CONTROL':
        return False, False, 'control'
    rules = calibration['rules']
    if method == 'HIGH-SCORE':
        rule = rules['SCORE']
        if rule['status'] == 'calibrated':
            cutoff_score = float(rule['cutoff_score'])
            cutoff_hash = int(rule['cutoff_request_hash64'])
            accept = float(score) > cutoff_score or (
                float(score) == cutoff_score and int(request_hash) <= cutoff_hash)
            return (True, True, 'score_tail') if accept else (False, False, 'below_score_cutoff')
    elif method == 'HIGH-RECENCY':
        rule = rules['RECENCY']
        if block_gap_seconds is None:
            return False, False, 'no_block_history'
        if rule['status'] == 'accept_all_valid_due_to_insufficient_history':
            return True, True, 'all_valid_recency_shortage'
        if rule['status'] == 'calibrated':
            cutoff_gap = float(rule['cutoff_block_gap_seconds'])
            cutoff_hash = int(rule['cutoff_request_hash64'])
            accept = float(block_gap_seconds) < cutoff_gap or (
                float(block_gap_seconds) == cutoff_gap and int(request_hash) <= cutoff_hash)
            return (True, True, 'recency_tail') if accept else (False, False, 'above_recency_cutoff')
    elif method == 'HIGH-RANDOM':
        rule = rules['RANDOM']
        if rule['status'] == 'calibrated' and int(request_hash) <= int(rule['cutoff_request_hash64']):
            return True, True, 'request_hash_sample'
    else:
        raise ValueError(f'unknown BC07 method: {method}')
    return False, False, 'no_calibration_cutoff'
