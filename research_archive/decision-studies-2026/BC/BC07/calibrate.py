"""Freeze q=1% supplement cutoffs from the BC06 warmup decision snapshots."""
import gzip
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OUT = BASE / 'outputs'
SNAPSHOT = OUT / 'BC06_candidate_decisions.jsonl.gz'
SNAPSHOT_MANIFEST = OUT / 'BC06_snapshot_manifest.json'
SOURCE_CONFIG = BASE / 'work' / 'bc03' / 'STATIC-HIGH' / 'config.json'
EVAL_START = 86401.23277902603
THETA0 = 0.798545
THETAH = 2 * THETA0 / (1 + THETA0)
Q = 0.01
METHODS = ['HIGH-CONTROL', 'HIGH-SCORE', 'HIGH-RECENCY', 'HIGH-RANDOM']

sys.path.insert(0, str(HERE))
from policy import HASH_SEED, request_hash64


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def boundary(sorted_rows, target_count, key_fn):
    """Choose the first lexicographic key whose complete tie group reaches target."""
    if not sorted_rows or target_count <= 0:
        return None, 0
    i = 0
    accumulated = 0
    while i < len(sorted_rows):
        key = key_fn(sorted_rows[i])
        j = i + 1
        while j < len(sorted_rows) and key_fn(sorted_rows[j]) == key:
            j += 1
        accumulated += j - i
        if accumulated >= target_count:
            return key, accumulated
        i = j
    return key_fn(sorted_rows[-1]), len(sorted_rows)


def main():
    work_cal = HERE / 'calibration.json'
    output_cal = OUT / 'BC07_calibration.json'
    if work_cal.exists() or output_cal.exists():
        raise RuntimeError('BC07 calibration already exists; refusing to resample or change cutoffs')
    manifest = json.loads(SNAPSHOT_MANIFEST.read_text(encoding='utf-8'))
    if not manifest.get('snapshot_frozen_before_BC05_target_link'):
        raise AssertionError('BC06 snapshot is not marked as sealed before outcome linkage')
    snap_manifest_work = json.loads((BASE / 'work' / 'bc06' / 'snapshot_manifest.json').read_text(encoding='utf-8'))
    if sha(SNAPSHOT) != manifest['files']['candidate_decisions.jsonl.gz']['sha256']:
        raise AssertionError('BC06 candidate snapshot hash differs from its manifest')
    if sha(SNAPSHOT_MANIFEST) != sha(BASE / 'work' / 'bc06' / 'snapshot_manifest.json'):
        raise AssertionError('BC06 snapshot manifests differ')
    if sha(BASE / 'work' / 'bc06' / 'HIGH-CONTROL' / 'candidate_decisions.jsonl.gz') != snap_manifest_work['files']['candidate_decisions.jsonl.gz']['sha256']:
        raise AssertionError('work candidate snapshot differs from frozen manifest')

    pool = []
    request_hashes = {}
    counts = {
        'snapshot_actual_candidates_read': 0,
        'warmup_actual_demand_candidates': 0,
        'warmup_demand_candidates_rejected_by_high': 0,
        'warmup_pool_with_finite_block_gap': 0,
        'warmup_pool_without_block_history': 0,
        'warmup_non_demand_candidates_ignored': 0,
        'warmup_high_accepts_excluded': 0,
        'evaluation_candidates_not_used_for_calibration': 0,
    }
    with gzip.open(SNAPSHOT, 'rt', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            counts['snapshot_actual_candidates_read'] += 1
            elapsed = float(row['candidate_created_trace_elapsed_s'])
            if elapsed >= EVAL_START:
                counts['evaluation_candidates_not_used_for_calibration'] += 1
                continue
            if row['candidate_source'] != 'demand':
                counts['warmup_non_demand_candidates_ignored'] += 1
                continue
            counts['warmup_actual_demand_candidates'] += 1
            score = row.get('author_model_score')
            if score is None or not math.isfinite(float(score)):
                raise AssertionError(f'missing/nonfinite score in frozen candidate row {line_no}')
            score = float(score)
            if score > THETAH:
                counts['warmup_high_accepts_excluded'] += 1
                continue
            counts['warmup_demand_candidates_rejected_by_high'] += 1
            seq = int(row['actual_decision_request_sequence'])
            h = request_hashes.get(seq)
            if h is None:
                h = request_hash64(seq)
                request_hashes[seq] = h
            gap = row['candidate_key_history_at_request_start'].get('seconds_since_block_last_get')
            if gap is None:
                counts['warmup_pool_without_block_history'] += 1
                gap_value = None
            else:
                gap_value = float(gap)
                if not math.isfinite(gap_value) or gap_value < 0:
                    raise AssertionError(('invalid block gap', line_no, gap))
                counts['warmup_pool_with_finite_block_gap'] += 1
            pool.append((score, gap_value, h))

    if counts['snapshot_actual_candidates_read'] != manifest['counts']['actual_candidate_decisions']:
        raise AssertionError(('snapshot row count', counts, manifest['counts']))
    if not pool:
        raise AssertionError('empty eligible warmup calibration pool')
    desired = max(1, math.ceil(Q * len(pool)))
    selected = {}

    pool.sort(key=lambda x: (-x[0], x[2]))
    score_key, score_n = boundary(pool, desired, lambda x: (-x[0], x[2]))
    if score_key is None:
        score_rule = {'status': 'unavailable', 'calibration_selected_chunks': 0}
    else:
        score_rule = {
            'status': 'calibrated',
            'cutoff_score': -score_key[0],
            'cutoff_request_hash64': score_key[1],
            'sort_order': 'score descending, request SHA256 hash ascending',
            'calibration_selected_chunks': score_n,
        }
        selected['SCORE'] = score_n

    finite = [x for x in pool if x[1] is not None]
    if len(finite) < desired:
        recency_rule = {
            'status': 'accept_all_valid_due_to_insufficient_history' if finite else 'unavailable',
            'calibration_selected_chunks': len(finite),
            'valid_history_chunks': len(finite),
            'requested_calibration_chunks': desired,
            'note': 'No first-access row is promoted to a zero-second gap.'
        }
        selected['RECENCY'] = len(finite)
    else:
        finite.sort(key=lambda x: (x[1], x[2]))
        recency_key, recency_n = boundary(finite, desired, lambda x: (x[1], x[2]))
        recency_rule = {
            'status': 'calibrated',
            'cutoff_block_gap_seconds': recency_key[0],
            'cutoff_request_hash64': recency_key[1],
            'sort_order': 'block gap ascending, request SHA256 hash ascending; missing gaps excluded',
            'calibration_selected_chunks': recency_n,
            'valid_history_chunks': len(finite),
            'requested_calibration_chunks': desired,
        }
        selected['RECENCY'] = recency_n

    pool.sort(key=lambda x: x[2])
    random_key, random_n = boundary(pool, desired, lambda x: (x[2],))
    random_rule = {
        'status': 'calibrated' if random_key is not None else 'unavailable',
        'cutoff_request_hash64': random_key[0] if random_key is not None else None,
        'sort_order': 'request SHA256 hash ascending; all candidate chunks in one request share one hash',
        'calibration_selected_chunks': random_n,
    }
    selected['RANDOM'] = random_n

    for key, rule in [('SCORE', score_rule), ('RECENCY', recency_rule), ('RANDOM', random_rule)]:
        n = int(rule.get('calibration_selected_chunks', 0))
        rule['calibration_pool_chunks'] = len(pool)
        rule['calibration_fraction_of_high_rejected_demand_chunks'] = n / len(pool)
        rule['calibration_q'] = Q

    calibration = {
        'protocol': 'BC07; one frozen warmup calibration; no model training or outcome labels',
        'q': Q,
        'parent_threshold_schedule': {
            'warmup': THETA0,
            'evaluation': THETAH,
            'rule': 'score > threshold; parent HIGH; supplemental decisions disabled during warmup',
        },
        'evaluation_start_trace_elapsed_s': EVAL_START,
        'calibration_scope': {
            'source': 'BC06 frozen actual candidate snapshot',
            'input_file': str(SNAPSHOT.resolve()),
            'input_sha256': sha(SNAPSHOT),
            'snapshot_manifest_sha256': sha(SNAPSHOT_MANIFEST),
            'selection': 'candidate_source=demand AND candidate_created_trace_elapsed_s < evaluation_start AND author_model_score <= evaluation_HIGH_threshold',
            'warmup_uses_base_policy': True,
            'interpretation': 'scores captured under the actual BASE warmup; HIGH rejection membership is recomputed from the saved scores, not claimed as an observed warmup HIGH decision',
            'counts': counts,
            'high_rejected_demand_pool_chunks': len(pool),
            'requested_supplement_chunks': desired,
            'hash_algorithm': 'SHA-256 over UTF-8 ASCII string "seed:request_sequence"; first 8 digest bytes interpreted unsigned big-endian',
            'hash_seed': HASH_SEED,
            'tie_rule': 'preserve the complete group with identical primary cutoff value and request hash; do not split candidates to hit exactly q',
        },
        'rules': {'SCORE': score_rule, 'RECENCY': recency_rule, 'RANDOM': random_rule},
        'calibration_actual_fractions': selected,
        'future_labels_read': False,
        'peak_window_or_position_read': False,
        'write_budget_used_for_selection': False,
        'calibration_datetime_local': '2026-09-30 Asia/Shanghai',
    }
    work_cal.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copyfile(work_cal, output_cal)

    base_config = json.loads(SOURCE_CONFIG.read_text(encoding='utf-8'))
    for method in METHODS:
        arm_dir = HERE / 'arms' / method
        if arm_dir.exists() and any(arm_dir.iterdir()):
            raise RuntimeError(f'output directory already contains an attempted replay: {arm_dir}')
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / 'raw').mkdir(exist_ok=True)
        config = dict(base_config)
        config['output_dir'] = str((arm_dir / 'raw').resolve()).replace('\\', '/')
        (arm_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({'calibration': calibration, 'arms': METHODS,
                      'calibration_path': str(work_cal.resolve()),
                      'output_calibration_path': str(output_cal.resolve())}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
