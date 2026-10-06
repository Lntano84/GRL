"""BC07 small-prefix/interface checks; this script never starts a replay."""
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OUT = BASE / 'outputs'
CAL = json.loads((HERE / 'calibration.json').read_text(encoding='utf-8'))
SNAPSHOT = OUT / 'BC06_candidate_decisions.jsonl.gz'
MODEL = BASE / 'work' / 'bc01' / 'Baleen-FAST24' / 'tmp' / 'example' / '201910_Region1_0_0.1' / 'ea_5892.86_wr_35.599_admit_threshold_binary.model'
EVAL_START = CAL['evaluation_start_trace_elapsed_s']
METHODS = ['HIGH-CONTROL', 'HIGH-SCORE', 'HIGH-RECENCY', 'HIGH-RANDOM']
sys.path.insert(0, str(HERE))
from policy import decide, request_hash64


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((OUT / 'BC06_snapshot_manifest.json').read_text(encoding='utf-8'))
    assert sha(SNAPSHOT) == manifest['files']['candidate_decisions.jsonl.gz']['sha256']
    assert CAL['future_labels_read'] is False and CAL['peak_window_or_position_read'] is False
    assert (HERE / 'calibration.json').read_bytes() == (OUT / 'BC07_calibration.json').read_bytes()

    baseline_cfg = json.loads((BASE / 'work' / 'bc03' / 'STATIC-HIGH' / 'config.json').read_text(encoding='utf-8'))
    config_errors = {}
    for method in METHODS:
        cfg = json.loads((HERE / 'arms' / method / 'config.json').read_text(encoding='utf-8'))
        expected = dict(baseline_cfg)
        expected['output_dir'] = str((HERE / 'arms' / method / 'raw').resolve()).replace('\\', '/')
        config_errors[method] = cfg != expected
        assert cfg == expected, f'{method} changes more than the output directory from HIGH'
    assert set(config_errors.values()) == {False}

    policy_source = (HERE / 'policy.py').read_text(encoding='utf-8').lower()
    replay_source = (HERE / 'replay.py').read_text(encoding='utf-8').lower()
    forbidden = ['bc05', 'frozen_targets', 'label_links', 'corresponding_peak', 'window 578']
    assert not any(token in policy_source or token in replay_source for token in forbidden), 'future outcome/peak access in online strategy interface'

    booster = lgb.Booster(model_file=str(MODEL))
    samples = {'warmup': [], 'evaluation': []}
    small_prefix_rows = 0
    with gzip.open(SNAPSHOT, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row.get('actual_decision_request_sequence') is None:
                continue
            eval_row = float(row['candidate_created_trace_elapsed_s']) >= EVAL_START
            phase = 'evaluation' if eval_row else 'warmup'
            if len(samples[phase]) < 32:
                samples[phase].append(row)
            else:
                continue
            if all(len(v) >= 32 for v in samples.values()):
                break

    if any(len(v) < 32 for v in samples.values()):
        raise AssertionError(('could not obtain warmup/evaluation prefixes', {k: len(v) for k, v in samples.items()}))
    for phase_rows in samples.values():
        for row in phase_rows:
            small_prefix_rows += 1
            threshold = float(row['threshold'])
            score = float(row['author_model_score'])
            parent = score > threshold
            if parent != bool(row['author_decision_accept']):
                raise AssertionError(('saved HIGH decision mismatch', row['candidate_id']))
            gap = row['legal_key_history_at_decision_request_start'].get('seconds_since_block_last_get')
            seq = int(row['actual_decision_request_sequence'])
            h = request_hash64(seq)
            is_eval = float(row['candidate_created_trace_elapsed_s']) >= EVAL_START
            source = row['candidate_source']
            for method in METHODS:
                d0, changed0, _ = decide(method, parent_accept=parent, score=score,
                                         threshold=threshold, candidate_source=source,
                                         block_gap_seconds=gap, request_hash=h,
                                         is_evaluation=is_eval, check_only=False,
                                         calibration=CAL)
                # A check_only request is qualification-only and cannot receive
                # supplemental demand admission under any policy arm.
                dc, changedc, _ = decide(method, parent_accept=parent, score=score,
                                         threshold=threshold, candidate_source=source,
                                         block_gap_seconds=gap, request_hash=h,
                                         is_evaluation=is_eval, check_only=True,
                                         calibration=CAL)
                if dc != parent or changedc:
                    raise AssertionError(('check_only decision changed', method, row['candidate_id']))
                if not is_eval and (d0 != parent or changed0):
                    raise AssertionError(('warmup decision changed', method, row['candidate_id']))
                if method == 'HIGH-CONTROL' and (d0 != parent or changed0):
                    raise AssertionError(('control rule differs from HIGH', row['candidate_id']))

    # Recompute one real author score and decision on the exact saved feature
    # vector while changing a synthetic future-outcome side table. The policy
    # API accepts no such file, so the two outputs must be identical.
    eval_reject = next((r for r in samples['evaluation']
                        if r['candidate_source'] == 'demand' and not r['author_decision_accept']), None)
    if eval_reject is None:
        raise AssertionError('evaluation prefix lacks a rejected demand candidate')
    vector = np.asarray(eval_reject['author_model_input_vector'], dtype=float).reshape(1, -1)
    prefix_feature_hash = hashlib.sha256(vector.tobytes()).hexdigest()
    test_sidecar = HERE / 'preflight_future_side_table.json'
    test_sidecar.write_text(json.dumps({'synthetic_future_outcome': 'hit'}, indent=2), encoding='utf-8')
    sidecar_hash_before = sha(test_sidecar)
    score_a = float(booster.predict(vector)[0])
    decision_a = score_a > float(eval_reject['threshold'])
    result_a = decide('HIGH-RECENCY', parent_accept=decision_a, score=score_a,
                      threshold=float(eval_reject['threshold']), candidate_source='demand',
                      block_gap_seconds=eval_reject['legal_key_history_at_decision_request_start'].get('seconds_since_block_last_get'),
                      request_hash=request_hash64(eval_reject['actual_decision_request_sequence']),
                      is_evaluation=True, check_only=False, calibration=CAL)
    test_sidecar.write_text(json.dumps({'synthetic_future_outcome': 'evicted'}, indent=2), encoding='utf-8')
    sidecar_hash_after = sha(test_sidecar)
    if sidecar_hash_before == sidecar_hash_after:
        raise AssertionError('future side-table variant did not change')
    score_b = float(booster.predict(vector)[0])
    decision_b = score_b > float(eval_reject['threshold'])
    result_b = decide('HIGH-RECENCY', parent_accept=decision_b, score=score_b,
                      threshold=float(eval_reject['threshold']), candidate_source='demand',
                      block_gap_seconds=eval_reject['legal_key_history_at_decision_request_start'].get('seconds_since_block_last_get'),
                      request_hash=request_hash64(eval_reject['actual_decision_request_sequence']),
                      is_evaluation=True, check_only=False, calibration=CAL)
    assert abs(score_a - float(eval_reject['author_model_score'])) <= 1e-12
    assert score_a == score_b and decision_a == decision_b and result_a == result_b
    assert hashlib.sha256(vector.tobytes()).hexdigest() == prefix_feature_hash

    result = {
        'passed': True,
        'full_replay_started': False,
        'checks': {
            'BC06_snapshot_manifest_and_candidate_hash': True,
            'four_configs_differ_from_STATIC_HIGH_only_by_output_directory': True,
            'closing_control_matches_saved_HIGH_decisions_on_small_prefix': True,
            'all_methods_leave_check_only_unchanged': True,
            'all_methods_leave_warmup_parent_decisions_unchanged': True,
            'online_policy_source_has_no_label_or_peak_interface': True,
            'online_replay_source_has_no_result_sidecar_reference': True,
            'saved_prefix_author_score_reproduced': True,
            'saved_prefix_rule_output_invariant_to_unread_future_outcome_change': True,
        },
        'small_prefix_rows': small_prefix_rows,
        'prefix_examples_per_phase': {k: len(v) for k, v in samples.items()},
        'one_evaluation_prefix_candidate': {
            'candidate_id': eval_reject['candidate_id'],
            'request_sequence': eval_reject['actual_decision_request_sequence'],
            'author_score_recomputed': score_a,
            'parent_decision': decision_a,
            'recency_rule_decision': result_a[0],
            'supplement_applied': result_a[1],
            'synthetic_future_side_table_sha256_before': sidecar_hash_before,
            'synthetic_future_side_table_sha256_after': sidecar_hash_after,
            'synthetic_side_table_changed': sidecar_hash_before != sidecar_hash_after,
            'saved_feature_vector_sha256_unchanged': hashlib.sha256(vector.tobytes()).hexdigest() == prefix_feature_hash,
        },
        'calibration_sha256': sha(HERE / 'calibration.json'),
        'snapshot_sha256': sha(SNAPSHOT),
        'policy_sha256': sha(HERE / 'policy.py'),
        'replay_sha256': sha(HERE / 'replay.py'),
        'notes': 'This is an interface/prefix check, not an end-to-end leakage proof; the full HIGH-CONTROL replay is the execution-level reference.'
    }
    (HERE / 'preflight.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil_target = OUT / 'BC07_preflight.json'
    shutil_target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
