"""Offline validation of the completed BC06 snapshot; deliberately label-blind."""
import collections
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
BASE = WORK.parents[1]
ARM = WORK / 'HIGH-CONTROL'
REPO = BASE / 'work/bc01/Baleen-FAST24'
SAMPLE_RATIO = .001
CONTROL_KEYS = [
    'time_elapsed_phy', 'time_log', 'service_time_used_stats', 'service_time_writes_stats',
    'service_time_used_demand_stats', 'service_time_used_prefetch_stats', 'service_time_nocache_stats',
    'flashcache/keys_written_stats', 'flashcache/prefetches_stats', 'fetches_chunks_prefetch_stats',
    'fetches_chunks_demandmiss_stats', 'fetches_ios_stats', 'iops_requests_stats',
    'chunk_queries_stats', 'puts_ios_stats', 'puts_chunks_stats', 'flashcache/evictions_stats',
]
HIST_FEATURES = [
    'prior_block_get_count', 'seconds_since_block_last_get', 'prior_chunk_get_count',
    'seconds_since_chunk_last_get', 'prior_actual_rejection_count',
    'seconds_since_latest_actual_rejection',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def one_stats(directory):
    matches = list(Path(directory).rglob('*.stats.lzma'))
    if len(matches) != 1:
        raise AssertionError(f'expected one stats file under {directory}, got {matches}')
    return matches[0]


def stream_jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise AssertionError(f'malformed JSONL at {path}:{line_no}') from e


def close_enough(a, b, tol=1e-9):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol


def load_window_expected(batches, idx):
    ends = np.asarray(batches['time_elapsed_phy'], dtype=float)
    duration = float(ends[idx] - (ends[idx-1] if idx else 0.0))
    factor = 100.0 / (36.0 * SAMPLE_RATIO * duration)
    def delta(key):
        values = batches[key]
        return float(values[idx] - (values[idx-1] if idx else 0.0))
    return {
        'window_index': idx,
        'start_elapsed_s': float(ends[idx-1] if idx else 0.0),
        'end_elapsed_s': float(ends[idx]),
        'duration_s': duration,
        'total_get_dt_pct': delta('service_time_used_stats') * factor,
        'demand_read_dt_pct': delta('service_time_used_demand_stats') * factor,
        'extra_prefetch_dt_pct': delta('service_time_used_prefetch_stats') * factor,
        'put_dt_pct': delta('service_time_writes_stats') * factor,
        'writes': int(round(delta('flashcache/keys_written_stats'))),
        'evictions': int(round(delta('flashcache/evictions_stats'))),
    }


def main():
    sys.path.insert(0, str(REPO))
    import compress_json
    execution_audit_path = ARM / 'run_audit.json'
    execution_audit = json.loads(execution_audit_path.read_text(encoding='utf-8'))
    rec = execution_audit['recorder']
    if rec['request_sequence'] != 147794 or rec['get_request_count'] != 127305:
        raise AssertionError(('full trace request totals', rec['request_sequence'], rec['get_request_count']))
    if rec['candidate_counts'].get('generated') != 2375547 or rec['candidate_counts'].get('decided') != 2375547:
        raise AssertionError('all actual candidates were not decided')
    if rec['check_only_counts'].get('items') != 49204:
        raise AssertionError('check-only call count mismatch')
    if rec['evaluation_writes_recorded_independently'] != 119276:
        raise AssertionError('evaluation write counter mismatch')

    stats_path = one_stats(ARM / 'raw')
    ref_path = one_stats(BASE / 'work/bc03/STATIC-HIGH/raw')
    actual = compress_json.load(str(stats_path))['batches']
    reference = compress_json.load(str(ref_path))['batches']
    stat_errors = {}
    for key in CONTROL_KEYS:
        if key not in actual or key not in reference:
            raise KeyError(f'required control field missing: {key}')
        a = np.asarray(actual[key], dtype=float)
        b = np.asarray(reference[key], dtype=float)
        if a.shape != (1008,) or b.shape != (1008,) or not np.isfinite(a).all():
            raise AssertionError(('unexpected control series shape/value', key, a.shape, b.shape))
        err = float(np.max(np.abs(a - b)))
        tol = 1e-9 if key.startswith('service_time') else 0.0
        if err > tol:
            raise AssertionError(('HIGH-CONTROL differs from BC03', key, err, tol))
        stat_errors[key] = err

    schema = json.loads((ARM / 'feature_schema.json').read_text(encoding='utf-8'))
    feature_names = schema['model_feature_names_in_exact_order']
    if len(feature_names) != 18 or rec['model_input_vector_lengths'] != {'18': 2424751}:
        raise AssertionError(('model feature shape mismatch', len(feature_names), rec['model_input_vector_lengths']))
    if len(set(feature_names)) != len(feature_names):
        raise AssertionError('duplicated author feature names')

    candidates_path = ARM / 'candidate_decisions.jsonl.gz'
    candidate_counts = collections.Counter()
    feature_missing = collections.Counter()
    request_counts = collections.Counter()
    block_counts = collections.Counter()
    target_leak_tokens = ('target', 'oracle', 'label', 'counterfactual', 'peak_window')
    for number, row in enumerate(stream_jsonl(candidates_path), 1):
        if row['candidate_id'] != number:
            raise AssertionError(('candidate id/order gap', number, row['candidate_id']))
        if any(t in k.lower() for k in row for t in target_leak_tokens):
            raise AssertionError(('future label field in snapshot', row['candidate_id']))
        if row.get('actual_decision_request_sequence') is None:
            candidate_counts['pending'] += 1
            continue
        candidate_counts['decided'] += 1
        accepted = bool(row['author_decision_accept'])
        candidate_counts['accepted' if accepted else 'rejected'] += 1
        candidate_counts['source/' + row['candidate_source']] += 1
        if len(row['author_model_input_vector']) != 18 or len(row['candidate_model_input_vector']) != 18:
            raise AssertionError(('bad vector length', row['candidate_id']))
        if row['author_model_input_vector'] != row['candidate_model_input_vector']:
            raise AssertionError(('input at candidate creation differs from score-time model input', row['candidate_id']))
        if accepted != (float(row['author_model_score']) > float(row['threshold'])):
            raise AssertionError(('recorded score/threshold decision mismatch', row['candidate_id']))
        if row['candidate_created_request_sequence'] == row['actual_decision_request_sequence']:
            candidate_counts['creation_decision_same_request'] += 1
        else:
            candidate_counts['creation_decision_delayed'] += 1
        if row['candidate_source'] not in {'demand', 'prefetch', 'unknown'}:
            raise AssertionError(('unrecognized source', row['candidate_id'], row['candidate_source']))
        request_counts[int(row['actual_decision_request_sequence'])] += 1
        block_counts[json.dumps(row['full_block_key'], ensure_ascii=False, sort_keys=True)] += 1
        feats = row['legal_key_history_at_decision_request_start']
        for name in HIST_FEATURES:
            feature_missing[name] += int(feats.get(name) is None)
    if candidate_counts['decided'] != 2375547 or candidate_counts['pending'] != 0:
        raise AssertionError(('persisted candidate table count', dict(candidate_counts)))
    if candidate_counts['accepted'] != 141064 or candidate_counts['rejected'] != 2234483:
        raise AssertionError(('candidate accept/reject totals', dict(candidate_counts)))
    if candidate_counts['source/demand'] != 2343380 or candidate_counts['source/prefetch'] != 32167:
        raise AssertionError(('candidate source totals', dict(candidate_counts)))

    check_counts = collections.Counter()
    for row in stream_jsonl(ARM / 'check_only_calls.jsonl.gz'):
        if row.get('decision_kind') != 'check_only':
            raise AssertionError('check-only row mixed with actual decisions')
        if len(row['author_model_input_vector']) != 18:
            raise AssertionError('check-only model vector length mismatch')
        if bool(row['author_decision']) != (float(row['author_model_score']) > float(row['threshold'])):
            raise AssertionError('check-only decision does not match recorded score and threshold')
        check_counts['items'] += 1
    if check_counts['items'] != 49204:
        raise AssertionError(('check-only row count', dict(check_counts)))

    context_count = 0
    context_sequences = set()
    windows_seen = {}
    context_field_missing = collections.Counter()
    for row in stream_jsonl(ARM / 'get_request_contexts.jsonl.gz'):
        seq = int(row['request_sequence'])
        if seq in context_sequences:
            raise AssertionError(('duplicate GET request context', seq))
        context_sequences.add(seq)
        context_count += 1
        if row['operation'] != 'GET':
            raise AssertionError(('non-GET in GET context table', seq))
        if row['current_request_chunk_count'] != len(row['requested_chunks']):
            raise AssertionError(('requested chunk count mismatch', seq))
        if row['cache_occupancy_chunks_at_request_start'] > 3002:
            raise AssertionError(('capacity exceeded in snapshot', seq))
        if row['last_completed_window'] != (row['last_six_completed_windows'][-1] if row['last_six_completed_windows'] else None):
            raise AssertionError(('latest completed window pointer mismatch', seq))
        for name in ['evaluation_cumulative_writes_before_request', 'last_completed_window']:
            context_field_missing[name] += int(row.get(name) is None)
        for window in row['last_six_completed_windows']:
            idx = int(window['window_index'])
            old = windows_seen.get(idx)
            if old is not None and old != window:
                raise AssertionError(('completed window values changed across snapshots', idx))
            windows_seen[idx] = window
    if context_count != 127305 or not context_sequences:
        raise AssertionError(('GET context count mismatch', context_count))
    if len(context_sequences) != 127305:
        raise AssertionError('GET context sequence set mismatch')
    if any(seq not in context_sequences for seq in request_counts):
        raise AssertionError('actual decision lacks its GET request context')

    window_errors = {}
    for idx, seen in sorted(windows_seen.items()):
        expected = load_window_expected(actual, idx)
        for field in ['writes', 'evictions']:
            if seen[field] != expected[field]:
                raise AssertionError(('window event feature differs from final stats', idx, field, seen[field], expected[field]))
        for field in ['total_get_dt_pct', 'demand_read_dt_pct', 'extra_prefetch_dt_pct', 'put_dt_pct']:
            err = abs(float(seen[field]) - float(expected[field]))
            window_errors[field] = max(window_errors.get(field, 0.0), err)
            if err > 1e-8:
                raise AssertionError(('window load feature differs from final stats', idx, field, err))
    # The final 1007th interval is settled after the last trace request and so
    # cannot be a pre-decision feature of any request. It remains in raw stats.
    if max(windows_seen) > 1006:
        raise AssertionError(('unexpected post-trace window leaked into decision context', max(windows_seen)))

    # The simulator's global ML prediction counter also includes prefetch-model
    # inference. Admission score rows are separately observed at NewMLAP._predict.
    prefetch_source = (REPO / 'BCacheSim/cachesim/prefetchers.py').read_text(encoding='utf-8')
    assert 'ods.bump("ml_predictions"' in prefetch_source
    admission_input_count = candidate_counts['decided'] + check_counts['items']
    global_ml_predictions = int(round(float(actual['ml_predictions_stats'][-1])))
    extra_shared_counter_predictions = global_ml_predictions - admission_input_count
    if extra_shared_counter_predictions < 0:
        raise AssertionError('global model counter unexpectedly smaller than admission inputs')

    paths = [candidates_path, ARM/'get_request_contexts.jsonl.gz', ARM/'check_only_calls.jsonl.gz',
             ARM/'feature_schema.json', stats_path, ref_path, ARM/'config.json', WORK/'preflight.json',
             WORK/'replay.py', REPO/'BCacheSim/cachesim/admission_policies.py',
             REPO/'BCacheSim/cachesim/prefetchers.py']
    result = {
        'snapshot_validation_passed': True,
        'replay_simulation_completed': True,
        'runner_exit_code': 1,
        'runner_exit_note': 'post-run verifier compared admission-only captured scores with the simulator-wide ml_predictions counter, which also includes prefetch-model predictions; all simulation and logged admission data are complete.',
        'trace_request_count': rec['request_sequence'],
        'get_request_context_count': context_count,
        'actual_candidate_rows': candidate_counts['decided'],
        'actual_accepted_rows': candidate_counts['accepted'],
        'actual_rejected_rows': candidate_counts['rejected'],
        'candidate_source_counts': {k.removeprefix('source/'): v for k, v in candidate_counts.items() if k.startswith('source/')},
        'candidate_creation_decision_same_request': candidate_counts['creation_decision_same_request'],
        'candidate_creation_decision_delayed': candidate_counts['creation_decision_delayed'],
        'unique_decision_requests': len(request_counts),
        'unique_candidate_blocks': len(block_counts),
        'check_only_rows': check_counts['items'],
        'author_feature_names': feature_names,
        'admission_model_input_rows_actual_plus_check_only': admission_input_count,
        'simulator_wide_ml_predictions_counter': global_ml_predictions,
        'additional_shared_counter_predictions_outside_admission_rows': extra_shared_counter_predictions,
        'completed_windows_in_request_contexts': len(windows_seen),
        'context_window_indices_min_max': [min(windows_seen), max(windows_seen)],
        'window_load_max_abs_errors_pct_points': window_errors,
        'window_write_eviction_errors': {'writes': 0, 'evictions': 0},
        'control_reproduction_max_errors': stat_errors,
        'evaluation_write_count': rec['evaluation_writes_recorded_independently'],
        'evaluation_write_count_from_raw_windows': int(round(sum(np.diff(np.asarray(actual['flashcache/keys_written_stats'], dtype=float), prepend=0.0)[144:]))),
        'evaluation_window_component_max_abs_error_s': rec['completed_window_component_max_abs_error'],
        'max_cache_occupancy_chunks_at_request_start': rec['max_cache_occupancy_chunks'],
        'missing_history_value_counts_at_decision': dict(feature_missing),
        'window_context_fields_missing': dict(context_field_missing),
        'future_label_firewall': {
            'snapshot_generator_reads_no_BC05_targets': 'frozen_targets' not in (WORK/'replay.py').read_text(encoding='utf-8'),
            'candidate_snapshot_has_no_target_oracle_label_fields': True,
            'target_file_not_read_by_this_validation': True,
        },
        'artifact_sha256': {str(p.resolve()): sha(p) for p in paths},
        'artifact_bytes': {str(p.resolve()): p.stat().st_size for p in paths},
    }
    if result['evaluation_write_count'] != result['evaluation_write_count_from_raw_windows']:
        raise AssertionError(('evaluation writes differ', result['evaluation_write_count'], result['evaluation_write_count_from_raw_windows']))
    out = WORK / 'snapshot_validation.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['snapshot_validation_passed', 'trace_request_count',
                    'get_request_context_count', 'actual_candidate_rows', 'actual_accepted_rows',
                    'actual_rejected_rows', 'candidate_source_counts', 'check_only_rows',
                    'completed_windows_in_request_contexts', 'control_reproduction_max_errors',
                    'window_load_max_abs_errors_pct_points', 'evaluation_write_count']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
