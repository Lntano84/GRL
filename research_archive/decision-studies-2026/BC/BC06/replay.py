"""BC06: one read-only HIGH-CONTROL replay with legal candidate snapshots."""
import collections
import enum
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
WORK = HERE
ARM = WORK / 'HIGH-CONTROL'
REPO = BASE / 'work/bc01/Baleen-FAST24'
START = 1572074461.57806
EVAL_START = 86401.23277902603
THETA0 = 0.798545
THETAH = 2 * THETA0 / (1 + THETA0)
CAPACITY = 3002
EVAL_WRITE_CAP = 148731
SAMPLE_RATIO = .001  # author stats encode 0.1 percent as a fraction
MODEL_PATH = REPO / 'tmp/example/201910_Region1_0_0.1/ea_5892.86_wr_35.599_admit_threshold_binary.model'
REQUIRED_BATCH_FIELDS = [
    'time_elapsed_phy', 'service_time_used_stats',
    'service_time_used_demand_stats', 'service_time_used_prefetch_stats',
    'service_time_writes_stats', 'flashcache/keys_written_stats',
    'flashcache/evictions_stats',
]

sys.path.insert(0, str(REPO))
import compress_json
from BCacheSim.cachesim import admission_policies as aps, sim_cache, eviction_policies as evp
from BCacheSim.cachesim.simulate_ap import get_parsed_args
from BCacheSim.cachesim.utils import ods


def typed(v):
    if isinstance(v, enum.Enum):
        return {'enum': type(v).__name__, 'name': v.name, 'value': typed(v.value)}
    if isinstance(v, tuple):
        return {'tuple': [typed(x) for x in v]}
    if isinstance(v, list):
        return {'list': [typed(x) for x in v]}
    if isinstance(v, (str, bool, int)) or v is None:
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            return {'float': str(v)}
        return v
    if isinstance(v, np.generic):
        return typed(v.item())
    raise TypeError(f'unsupported audit key type: {type(v)}')


def token(v):
    return json.dumps(typed(v), ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Recorder:
    def __init__(self):
        self.start = time.perf_counter()
        self.seq = 0
        self.request = None
        self.candidate_id = 0
        self.batch_id = 0
        self.pending = {}
        self.current_rejections = []
        self.current_writes = 0
        self.eval_writes = 0
        self.all_candidate_counts = collections.Counter()
        self.decisions = collections.Counter()
        self.checkonly = collections.Counter()
        self.unique_creation_requests = set()
        self.unique_decision_requests = set()
        self.unique_blocks = set()
        self.unique_chunks = set()
        self.feature_vectors_by_length = collections.Counter()
        self.source_counts = collections.Counter()
        self.history = {
            'block_get_count': collections.Counter(),
            'chunk_get_count': collections.Counter(),
            'block_last_get_elapsed_s': {},
            'chunk_last_get_elapsed_s': {},
            'chunk_rejection_count': collections.Counter(),
            'chunk_last_rejection_elapsed_s': {},
        }
        self.progress_windows = 0
        self.prediction_capture = None
        self.feature_names = None
        self.predicted_items = collections.Counter()
        self.completed_windows = []
        self.window_write_count = 0
        self.window_eviction_count = 0
        self.requests_out = gzip.open(ARM / 'get_request_contexts.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1)
        self.candidates_out = gzip.open(ARM / 'candidate_decisions.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1)
        self.checkonly_out = gzip.open(ARM / 'check_only_calls.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1)
        self.reference = self._read_reference()
        self.ends = np.asarray(self.reference['time_elapsed_phy'], dtype=float)
        assert self.ends.shape == (1008,) and abs(self.ends[143] - EVAL_START) < 1e-8
        self.threshold_counts = collections.Counter()
        self.max_occupancy = 0
        self.completed_window_errors = []

    def _read_reference(self):
        ref = list((BASE / 'work/bc03/STATIC-HIGH/raw').rglob('*.stats.lzma'))
        if len(ref) != 1:
            raise AssertionError(f'expected one BC03 STATIC-HIGH reference, found {len(ref)}')
        data = compress_json.load(str(ref[0]))['batches']
        for name in REQUIRED_BATCH_FIELDS:
            if name not in data:
                raise KeyError(f'required window field missing from reference: {name}')
            if len(data[name]) != 1008:
                raise AssertionError(f'{name} expected 1008 windows, found {len(data[name])}')
        return data

    def _completed_window(self, batches, j):
        if j < 0:
            return None
        ends = batches['time_elapsed_phy']
        duration = float(ends[j] - (ends[j-1] if j else 0.0))
        if duration <= 0:
            raise AssertionError(('nonpositive completed window duration', j, duration))
        delta = {}
        for field in REQUIRED_BATCH_FIELDS[1:5]:
            values = batches[field]
            if len(values) <= j:
                raise KeyError(f'incomplete required window field {field} at {j}')
            delta[field] = float(values[j] - (values[j-1] if j else 0.0))
        comp = (delta['service_time_used_demand_stats'] + delta['service_time_used_prefetch_stats'])
        err = abs(delta['service_time_used_stats'] - comp)
        self.completed_window_errors.append(err)
        if err > 1e-8:
            raise AssertionError(('total GET DT components mismatch', j, err))
        factor = 100.0 / (36.0 * SAMPLE_RATIO * duration)
        row = {
            'window_index': int(j),
            'start_elapsed_s': float(ends[j-1] if j else 0.0),
            'end_elapsed_s': float(ends[j]),
            'duration_s': duration,
            'total_get_dt_pct': delta['service_time_used_stats'] * factor,
            'demand_read_dt_pct': delta['service_time_used_demand_stats'] * factor,
            'extra_prefetch_dt_pct': delta['service_time_used_prefetch_stats'] * factor,
            'put_dt_pct': delta['service_time_writes_stats'] * factor,
            # Event totals come from read-only hooks on every successful write
            # and eviction. Some ODS counters are absent until their first
            # event, so a missing live key is never treated as a measured zero.
            'writes': int(self.window_write_count),
            'evictions': int(self.window_eviction_count),
        }
        if row['writes'] < 0 or row['evictions'] < 0:
            raise AssertionError(('negative completed-window counter', row))
        return row

    def finish_window(self, ts):
        batches = ods.batches
        n = len(batches.get('time_elapsed_phy', []))
        if n == len(self.completed_windows):
            return
        if n != len(self.completed_windows) + 1:
            raise AssertionError(('checkpoint/window sequence skipped', n, len(self.completed_windows)))
        row = self._completed_window(batches, n - 1)
        row['window_end_checkpoint_physical_time_s'] = float(ts.physical)
        self.completed_windows.append(row)
        self.window_write_count = 0
        self.window_eviction_count = 0

    def request_context(self, sim, acc):
        relative = float(acc.ts.physical - START)
        nwin = len(ods.batches.get('time_elapsed_phy', []))
        if len(self.completed_windows) != nwin:
            raise AssertionError(('window feature history not synchronized', len(self.completed_windows), nwin))
        if not 0 <= nwin <= 1008:
            raise AssertionError(('bad window count', nwin))
        assigned = int(np.searchsorted(self.ends, relative, side='right'))
        # The author loop handles the final request before the final
        # _checkpoint. BC04 uses this same endpoint exception: at exactly the
        # last trace timestamp ODS still exposes window 1007, while a pure
        # right-sided search over the settled endpoints returns 1008.
        final_endpoint_pending = (nwin == 1007 and abs(relative - self.ends[-1]) <= 1e-8)
        if nwin != assigned and not final_endpoint_pending:
            raise AssertionError(('author checkpoint assignment changed', nwin, assigned, relative))
        completed = list(self.completed_windows[-6:])
        cache_len = len(sim.cache.cache)
        self.max_occupancy = max(self.max_occupancy, cache_len)
        if cache_len > CAPACITY:
            raise AssertionError(('cache capacity exceeded', cache_len))
        return {
            'request_sequence': self.seq,
            'physical_time_s': float(acc.ts.physical),
            'trace_elapsed_s': relative,
            'window_index_at_request_start': int(nwin),
            'operation': 'GET',
            'current_request_chunk_count': len(acc.chunks),
            'current_request_chunk_start': int(acc.chunk_range[0]),
            'current_request_chunk_end_exclusive': int(acc.chunk_range[1]),
            'cache_occupancy_chunks_at_request_start': cache_len,
            'cache_capacity_chunks': CAPACITY,
            'cache_occupancy_fraction_at_request_start': cache_len / CAPACITY,
            'evaluation_cumulative_writes_before_request': self.eval_writes if relative >= EVAL_START else None,
            'evaluation_write_cap_chunks': EVAL_WRITE_CAP if relative >= EVAL_START else None,
            'evaluation_write_budget_fraction_before_request': self.eval_writes / EVAL_WRITE_CAP if relative >= EVAL_START else None,
            'last_completed_window': completed[-1] if completed else None,
            'last_six_completed_windows': completed,
            'completed_window_count': nwin,
        }

    def key_features(self, key, context):
        block, chunk = key
        s = context['trace_elapsed_s']
        b_last = self.history['block_last_get_elapsed_s'].get(block)
        c_last = self.history['chunk_last_get_elapsed_s'].get(key)
        r_last = self.history['chunk_last_rejection_elapsed_s'].get(key)
        return {
            'prior_block_get_count': int(self.history['block_get_count'].get(block, 0)),
            'seconds_since_block_last_get': None if b_last is None else max(0.0, s - b_last),
            'prior_chunk_get_count': int(self.history['chunk_get_count'].get(key, 0)),
            'seconds_since_chunk_last_get': None if c_last is None else max(0.0, s - c_last),
            'prior_actual_rejection_count': int(self.history['chunk_rejection_count'].get(key, 0)),
            'seconds_since_latest_actual_rejection': None if r_last is None else max(0.0, s - r_last),
        }

    def begin_get(self, sim, acc):
        if self.request is not None:
            raise AssertionError('nested GET request')
        self.request = self.request_context(sim, acc)
        self.request['full_block_key'] = typed(acc.block_id)
        self.request['requested_chunks'] = [int(c) for c in acc.chunks]
        self.request['author_current_request_metadata'] = {
            'op': typed(acc.features.op),
            'namespace': typed(getattr(acc.features, 'namespace', None)),
            'user': typed(getattr(acc.features, 'user', None)),
            'pipeline': typed(getattr(acc.features, 'pipeline', None)),
            'offset_bytes': typed(getattr(acc.features, 'offset', None)),
            'request_end_offset_bytes': (typed(acc.features.offset + acc.features.size)
                                         if getattr(acc.features, 'offset', None) is not None and
                                         getattr(acc.features, 'size', None) is not None else None),
            'request_size_bytes': typed(getattr(acc.features, 'size', None)),
            'repeat_count': typed(getattr(acc.features, 'repeat', None)),
        }
        self.request['_new_rejections'] = []
        self.request['_writes'] = 0

    def record_candidate(self, queue, key, ts, feature_vector, metadata):
        if self.request is None:
            raise AssertionError('candidate generated outside GET request')
        self.candidate_id += 1
        feature_vector = [float(x) if isinstance(x, (float, np.floating)) else int(x) if isinstance(x, (int, np.integer)) else typed(x)
                          for x in list(feature_vector)]
        prefetch_present = metadata is not None and 'prefetch' in metadata
        if prefetch_present:
            source = 'prefetch' if bool(metadata['prefetch']) else 'demand'
            source_basis = 'explicit_prefetch_metadata'
        elif self.request is not None:
            # The only other QueueCache.insert call site in this frozen config
            # is sim_cache.run_get's demand-miss loop. Prefetcher inserts carry
            # explicit prefetch=True metadata; unknown outside that path stays
            # unknown rather than being inferred.
            source = 'demand'
            source_basis = 'sim_cache_run_get_miss_insert_callsite'
        else:
            source = 'unknown'
            source_basis = 'missing_source_context'
        self.source_counts[source] += 1
        self.unique_creation_requests.add(self.seq)
        self.unique_blocks.add(key[0])
        self.unique_chunks.add(key)
        self.all_candidate_counts['generated'] += 1
        row = {
            'candidate_id': self.candidate_id,
            'full_block_key': typed(key[0]),
            'chunk_id': int(key[1]),
            'chunk_key': typed(key),
            'candidate_created_request_sequence': self.seq,
            'candidate_created_physical_time_s': float(ts.physical),
            'candidate_created_trace_elapsed_s': float(ts.physical - START),
            'candidate_creation_window_index': self.request['window_index_at_request_start'],
            'candidate_source': source,
            'candidate_source_basis': source_basis,
            'candidate_source_metadata_field_present': prefetch_present,
            'candidate_metadata_fields_present': sorted(list((metadata or {}).keys())),
            'candidate_metadata_size': (metadata or {}).get('size'),
            'candidate_metadata_acc_chunk_range': typed((metadata or {}).get('acc_chunk_range')),
            'candidate_metadata_promotion': (metadata or {}).get('promotion'),
            'candidate_metadata_at_episode_start': (metadata or {}).get('at_ep_start'),
            'candidate_model_input_vector': feature_vector,
            'candidate_request_context_sequence': self.seq,
            'candidate_key_history_at_request_start': self.key_features(key, self.request),
            'candidate_current_request_chunk_count': self.request['current_request_chunk_count'],
        }
        self.pending[key] = row
        self.all_candidate_counts['currently_pending'] = len(self.pending)

    def capture_predict(self, self_ap, batch, ts, result):
        # Mirror LearnedAP._predict's exact preprocessing, then check dimensions.
        xs = list(batch.values())
        xs = [x[:-3] if len(x) == 12 else x for x in xs]
        matrix = np.asarray(xs)
        if matrix.ndim != 2 or matrix.shape[0] != len(batch):
            raise AssertionError(('model input shape', matrix.shape, len(batch)))
        scores = np.asarray(result, dtype=float).reshape(-1)
        if len(scores) != len(batch):
            raise AssertionError(('prediction count mismatch', len(scores), len(batch)))
        self.feature_vectors_by_length[int(matrix.shape[1])] += len(batch)
        names = list(self_ap.gbm.feature_name())
        if self.feature_names is None:
            self.feature_names = names
            if len(names) != matrix.shape[1]:
                raise AssertionError(('model feature names and vector length differ', len(names), matrix.shape[1]))
            (ARM / 'feature_schema.json').write_text(json.dumps({
                'model_feature_names_in_exact_order': names,
                'model_feature_count': len(names),
                'model_file': str(MODEL_PATH),
                'new_legal_features_are_separate_from_author_vector': True,
            }, ensure_ascii=False, indent=2), encoding='utf-8')
        self.prediction_capture = {
            'keys': list(batch.keys()),
            'matrix': matrix,
            'scores': scores,
            'threshold': float(self_ap.threshold),
            'physical_time_s': float(ts.physical),
            'trace_elapsed_s': float(ts.physical - START),
            'request_sequence': self.seq,
        }

    def record_batch(self, self_ap, batch, ts, metadata, check_only, decisions):
        capture = self.prediction_capture
        if capture is None or capture['request_sequence'] != self.seq:
            raise AssertionError('author prediction capture missing or outside current request')
        if capture['keys'] != list(batch.keys()):
            raise AssertionError('candidate order changed between model input and decisions')
        if self.request is None:
            raise AssertionError('admission call outside a GET request')
        threshold = float(self_ap.threshold)
        predictions = capture['scores']
        self.batch_id += 1
        if check_only:
            for idx, key in enumerate(capture['keys']):
                score = float(predictions[idx])
                decision = bool(decisions[key])
                expected = score > threshold
                if decision != expected:
                    raise AssertionError(('check-only decision mismatch', key, score, threshold, decision))
                row = {
                    'decision_kind': 'check_only', 'decision_batch_id': self.batch_id,
                    'request_sequence': self.seq, 'physical_time_s': float(ts.physical),
                    'full_block_key': typed(key[0]), 'chunk_id': int(key[1]), 'chunk_key': typed(key),
                    'author_model_input_vector': [float(x) for x in capture['matrix'][idx].tolist()],
                    'author_model_score': score, 'threshold': threshold,
                    'author_decision': decision, 'current_request_chunk_count': self.request['current_request_chunk_count'],
                    'legal_key_history_at_request_start': self.key_features(key, self.request),
                }
                self.checkonly_out.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False) + '\n')
            self.checkonly['calls'] += 1
            self.checkonly['items'] += len(batch)
            self.checkonly['accept'] += sum(bool(v) for v in decisions.values())
            self.checkonly['reject'] += sum(not bool(v) for v in decisions.values())
            self.predicted_items['check_only'] += len(batch)
            return

        self.decisions['batches'] += 1
        self.decisions['items'] += len(batch)
        self.unique_decision_requests.add(self.seq)
        for idx, key in enumerate(capture['keys']):
            if key not in self.pending:
                raise AssertionError(('actual decision has no logged QueueCache candidate', key, self.seq))
            candidate = self.pending.pop(key)
            score = float(predictions[idx])
            captured_vector = [float(x) for x in capture['matrix'][idx].tolist()]
            if len(candidate['candidate_model_input_vector']) != len(captured_vector) or not np.array_equal(
                    np.asarray(candidate['candidate_model_input_vector'], dtype=float),
                    np.asarray(captured_vector, dtype=float)):
                raise AssertionError(('candidate creation vector differs from actual author model input', key, self.seq))
            decision = bool(decisions[key])
            expected = score > threshold
            if decision != expected:
                raise AssertionError(('actual decision mismatch', key, score, threshold, decision))
            candidate.update({
                'decision_batch_id': self.batch_id,
                'decision_batch_position': idx,
                'decision_batch_size': len(batch),
                'actual_decision_request_sequence': self.seq,
                'actual_decision_physical_time_s': float(ts.physical),
                'actual_decision_trace_elapsed_s': float(ts.physical - START),
                'actual_decision_window_index': self.request['window_index_at_request_start'],
                'candidate_to_decision_delay_s': float(ts.physical - candidate['candidate_created_physical_time_s']),
                'author_model_input_vector': captured_vector,
                'author_model_score': score,
                'threshold': threshold,
                'author_decision_accept': decision,
                'author_decision_rule': 'score > threshold',
                'legal_key_history_at_decision_request_start': self.key_features(key, self.request),
                'decision_request_context_sequence': self.seq,
                'decision_current_request_chunk_count': self.request['current_request_chunk_count'],
            })
            self.candidates_out.write(json.dumps(candidate, separators=(',', ':'), ensure_ascii=False) + '\n')
            self.decisions['accepted' if decision else 'rejected'] += 1
            self.decisions['source/' + candidate['candidate_source']] += 1
            self.decisions['created_and_decided_same_request'] += int(candidate['candidate_created_request_sequence'] == self.seq)
            self.decisions['delay_positive'] += int(ts.physical > candidate['candidate_created_physical_time_s'])
            if not decision:
                self.request['_new_rejections'].append(key)
        self.all_candidate_counts['decided'] += len(batch)
        self.predicted_items['actual'] += len(batch)
        self.all_candidate_counts['currently_pending'] = len(self.pending)

    def finish_get(self, acc):
        if self.request is None:
            raise AssertionError('GET finish without begin')
        now = float(acc.ts.physical - START)
        block = acc.block_id
        self.history['block_get_count'][block] += 1
        self.history['block_last_get_elapsed_s'][block] = now
        for chunk in acc.chunks:
            key = (block, int(chunk))
            self.history['chunk_get_count'][key] += 1
            self.history['chunk_last_get_elapsed_s'][key] = now
        # Commit actual model rejections only after every decision in this GET,
        # keeping them out of all snapshots from the current request.
        for key in self.request['_new_rejections']:
            self.history['chunk_rejection_count'][key] += 1
            self.history['chunk_last_rejection_elapsed_s'][key] = now
        if now >= EVAL_START:
            self.eval_writes += int(self.request['_writes'])
        row = {k: v for k, v in self.request.items() if not k.startswith('_')}
        self.requests_out.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False) + '\n')
        self.request = None

    def write_success(self):
        if self.request is not None:
            self.request['_writes'] += 1
        self.window_write_count += 1

    def progress(self, ts):
        windows = len(ods.batches.get('time_elapsed_phy', []))
        self.progress_windows = windows
        (ARM / 'progress.json').write_text(json.dumps({
            'arm': 'HIGH-CONTROL', 'completed_windows': windows,
            'trace_elapsed_s': float(ts.physical - START),
            'request_sequence': self.seq, 'candidate_items_decided': self.decisions['items'],
            'wall_seconds': time.perf_counter() - self.start,
        }), encoding='utf-8')

    def close(self):
        # Surface any final candidates that the author never sent to a score call.
        for row in self.pending.values():
            row.update({
                'actual_decision_request_sequence': None,
                'actual_decision_physical_time_s': None,
                'author_model_input_vector': None,
                'author_model_score': None,
                'threshold': None,
                'author_decision_accept': None,
                'candidate_status': 'pending_at_trace_end',
            })
            self.candidates_out.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False) + '\n')
        self.candidates_out.close()
        self.requests_out.close()
        self.checkonly_out.close()

    def audit(self):
        return {
            'completed': False,
            'arm': 'HIGH-CONTROL',
            'elapsed_wall_seconds': time.perf_counter() - self.start,
            'request_sequence': self.seq,
            'get_request_count': self.decisions.get('get_requests', None),
            'candidate_counts': dict(self.all_candidate_counts),
            'actual_decision_counts': dict(self.decisions),
            'check_only_counts': dict(self.checkonly),
            'candidate_source_counts': dict(self.source_counts),
            'unique_candidate_creation_requests': len(self.unique_creation_requests),
            'unique_actual_decision_requests': len(self.unique_decision_requests),
            'unique_candidate_blocks': len(self.unique_blocks),
            'unique_candidate_chunks': len(self.unique_chunks),
            'model_input_vector_lengths': dict(self.feature_vectors_by_length),
            'predicted_items_by_call_kind': dict(self.predicted_items),
            'model_feature_names': self.feature_names,
            'max_cache_occupancy_chunks': self.max_occupancy,
            'evaluation_writes_recorded_independently': self.eval_writes,
            'completed_windows_recorded': len(self.completed_windows),
            'completed_window_component_max_abs_error': max(self.completed_window_errors, default=0.0),
            'wall_seconds': time.perf_counter() - self.start,
        }


def main():
    os.chdir(REPO)
    cfg = json.loads((ARM / 'config.json').read_text(encoding='utf-8'))
    assert cfg['size_gb'] == 366.475 and cfg['eviction_policy'] == 'LRU' and cfg['write_mbps'] == 0
    assert cfg['batch_size'] == 16 and cfg['ap_feat_subset'] == 'meta+block+chunk'
    recorder = Recorder()
    originals = {
        'stats': sim_cache.CacheSimulator._stats,
        'run_get': sim_cache.CacheSimulator.run_get,
        'run_put': sim_cache.CacheSimulator.run_put,
        'insert': evp.QueueCache.insert,
        'admit': evp.QueueCache.admit,
        'eviction': evp.EvictionPolicy.log_eviction,
        'checkpoint': sim_cache.CacheSimulator._checkpoint,
        'accept': aps.NewMLAP.batchAccept,
        'predict': aps.LearnedAP._predict,
    }

    def stats(self, ts):
        originals['stats'](self, ts)
        recorder.seq += 1
        assert self.start_ts.physical == START

    def run_get(self, acc):
        recorder.begin_get(self, acc)
        result = originals['run_get'](self, acc)
        recorder.finish_get(acc)
        recorder.decisions['get_requests'] += 1
        return result

    def run_put(self, acc):
        # PUT creates no admission history and does not update GET counts.
        return originals['run_put'](self, acc)

    def insert(self, key, ts, keyfeaturelist, *, metadata=None):
        recorder.record_candidate(self, key, ts, keyfeaturelist, metadata)
        return originals['insert'](self, key, ts, keyfeaturelist, metadata=metadata)

    def admit(self, key, ts, **kwargs):
        before = self.keys_written
        result = originals['admit'](self, key, ts, **kwargs)
        if self.keys_written - before != 1:
            raise AssertionError(('actual admission write counter mismatch', key, before, self.keys_written))
        if recorder.request is None:
            raise AssertionError('cache write occurred outside GET; current action scope changed')
        recorder.write_success()
        return result

    def eviction(self, ts, evicted):
        before = self.evictions
        result = originals['eviction'](self, ts, evicted)
        if self.evictions - before != 1:
            raise AssertionError(('actual eviction counter mismatch', before, self.evictions))
        recorder.window_eviction_count += 1
        return result

    def predict(self_ap, batch, ts):
        result = originals['predict'](self_ap, batch, ts)
        recorder.capture_predict(self_ap, batch, ts, result)
        return result

    def accept(self, batch, ts, *, metadata=None, check_only=False):
        # Record only after the untouched author implementation has made its
        # decision and updated its ordinary counters.
        self.threshold = THETA0 if ts.physical - START < EVAL_START else THETAH
        result = originals['accept'](self, batch, ts, metadata=metadata, check_only=check_only)
        recorder.record_batch(self, batch, ts, metadata, check_only, result)
        return result

    def checkpoint(self, ts, *args, **kwargs):
        result = originals['checkpoint'](self, ts, *args, **kwargs)
        recorder.finish_window(ts)
        recorder.progress(ts)
        return result

    sim_cache.CacheSimulator._stats = stats
    sim_cache.CacheSimulator.run_get = run_get
    sim_cache.CacheSimulator.run_put = run_put
    evp.QueueCache.insert = insert
    evp.QueueCache.admit = admit
    evp.EvictionPolicy.log_eviction = eviction
    aps.LearnedAP._predict = predict
    aps.NewMLAP.batchAccept = accept
    sim_cache.CacheSimulator._checkpoint = checkpoint
    sys.argv = ['bc06/replay.py', '--config', str(ARM / 'config.json')]
    args = get_parsed_args()
    assert args.peak_strategy is None and args.write_mbps == 0 and args.ap == 'mlnew'
    if (ARM / 'run_audit.json').exists():
        raise RuntimeError('refusing a duplicate BC06 replay')
    audit = {'completed': False, 'mode': 'read_only_observation', 'started_unix': time.time()}
    try:
        sim_cache.simulate_cache_driver(args)
        if recorder.request is not None:
            raise AssertionError('unfinished GET context after simulation')
        if recorder.seq != 147794 or recorder.decisions['get_requests'] != 127305:
            raise AssertionError(('trace request counts changed', recorder.seq, recorder.decisions['get_requests']))
        if recorder.decisions['accepted'] != ods.get('ap_NewMLAP_accepts'):
            raise AssertionError(('accepted candidate counter mismatch', recorder.decisions['accepted'], ods.get('ap_NewMLAP_accepts')))
        if recorder.decisions['rejected'] != ods.get('ap_NewMLAP_rejects'):
            raise AssertionError(('rejected candidate counter mismatch', recorder.decisions['rejected'], ods.get('ap_NewMLAP_rejects')))
        if recorder.decisions['items'] != recorder.decisions['accepted'] + recorder.decisions['rejected']:
            raise AssertionError('actual decision item total mismatch')
        if recorder.decisions['items'] + recorder.checkonly['items'] != ods.get('ml_predictions'):
            raise AssertionError(('prediction item count mismatch', recorder.decisions['items'],
                                  recorder.checkonly['items'], ods.get('ml_predictions')))
        if recorder.eval_writes != 119276:
            raise AssertionError(('HIGH-CONTROL evaluation write count changed', recorder.eval_writes))
        if recorder.all_candidate_counts['generated'] != recorder.decisions['items'] + len(recorder.pending):
            raise AssertionError(('candidate event accounting mismatch', recorder.all_candidate_counts['generated'],
                                  recorder.decisions['items'], len(recorder.pending)))
        stats = list((ARM / 'raw').rglob('*.stats.lzma'))
        if len(stats) != 1:
            raise AssertionError(('expected one raw stats file', stats))
        batches = compress_json.load(str(stats[0]))['batches']
        for field in REQUIRED_BATCH_FIELDS:
            if field not in batches or field not in recorder.reference:
                raise KeyError(f'missing required comparison field {field}')
            a = np.asarray(batches[field], dtype=float)
            b = np.asarray(recorder.reference[field], dtype=float)
            if a.shape != b.shape or not np.isfinite(a).all():
                raise AssertionError(('bad stats field shape/values', field, a.shape, b.shape))
            error = float(np.max(np.abs(a - b)))
            tolerance = 1e-9 if field.startswith('service_time') else 0.0
            if error > tolerance:
                raise AssertionError(('HIGH-CONTROL reproduction failure', field, error, tolerance))
            audit.setdefault('control_reproduction_max_errors', {})[field] = error
        if len(recorder.completed_windows) != 1008:
            raise AssertionError(('completed-window snapshot count mismatch', len(recorder.completed_windows)))
        for field, column in [('flashcache/keys_written_stats', 'writes'),
                              ('flashcache/evictions_stats', 'evictions')]:
            values = np.asarray(batches[field], dtype=float)
            observed = np.asarray([w[column] for w in recorder.completed_windows], dtype=float)
            expected = np.diff(values, prepend=0.0)
            if observed.shape != expected.shape:
                raise AssertionError(('window event count shape mismatch', field, observed.shape, expected.shape))
            error = float(np.max(np.abs(observed - expected)))
            if error != 0.0:
                raise AssertionError(('window event counter reconciliation failed', field, error))
            audit.setdefault('window_event_reconciliation', {})[field] = error
        if int(round(batches['flashcache/keys_written_stats'][-1])) != 141064:
            raise AssertionError('unexpected full-trace writes; likely wrong arm')
        recorder.close()
        audit.update({
            'completed': True,
            'trace_request_sequence': recorder.seq,
            'candidate_file': str(ARM / 'candidate_decisions.jsonl.gz'),
            'candidate_file_sha256': sha(ARM / 'candidate_decisions.jsonl.gz'),
            'candidate_file_size_bytes': (ARM / 'candidate_decisions.jsonl.gz').stat().st_size,
            'request_context_file_sha256': sha(ARM / 'get_request_contexts.jsonl.gz'),
            'check_only_file_sha256': sha(ARM / 'check_only_calls.jsonl.gz'),
            'stats_path': str(stats[0]),
            'stats_sha256': sha(stats[0]),
            'recorder': recorder.audit(),
            'reference_stats_path': str(list((BASE / 'work/bc03/STATIC-HIGH/raw').rglob('*.stats.lzma'))[0]),
            'run_completed_unix': time.time(),
        })
    finally:
        for name, original in originals.items():
            if name == 'stats': sim_cache.CacheSimulator._stats = original
            elif name == 'run_get': sim_cache.CacheSimulator.run_get = original
            elif name == 'run_put': sim_cache.CacheSimulator.run_put = original
            elif name == 'insert': evp.QueueCache.insert = original
            elif name == 'admit': evp.QueueCache.admit = original
            elif name == 'eviction': evp.EvictionPolicy.log_eviction = original
            elif name == 'checkpoint': sim_cache.CacheSimulator._checkpoint = original
            elif name == 'accept': aps.NewMLAP.batchAccept = original
            elif name == 'predict': aps.LearnedAP._predict = original
        if not recorder.candidates_out.closed:
            recorder.close()
        audit['recorder'] = recorder.audit()
        audit['completed'] = bool(audit.get('completed'))
        (ARM / 'run_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
