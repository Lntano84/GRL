"""One BC05 parent replay, optionally applying the frozen target override."""
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
WORK = HERE
REPO = BASE / 'work/bc01/Baleen-FAST24'
START = 1572074461.57806
EVAL_START = 86401.23277902603
THETA0 = 0.798545
THETAH = 2 * THETA0 / (1 + THETA0)
CAPACITY = 3002

sys.path.insert(0, str(REPO))
import compress_json
from BCacheSim.cachesim import admission_policies as aps, sim_cache, eviction_policies as evp
from BCacheSim.cachesim.simulate_ap import get_parsed_args
from BCacheSim.cachesim.utils import ods
from BCacheSim.episodic_analysis.episodes import service_time


def enc(v):
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return {'float_hex': v.hex()}
    if isinstance(v, np.generic):
        return enc(v.item())
    if isinstance(v, np.ndarray):
        return {'ndarray': str(v.dtype), 'shape': v.shape, 'values': enc(v.tolist())}
    if isinstance(v, dict):
        pairs = [[enc(k), enc(x)] for k, x in v.items()]
        if not isinstance(v, collections.OrderedDict):
            pairs.sort(key=lambda p: json.dumps(p[0], sort_keys=True))
        return {'mapping_type': type(v).__name__, 'items': pairs}
    if isinstance(v, (set, frozenset)):
        items = [enc(x) for x in v]
        return {'set': sorted(items, key=lambda x: json.dumps(x, sort_keys=True))}
    if isinstance(v, (list, tuple, collections.deque)):
        return {'sequence_type': type(v).__name__, 'items': [enc(x) for x in v]}
    if hasattr(v, '__dict__'):
        return {'class': type(v).__module__ + '.' + type(v).__name__, 'state': enc(vars(v))}
    raise TypeError(f'Uncaptured snapshot type: {type(v)}')


def sha_obj(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def state_snapshot(sim):
    cache = sim.cache
    excluded = {'ap', 'prefetcher', 'evictions_log', 'on_evict'}
    assert cache.cache_size == CAPACITY and sim.ram_cache is None
    assert cache.evictions_log is None and cache.on_evict is None
    assert cache.offline_feat_df is None and cache.episodes is None
    state = {k: v for k, v in vars(cache).items() if k not in excluded}
    state['admit_buffer_order'] = list(cache.admit_buffer)
    ap_state = {k: v for k, v in vars(cache.ap).items() if k != 'gbm'}
    pf_state = {k: v for k, v in vars(sim.prefetcher).items()
                if k not in {'cache', 'insert_cache', 'ram_cache', 'ap', 'model'}}
    pf_model = sim.prefetcher.model
    pf_state['model_non_booster_state'] = {k: v for k, v in vars(pf_model).items() if k != 'models'}
    return {
        'cache_and_history': enc(state), 'ap_state': enc(ap_state), 'prefetch_state': enc(pf_state),
        'simulator_time_state': enc({k: getattr(sim, k) for k in ['start_ts', 'last_log_tracetime', 'last_util_peak']}),
        'cumulative_counters': enc(ods.counters), 'frequency_counters': enc(ods.freq),
        'checkpoint_index': ods.idx,
    }


def token_key(key):
    return json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


class Recorder:
    def __init__(self, arm):
        self.arm = arm
        self.cf = arm.endswith('-CF')
        self.parent = 'BASE' if arm.startswith('BASE') else 'HIGH'
        self.targets = json.loads((WORK / 'frozen_targets.json').read_text(encoding='utf-8'))
        self.by_seq = collections.defaultdict(list)
        self.by_key = {}
        self.by_peak_seq = collections.defaultdict(list)
        for t in self.targets:
            t['native_key'] = (t['full_block_key'], t['chunk'])
            t['key_token'] = token_key(list(t['native_key']))
            t['state'] = {
                'event_id': t['event_id'], 'reject_request_sequence': t['reject_request_sequence'],
                'peak_request_sequence': t['peak_request_sequence'], 'key': t['key_token'],
                'pending_before_request': None, 'cached_before_request': None,
                'candidate_inserted_at_request': False, 'decision_seen': False,
                'check_only_seen': False, 'parent_decision': None, 'intervention_changed': False,
                'write_success': False, 'write_request_sequence': None,
                'write_was_forced': False,
                'evicted_after_write': False, 'eviction_request_sequence': None,
                'evicted_before_peak_request': False,
                'peak_chunk_missing': None, 'peak_request_window': 578,
                'status': None,
            }
            self.by_seq[t['reject_request_sequence']].append(t)
            self.by_peak_seq[t['peak_request_sequence']].append(t)
            key_id = (t['reject_request_sequence'], t['key_token'])
            assert key_id not in self.by_key
            self.by_key[key_id] = t
        self.current_seq = 0
        self.current_ts = None
        self.current_acc = None
        self.inserted_this_request = set()
        self.actual_decisions = set()
        self.active_forced = collections.defaultdict(list)
        self.calls = collections.Counter()
        self.thresholds = collections.Counter()
        self.max_cache_len = 0
        self.start = time.perf_counter()
        self.snapshot_hashes = None
        self.snapshot_sha256 = None
        self.first_target_seq = min(self.by_seq)
        self.target_seqs_seen = set()
        self.request_costs = {}
        self.parent_actual = collections.Counter()
        self.effective_actual = collections.Counter()
        self.check_only_items = 0

    def threshold(self, relative):
        if self.parent == 'BASE' or relative < EVAL_START:
            return THETA0
        return THETAH

    def begin(self, sim, acc):
        self.current_acc = acc
        self.current_ts = acc.ts
        self.inserted_this_request = set()
        self.actual_decisions = set()
        seq = self.current_seq
        if seq == self.first_target_seq:
            snap = state_snapshot(sim)
            self.snapshot_hashes = {k: sha_obj(v) for k, v in snap.items()}
            self.snapshot_sha256 = sha_obj(snap)
        if seq in self.by_seq:
            self.target_seqs_seen.add(seq)
            for t in self.by_seq[seq]:
                key = t['native_key']
                st = t['state']
                st['cached_before_request'] = key in sim.cache.cache
                st['pending_before_request'] = key in sim.cache.admit_buffer
        self.max_cache_len = max(self.max_cache_len, len(sim.cache.cache))

    def end(self, acc):
        seq = self.current_seq
        for t in self.by_seq.get(seq, []):
            st = t['state']
            if st['decision_seen']:
                if st['intervention_changed']:
                    st['status'] = 'overridden'
                    if not st['write_success']:
                        st['status'] = 'overridden_no_write'
                elif st['parent_decision']:
                    st['status'] = 'parent_accept_unchanged'
                else:
                    st['status'] = 'parent_reject_unchanged'
            elif st['cached_before_request']:
                st['status'] = 'already_present'
            elif st['pending_before_request']:
                st['status'] = 'already_pending_no_decision'
            elif st['candidate_inserted_at_request']:
                st['status'] = 'candidate_deferred'
            else:
                st['status'] = 'no_actual_candidate'
            assert st['status'] is not None
        self.current_acc = None
        self.current_ts = None

    def score_decisions(self, self_ap, batch, ts, check_only):
        rel = ts.physical - START
        threshold = self.threshold(rel)
        self_ap.threshold = threshold
        phase = 'warmup' if rel < EVAL_START else 'evaluation'
        self.thresholds[f'{phase}|{check_only}|{threshold.hex()}'] += len(batch)
        predictions = self_ap._predict(batch, ts)
        original = {k: bool(p > threshold) for k, p in zip(batch.keys(), predictions)}
        effective = dict(original)
        self.calls[f'{phase}|{check_only}'] += 1
        if not check_only:
            assert self.current_seq > 0, 'actual admission outside request sequence'
            self.parent_actual.update(original.values())
            for target in self.by_seq.get(self.current_seq, []):
                native = target['native_key']
                st = target['state']
                key = self.by_key[(self.current_seq, target['key_token'])]['native_key']
                if key not in original:
                    continue
                assert not st['decision_seen'], ('target decided more than once', target['event_id'])
                st['decision_seen'] = True
                st['parent_decision'] = original[key]
                self.actual_decisions.add(target['key_token'])
                if self.cf and not original[key]:
                    effective[key] = True
                    st['intervention_changed'] = True
        else:
            self.check_only_items += len(batch)
            for target in self.by_seq.get(self.current_seq, []):
                if target['native_key'] in original:
                    target['state']['check_only_seen'] = True
        if not check_only:
            self.effective_actual.update(effective.values())
            self_ap.count_decisions(effective)
        return effective

    def mark_candidate(self, key):
        if self.current_seq in self.by_seq and self.current_acc is not None:
            tid = self.by_key.get((self.current_seq, token_key(list(key))))
            if tid is not None:
                tid['state']['candidate_inserted_at_request'] = True
                self.inserted_this_request.add(tid['key_token'])

    def mark_write_before(self, key, ts):
        target = self.by_key.get((self.current_seq, token_key(list(key))))
        if target is not None and target['state']['decision_seen'] and (
                target['state']['parent_decision'] or target['state']['intervention_changed']):
            st = target['state']
            assert not st['write_success']
            st['write_attempted'] = True
            st['write_was_forced'] = st['intervention_changed']
            st['write_request_sequence'] = self.current_seq
            self.active_forced[key].append(target)

    def mark_write_after(self, key):
        target = self.by_key.get((self.current_seq, token_key(list(key))))
        if target is not None and target['state'].get('write_attempted'):
            target['state']['write_success'] = True
            target['state']['write_trace_elapsed_s'] = self.current_ts.physical - START

    def mark_eviction(self, key):
        active = self.active_forced.pop(key, [])
        for target in active:
            st = target['state']
            if st['write_success']:
                st['evicted_after_write'] = True
                st['eviction_request_sequence'] = self.current_seq
                st['evicted_before_peak_request'] = self.current_seq < target['peak_request_sequence']

    def request_cost(self, need_fetch, need_prefetch, acc, before, after):
        if need_fetch:
            demand_span = max(need_fetch) - min(need_fetch) + 1
            all_chunks = list(need_fetch) + list(need_prefetch)
            extra_span = max(all_chunks) - min(all_chunks) + 1 - demand_span
            demand_s = float(service_time(1, demand_span))
            extra_s = float(service_time(0, extra_span))
        else:
            assert not need_prefetch
            demand_span = extra_span = 0
            demand_s = extra_s = 0.0
        errors = [abs(after[0]-before[0]-demand_s), abs(after[1]-before[1]-extra_s),
                  abs(after[2]-before[2]-demand_s-extra_s)]
        assert max(errors) <= 1e-9, ('request-cost accounting', errors)
        row = {'arm': self.arm, 'request_sequence': self.current_seq,
               'trace_elapsed_s': acc.ts.physical-START,
               'window_index': len(ods.batches.get('time_elapsed_phy', [])),
               'need_fetch_chunks': len(need_fetch), 'need_prefetch_chunks': len(need_prefetch),
               'demand_span_chunks': demand_span, 'extra_prefetch_span_chunks': extra_span,
               'demand_service_time_s': demand_s, 'extra_prefetch_service_time_s': extra_s,
               'total_get_service_time_s': demand_s+extra_s}
        if self.current_seq in self.by_seq:
            self.request_costs[('intervention', self.current_seq)] = {**row, 'role': 'intervention_request'}
        if self.current_seq in self.by_peak_seq:
            self.request_costs[('peak', self.current_seq)] = {**row, 'role': 'corresponding_peak_request'}
            for t in self.by_peak_seq[self.current_seq]:
                t['state']['peak_chunk_missing'] = t['chunk'] in need_fetch
                t['state']['peak_request_demand_service_time_s'] = demand_s
                t['state']['peak_request_prefetch_service_time_s'] = extra_s

    def save(self):
        states = [t['state'] for t in self.targets]
        assert self.target_seqs_seen == set(self.by_seq), ('target requests unseen', set(self.by_seq)-self.target_seqs_seen)
        assert all(s['status'] for s in states)
        assert all(s['peak_chunk_missing'] is not None for s in states)
        assert self.snapshot_hashes and self.snapshot_sha256
        directory = WORK / self.arm
        (directory / 'intervention_audit.json').write_text(json.dumps({
            'arm': self.arm, 'parent': self.parent, 'counterfactual': self.cf,
            'first_target_request_sequence': self.first_target_seq,
            'pre_first_target_state_hashes': self.snapshot_hashes,
            'pre_first_target_state_sha256': self.snapshot_sha256,
            'target_request_sequences_seen': len(self.target_seqs_seen),
            'target_status_counts': dict(collections.Counter(s['status'] for s in states)),
            'parent_actual_accepts': self.parent_actual[True],
            'parent_actual_rejects': self.parent_actual[False],
            'effective_actual_accepts': self.effective_actual[True],
            'effective_actual_rejects': self.effective_actual[False],
            'check_only_items_not_counted_as_actual': self.check_only_items,
            'target_events': states,
            'request_costs': sorted(self.request_costs.values(), key=lambda r: (r['request_sequence'], r['role'])),
            'threshold_items': dict(self.thresholds), 'admission_calls': dict(self.calls),
            'max_cache_entries_observed_at_request_boundaries': self.max_cache_len,
            'wall_seconds': time.perf_counter() - self.start,
        }, ensure_ascii=False, indent=2), encoding='utf-8')


def main(arm):
    expected = ['BASE-CONTROL', 'BASE-CF', 'HIGH-CONTROL', 'HIGH-CF']
    if arm not in expected:
        raise ValueError(arm)
    os.chdir(REPO)
    cfg = json.loads((WORK / arm / 'config.json').read_text(encoding='utf-8'))
    assert cfg['size_gb'] == 366.475 and cfg['eviction_policy'] == 'LRU' and cfg['write_mbps'] == 0
    recorder = Recorder(arm)
    original_stats = sim_cache.CacheSimulator._stats
    original_get = sim_cache.CacheSimulator.run_get
    original_put = sim_cache.CacheSimulator.run_put
    original_log_st = sim_cache.CacheSimulator._log_st
    original_insert = evp.QueueCache.insert
    original_admit = evp.QueueCache.admit
    original_eviction = evp.EvictionPolicy.log_eviction
    original_checkpoint = sim_cache.CacheSimulator._checkpoint
    original_accept = aps.NewMLAP.batchAccept

    def stats(self, ts):
        original_stats(self, ts)
        recorder.current_seq += 1
        assert self.start_ts.physical == START

    def run_get(self, acc):
        recorder.begin(self, acc)
        try:
            result = original_get(self, acc)
            recorder.peak_cost([], acc) if False else None
            recorder.end(acc)
            return result
        except Exception:
            raise

    def run_put(self, acc):
        recorder.current_acc = acc
        recorder.current_ts = acc.ts
        try:
            return original_put(self, acc)
        finally:
            recorder.current_acc = None
            recorder.current_ts = None

    def log_st(self, need_fetch, need_prefetch, all_chunks_hit, acc):
        before = [ods.get(k) for k in ['service_time_used_demand', 'service_time_used_prefetch', 'service_time_used']]
        result = original_log_st(self, need_fetch, need_prefetch, all_chunks_hit, acc)
        after = [ods.get(k) for k in ['service_time_used_demand', 'service_time_used_prefetch', 'service_time_used']]
        recorder.request_cost(need_fetch, need_prefetch, acc, before, after)
        return result

    def insert(self, key, ts, keyfeaturelist, *, metadata=None):
        recorder.mark_candidate(key)
        return original_insert(self, key, ts, keyfeaturelist, metadata=metadata)

    def admit(self, key, ts, **kwargs):
        before = self.keys_written
        recorder.mark_write_before(key, ts)
        result = original_admit(self, key, ts, **kwargs)
        assert self.keys_written - before == 1
        assert len(self.cache) <= CAPACITY
        recorder.max_cache_len = max(recorder.max_cache_len, len(self.cache))
        recorder.mark_write_after(key)
        return result

    def accept(self, batch, ts, *, metadata=None, check_only=False):
        return recorder.score_decisions(self, batch, ts, check_only)

    def eviction(self, ts, evicted):
        result = original_eviction(self, ts, evicted)
        recorder.mark_eviction(evicted[0])
        return result

    def checkpoint(self, ts, *args, **kwargs):
        result = original_checkpoint(self, ts, *args, **kwargs)
        (WORK / arm / 'progress.json').write_text(json.dumps({
            'arm': arm, 'windows': len(ods.batches.get('time_elapsed_phy', [])),
            'trace_elapsed_s': ts.physical - START, 'wall_seconds': time.perf_counter() - recorder.start,
        }), encoding='utf-8')
        return result

    sim_cache.CacheSimulator._stats = stats
    sim_cache.CacheSimulator.run_get = run_get
    sim_cache.CacheSimulator.run_put = run_put
    sim_cache.CacheSimulator._log_st = log_st
    evp.QueueCache.insert = insert
    evp.QueueCache.admit = admit
    evp.EvictionPolicy.log_eviction = eviction
    aps.NewMLAP.batchAccept = accept
    sim_cache.CacheSimulator._checkpoint = checkpoint
    sys.argv = ['bc05/replay.py', '--config', str(WORK / arm / 'config.json')]
    args = get_parsed_args()
    assert args.peak_strategy is None and args.write_mbps == 0 and args.ap == 'mlnew'
    audit = {'arm': arm, 'completed': False, 'frozen_targets_sha256': hashlib.sha256((WORK / 'frozen_targets.json').read_bytes()).hexdigest()}
    try:
        sim_cache.simulate_cache_driver(args)
        assert recorder.current_acc is None
        assert recorder.effective_actual[True] == ods.get('ap_NewMLAP_accepts')
        assert recorder.effective_actual[False] == ods.get('ap_NewMLAP_rejects')
        recorder.save()
        raw = list((WORK / arm / 'raw').rglob('*.stats.lzma'))
        assert len(raw) == 1, raw
        stats_data = compress_json.load(str(raw[0]))['batches']
        assert len(stats_data['time_elapsed_phy']) == 1008
        audit.update(completed=True, request_sequence=recorder.current_seq,
                     stats_path=str(raw[0]), stats_sha256=hashlib.sha256(raw[0].read_bytes()).hexdigest(),
                     intervention_audit_sha256=hashlib.sha256((WORK / arm / 'intervention_audit.json').read_bytes()).hexdigest(),
                     wall_seconds=time.perf_counter() - recorder.start)
    finally:
        (WORK / arm / 'run_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
