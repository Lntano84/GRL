"""One BC07 HIGH-parent replay with an online, demand-only supplement rule."""
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
REPO = BASE / 'work' / 'bc01' / 'Baleen-FAST24'
CAL_PATH = HERE / 'calibration.json'
START = 1572074461.57806
EVAL_START = 86401.23277902603
THETA0 = 0.798545
THETAH = 2 * THETA0 / (1 + THETA0)
CAPACITY = 3002
EVAL_WRITE_CAP = 148731
SAMPLE_RATIO_LOGICAL = .001
ARMS = ['HIGH-CONTROL', 'HIGH-SCORE', 'HIGH-RECENCY', 'HIGH-RANDOM']
REQUIRED = [
    'time_elapsed_phy', 'time_log', 'service_time_used_stats', 'service_time_writes_stats',
    'service_time_used_demand_stats', 'service_time_used_prefetch_stats', 'service_time_nocache_stats',
    'flashcache/keys_written_stats', 'flashcache/prefetches_stats', 'fetches_chunks_prefetch_stats',
    'fetches_chunks_demandmiss_stats', 'fetches_ios_stats', 'iops_requests_stats',
    'chunk_queries_stats', 'puts_ios_stats', 'puts_chunks_stats', 'flashcache/evictions_stats',
]

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from policy import decide, request_hash64

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
    if isinstance(v, dict):
        return {'dict': [[typed(k), typed(x)] for k, x in v.items()]}
    if isinstance(v, (str, bool, int)) or v is None:
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            return {'float': str(v)}
        return v
    if isinstance(v, np.generic):
        return typed(v.item())
    raise TypeError(f'unsupported audit key type: {type(v)}')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Recorder:
    def __init__(self, arm):
        self.arm = arm
        self.method = arm
        self.calibration = json.loads(CAL_PATH.read_text(encoding='utf-8'))
        self.dir = HERE / 'arms' / arm
        self.seq = 0
        self.get_requests = 0
        self.current_acc = None
        self.current_request_hash = None
        self.current_elapsed_s = None
        self.last_block_get_elapsed_s = {}
        self.candidate_source_by_key = {}
        self.supplement_events_by_key = collections.defaultdict(list)
        self.supplement_events = []
        self.parent = collections.Counter()
        self.effective = collections.Counter()
        self.phase_source_counts = collections.Counter()
        self.eval_demand_candidates = 0
        self.eval_parent_rejected_demand = 0
        self.eval_all_parent_rejected = 0
        self.eval_supplemented = 0
        self.eval_supplement_writes = 0
        self.eval_demands_accepted_effective = 0
        self.eval_unknown_source = 0
        self.check_only_calls = 0
        self.check_only_items = 0
        self.check_only_changed = 0
        self.actual_batches = 0
        self.actual_items = 0
        self.all_inserted = 0
        self.writes = 0
        self.eval_writes_live = 0
        self.evictions = 0
        self.max_cache_entries = 0
        self.reason_counts = collections.Counter()
        self.supplement_requests = set()
        self.supplement_blocks = set()
        self.wall_start = time.perf_counter()
        self.progress_windows = 0

    def note_progress(self, ts):
        windows = len(ods.batches.get('time_elapsed_phy', []))
        self.progress_windows = windows
        (self.dir / 'progress.json').write_text(json.dumps({
            'arm': self.arm, 'completed_windows': windows, 'trace_elapsed_s': float(ts.physical-START),
            'request_sequence': self.seq, 'get_requests': self.get_requests,
            'supplemented': self.eval_supplemented, 'wall_seconds': time.perf_counter()-self.wall_start,
        }, ensure_ascii=False, indent=2), encoding='utf-8')

    def source_for(self, metadata):
        if metadata is not None and 'prefetch' in metadata:
            return 'prefetch' if bool(metadata['prefetch']) else 'demand'
        return 'demand'

    def on_insert(self, key, metadata):
        if key in self.candidate_source_by_key:
            raise AssertionError(('duplicate pending candidate key', typed(key), self.seq))
        self.candidate_source_by_key[key] = self.source_for(metadata)
        self.all_inserted += 1

    def run_decisions(self, self_ap, batch, ts, check_only):
        if self.current_acc is None:
            raise AssertionError('admission decision outside GET request')
        relative = float(ts.physical - START)
        is_eval = relative >= EVAL_START
        threshold = THETAH if is_eval else THETA0
        self_ap.threshold = threshold
        scores = np.asarray(self_ap._predict(batch, ts), dtype=float).reshape(-1)
        if len(scores) != len(batch):
            raise AssertionError(('prediction count differs from candidate batch', len(scores), len(batch)))
        decisions = {}
        for key, score_val in zip(batch.keys(), scores):
            score = float(score_val)
            parent_accept = score > threshold
            if check_only:
                source = self.candidate_source_by_key.get(key, 'unknown')
                gap = None
                h = int(self.current_request_hash)
            else:
                if key not in self.candidate_source_by_key:
                    raise AssertionError(('actual decision has no source-tagged insert', typed(key), self.seq))
                source = self.candidate_source_by_key.pop(key)
                last = self.last_block_get_elapsed_s.get(key[0])
                gap = None if last is None else max(0.0, relative - last)
                h = int(self.current_request_hash)

            decision, supplemented, reason = decide(
                self.method, parent_accept=parent_accept, score=score, threshold=threshold,
                candidate_source=source, block_gap_seconds=gap, request_hash=h,
                is_evaluation=is_eval, check_only=check_only, calibration=self.calibration)
            if check_only:
                self.check_only_items += 1
                if decision != parent_accept or supplemented:
                    self.check_only_changed += 1
            else:
                self.actual_items += 1
                self.phase_source_counts[('evaluation' if is_eval else 'warmup', source)] += 1
                self.parent[parent_accept] += 1
                self.effective[decision] += 1
                if is_eval:
                    self.eval_all_parent_rejected += int(not parent_accept)
                    if source == 'demand':
                        self.eval_demand_candidates += 1
                        self.eval_parent_rejected_demand += int(not parent_accept)
                        self.eval_demands_accepted_effective += int(decision)
                    if source == 'unknown':
                        self.eval_unknown_source += 1
                if supplemented:
                    if check_only or not is_eval or source != 'demand' or parent_accept:
                        raise AssertionError('supplement escaped its eligible online scope')
                    self.eval_supplemented += 1
                    block, chunk = key
                    event = {
                        'request_sequence': self.seq,
                        'trace_elapsed_s': relative,
                        'block_key': typed(block),
                        'chunk_id': int(chunk),
                        'method': self.method,
                        'score': score,
                        'parent_threshold': threshold,
                        'block_gap_seconds_at_request_start': gap,
                        'request_hash64': h,
                        'decision_reason': reason,
                        'parent_decision_accept': False,
                        'effective_decision_accept': True,
                        'successful_cache_write': False,
                    }
                    self.supplement_events.append(event)
                    self.supplement_events_by_key[key].append(event)
                    self.supplement_requests.add(self.seq)
                    self.supplement_blocks.add(json.dumps(typed(block), sort_keys=True, ensure_ascii=False))
                self.reason_counts[reason] += 1
            decisions[key] = decision

        if check_only:
            self.check_only_calls += 1
        else:
            self.actual_batches += 1
            self_ap.count_decisions(decisions)
        return decisions

    def write_event(self, key, ts, delta):
        self.writes += delta
        if ts.physical - START >= EVAL_START:
            self.eval_writes_live += delta
        self.max_cache_entries = max(self.max_cache_entries, len(getattr(self._cache_ref, 'cache', {})))
        if key in self.supplement_events_by_key:
            events = self.supplement_events_by_key[key]
            if not events:
                raise AssertionError(('empty supplement event queue', typed(key)))
            event = events.pop(0)
            if not events:
                del self.supplement_events_by_key[key]
            event['successful_cache_write'] = delta == 1
            event['write_trace_elapsed_s'] = float(ts.physical - START)
            event['write_request_sequence'] = self.seq
            if delta == 1 and ts.physical - START >= EVAL_START:
                self.eval_supplement_writes += 1

    def end_get(self, acc):
        if self.supplement_events_by_key:
            leftovers = {typed(k): len(v) for k, v in self.supplement_events_by_key.items() if v}
            if leftovers:
                raise AssertionError(('supplemented decisions did not reach cache write', leftovers))
        self.last_block_get_elapsed_s[acc.block_id] = float(acc.ts.physical - START)
        self.current_acc = None
        self.current_request_hash = None
        self.current_elapsed_s = None

    def close_events(self):
        path = self.dir / 'supplement_events.jsonl'
        with path.open('w', encoding='utf-8', newline='\n') as f:
            for event in self.supplement_events:
                f.write(json.dumps(event, ensure_ascii=False, separators=(',', ':')) + '\n')

    def final_audit(self, completed, stats_path=None, error=None):
        obj = {
            'completed': bool(completed), 'arm': self.arm,
            'replay_mode': 'online branch-local decisions; demand candidates only; no learned model training',
            'trace_request_sequence': self.seq, 'get_requests': self.get_requests,
            'actual_candidate_decisions': self.actual_items, 'actual_decision_batches': self.actual_batches,
            'parent_accepts': self.parent[True], 'parent_rejects': self.parent[False],
            'effective_accepts': self.effective[True], 'effective_rejects': self.effective[False],
            'phase_source_candidate_counts': {f'{a}|{b}': n for (a, b), n in self.phase_source_counts.items()},
            'evaluation': {
                'demand_candidates': self.eval_demand_candidates,
                'high_parent_rejected_demand_candidates': self.eval_parent_rejected_demand,
                'all_sources_high_parent_rejected': self.eval_all_parent_rejected,
                'supplemented_parent_rejections': self.eval_supplemented,
                'supplement_successful_cache_writes': self.eval_supplement_writes,
                'supplement_rate_of_parent_rejected_demand': (self.eval_supplemented/self.eval_parent_rejected_demand if self.eval_parent_rejected_demand else None),
                'supplement_rate_of_all_demand_candidates': (self.eval_supplemented/self.eval_demand_candidates if self.eval_demand_candidates else None),
                'effective_demand_accepts': self.eval_demands_accepted_effective,
                'unknown_source_candidates': self.eval_unknown_source,
                'write_count_live': self.eval_writes_live,
                'write_cap': EVAL_WRITE_CAP,
                'resource_eligible_live_counter': self.eval_writes_live <= EVAL_WRITE_CAP,
            },
            'check_only': {'calls': self.check_only_calls, 'items': self.check_only_items,
                           'changed_decisions': self.check_only_changed},
            'supplement_unique_requests': len(self.supplement_requests),
            'supplement_unique_blocks': len(self.supplement_blocks),
            'supplement_event_count': len(self.supplement_events),
            'supplement_write_success_count': sum(bool(e['successful_cache_write']) for e in self.supplement_events),
            'write_count_full_trace_live': self.writes,
            'max_cache_entries_observed': self.max_cache_entries,
            'max_cache_capacity': CAPACITY,
            'rule_reason_counts': dict(self.reason_counts),
            'check_only_unchanged': self.check_only_changed == 0,
            'supplement_events_path': str(self.dir / 'supplement_events.jsonl'),
            'wall_seconds': time.perf_counter()-self.wall_start,
            'stats_path': str(stats_path) if stats_path else None,
            'error': repr(error) if error else None,
        }
        return obj


def run_one(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    os.chdir(REPO)
    arm_dir = HERE / 'arms' / arm
    if (arm_dir / 'run_audit.json').exists():
        raise RuntimeError(f'refusing a duplicate BC07 replay: {arm}')
    config = json.loads((arm_dir / 'config.json').read_text(encoding='utf-8'))
    high_config = json.loads((BASE / 'work' / 'bc03' / 'STATIC-HIGH' / 'config.json').read_text(encoding='utf-8'))
    expected = dict(high_config)
    expected['output_dir'] = str((arm_dir / 'raw').resolve()).replace('\\', '/')
    if config != expected:
        raise AssertionError('BC07 arm config changed more than its output directory')
    recorder = Recorder(arm)
    originals = {
        'stats': sim_cache.CacheSimulator._stats,
        'run_get': sim_cache.CacheSimulator.run_get,
        'insert': evp.QueueCache.insert,
        'admit': evp.QueueCache.admit,
        'checkpoint': sim_cache.CacheSimulator._checkpoint,
        'accept': aps.NewMLAP.batchAccept,
    }

    def stats(self, ts):
        originals['stats'](self, ts)
        recorder.seq += 1
        if self.start_ts.physical != START:
            raise AssertionError(('trace start changed', self.start_ts.physical, START))

    def run_get(self, acc):
        if recorder.current_acc is not None:
            raise AssertionError('nested GET')
        recorder.current_acc = acc
        relative = float(acc.ts.physical - START)
        recorder.current_elapsed_s = relative
        recorder.current_request_hash = request_hash64(recorder.seq)
        recorder.get_requests += 1
        recorder.max_cache_entries = max(recorder.max_cache_entries, len(self.cache.cache))
        try:
            result = originals['run_get'](self, acc)
        except BaseException:
            raise
        recorder.end_get(acc)
        return result

    def insert(self, key, ts, keyfeaturelist, *, metadata=None):
        recorder.on_insert(key, metadata)
        return originals['insert'](self, key, ts, keyfeaturelist, metadata=metadata)

    def admit(self, key, ts, **kwargs):
        before = self.keys_written
        result = originals['admit'](self, key, ts, **kwargs)
        delta = self.keys_written - before
        if delta != 1:
            raise AssertionError(('accepted cache write count changed', typed(key), delta))
        recorder._cache_ref = self
        recorder.write_event(key, ts, delta)
        if len(self.cache) > CAPACITY:
            raise AssertionError(('cache capacity exceeded', len(self.cache)))
        return result

    def accept(self_ap, batch, ts, *, metadata=None, check_only=False):
        return recorder.run_decisions(self_ap, batch, ts, check_only)

    def checkpoint(self, ts, *args, **kwargs):
        result = originals['checkpoint'](self, ts, *args, **kwargs)
        recorder.note_progress(ts)
        return result

    sim_cache.CacheSimulator._stats = stats
    sim_cache.CacheSimulator.run_get = run_get
    evp.QueueCache.insert = insert
    evp.QueueCache.admit = admit
    aps.NewMLAP.batchAccept = accept
    sim_cache.CacheSimulator._checkpoint = checkpoint
    sys.argv = ['bc07/replay.py', '--config', str(arm_dir / 'config.json')]
    args = get_parsed_args()
    if args.peak_strategy is not None or args.write_mbps != 0 or args.ap != 'mlnew':
        raise AssertionError('BC07 config selected an unintended simulator mode')
    audit = {'completed': False, 'arm': arm, 'started_unix': time.time(),
             'calibration_sha256': sha(CAL_PATH), 'policy_sha256': sha(HERE/'policy.py'),
             'config_sha256': sha(arm_dir/'config.json')}
    error = None
    stats_path = None
    try:
        sim_cache.simulate_cache_driver(args)
        if recorder.current_acc is not None:
            raise AssertionError('unfinished GET after simulation')
        if recorder.seq != 147794 or recorder.get_requests != 127305:
            raise AssertionError(('trace request/get count changed', recorder.seq, recorder.get_requests))
        if recorder.candidate_source_by_key:
            raise AssertionError(('unresolved actual candidate sources', len(recorder.candidate_source_by_key)))
        if recorder.check_only_changed:
            raise AssertionError('check_only decisions changed')
        if recorder.effective[True] != ods.get('ap_NewMLAP_accepts'):
            raise AssertionError(('accept counter mismatch', recorder.effective[True], ods.get('ap_NewMLAP_accepts')))
        if recorder.effective[False] != ods.get('ap_NewMLAP_rejects'):
            raise AssertionError(('reject counter mismatch', recorder.effective[False], ods.get('ap_NewMLAP_rejects')))
        if recorder.eval_supplemented != recorder.eval_supplement_writes:
            raise AssertionError(('supplement decisions and writes differ', recorder.eval_supplemented, recorder.eval_supplement_writes))
        found = list((arm_dir / 'raw').rglob('*.stats.lzma'))
        if len(found) != 1:
            raise AssertionError(('expected one stats file', found))
        stats_path = found[0]
        batches = compress_json.load(str(stats_path))['batches']
        for name in REQUIRED:
            if name not in batches:
                raise KeyError(f'required stats field missing: {name}')
            values = np.asarray(batches[name], dtype=float)
            if values.shape != (1008,) or not np.isfinite(values).all():
                raise AssertionError(('invalid stats field', name, values.shape))
        eval_writes = int(round(float(batches['flashcache/keys_written_stats'][-1] - batches['flashcache/keys_written_stats'][143])))
        if eval_writes != recorder.eval_writes_live:
            raise AssertionError(('live and raw evaluation writes differ', recorder.eval_writes_live, eval_writes))
        if int(round(float(batches['flashcache/keys_written_stats'][-1]))) != recorder.writes:
            raise AssertionError(('live and raw full trace writes differ', recorder.writes, batches['flashcache/keys_written_stats'][-1]))
        recorder.close_events()
        audit.update(recorder.final_audit(True, stats_path=stats_path))
        audit['completed'] = True
        audit['run_completed_unix'] = time.time()
        audit['stats_sha256'] = sha(stats_path)
        audit['supplement_events_sha256'] = sha(arm_dir / 'supplement_events.jsonl')
    except BaseException as exc:
        error = exc
        recorder.close_events()
        audit.update(recorder.final_audit(False, stats_path=stats_path, error=exc))
        raise
    finally:
        sim_cache.CacheSimulator._stats = originals['stats']
        sim_cache.CacheSimulator.run_get = originals['run_get']
        evp.QueueCache.insert = originals['insert']
        evp.QueueCache.admit = originals['admit']
        sim_cache.CacheSimulator._checkpoint = originals['checkpoint']
        aps.NewMLAP.batchAccept = originals['accept']
        (arm_dir / 'decision_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return audit


if __name__ == '__main__':
    run_one(sys.argv[1])
