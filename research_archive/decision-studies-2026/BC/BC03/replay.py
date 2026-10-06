"""Run one original simulator in a fresh process, patching only NewMLAP's threshold.

Both actual admission and prefetch check_only use the invocation timestamp.
Full pre-evaluation behavioral state is captured without flushing any queues.
"""
import collections
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import *

sys.path.insert(0, str(REPO))
import numpy as np
from BCacheSim.cachesim import admission_policies as aps, sim_cache
from BCacheSim.cachesim.utils import ods
from BCacheSim.cachesim.simulate_ap import get_parsed_args

ORIGINAL_ACCEPT = aps.NewMLAP.batchAccept

def encoded(value):
    """Exact floats; preserve ordered containers and sequence order, sort sets/maps.

Mutable dictionary ordering is irrelevant except admission-buffer batch order,
which is separately recorded. Cache LRU order is an OrderedDict and preserved.
Unknown objects fail rather than silently disappearing from the snapshot.
"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return {'float_hex': value.hex()}
    if isinstance(value, np.generic):
        return encoded(value.item())
    if isinstance(value, np.ndarray):
        return {'ndarray': str(value.dtype), 'shape': value.shape, 'values': encoded(value.tolist())}
    if isinstance(value, dict):
        pairs = [[encoded(k), encoded(v)] for k, v in value.items()]
        if not isinstance(value, collections.OrderedDict):
            pairs.sort(key=lambda p: json.dumps(p[0], sort_keys=True))
        return {'mapping_type': type(value).__name__, 'items': pairs}
    if isinstance(value, (set, frozenset)):
        return {'set': sorted([encoded(v) for v in value], key=lambda x: json.dumps(x, sort_keys=True))}
    if isinstance(value, (list, tuple, collections.deque)):
        return {'sequence_type': type(value).__name__, 'items': [encoded(v) for v in value]}
    if hasattr(value, '__dict__'):
        return {'class': type(value).__module__ + '.' + type(value).__name__, 'state': encoded(vars(value))}
    raise TypeError(f'Uncaptured snapshot type: {type(value)}')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def state_snapshot(sim):
    cache = sim.cache
    excluded = {'ap', 'prefetcher', 'evictions_log', 'on_evict'}
    assert cache.cache_size == 3002 and sim.ram_cache is None
    assert cache.evictions_log is None and cache.on_evict is None
    assert cache.offline_feat_df is None and cache.episodes is None
    state = {k: v for k, v in vars(cache).items() if k not in excluded}
    # Capture admission order explicitly; sorted metadata maps do not lose batch order.
    state['admit_buffer_order'] = list(cache.admit_buffer)
    ap_state = {k: v for k, v in vars(cache.ap).items() if k != 'gbm'}
    pf_state = {k: v for k, v in vars(sim.prefetcher).items()
                if k not in {'cache', 'insert_cache', 'ram_cache', 'ap', 'model'}}
    pf_model = sim.prefetcher.model
    pf_state['model_non_booster_state'] = {k: v for k, v in vars(pf_model).items() if k != 'models'}
    return { 'cache_and_history': encoded(state), 'ap_state': encoded(ap_state),
             'prefetch_state': encoded(pf_state),
             'simulator_time_state': encoded({k: getattr(sim, k) for k in ['start_ts', 'last_log_tracetime', 'last_util_peak']}),
             'cumulative_counters': encoded(ods.counters), 'frequency_counters': encoded(ods.freq),
             'checkpoint_index': ods.idx }

def scheduled_accept(arm, audit):
    def call(self, batch, ts, *, metadata=None, check_only=False):
        relative = ts.physical - START
        threshold = threshold_at(arm, relative)
        self.threshold = threshold
        phase = 'warmup' if relative < EVAL_START else 'evaluation'
        key = f'{phase}|check_only={check_only}|theta={threshold.hex()}'
        group = audit['calls'].setdefault(key, {'calls': 0, 'items': 0, 'first_s': relative, 'last_s': relative})
        group['calls'] += 1
        group['items'] += len(batch)
        group['last_s'] = relative
        if phase == 'warmup' and threshold != THETA0:
            raise AssertionError('changed warmup threshold')
        return ORIGINAL_ACCEPT(self, batch, ts, metadata=metadata, check_only=check_only)
    return call

def main(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    os.chdir(REPO)
    directory = WORK / arm
    directory.mkdir(parents=True, exist_ok=True)
    audit = {'arm': arm, 'calls': {}, 'snapshot_captured': False}
    started = time.perf_counter()
    aps.NewMLAP.batchAccept = scheduled_accept(arm, audit)
    original_stats = sim_cache.CacheSimulator._stats
    original_checkpoint = sim_cache.CacheSimulator._checkpoint
    def stats(self, ts):
        original_stats(self, ts)
        relative = (ts-self.start_ts).physical
        if not audit['snapshot_captured'] and relative >= EVAL_START:
            assert relative == EVAL_START, (relative, EVAL_START)
            assert self.start_ts.physical == START
            snapshot = state_snapshot(self)
            hashes = {k: digest(v) for k, v in snapshot.items()}
            audit['warmup_state_hashes'] = hashes
            audit['warmup_snapshot_sha256'] = digest(snapshot)
            audit['evaluation_start_s'] = relative
            audit['snapshot_captured'] = True
            with gzip.open(directory / 'warmup_state.json.gz', 'wt', encoding='utf-8') as f:
                json.dump(snapshot, f, separators=(',', ':'))
            if arm != 'STATIC-BASE':
                reference = json.loads((WORK / 'STATIC-BASE' / 'audit.json').read_text())
                if hashes != reference['warmup_state_hashes']:
                    raise AssertionError(f'warmup state mismatch: {hashes} vs {reference["warmup_state_hashes"]}')
            (directory / 'audit.json').write_text(json.dumps(audit, indent=2))
    def checkpoint(self, ts, *args, **kwargs):
        result = original_checkpoint(self, ts, *args, **kwargs)
        progress = {'arm': arm, 'relative_time_s': (ts-self.start_ts).physical,
                    'windows': len(ods.batches.get('time_elapsed_phy', [])),
                    'wall_seconds': time.perf_counter()-started}
        (directory / 'progress.json').write_text(json.dumps(progress))
        return result
    sim_cache.CacheSimulator._stats = stats
    sim_cache.CacheSimulator._checkpoint = checkpoint
    sys.argv = ['bc03/replay.py', '--config', str(directory / 'config.json')]
    args = get_parsed_args()
    assert args.peak_strategy is None and args.write_mbps == 0 and args.ap == 'mlnew'
    try:
        sim_cache.simulate_cache_driver(args)
        assert audit['snapshot_captured']
        assert all(any(f'evaluation|check_only={mode}|' in k for k in audit['calls']) for mode in [True, False])
        audit['completed'] = True
    finally:
        audit['wall_seconds'] = time.perf_counter()-started
        (directory / 'audit.json').write_text(json.dumps(audit, indent=2))

if __name__ == '__main__':
    main(sys.argv[1])
