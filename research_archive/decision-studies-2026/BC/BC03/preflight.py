"""User-requested interval and original admission-interface checks."""
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import *
from replay import aps, ods, scheduled_accept
from BCacheSim.cachesim.ep_helpers import Timestamp
from BCacheSim.cachesim.prefetchers import Prefetcher

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cases = []
    for arm in ARMS:
        assert threshold_at(arm, math.nextafter(EVAL_START, -math.inf)) == THETA0
        for relative in [0, 3600, 86400]:
            assert threshold_at(arm, relative) == THETA0
    for arm, (left, right) in INTERVALS.items():
        times = [math.nextafter(left, -math.inf), left, math.nextafter(right, -math.inf), right]
        expected = [THETAH, THETAL, THETAL, THETAH]
        for relative, theta in zip(times, expected):
            assert threshold_at(arm, relative) == theta
            cases.append({'arm': arm, 'time_hex': relative.hex(), 'theta_hex': theta.hex()})
    # Exercise the actual original LearnedAP.batchAccept with a deterministic
    # prediction fixture, including equality and the closest float above it.
    audit = {'calls': {}}
    aps.NewMLAP.batchAccept = scheduled_accept('LEAD', audit)
    ap = object.__new__(aps.NewMLAP)
    for relative in [100, EVAL_START, INTERVALS['LEAD'][0], INTERVALS['LEAD'][1]]:
        theta = threshold_at('LEAD', relative)
        ap._predict = lambda batch, ts, theta=theta: [theta, math.nextafter(theta, math.inf)]
        batch = {('A', 1): [0], ('A', 2): [0]}
        ts = Timestamp(physical=START+relative, logical=0)
        ods.counters.clear()
        expected = {('A', 1): False, ('A', 2): True}
        before = dict(ods.counters)
        assert ap.batchAccept(batch, ts, metadata={}, check_only=True) == expected
        assert ods.counters == before, 'check_only counted acceptance/rejection'
        assert ap.batchAccept(batch, ts, metadata={}, check_only=False) == expected
        assert ods.get('ap_NewMLAP_accepts') == 1 and ods.get('ap_NewMLAP_rejects') == 1
    # Drive the genuine prefetch qualifier, proving it routes through check_only.
    ap._predict = lambda batch, ts: [THETAL, math.nextafter(THETAL, math.inf)]
    pf = object.__new__(Prefetcher)
    pf.ap = ap
    pf.cache = SimpleNamespace(collect_features=lambda key, acc: [0])
    acc = SimpleNamespace(block_id='A', ts=Timestamp(physical=START+INTERVALS['LEAD'][0], logical=1))
    ods.counters.clear()
    assert pf.filter_ap([1, 2], acc, {1: {}, 2: {}}) == [2]
    assert ods.get('ap_NewMLAP_accepts') == 0 and ods.get('ap_NewMLAP_rejects') == 0
    # Actual and qualifier calls at the same timestamp use exactly the same L.
    assert ap.threshold == THETAL
    config = json.loads(CONFIG.read_text())
    assert config['ap_threshold'] == THETA0 and config['batch_size'] == 16
    assert config['eviction_policy'] == 'LRU' and config['write_mbps'] == 0
    config_hashes = {}
    for arm in ARMS:
        directory = WORK / arm
        directory.mkdir(parents=True, exist_ok=True)
        target = {**config, 'output_dir': str(directory / 'raw').replace('\\', '/')}
        path = directory / 'config.json'
        path.write_text(json.dumps(target, indent=2))
        assert {k for k in config if config[k] != target[k]} == {'output_dir'}
        config_hashes[arm] = sha(path)
    model_dir = REPO / 'tmp/example/201910_Region1_0_0.1'
    models = list(model_dir.glob('*.model'))
    assert len(models) == 5
    pinned = [CONFIG, REFERENCE, REPO/'data/tectonic/201910/Region1/full_0_0.1.trace', *models]
    result = {'passed': True, 'boundary_cases': cases,
              'checks': ['all_warmup_thresholds_original', 'left_closed_right_open',
                         'strict_pred_greater_than_threshold', 'check_only_preserved',
                         'original_prefetch_qualifier_routes_to_same_interface',
                         'only_output_dir_differs_in_source_config'],
              'theta0': THETA0, 'thetaL': THETAL, 'thetaH': THETAH,
              'intervals': INTERVALS, 'evaluation_start_s': EVAL_START,
              'write_budget_chunks': WRITE_BUDGET, 'compute_budget_seconds': 7200,
              'inputs_sha256': {str(p.relative_to(REPO)): sha(p) for p in pinned},
              'config_sha256': config_hashes}
    (WORK / 'preflight.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'preflight_passed': True, 'thetaL': THETAL, 'thetaH': THETAH, 'checks': result['checks']}))

if __name__ == '__main__':
    main()
