"""Run the four frozen BC07 arms sequentially under one sub-hour budget."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parent
BASE = WORK.parents[1]
OUT = BASE / 'outputs'
PYTHON = BASE / 'work' / 'bc01' / 'python311-embed' / 'python.exe'
REPO = BASE / 'work' / 'bc01' / 'Baleen-FAST24'
ARMS = ['HIGH-CONTROL', 'HIGH-SCORE', 'HIGH-RECENCY', 'HIGH-RANDOM']
BUDGET_SECONDS = 3500
KEYS = [
    'time_elapsed_phy', 'time_log', 'service_time_used_stats', 'service_time_writes_stats',
    'service_time_used_demand_stats', 'service_time_used_prefetch_stats', 'service_time_nocache_stats',
    'flashcache/keys_written_stats', 'flashcache/prefetches_stats', 'fetches_chunks_prefetch_stats',
    'fetches_chunks_demandmiss_stats', 'fetches_ios_stats', 'iops_requests_stats',
    'chunk_queries_stats', 'puts_ios_stats', 'puts_chunks_stats', 'flashcache/evictions_stats',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_stats(path):
    sys.path.insert(0, str(REPO))
    import compress_json
    return compress_json.load(str(path))['batches']


def one_stats(arm):
    files = list((WORK / 'arms' / arm / 'raw').rglob('*.stats.lzma'))
    if len(files) != 1:
        raise AssertionError(f'{arm}: expected one stats artifact, found {len(files)}')
    return files[0]


def comparison_gate(arm):
    import numpy as np
    actual_path = one_stats(arm)
    actual = load_stats(actual_path)
    reference_path = next((BASE / 'work' / 'bc03' / 'STATIC-HIGH' / 'raw').rglob('*.stats.lzma'))
    reference = load_stats(reference_path)
    errors = {}
    limit = 1008 if arm == 'HIGH-CONTROL' else 144
    for key in KEYS:
        if key not in actual or key not in reference:
            raise KeyError(f'{arm}: missing required comparison field {key}')
        a = np.asarray(actual[key], dtype=float)
        b = np.asarray(reference[key], dtype=float)
        if a.shape != (1008,) or b.shape != (1008,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError(('invalid required series', arm, key, a.shape, b.shape))
        err = float(np.max(np.abs(a[:limit] - b[:limit])))
        tolerance = 1e-9 if key.startswith('service_time') else 0.0
        if err > tolerance:
            raise AssertionError(('HIGH parent replay mismatch', arm, key, err, tolerance, limit))
        errors[key] = err
    audit_path = WORK / 'arms' / arm / 'decision_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if not audit['completed'] or audit['trace_request_sequence'] != 147794 or audit['get_requests'] != 127305:
        raise AssertionError(('incomplete BC07 replay audit', arm))
    if audit['check_only']['changed_decisions'] != 0 or audit['evaluation']['unknown_source_candidates'] != 0:
        raise AssertionError(('check-only or candidate source qualification failure', arm, audit['check_only'], audit['evaluation']))
    if audit['max_cache_entries_observed'] > 3002:
        raise AssertionError(('capacity exceeded', arm, audit['max_cache_entries_observed']))
    return {
        'passed': True,
        'arm': arm,
        'reference': 'BC03 STATIC-HIGH',
        'compared_windows': limit,
        'max_errors_by_required_field': errors,
        'decision_audit_sha256': sha(audit_path),
        'stats_sha256': sha(actual_path),
    }


def read_progress(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def main():
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    if not json.loads((WORK / 'preflight.json').read_text(encoding='utf-8'))['passed']:
        raise AssertionError('BC07 preflight did not pass')
    cal = WORK / 'calibration.json'
    if not cal.exists() or sha(cal) != sha(OUT / 'BC07_calibration.json'):
        raise AssertionError('frozen calibration is missing or output copy differs')
    if (WORK / 'run_status.json').exists():
        raise RuntimeError('BC07 runner already attempted; refusing to duplicate replay work')
    for arm in ARMS:
        d = WORK / 'arms' / arm
        if (d / 'run_audit.json').exists() or (d / 'decision_audit.json').exists():
            raise RuntimeError(f'{arm} already has run output; refusing a duplicate')
        cfg = json.loads((d / 'config.json').read_text(encoding='utf-8'))
        ref = json.loads((BASE / 'work' / 'bc03' / 'STATIC-HIGH' / 'config.json').read_text(encoding='utf-8'))
        ref['output_dir'] = str((d / 'raw').resolve()).replace('\\', '/')
        if cfg != ref:
            raise AssertionError(f'{arm} config is not an output-directory-only HIGH copy')

    start = time.monotonic()
    deadline = start + BUDGET_SECONDS
    status = {
        'status': 'running', 'started_unix': time.time(), 'budget_seconds': BUDGET_SECONDS,
        'calibration_sha256': sha(cal), 'preflight_sha256': sha(WORK / 'preflight.json'),
        'arms': {}, 'completed': [], 'attempted': [],
    }

    def save():
        status['wall_seconds'] = time.monotonic() - start
        status['remaining_budget_seconds'] = max(0.0, deadline - time.monotonic())
        (WORK / 'run_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        (OUT / 'BC07_run_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')

    env = dict(os.environ)
    env['PATH'] = 'C:/Program Files/Git/usr/bin;' + env.get('PATH', '')
    env['PYTHONHASHSEED'] = '0'
    save()
    for arm in ARMS:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            status['status'] = 'budget_exhausted'
            break
        arm_dir = WORK / 'arms' / arm
        item = {'status': 'running', 'started_unix': time.time()}
        status['arms'][arm] = item
        status['attempted'].append(arm)
        status['current_arm'] = arm
        save()
        print(f'START {arm}; remaining={remaining:.1f}s', flush=True)
        with (arm_dir / 'console.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen([str(PYTHON), '-B', str(WORK / 'replay.py'), arm],
                                    cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
            arm_start = time.monotonic()
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    proc.kill()
                    proc.wait()
                    item.update(status='budget_interrupted', exit_code=proc.returncode,
                                wall_seconds=time.monotonic() - arm_start,
                                latest_progress=read_progress(arm_dir / 'progress.json'))
                    status['status'] = 'budget_exhausted'
                    save()
                    break
                try:
                    code = proc.wait(timeout=min(left, 30.0))
                    break
                except subprocess.TimeoutExpired:
                    item['latest_progress'] = read_progress(arm_dir / 'progress.json')
                    save()
                    print(f'PROGRESS {arm}: {item["latest_progress"]}', flush=True)
            if status['status'] == 'budget_exhausted':
                break
        item.update(exit_code=code, wall_seconds=time.monotonic() - arm_start)
        if code != 0:
            item['status'] = 'failed'
            item['console_tail'] = (arm_dir / 'console.log').read_text(encoding='utf-8', errors='replace')[-6000:]
            status['status'] = 'replay_failure'
            save()
            print(f'FAILED {arm} exit={code}', flush=True)
            break
        try:
            gate = comparison_gate(arm)
        except Exception as exc:
            item.update(status='qualification_failure', gate_error=repr(exc))
            status['status'] = 'protocol_failure'
            save()
            print(f'GATE FAILED {arm}: {exc!r}', flush=True)
            break
        item.update(status='completed', gate=gate)
        status['completed'].append(arm)
        save()
        print(f'COMPLETE {arm}; wall={item["wall_seconds"]:.1f}s; remaining={status["remaining_budget_seconds"]:.1f}s', flush=True)
    else:
        status['status'] = 'completed'
    status['current_arm'] = None
    status['uncompleted'] = [arm for arm in ARMS if arm not in status['completed']]
    save()
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
