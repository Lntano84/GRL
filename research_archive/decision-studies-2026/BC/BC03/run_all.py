"""Exactly seven complete replay attempts with one shared two-hour wall budget."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import *
sys.path.insert(0, str(REPO))
import compress_json
import numpy as np

KEYS = ['time_elapsed_phy', 'service_time_used_stats', 'service_time_writes_stats',
        'service_time_nocache_stats', 'flashcache/keys_written_stats',
        'fetches_chunks_prefetch_stats', 'fetches_chunks_demandmiss_stats',
        'iops_requests_stats', 'chunk_queries_stats', 'puts_ios_stats', 'puts_chunks_stats']

def find_stats(arm):
    found = list((WORK/arm/'raw').rglob('*.stats.lzma'))
    if len(found) != 1:
        raise AssertionError(f'{arm}: expected one completed stats, found {found}')
    return found[0]

def base_gate():
    ref = compress_json.load(str(REFERENCE))['batches']
    now = compress_json.load(str(find_stats('STATIC-BASE')))['batches']
    errors = {}
    for key in KEYS:
        a, b = np.asarray(ref[key], dtype=float), np.asarray(now[key], dtype=float)
        assert len(a) == len(b) == 1008 and np.isfinite(a).all() and np.isfinite(b).all()
        err = float(np.max(np.abs(a-b)))
        tol = 1e-9 if key.startswith('service_time') else 0
        assert err <= tol, f'STATIC-BASE mismatch {key}: {err}'
        errors[key] = err
    (WORK/'static_base_gate.json').write_text(json.dumps({'passed': True, 'max_errors': errors}, indent=2))
    return errors

def main():
    assert json.loads((WORK/'preflight.json').read_text())['passed']
    assert not (WORK/'run_status.json').exists(), 'Do not launch duplicate attempts'
    env = dict(os.environ)
    env['PATH'] = 'C:/Program Files/Git/usr/bin;' + env['PATH']
    env['PYTHONHASHSEED'] = '0'
    started = time.monotonic()
    deadline = started + 7200
    status = {'started_unix': time.time(), 'budget_seconds': 7200, 'completed': [], 'arms': {}, 'status': 'running'}
    def save():
        status['wall_seconds'] = time.monotonic()-started
        (WORK/'run_status.json').write_text(json.dumps(status, indent=2))
    save()
    for arm in ARMS:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            status['status'] = 'budget_exhausted'
            break
        directory = WORK/arm
        arm_start = time.monotonic()
        status['current_arm'] = arm
        status['arms'][arm] = {'status': 'running', 'started_unix': time.time()}
        save()
        print(f'START {arm}', flush=True)
        with (directory/'console.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen([str(PYTHON), '-B', str(WORK/'replay.py'), arm],
                                    cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                status['arms'][arm]['status'] = 'budget_interrupted'
                status['status'] = 'budget_exhausted'
                save()
                break
        status['arms'][arm].update(exit_code=code, wall_seconds=time.monotonic()-arm_start)
        if code != 0:
            status['arms'][arm]['status'] = 'failed'
            status['status'] = 'interface_or_replay_failure'
            save()
            print(f'FAILED {arm}', flush=True)
            break
        assert json.loads((directory/'audit.json').read_text())['completed']
        find_stats(arm)
        if arm == 'STATIC-BASE':
            try:
                status['static_base_gate'] = base_gate()
            except Exception as error:
                status['status'] = 'baseline_reproduction_failure'
                status['error'] = repr(error)
                save()
                raise
        status['arms'][arm]['status'] = 'completed'
        status['completed'].append(arm)
        save()
        print(f'COMPLETE {arm}: {status["arms"][arm]["wall_seconds"]:.2f}s', flush=True)
    else:
        status['status'] = 'completed'
    status['uncompleted'] = [arm for arm in ARMS if arm not in status['completed']]
    save()
    print(json.dumps(status), flush=True)

if __name__ == '__main__':
    main()
