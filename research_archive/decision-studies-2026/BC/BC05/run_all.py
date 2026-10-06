"""Exactly four sequential BC05 replay attempts under a shared one-hour cap."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
PYTHON = BASE/'work/bc01/python311-embed/python.exe'
REPO = BASE/'work/bc01/Baleen-FAST24'
ARMS = ['BASE-CONTROL', 'BASE-CF', 'HIGH-CONTROL', 'HIGH-CF']
BUDGET = 3600
KEYS = ['time_elapsed_phy', 'time_log', 'service_time_used_stats', 'service_time_writes_stats',
        'service_time_used_demand_stats', 'service_time_used_prefetch_stats', 'service_time_nocache_stats',
        'flashcache/keys_written_stats', 'flashcache/prefetches_stats', 'fetches_chunks_prefetch_stats',
        'fetches_chunks_demandmiss_stats', 'fetches_ios_stats', 'iops_requests_stats',
        'chunk_queries_stats', 'puts_ios_stats', 'puts_chunks_stats']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stats(path):
    import sys
    sys.path.insert(0, str(REPO))
    import compress_json
    data = compress_json.load(str(path))['batches']
    return data


def one_stats(arm):
    found = list((WORK/arm/'raw').rglob('*.stats.lzma'))
    if len(found) != 1:
        raise AssertionError(f'{arm}: expected one stats artifact, found {found}')
    return found[0]


def gate_control(arm):
    import numpy as np
    ref_arm = 'STATIC-BASE' if arm == 'BASE-CONTROL' else 'STATIC-HIGH'
    ref = list((BASE/'work/bc03'/ref_arm/'raw').rglob('*.stats.lzma'))
    assert len(ref) == 1
    actual = load_stats(one_stats(arm))
    expected = load_stats(ref[0])
    errors = {}
    for key in KEYS:
        if key not in actual or key not in expected:
            raise KeyError('required control field missing: '+key)
        a, b = np.asarray(actual[key], dtype=float), np.asarray(expected[key], dtype=float)
        assert a.shape == b.shape == (1008,) and np.isfinite(a).all() and np.isfinite(b).all(), key
        error = float(np.max(np.abs(a-b)))
        tolerance = 1e-9 if key.startswith('service_time') else 0.0
        assert error <= tolerance, f'{arm} differs from {ref_arm} on {key}: {error}'
        errors[key] = error
    record = {'passed': True, 'arm': arm, 'reference_arm': ref_arm, 'max_errors': errors,
              'max_required_field_error': max(errors.values())}
    event_audit = json.loads((WORK/arm/'intervention_audit.json').read_text(encoding='utf-8'))
    if arm == 'BASE-CONTROL':
        assert event_audit['target_status_counts'] == {'parent_reject_unchanged': 1489}, event_audit['target_status_counts']
        assert event_audit['parent_actual_rejects'] >= 1489
        record['frozen_targets_rejected_at_expected_request'] = 1489
    (WORK/arm/'control_reproduction_gate.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def gate_preintervention(cf_arm, control_arm):
    cf = json.loads((WORK/cf_arm/'intervention_audit.json').read_text(encoding='utf-8'))
    control = json.loads((WORK/control_arm/'intervention_audit.json').read_text(encoding='utf-8'))
    assert cf['first_target_request_sequence'] == control['first_target_request_sequence']
    assert cf['pre_first_target_state_hashes'] == control['pre_first_target_state_hashes'], 'state component mismatch'
    assert cf['pre_first_target_state_sha256'] == control['pre_first_target_state_sha256'], 'full state mismatch'
    record = {'passed': True, 'cf_arm': cf_arm, 'control_arm': control_arm,
              'request_sequence': cf['first_target_request_sequence'],
              'snapshot_sha256': cf['pre_first_target_state_sha256'],
              'component_hashes': cf['pre_first_target_state_hashes']}
    (WORK/(cf_arm+'_preintervention_gate.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def validate_inputs():
    assert PYTHON.exists(), f'configured embedded Python missing: {PYTHON}'
    assert (WORK/'replay.py').exists()
    manifest = json.loads((WORK/'frozen_manifest.json').read_text(encoding='utf-8'))
    targets = WORK/'frozen_targets.json'
    assert sha(targets) == manifest['targets_sha256']
    for path, digest in manifest['source_sha256'].items():
        assert sha(Path(path)) == digest, f'input changed after freezing: {path}'
    assert json.loads((WORK/'preflight.json').read_text(encoding='utf-8'))['passed']
    for arm in ARMS:
        assert not (WORK/arm/'run_audit.json').exists(), f'{arm} already attempted; refusing duplicate'
    return manifest


def read_progress_safe(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def main():
    manifest = validate_inputs()
    started = time.monotonic()
    deadline = started+BUDGET
    status = {'status': 'running', 'started_unix': time.time(), 'budget_seconds': BUDGET,
              'frozen_targets_sha256': manifest['targets_sha256'], 'attempted': [], 'arms': {}}

    def save():
        status['wall_seconds'] = time.monotonic()-started
        status['remaining_budget_seconds'] = max(0.0, deadline-time.monotonic())
        (WORK/'run_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')

    save()
    env = dict(os.environ)
    env['PATH'] = 'C:/Program Files/Git/usr/bin;' + env.get('PATH', '')
    env['PYTHONHASHSEED'] = '0'
    for arm in ARMS:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            status['status'] = 'budget_exhausted'
            break
        d = WORK/arm
        arm_start = time.monotonic()
        item = {'status': 'running', 'started_unix': time.time()}
        status['arms'][arm] = item
        status['attempted'].append(arm)
        status['current_arm'] = arm
        save()
        print(f'START {arm}; remaining={remaining:.1f}s', flush=True)
        with (d/'console.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen([str(PYTHON), '-B', str(WORK/'replay.py'), arm],
                                    cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    left = deadline-time.monotonic()
                    if left <= 0:
                        proc.kill()
                        proc.wait()
                        item.update(status='budget_interrupted', exit_code=proc.returncode,
                                    wall_seconds=time.monotonic()-arm_start)
                        status['status'] = 'budget_exhausted'
                        save()
                        break
                    try:
                        code = proc.wait(timeout=min(left, 30.0))
                        break
                    except subprocess.TimeoutExpired:
                        progress = d/'progress.json'
                        item['latest_progress'] = read_progress_safe(progress)
                        save()
                        print(f'PROGRESS {arm}: {item["latest_progress"]}', flush=True)
            except BaseException as error:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                item.update(status='coordinator_failure', error=repr(error),
                            wall_seconds=time.monotonic()-arm_start)
                status['status'] = 'coordinator_failure'
                save()
                raise
            if status['status'] == 'budget_exhausted':
                break
        item.update(exit_code=code, wall_seconds=time.monotonic()-arm_start)
        if code != 0:
            item['status'] = 'failed'
            status['status'] = 'replay_failure'
            item['console_tail'] = (d/'console.log').read_text(encoding='utf-8', errors='replace')[-5000:]
            save()
            print(f'FAILED {arm}, exit={code}', flush=True)
            break
        run_audit = json.loads((d/'run_audit.json').read_text(encoding='utf-8'))
        assert run_audit['completed'] and run_audit['request_sequence'] == 147794
        one_stats(arm)
        try:
            if arm.endswith('CONTROL'):
                gate = gate_control(arm)
                item['control_gate'] = gate
            elif arm == 'BASE-CF':
                item['preintervention_gate'] = gate_preintervention(arm, 'BASE-CONTROL')
            elif arm == 'HIGH-CF':
                item['preintervention_gate'] = gate_preintervention(arm, 'HIGH-CONTROL')
        except Exception as error:
            item['status'] = 'protocol_gate_failure'
            item['gate_error'] = repr(error)
            status['status'] = 'protocol_gate_failure'
            save()
            print(f'GATE FAILED {arm}: {error!r}', flush=True)
            break
        item['status'] = 'completed'
        status['completed'] = status.get('completed', []) + [arm]
        save()
        print(f'COMPLETE {arm}: {item["wall_seconds"]:.1f}s', flush=True)
    else:
        status['status'] = 'completed'
    status['uncompleted'] = [arm for arm in ARMS if arm not in status.get('completed', [])]
    status['current_arm'] = None
    save()
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
