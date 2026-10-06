"""Resume the frozen four-arm run after archiving one instrumentation failure."""
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
ARMS_TO_RUN = ['HIGH-SCORE', 'HIGH-RECENCY', 'HIGH-RANDOM']
sys.path.insert(0, str(WORK))
from run_all import comparison_gate, read_progress
from summarize import main as summarize_main
from calibrate import sha


def main():
    status_path = WORK / 'run_status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status.get('status') != 'replay_failure' or status.get('completed') != ['HIGH-CONTROL']:
        raise AssertionError(('unexpected resume starting status', status.get('status'), status.get('completed')))
    if status['arms']['HIGH-SCORE'].get('status') != 'failed':
        raise AssertionError('expected exactly one failed partial HIGH-SCORE attempt')
    archive = WORK / 'arms' / 'HIGH-SCORE' / 'attempts' / '01_failed_empty_event_queue'
    for item in ['console.log', 'decision_audit.json', 'progress.json', 'supplement_events.jsonl', 'raw', 'run_status_at_failure.json']:
        if not (archive / item).exists():
            raise FileNotFoundError(archive / item)
    if sha(WORK / 'calibration.json') != status['calibration_sha256']:
        raise AssertionError('calibration changed after the frozen run started')
    if not json.loads((WORK / 'preflight.json').read_text(encoding='utf-8'))['passed']:
        raise AssertionError('updated source preflight failed')

    initial_remaining = float(status['remaining_budget_seconds'])
    resume_cap = min(2750.0, initial_remaining)
    if resume_cap < 2100:
        raise RuntimeError(f'insufficient one-hour budget remaining for three full arms: {resume_cap:.1f}s')
    resume_start = time.monotonic()
    deadline = resume_start + resume_cap
    initial_wall = float(status['wall_seconds'])
    initial_completed = list(status['completed'])
    status['attempt_history'] = status.get('attempt_history', []) + [{
        'arm': 'HIGH-SCORE', 'attempt': 1, 'status': 'failed_instrumentation',
        'archive': str(archive.resolve()), 'progress_windows': 167,
        'failure': 'empty event queue key remained in bookkeeping after the matching supplement write; recorder now deletes drained keys',
        'replay_wall_seconds': status['arms']['HIGH-SCORE']['wall_seconds'],
        'no_strategy_or_calibration_change': True,
    }]
    status['status'] = 'running_after_instrumentation_fix'
    status['resume_started_unix'] = time.time()
    status['resume_budget_seconds'] = resume_cap
    status['resume_policy_sha256'] = sha(WORK / 'replay.py')
    status['resume_preflight_sha256'] = sha(WORK / 'preflight.json')
    status['current_arm'] = None
    status['arms']['HIGH-SCORE'] = {'status': 'pending_rerun', 'prior_failed_attempt_archive': str(archive.resolve())}

    def save():
        resumed = time.monotonic() - resume_start
        status['resume_wall_seconds'] = resumed
        status['wall_seconds'] = initial_wall + resumed
        status['remaining_budget_seconds'] = max(0.0, initial_remaining - resumed)
        status['completed'] = initial_completed + [a for a in ARMS_TO_RUN if status['arms'].get(a, {}).get('status') == 'completed']
        status['uncompleted'] = [a for a in ['HIGH-CONTROL', *ARMS_TO_RUN] if a not in status['completed']]
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        (OUT / 'BC07_run_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')

    env = dict(os.environ)
    env['PATH'] = 'C:/Program Files/Git/usr/bin;' + env.get('PATH', '')
    env['PYTHONHASHSEED'] = '0'
    status['attempted'].append('HIGH-SCORE#2')
    save()

    for arm in ARMS_TO_RUN:
        arm_dir = WORK / 'arms' / arm
        for existing in ['decision_audit.json', 'supplement_events.jsonl', 'progress.json']:
            if (arm_dir / existing).exists():
                raise RuntimeError(f'replay output already exists for {arm}: {existing}')
        item = {'status': 'running', 'started_unix': time.time(),
                'attempt': 2 if arm == 'HIGH-SCORE' else 1,
                'prior_failed_attempt_archive': str(archive.resolve()) if arm == 'HIGH-SCORE' else None}
        status['arms'][arm] = item
        status['current_arm'] = arm
        if arm != 'HIGH-SCORE':
            status['attempted'].append(arm)
        save()
        remaining = deadline - time.monotonic()
        print(f'START {arm}; remaining={remaining:.1f}s', flush=True)
        arm_start = time.monotonic()
        with (arm_dir / 'console.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen([str(PYTHON), '-B', str(WORK / 'replay.py'), arm],
                                    cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
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
            status['status'] = 'replay_failure_after_instrumentation_fix'
            save()
            print(f'FAILED {arm} exit={code}', flush=True)
            break
        try:
            gate = comparison_gate(arm)
        except Exception as exc:
            item.update(status='qualification_failure', gate_error=repr(exc))
            status['status'] = 'protocol_failure_after_instrumentation_fix'
            save()
            print(f'GATE FAILED {arm}: {exc!r}', flush=True)
            break
        item.update(status='completed', gate=gate)
        status['completed'].append(arm)
        save()
        print(f'COMPLETE {arm}; wall={item["wall_seconds"]:.1f}s; remaining={status["remaining_budget_seconds"]:.1f}s', flush=True)
    else:
        status['status'] = 'completed'
        status['completed'] = ['HIGH-CONTROL', *ARMS_TO_RUN]
        status['uncompleted'] = []
        status['current_arm'] = None
        save()
        summarize_main()
    if status['status'] != 'completed':
        status['current_arm'] = None
        save()
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
