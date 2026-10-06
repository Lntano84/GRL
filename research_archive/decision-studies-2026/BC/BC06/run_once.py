"""Run the single authorized BC06 HIGH-CONTROL replay with a hard wall cap."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parent
BASE = WORK.parents[1]
ARM = WORK / 'HIGH-CONTROL'
PYTHON = BASE / 'work/bc01/python311-embed/python.exe'
REPO = BASE / 'work/bc01/Baleen-FAST24'
BUDGET_SECONDS = 1800


def save(state):
    state['wall_seconds'] = time.monotonic() - state['_started_monotonic']
    state['remaining_budget_seconds'] = max(0.0, BUDGET_SECONDS - state['wall_seconds'])
    (WORK / 'run_status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    preflight = json.loads((WORK / 'preflight.json').read_text(encoding='utf-8'))
    assert preflight['passed'] and preflight['replay_count_authorized'] == 1
    assert PYTHON.is_file() and (WORK / 'replay.py').is_file()
    if (ARM / 'run_audit.json').exists() or list((ARM / 'raw').rglob('*.stats.lzma')) if (ARM / 'raw').exists() else False:
        raise RuntimeError('active BC06 arm has prior replay output; refusing duplicate')
    started_mono = time.monotonic()
    state = {'status': 'running', 'arm': 'HIGH-CONTROL', 'budget_seconds': BUDGET_SECONDS,
             'attempts': 1, 'started_unix': time.time(), '_started_monotonic': started_mono}
    save(state)
    env = dict(os.environ)
    env['PATH'] = 'C:/Program Files/Git/usr/bin;' + env.get('PATH', '')
    env['PYTHONHASHSEED'] = '0'
    if not any((Path(p) / 'md5sum.exe').exists() for p in env['PATH'].split(os.pathsep) if p):
        raise RuntimeError('Git md5sum.exe missing from replay PATH')
    log_path = ARM / 'console.log'
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([str(PYTHON), '-B', str(WORK / 'replay.py')], cwd=str(REPO),
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            remaining = BUDGET_SECONDS - (time.monotonic() - started_mono)
            if remaining <= 0:
                proc.kill()
                proc.wait()
                state.update(status='budget_exhausted', exit_code=proc.returncode)
                save(state)
                break
            time.sleep(min(30.0, remaining))
            progress_path = ARM / 'progress.json'
            if progress_path.exists():
                try:
                    state['latest_progress'] = json.loads(progress_path.read_text(encoding='utf-8'))
                except (OSError, json.JSONDecodeError):
                    pass
            save(state)
        else:
            code = proc.returncode
            audit_path = ARM / 'run_audit.json'
            audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else None
            if code == 0 and audit and audit.get('completed') is True:
                state.update(status='completed', exit_code=0, replay_audit=str(audit_path),
                             replay_wall_seconds=audit.get('recorder', {}).get('wall_seconds'))
            else:
                state.update(status='failed', exit_code=code, replay_audit=audit,
                             console_tail=log_path.read_text(encoding='utf-8', errors='replace')[-6000:])
            save(state)
    state.pop('_started_monotonic', None)
    (WORK / 'run_status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in state.items() if k != 'console_tail'}, ensure_ascii=False, indent=2))
    if state['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
