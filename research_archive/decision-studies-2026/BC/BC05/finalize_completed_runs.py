"""Reconcile already-finished child replays after a coordinator log-read race."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import ARMS, WORK, one_stats, gate_control, gate_preintervention


def main():
    status_path = WORK/'run_status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    assert status['frozen_targets_sha256'] == json.loads((WORK/'frozen_manifest.json').read_text(encoding='utf-8'))['targets_sha256']
    for arm in ARMS:
        audit_path = WORK/arm/'run_audit.json'
        assert audit_path.exists(), f'child replay did not finish: {arm}'
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        assert audit['completed'] and audit['request_sequence'] == 147794
        one_stats(arm)
        entry = status['arms'].setdefault(arm, {})
        entry.update(status='completed', exit_code=0, wall_seconds=audit['wall_seconds'],
                     child_run_audit=str(audit_path))
    gates = {'BASE-CONTROL': gate_control('BASE-CONTROL'),
             'HIGH-CONTROL': gate_control('HIGH-CONTROL'),
             'BASE-CF': gate_preintervention('BASE-CF', 'BASE-CONTROL'),
             'HIGH-CF': gate_preintervention('HIGH-CF', 'HIGH-CONTROL')}
    start = float(status['started_unix'])
    completed_at = max(float(status['arms'][arm]['started_unix']) +
                       float(status['arms'][arm]['wall_seconds']) for arm in ARMS)
    experiment_elapsed = completed_at-start
    assert experiment_elapsed <= status['budget_seconds'], f'four replay processes completed after budget: {experiment_elapsed}'
    finalized_at = time.time()
    status.update(status='completed', completed=ARMS, uncompleted=[], current_arm=None,
                  wall_seconds=experiment_elapsed, remaining_budget_seconds=status['budget_seconds']-experiment_elapsed,
                  child_process_wall_seconds=sum(float(status['arms'][arm]['wall_seconds']) for arm in ARMS),
                  replay_completion_unix=completed_at, finalized_at_unix=finalized_at,
                  post_completion_finalize_delay_seconds=max(0.0, finalized_at-completed_at),
                  finalization='all four child replay processes completed; coordinator had exited on a transient partial progress.json read; no replay was restarted',
                  gates=gates)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': status['status'], 'wall_seconds': experiment_elapsed,
                      'completed': status['completed'], 'gates': gates}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
