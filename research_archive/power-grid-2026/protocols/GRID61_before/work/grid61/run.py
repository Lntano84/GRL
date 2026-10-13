"""One 32-run supplemental timing repetition; original GRID55 remains immutable."""
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid61'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    temp = p.with_suffix(p.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, p)


def verify():
    for p, digest in read(OUT / 'code_freeze.json').items():
        assert sha(ROOT / p) == digest, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    verify()
    design = read(OUT / 'design.json')
    if args.worker is not None:
        # Execute the already-frozen runner's worker branch verbatim, except its
        # output directory. Its original worker/controller/timers are unchanged.
        source_path = ROOT / 'work/grid55/run.py'
        source = source_path.read_text(encoding='utf-8')
        old = "OUT=ROOT/'outputs/grid55'"
        assert source.count(old) == 1
        source = source.replace(old, "OUT=ROOT/'outputs/grid61'")
        marker = '    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose'
        assert source.count(marker)==1
        patch = "    sys.path.insert(0,str(ROOT/'work/grid61'))\n    from mechanism_adapter import install_mechanism_adapter\n    install_mechanism_adapter(engine.Search)\n"
        source=source.replace(marker,patch+marker)
        sys.argv = [str(source_path), '--worker', str(args.worker)]
        exec(compile(source, str(source_path), 'exec'),
             {'__file__': str(source_path), '__name__': '__main__'})
        return
    assert not (OUT / 'finished.json').exists(), 'Supplement already complete; never repeat.'
    assert not (OUT / 'RUN.lock').exists(), 'Live or unresolved prior runner lock.'
    with (OUT / 'RUN.lock').open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()))
        f.flush()
        os.fsync(f.fileno())
    start = time.perf_counter()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    summaries = read(OUT / 'manifest.json')['summaries'] if (OUT / 'manifest.json').exists() else []
    existing = {(s['repeat'], s['week'], s['method']) for s in summaries}
    # Scoped execution request: no power-plan edits, restored even after error.
    lease = ctypes.windll.kernel32.SetThreadExecutionState
    lease.argtypes = [ctypes.c_uint32]
    lease.restype = ctypes.c_uint32
    lease_previous = lease(0x80000003)  # CONTINUOUS | SYSTEM_REQUIRED | DISPLAY_REQUIRED
    assert lease_previous, 'Cannot acquire bounded awake lease.'
    save(OUT / 'runtime.json', {'started_utc': started, 'pid': os.getpid(),
         'awake_lease_previous': int(lease_previous), 'permanent_power_plan_changed': False})
    (OUT / 'RUN_STATE.md').write_text('RUNNING: one serial 32-run repetition with scoped awake lease.\n', encoding='utf-8')
    try:
        for i, item in enumerate(design['execution_order']):
            ident = (item['repeat'], item['week'], item['method'])
            if ident in existing:
                continue
            folder = OUT / 'runs' / f"r{item['repeat']}__{item['week']}__{item['method']}"
            result = OUT / f'worker_result_{i:02d}.json'
            # Recover a fully persisted child receipt; never blindly rerun it.
            if result.exists() and (folder / 'components.json').exists():
                summary = read(result)
                assert summary == read(folder / 'summary.json')
            else:
                assert not folder.exists() and not result.exists(), 'Partial child requires review.'
                remaining = design['caps']['wall_s'] - (time.perf_counter() - start)
                assert remaining > 0
                with (OUT / f'worker_{i:02d}.log').open('x', encoding='utf-8') as f:
                    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(i)],
                        cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                        timeout=min(design['caps']['episode_s'] + 60, remaining),
                        creationflags=subprocess.CREATE_NO_WINDOW)
                assert proc.returncode == 0, (i, proc.returncode)
                summary = read(result)
            summaries.append(summary)
            existing.add(ident)
            save(OUT / 'manifest.json', {'planned': 32, 'summaries': summaries})
            print(json.dumps({'completed': len(summaries), 'planned': 32, 'method': item['method'],
                              'control_s': summary['controller_s'], 'steps': summary['steps']}), flush=True)
        assert len(summaries) == 32
        assert sum(s['steps'] for s in summaries) <= design['caps']['physical_steps']
        assert sum(s['public_forecasts'] for s in summaries) <= design['caps']['public_forecasts']
        save(OUT / 'finished.json', {'passed_engineering': True, 'summaries': summaries,
             'started_utc': started, 'ended_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'wall_s': time.perf_counter() - start, 'new_fits': 0})
        (OUT / 'RUN_STATE.md').write_text('32 RUNS COMPLETE: power-log, saved-data and timing-view audit pending.\n', encoding='utf-8')
    except BaseException:
        import traceback
        save(OUT / 'runner_error.json', {'traceback': traceback.format_exc(), 'completed': len(summaries)})
        (OUT / 'RUN_STATE.md').write_text('STOPPED: preserve partial data; no automatic repeat loop.\n', encoding='utf-8')
        raise
    finally:
        returned = lease(0x80000000)
        save(OUT / 'awake_lease_release.json', {'returned': int(returned),
             'released_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
        (OUT / 'RUN.lock').unlink()


if __name__ == '__main__':
    main()
