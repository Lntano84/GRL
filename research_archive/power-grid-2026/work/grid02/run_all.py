"""Serial, checkpointed runs. Qualification must pass; no automatic retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid02'
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['smoke', 'full'], required=True)
args = parser.parse_args()
prior_wall = 0.0
qualification = json.loads((OUT / 'preflight.json').read_text(encoding='utf-8'))
assert qualification['status'] == 'PASS_NO_PHYSICAL_STEPS'
if args.phase == 'full':
    smokes = json.loads((OUT / 'run_manifest_smoke.json').read_text(encoding='utf-8'))
    assert len(smokes) == 2 and all(x['exit'] == 0 for x in smokes)
    prior_wall = sum(x['wall_s'] for x in smokes)
    for scenario in [0, 1]:
        rows = [json.loads(x) for x in (OUT / f'NN20_scenario{scenario}_smoke/steps.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows) == 24
        assert all(not(x['done'] or x['illegal'] or x['ambiguous'] or x['exceptions']) for x in rows)
plan = [('NN20', 0), ('NN20', 1)] if args.phase == 'smoke' else [('FULL', 0), ('NN20', 0), ('NN352', 0), ('FULL', 1), ('NN20', 1), ('NN352', 1)]
manifest_path = OUT / f'run_manifest_{args.phase}.json'
assert not manifest_path.exists(), 'Refuse to repeat existing phase'
snapshot = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'work/grid02').rglob('*.py'))
            if '.venv' not in p.parts and '__pycache__' not in p.parts and 'tmp' not in p.parts}
(OUT / f'run_code_freeze_{args.phase}.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
started = time.monotonic()
records = []
for policy, scenario in plan:
    dst = OUT / f'{policy}_scenario{scenario}_{args.phase}'
    assert not dst.exists(), 'Never overwrite or silently retry a trajectory'
    dst.mkdir()
    ready = dst / 'import_ready.json'
    t = time.monotonic()
    deadline = t + 180
    seen = False
    environ = dict(os.environ)
    environ.update({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                    'MPLCONFIGDIR': str(ROOT / 'work/grid02/mplconfig')})
    with (dst / 'console.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, str(ROOT / 'work/grid02/run_one.py'), '--scenario', str(scenario), '--mode', args.phase, '--policy', policy], stdout=log, stderr=subprocess.STDOUT, env=environ)
        while process.poll() is None:
            if ready.exists() and not seen:
                seen = True
                deadline = time.monotonic() + 900
            if time.monotonic() > deadline or time.monotonic() - started + prior_wall > 3600:
                process.kill()
                process.wait()
                (dst / 'watchdog_timeout.json').write_text(json.dumps({'import_seen': seen, 'wall_s': time.monotonic()-t}), encoding='utf-8')
                raise TimeoutError('Frozen runtime cap exceeded')
            time.sleep(.1)
    records.append({'scenario': scenario, 'policy': policy, 'mode': args.phase, 'exit': process.returncode, 'wall_s': time.monotonic()-t})
    manifest_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(records[-1], flush=True)
    if process.returncode:
        sys.exit(process.returncode)
print('PHASE_COMPLETE', args.phase, time.monotonic()-started, flush=True)
