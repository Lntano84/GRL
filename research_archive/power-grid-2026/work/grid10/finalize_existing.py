"""Resume reporting after orchestration argument error, without simulator calls."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid10'
start = time.perf_counter()
assert not (OUT / 'finished.json').exists()
for version in ['raw', 'normalized']:
    for phase in ['train', 'evaluate']:
        assert json.loads((OUT / version / phase / 'gnn/finished.json').read_text())['passed']
for script, args in [('audit.py', ['--version', 'raw']),
                     ('audit.py', ['--version', 'normalized']), ('report.py', []),
                     ('training_diagnostic.py', []), ('compare_training.py', [])]:
    label = args[-1] if args else 'combined'
    with (OUT / f'{label}_{Path(script).stem}.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, str(ROOT / 'work/grid10' / script), *args],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'completed': script, 'label': label}), flush=True)
(OUT / 'finished.json').write_text(json.dumps({
    'passed': True, 'report_resume_wall_s': time.perf_counter() - start,
    'orchestration_error': 'finish.py retained GRID09 audit version names; argparse rejected them before audit. Training and evaluation had already completed. Only reporting resumed.',
    'additional_physical_steps': 0, 'additional_fits': 0,
}, indent=2), encoding='utf-8')
subprocess.run([sys.executable, str(ROOT / 'work/grid10/neural_audit.py')], cwd=ROOT, check=True)
