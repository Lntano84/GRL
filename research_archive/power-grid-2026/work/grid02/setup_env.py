import json
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[2]
DST = ROOT / 'work/grid02/.venv'
OUT = ROOT / 'outputs/grid02'
if DST.exists():
    raise RuntimeError('Refuse to overwrite environment')
venv.create(DST, with_pip=False)
layers = [ROOT / 'work/grid01/.venv/Lib/site-packages',
          ROOT / 'work/grid00/.venv/Lib/site-packages',
          Path('C:/Users/windows/Documents/Codex/2026-10-06/lg01-gnn-rins-rl-lp-20/work/lg01/.venv/Lib/site-packages')]
assert all(p.is_dir() for p in layers)
(DST / 'Lib/site-packages/grid02_layers.pth').write_text('\n'.join(map(str, layers)) + '\n', encoding='utf-8')
(OUT / 'environment_layers.json').write_text(json.dumps({'python': sys.version, 'read_only_layers_in_order': list(map(str, layers)), 'new_environment': str(DST),
 'packages_to_add': ['stable-baselines3==2.3.0', 'gymnasium==0.29.1', 'Farama-Notifications==0.0.4'],
 'note': 'Reuse torch 2.8.0+cpu from LG01 read-only; keep GRID01/GRID00 higher priority. SB3/gym versions match model archive.'}, indent=2), encoding='utf-8')
print('Created layered venv', DST)
