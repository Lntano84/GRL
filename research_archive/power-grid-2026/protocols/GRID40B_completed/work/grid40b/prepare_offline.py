"""Freeze downstream script only after training-only uniform config qualification."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).resolve().parent; OUT = ROOT / 'outputs/grid40b'
A = json.loads((OUT / 'audit.json').read_text(encoding='utf-8'))
assert A['passed_accounting'] and A['configuration_qualified']
src = (ROOT / 'work/grid40/offline.py').read_text(encoding='utf-8')
before = 'from adapter import OUT, Predictor, raw_features, read'
assert src.count(before) == 1
src = src.replace(before, "from adapter import OUT as BASE_OUT, Predictor, raw_features, read\nOUT = ROOT / 'outputs/grid40b'")
assert not (HERE / 'offline.py').exists(); (HERE / 'offline.py').write_text(src, encoding='utf-8')
(OUT / 'offline_code_freeze.json').write_text(json.dumps({'work/grid40b/offline.py': hashlib.sha256(src.encode('utf-8')).hexdigest()}, indent=2), encoding='utf-8')
print('Lower-lr uniform config selected by training fit; cached-state script frozen.')
