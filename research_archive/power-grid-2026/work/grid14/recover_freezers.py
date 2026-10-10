"""Recover two previous generator texts; only retain if original frozen SHA256 matches."""
import json, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid14'
current = (ROOT / 'work/grid14/freeze.py').read_text(encoding='utf-8')
v1_revision = "          'engineering_revision': 'V1 after preserved raw-byte copy-assertion failure: CVXPY changes scalar 15 int64 to 15.0 float64, with identical numeric values. Canonical numeric hashing plus exact numeric comparisons; candidate rules/weeks/budgets unchanged.',"
v1 = '\n'.join(v1_revision if "'engineering_revision':" in line else line
               for line in current.split('\n') if "'continuous_gate_atol':" not in line)
v1 = v1.replace('Exact discrete shadow-original action plus frozen numerical/forecast gates before interpreting outcomes.',
                'Exact shadow-original delivered-action gate before interpreting outcomes.')
v1 = v1.replace("ROOT / 'work/grid14/check_clone.py', ROOT / 'work/grid14/diagnose_delivery.py', ROOT / 'work/grid08/common.py'",
                "ROOT / 'work/grid14/check_clone.py', ROOT / 'work/grid08/common.py'")
v0 = '\n'.join(line for line in v1.split('\n') if "'engineering_revision':" not in line)
v0 = v0.replace("ROOT / 'work/grid14/check_clone.py', ROOT / 'work/grid08/common.py'", "ROOT / 'work/grid08/common.py'")
for folder, text in [('invalid_attempt_01', v0), ('invalid_attempt_02', v1)]:
    frozen = json.loads((OUT / folder / 'code_freeze.json').read_text())
    expected = frozen[str(Path('work/grid14/freeze.py'))]
    data = text.encode('utf-8'); actual = hashlib.sha256(data).hexdigest()
    assert actual == expected, (folder, actual, expected)
    (OUT / folder / 'freeze.py').write_bytes(data)
    assert hashlib.sha256((OUT / folder / 'run_probe.py').read_bytes()).hexdigest() == frozen[str(Path('work/grid14/run_probe.py'))]
    assert hashlib.sha256((OUT / folder / 'design.json').read_bytes()).hexdigest() == frozen[str(Path('outputs/grid14/design.json'))]
print('ARCHIVED_GENERATORS_MATCH_ORIGINAL_FROZEN_SHA256')
