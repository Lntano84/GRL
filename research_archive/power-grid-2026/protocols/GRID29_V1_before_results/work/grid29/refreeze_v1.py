"""Preserve V0 and freeze mathematically equivalent numeric reciprocal revision."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid29'
assert (OUT/'revisions/v0_non_dpp/FAILURE.md').exists()
assert not (OUT/'shadow.json').exists()
paths=[ROOT/'work/grid29/cache_dc.py',ROOT/'work/grid29/shadow.py',ROOT/'work/grid29/prepare.py',ROOT/'work/grid02/ljn_nn/modules/convex_optim.py']
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')
D=json.loads((OUT/'design.json').read_text())
D['revision']='V1: numeric reciprocal keyed by actual reactance values; V0 real Parameter path failed DPP qualification before any cache trajectory.'
D['manual_checks']+=['real reactance Parameter mutation','triangle flow changes under impedance mutation']
D['reference']='https://www.cvxpy.org/tutorial/dpp/index.html : quotient expr/p is not DPP; represent inverse outside the DSL.'
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
(OUT/'revision_v1_freeze.json').write_text(json.dumps({'before_numeric_run':True,'change':D['revision'],'checks':D['manual_checks'],
    'files':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__).resolve()]}},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('V1 PREPARED: V0 non-DPP rejected before cached numerical comparison; no live cache trajectory yet.\n')
print(json.dumps({'frozen':True,'revision':D['revision']}))
