import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid30'
assert (OUT/'revisions/v0_parameter_setter/FAILURE.md').exists()
assert not (OUT/'qualification.json').exists()
D=json.loads((OUT/'design.json').read_text())
D['revision']='V1:replica arrays copied in-place exactly as original parameter updater; negative author values retained, no projection. Scalar values still use setter.'
D['manual_gate']+=' Plus exact unchanged copy of negative original array values.'
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
paths=['work/grid30/cache_safe.py','work/grid30/qualify.py','work/grid30/prepare.py',
    'work/grid30/refreeze_v1.py','work/grid29/cache_dc.py','work/grid28/run.py']
(OUT/'code_freeze.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('V1 PREPARED:no-clipping replica copy;V0setter failure retained, no cached safe control deployed.\n',encoding='utf-8')
print(json.dumps({'refrozen':True,'first_numeric_result':False}))
