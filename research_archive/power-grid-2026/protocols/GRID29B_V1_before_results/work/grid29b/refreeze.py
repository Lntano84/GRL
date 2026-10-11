"""Script-name repair only, before first control/probe result."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid29b'
assert (OUT/'revisions/v0_stdlib_collision/FAILURE.md').exists()
assert not (OUT/'profile.json').exists()
paths=[ROOT/'work/grid29b/probe_dc.py',ROOT/'work/grid29b/prepare.py',ROOT/'work/grid29/cache_dc.py',ROOT/'work/grid28/run.py']
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')
D=json.loads((OUT/'design.json').read_text());D['revision']='V1: rename profile.py to probe_dc.py to avoid Python stdlib shadowing; no numerical or controller logic change.'
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('V1 PREPARED: initialization name collision preserved; no numeric/control outcome yet.\n')
print(json.dumps({'refrozen':True,'revision':D['revision']}))
