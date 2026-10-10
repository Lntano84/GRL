"""Offline manifest schema correction; never modifies source, policy, assets or runs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid02'
path=OUT/'adaptation_manifest.json'
backup=OUT/'adaptation_manifest_BEFORE_path_alias_fix.json'
assert not backup.exists()
raw=path.read_bytes()
manifest=json.loads(raw.decode('utf-8'))
migration=json.loads((OUT/'mapping_migration.json').read_text(encoding='utf-8'))
asset='assets/nn_act_space/action_12_unsafe_nn.npz'
assert manifest['hashes'][asset]==migration['adapted_sha256']
assert manifest['hashes'][asset.replace('/','\\')]==migration['source_sha256']
actual=hashlib.sha256((ROOT/'work/grid02/ljn_nn'/asset).read_bytes()).hexdigest()
assert actual==migration['adapted_sha256']
normalized={}
for key,value in manifest['hashes'].items():
    canonical=key.replace('\\','/')
    if canonical==asset:
        value=migration['adapted_sha256']
    if canonical in normalized:
        assert normalized[canonical]==value
    normalized[canonical]=value
for name,expected in normalized.items():
    assert hashlib.sha256((ROOT/'work/grid02/ljn_nn'/name).read_bytes()).hexdigest()==expected,name
backup.write_bytes(raw)
manifest['hashes']=normalized
manifest['manifest_schema_fix']='Normalize Windows path separators; remove duplicate NN mapping alias retaining the separately certified migrated hash. No asset or run changed.'
path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Manifest path aliases corrected; every actual source/asset hash verified. No simulations.')
