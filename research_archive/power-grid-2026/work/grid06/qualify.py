"""Zero-simulation qualification of the fixed source and previously sealed files."""
import ast
import hashlib
import json
from pathlib import Path
import time

started=time.perf_counter()
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid06'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
design=load(OUT/'design_freeze.json')
assert sha(OUT/'design_freeze.json')==(OUT/'design_freeze.sha256').read_text().strip()
for p,h in load(OUT/'reused_assets_frozen.json').items():assert sha(ROOT/p)==h,p
pre=load(ROOT/'outputs/grid05/preflight.json')
assert pre['passed'] and pre['physical_steps']==0 and pre['source_full_greedy_rewards_actions_and_selection_match']
for p in (ROOT/'work/grid06').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
text=(ROOT/'work/grid06/run_one.py').read_text(encoding='utf-8')
assert "choices=['NN20','FALLBACK90','NN352']" in text
assert "if args.policy=='NN352':" in text and 'agent.topo_12_unsafe.top_k=352' in text
assert 'force_expand_for_test=True' not in text
assert "install_rule(agent.topo_12_unsafe,agent.rho_safe,holder)" in text
raw=load(ROOT/'outputs/grid03/extraction_manifest.json')
raw_hash={x['path']:x['sha256'] for x in raw['files']}
root=ROOT/design['data_root'];files={}
for scene in design['selected_weeks']:
    folder=root/'chronics'/scene
    assert folder.is_dir()
    for p in sorted(folder.rglob('*')):
        if p.is_file():
            rel=p.relative_to(root).as_posix()
            assert rel in raw_hash and sha(p)==raw_hash[rel],rel
            files[str(p.relative_to(ROOT))]=sha(p)
for p in sorted(root.iterdir()):
    if p.is_file():files[str(p.relative_to(ROOT))]=sha(p)
(OUT/'data_files_frozen.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
result=dict(passed=True,physical_steps=0,native_simulates=0,selected_data_files_hash_checked=len(files),
    reused_GRID05_preflight=True,production_rule_unchanged=True,worker_AST_passed=True,wall_s=time.perf_counter()-started)
(OUT/'qualification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
