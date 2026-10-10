"""Seal completed outputs only after all their writers close."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid08'
assert json.loads((OUT/'completion_supervisor_finished.json').read_text())['passed']
for v in ['v0','v1','v2']:assert json.loads((OUT/v/'audit.json').read_text())['passed']
assert json.loads((OUT/'resume_audit.json').read_text())['passed']
assert json.loads((OUT/'guard_qualification/audit.json').read_text())['passed']
assert json.loads((OUT/'critic_capacity_probe/audit.json').read_text())['passed']
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
manifest.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'work/grid08').iterdir() if p.is_file() and p.suffix in ['.py','.md']})
(OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
for p,h in manifest.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert size<=1073741824
print(json.dumps({'passed':True,'files_checked':len(manifest),'output_bytes_including_manifest':size,
                  'physical_step_upper_bound':json.loads((OUT/'delivery.json').read_text())['resources']['cumulative_physical_steps_including_qualification']}))
