"""Read-only resume qualification; no inference, model change, or new episode."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid44'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'finished.json').exists()
S=read(OUT/'shutdown_checkpoint.json');M=read(OUT/'manifest.json');D=read(OUT/'design.json')
assert S['completed_runs']==len(M['summaries'])==36
assert S['remaining_runs']==60 and len(D['execution_order'])==96
for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
for r in S['artifact_hashes']:assert (ROOT/r['path']).stat().st_size==r['bytes'] and sha(ROOT/r['path'])==r['sha256'],r['path']
E=read(OUT/'extraction_complete.json')
for r in E['files']:assert (ROOT/r['path']).stat().st_size==r['bytes'] and sha(ROOT/r['path'])==r['sha256'],r['path']
for i,s in enumerate(M['summaries']):
    assert all(s[k]==D['execution_order'][i][k] for k in ['repeat','week','method','checkpoint','rule','policy'])
    assert read(OUT/f'worker_result_{i:02d}.json')==s
assert not (OUT/'worker_result_36.json').exists()
R={'passed':True,'resume_date':'2026-10-12','completed_unchanged':36,'pending':60,
   'trajectory_artifacts_checked':len(S['artifact_hashes']),'chronics_files_checked':len(E['files']),
   'frozen_scientific_code_unchanged':True,'new_inference_or_fit':0,
   'timing_note':'Resume after requested shutdown;compare methods within same date/repeat block. finished.wall_s measures resumed dispatcher only;sum run.wall_s for complete serial-worker accounting.'}
(OUT/'resume_qualification.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
print(json.dumps(R))
