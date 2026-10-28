from pathlib import Path
import json,shutil,hashlib,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,A,R=C.O,C.A,C.R
C.verify();assert C.read(O/'complete.json')['passed'] and C.read(O/'independent_audit.json')['passed'] and C.read(O/'delivery_check.json')['passed']
err=O/'engineering_error_delivery01.json';assert not err.exists();C.save(err.name,dict(type='NameError',detail='finalize.py final delivery copy references undefined W; all scientific analysis/audit/docs/collaboration already completed; delivery folder empty.',preserved_source='work/grid244_prior_distribution/finalize.py',source_sha256=C.sha(Path(__file__).parent/'finalize.py'),recovery='Copy only missing delivery files. Do not rerun analysis, audit, fitting, policy scores or full archives.'))
d=A/'delivery';assert d.exists() and not list(d.iterdir())
for p in [O/'CONTINUOUS_HANDOFF.md',O/'delivery_check.json',err,Path(__file__).parent/'finalize.py',Path(__file__)]:shutil.copy2(p,d/p.name)
(d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8');C.save('delivery_recovery01.json',dict(passed=True,no_new_analysis_fit_score=True,source_freeze_unchanged=True,delivered_files=len(list(d.iterdir()))));shutil.copy2(O/'delivery_recovery01.json',d/'delivery_recovery01.json');m=C.read(d/'archive_manifest.json');m['delivery_recovery01.json']=C.sha(d/'delivery_recovery01.json');(d/'archive_manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf8');print('DELIVERY_RECOVERED_ONLY',len(m))
