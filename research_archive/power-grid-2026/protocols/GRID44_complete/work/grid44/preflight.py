"""Metadata/formula checks only; no reserved-data observation or model inference."""
import ast,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid44'
D=json.loads((OUT/'design.json').read_text());assert len(D['execution_order'])==96
assert len({(r['repeat'],r['week'],r['method']) for r in D['execution_order']})==96
assert all(r['cache_dc'] and r['cache_safe'] for r in D['execution_order'])
assert all(r['policy']=='PATIENCE64' for r in D['methods'] if r['method']!='FULL')
assert next(r for r in D['methods'] if r['method']=='FULL')['policy']=='PROTECTED'
for p,h in json.loads((OUT/'code_freeze.json').read_text()).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
worker=(HERE/'worker.py').read_text();assert 'historical =' not in worker and 'reference = None' in worker
assert 'tied_complete' not in (HERE/'run.py').read_text()
spec=importlib.util.spec_from_file_location('grid44_audit_preflight',HERE/'audit.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
assert a.mean([.1,.3])==.2 and a.interval([.1]*8)==[.1,.1]
assert a.key({'reward':1.0,'pool_id':2})>a.key({'reward':1.0,'pool_id':3})
assert not (OUT/'manifest.json').exists() and not (OUT/'extraction_complete.json').exists()
R={'passed':True,'runs':96,'families':8,'timing_repeats':2,'no_data_or_model_inference':True,'checks':'Unique matrix,common cache/guard,syntax,frozen hashes,no reference in controller,no tie-candidate adoption,bootstrap constant and exact tie ordering.'}
(OUT/'preflight.json').write_text(json.dumps(R,indent=2),encoding='utf-8');print(json.dumps(R))
