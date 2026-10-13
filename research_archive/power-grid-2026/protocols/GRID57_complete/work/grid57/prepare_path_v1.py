"""Recover an unstarted preflight: design lives in outputs, not work."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid57';HERE=Path(__file__).resolve().parent
assert OUT.exists() and not (OUT/'code_freeze.json').exists() and not (OUT/'manifest.json').exists() and not (OUT/'runs').exists()
source=(HERE/'prepare.py').read_text(encoding='utf-8')
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'design.json',HERE/'run.py']}
failure={'stage':'preflight_before_freeze','error':'Referenced work/grid37/design.json instead of outputs/grid37/design.json','new_solver_runs':0,'new_fits':0,'partial_design_and_run_hashes':before}
(OUT/'prepare_v0_path_error.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
for needle in ['assert not OUT.exists();OUT.mkdir()',"ROOT/'work/grid37/design.json'","assert not (HERE/'run.py').exists();(HERE/'run.py').write_text(source,encoding='utf-8')"]:
    assert source.count(needle)==1,needle
source=source.replace('assert not OUT.exists();OUT.mkdir()',"assert OUT.exists() and not (OUT/'code_freeze.json').exists()")
source=source.replace("ROOT/'work/grid37/design.json'","ROOT/'outputs/grid37/design.json'")
source=source.replace("assert not (HERE/'run.py').exists();(HERE/'run.py').write_text(source,encoding='utf-8')","assert (HERE/'run.py').read_text(encoding='utf-8')==source")
exec(compile(source,str(HERE/'prepare.py'),'exec'),{'__file__':str(HERE/'prepare.py'),'__name__':'__main__'})
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==before[p.name] for p in [OUT/'design.json',HERE/'run.py'])
print(json.dumps({'recovered_preflight':True,'existing_design_runner_unchanged':True,'new_runs':0}))
