"""Pre-outcome correction: use already-extracted monthly validation subset."""
import json, shutil, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/grid08'
assert not (OUT/'preoutcome_revision.json').exists()
shutil.copy2(OUT/'design.json',OUT/'design_v0_UNRUN.json')
shutil.copy2(ROOT/'work/grid08/freeze.py',OUT/'freeze_v0_UNRUN.py')
d=json.loads((OUT/'design.json').read_text())
s=json.loads((ROOT/'outputs/grid03/scenario_split_frozen.json').read_text())
d['evaluation_weeks']=[next(x for x in s['extracted_subsets']['validation'] if x.startswith(f'2035-{m:02}-')) for m in [2,5,8,11]]
d['ppo']['blackout_penalty']=1000.
d['revision']='Before any simulation or fit: original selector incorrectly used entire archive split, including unextracted files. Use original monthly extracted validation list. Terminal penalty increased before outcomes to put survival ahead of avoiding cost by early termination.'
assert all((ROOT/'work/grid03/formal_env_initialized/chronics'/x).is_dir() for x in d['training_weeks']+d['evaluation_weeks'])
(OUT/'design.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
(OUT/'preoutcome_revision.json').write_text(json.dumps({'before_outcomes':True,'physical_steps':0,'model_fits':0,'evaluation_weeks':d['evaluation_weeks'],'design_sha256':hashlib.sha256((OUT/'design.json').read_bytes()).hexdigest()},indent=2))
print(d['evaluation_weeks'])
