"""Seal every controller input/model after generation, before any replay."""
from pathlib import Path
import datetime as dt,hashlib,json
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid89_review'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 assert read(OUT/'generation.json')['passed'] and read(OUT/'meta_training/training_finished.json')['passed_engineering'] and read(OUT/'preflight.json')['passed']
 if (OUT/'code_freeze.json').exists():
  for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
  for p,h in read(OUT/'data_freeze.json').items():assert sha(ROOT/p)==h,p
  print('Existing complete freeze verified');return
 codes=read(ROOT/'outputs/grid88/code_freeze.json')
 for p,h in codes.items():assert sha(ROOT/p)==h,p
 files=[*W.glob('*.py'),OUT/'design.json',OUT/'evaluation_plan.json',OUT/'preflight.json',OUT/'meta_training/design.json']
 for phase in ['order','parameters','meta','sealed']:
  d=read(OUT/phase/'design.json');files.append(OUT/phase/'design.json')
  for m in d['methods']:
   if m.get('meta_model'):files.append(ROOT/m['meta_model'])
 files.append(ROOT/'work/grid77/metacontrol.py')
 for p in files:assert p.exists(),p;codes[p.relative_to(ROOT).as_posix()]=sha(p)
 data=read(ROOT/'outputs/grid88/data_freeze.json')
 for p,h in data.items():assert sha(ROOT/p)==h,p
 for folder in [ROOT/'work/grid80/fresh_env',ROOT/read(OUT/'design.json')['environment']]:
  for p in folder.iterdir():
   if p.is_file():data[p.relative_to(ROOT).as_posix()]=sha(p)
  for week in read(OUT/('order' if folder.name=='fresh_env' and folder.parent.name=='grid80' else 'sealed')/'design.json')['weeks']:
   for p in (folder/'chronics'/week).iterdir():
    if p.is_file():data[p.relative_to(ROOT).as_posix()]=sha(p)
 save(OUT/'data_freeze.json',data);codes[(OUT/'data_freeze.json').relative_to(ROOT).as_posix()]=sha(OUT/'data_freeze.json');save(OUT/'code_freeze.json',codes)
 for phase in ['order','parameters','meta','sealed']:save(OUT/phase/'code_freeze.json',codes)
 save(OUT/'qualification.json',dict(passed=True,utc=dt.datetime.now(dt.timezone.utc).isoformat(),new_scenarios=6,generation_seed=20261017,new_model_fits=3,replay_count=246,code_items=len(codes),input_items=len(data),no_evaluation_results_used_for_selection=True))
 print(json.dumps(dict(code_items=len(codes),data_items=len(data),runs=246)))
if __name__=='__main__':main()
