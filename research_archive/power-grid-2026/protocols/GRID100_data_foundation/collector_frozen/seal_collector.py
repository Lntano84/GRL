"""Freeze qualified pilot runtime; final test and bulk teacher execution disabled."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid100_data_foundation'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert read(OUT/'collector_qualification.json')['passed'];assert not (OUT/'collector_code_freeze.json').exists()
    D=read(OUT/'design.json');pilot=D['pilot'];runtime=OUT/'teacher_runtime';runtime.mkdir(exist_ok=True)
    item=dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True,policy='FULL',order_mode='STATIC_ID',repeat=0,stage_phase='pilot',week=pilot['scenario'],method='FULL_TEACHER',partition='train')
    design=dict(execution_order=[item],timing='Data collection only: complete paid teacher feedback. No method speed comparison; all logging/setup/forecast costs retained.',caps=dict(episode_s=1200,wall_s=1320,physical_steps=2017,public_forecasts=200000))
    (runtime/'design.json').write_text(json.dumps(design,indent=2)+'\n',encoding='utf8')
    codes=read(ROOT/'outputs/grid91_mechanism/code_freeze.json')
    for n,h in codes.items():assert sha(ROOT/n)==h,n
    paths=list(W.glob('*.py'))+[OUT/'design.json',OUT/'split_manifest.json',OUT/'collector_qualification.json',runtime/'design.json']
    paths+=[ROOT/n for n in ['work/grid89_review/adapter.py','work/grid89_review/policies.py','work/grid55/run.py','work/grid55/worker.py']]
    codes.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    (OUT/'collector_code_freeze.json').write_text(json.dumps(codes,indent=2)+'\n',encoding='utf8');(runtime/'code_freeze.json').write_text(json.dumps(codes,indent=2)+'\n',encoding='utf8')
    archive=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID100_data_foundation/collector_frozen';archive.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+[OUT/'collector_qualification.json',OUT/'collector_code_freeze.json',runtime/'design.json']:shutil.copy2(p,archive/p.name)
    print('Pilot collector frozen',len(codes),'items; no physical replay yet')
if __name__=='__main__':main()
