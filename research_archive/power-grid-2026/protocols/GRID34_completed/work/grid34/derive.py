"""Minimal, asserted derivation of frozen shared-cache actual-control driver."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=(ROOT/'work/grid32/run.py').read_text(encoding='utf-8')
replacements={
 '"""GRID32 remaining exposed dates,2arms; derived from frozen GRID31 shared cache pilot."""':'"""GRID34 common-cache FULL and all existing rankers; staged quality pilot, resumable."""',
 "OUT=ROOT/'outputs/grid32'":"OUT=ROOT/'outputs/grid34'",
 "parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()":"parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);parser.add_argument('--phase',choices=['pilot','remaining','all'],default='pilot');args=parser.parse_args()",
 "summary=worker.worker(item,D);assert summary['exact_full_trajectory']":"summary=worker.worker(item,D)\n        if item['rule']=='FULL':assert summary['exact_full_trajectory']",
 "    assert not (OUT/'finished.json').exists()\n    stamp=time.perf_counter();summaries=[]":"    assert not (OUT/'finished.json').exists()\n    stamp=time.perf_counter();summaries=load(OUT/'manifest.json')['summaries'] if (OUT/'manifest.json').exists() else []\n    existing={(s['repeat'],s['week'],s['method']) for s in summaries}\n    if args.phase=='remaining':assert load(OUT/'pilot_audit.json')['all_rankers_qualified']",
 "    (OUT/'RUN_STATE.md').write_text('RUNNING:one serial fresh worker;6balanced original/DC/DC+safe component profiles.\\n',encoding='utf-8')":"    (OUT/'RUN_STATE.md').write_text('RUNNING:one serial fresh worker; all6arms shareDC+safe cache;phase='+args.phase+'\\n',encoding='utf-8')",
 "    for index,item in enumerate(D['execution_order']):\n        assert time.perf_counter()-stamp<D['caps']['wall_s']":"    for index,item in enumerate(D['execution_order']):\n        if args.phase!='all' and item['stage_phase']!=args.phase:continue\n        if (item['repeat'],item['week'],item['method']) in existing:continue\n        assert time.perf_counter()-stamp<D['caps']['wall_s']",
 "{'summaries':summaries,'planned':6}":"{'summaries':summaries,'planned':len(D['execution_order'])}",
 "'planned':6":"'planned':len(D['execution_order'])",
 "    assert sum(s['steps'] for s in summaries)<=D['caps']['physical_steps']":"    if len(summaries)<len(D['execution_order']):\n        save(OUT/'phase_complete.json',{'phase':args.phase,'runs':len(summaries),'passed_engineering':True})\n        (OUT/'RUN_STATE.md').write_text('PILOT RUNS COMPLETE: saved-data quality gate required before remaining.\\n',encoding='utf-8')\n        sys.exit(0)\n    assert sum(s['steps'] for s in summaries)<=D['caps']['physical_steps']"
}
for old,new in replacements.items():
    assert source.count(old)==1,(old[:120],source.count(old))
    source=source.replace(old,new)
dest=ROOT/'work/grid34/run.py';assert not dest.exists();dest.write_text(source,encoding='utf-8')
print('Derived GRID34 driver without changing historical GRID32 code.')
