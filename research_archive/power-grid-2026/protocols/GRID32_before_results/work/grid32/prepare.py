"""Verify shared cache compatibility on all other exposed development weeks."""
import hashlib,json,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid32'
assert json.loads((ROOT/'outputs/grid31/review.json').read_text())['passed']
source=(ROOT/'work/grid31/run.py').read_text(encoding='utf-8')
assert source.count("OUT=ROOT/'outputs/grid31'")==1
source=source.replace("OUT=ROOT/'outputs/grid31'","OUT=ROOT/'outputs/grid32'")
source=source.replace('GRID31 three measured arms; derived from retained GRID29C. Original/DC/DC+safe.',
    'GRID32 remaining exposed dates,2arms; derived from frozen GRID31 shared cache pilot.')
ast.parse(source)
assert not (ROOT/'work/grid32/run.py').exists();(ROOT/'work/grid32/run.py').write_text(source,encoding='utf-8')
OUT.mkdir(parents=True,exist_ok=False)
D=json.loads((ROOT/'outputs/grid31/design.json').read_text())
weeks=['2035-06-25_8','2035-09-03_4','2035-10-01_5']
methods=[D['methods'][0],D['methods'][2]]
D.update({'stage':'GRID32_SHARED_CACHE_EXPOSED_WEEK_COMPATIBILITY','weeks':weeks,'methods':methods,
    'execution_order':[{**m,'repeat':0,'week':week} for i,week in enumerate(weeks)
        for m in (list(reversed(methods)) if i%2==0 else methods)],
    'purpose':'Check whether exact control/cost compatibility of shared DC+safeQP reuse holds beyond March, across three previously exposed dev weeks; report date-specific gain without training/test-date changes.',
    'quality_gate':'Each original/cached path must equal sealed corresponding FULL actions/observations/cost; retain shared June failure at1901, no earlier/new failure. Any mismatch stops before speed claim.',
    'analysis':'All3date pairs, single paired repeat. Report per-date/pooled control/component sums, same physical paths, actual cache reuse and actual native solver states/iterations. March prior2repeats not merged as independent dates.',
    'scope':'Other3exposed development dates, same fixed GNN0/K128protected guard/cadence. Two overlap original training; no independent final generalization. Shared optimization must be applied to all strong comparators.',
    'caps':{'episode_s':1200,'wall_s':3000,'physical_steps':12500,'public_forecasts':90000},
    'new_fits':0,'reserved_evaluations':0})
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
paths=['work/grid32/run.py','work/grid32/prepare.py','work/grid30/cache_safe.py',
       'work/grid29/cache_dc.py','work/grid28/run.py']
(OUT/'code_freeze.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('PREPARED:6original/shared-cache runs on3other exposed weeks; no new fitting/final evaluation.\n',encoding='utf-8')
print(json.dumps({'prepared':True,'weeks':weeks,'runs':6,'new_fits':0,'reserved_evaluations':0}))
