import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid31'
assert json.loads((ROOT/'outputs/grid30/audit.json').read_text())['passed']
OUT.mkdir(parents=True,exist_ok=False)
D=json.loads((ROOT/'outputs/grid29c/design.json').read_text())
base=D['methods'][0]
methods=[{**base,'method':name,'cache_dc':dc,'cache_safe':safe} for name,dc,safe in
    [('ORIGINAL',False,False),('DC_ONLY',True,False),('DC_AND_SAFE',True,True)]]
D.update({'stage':'GRID31_SHARED_DC_SAFE_LIVE_PILOT','methods':methods,
    'purpose':'Measure added safe-QP compilation reuse using real delivered three arms, shared with all comparators; original and previous DC-only control retained.',
    'execution_order':[{**m,'repeat':repeat,'week':'2035-03-05_0'} for repeat,order in
        [(0,methods),(1,list(reversed(methods)))] for m in order],
    'quality_gate':'Every delivered action/observation/cost/forecast count must equal sealed protected FULL. Any mismatch stops stage before reporting speed.',
    'analysis':'Pre-specified all2balanced repeats/3comparisons. Report whole control, each solve phase/compile/backend/iterations, source/forecast, residual. No choosing fastest repeat or adding stage percentages.',
    'scope':'Exposed March only; sameGNN0,K128guard,cadence,solvers/tolerances/warm_startFalse. Safe-QP qualification passed153real calls/306reconstructed actions,maxerror4.33e-10MW. No graph-attribution claim.',
    'caps':{'episode_s':1200,'wall_s':2800,'physical_steps':12102,'public_forecasts':67950},
    'new_fits':0,'reserved_evaluations':0})
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
paths=['work/grid31/run.py','work/grid31/prepare.py','work/grid31/derive.py','work/grid30/cache_safe.py',
       'work/grid29/cache_dc.py','work/grid28/run.py']
(OUT/'code_freeze.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('PREPARED:6balanced three-arm exact-path measured pilot,no cached safe action yet deployed.\n',encoding='utf-8')
print(json.dumps({'prepared':True,'runs':6,'new_fits':0,'reserved_evaluations':0}))
