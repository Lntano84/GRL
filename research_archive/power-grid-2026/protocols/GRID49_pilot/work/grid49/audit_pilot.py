"""Five-arm old-March comparison;never select only the better neural seed."""
import gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid49'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
M=read(OUT/'manifest.json');assert len(M['summaries'])==5
for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
full=next(s for s in M['summaries'] if s['method']=='FULL');results=[]
for s in M['summaries']:
    folder=OUT/'runs'/f"r0__{s['week']}__{s['method']}"
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:r=[json.loads(l) for l in f]
    assert len(r)==s['steps'] and abs(math.fsum(x['raw_cost'] for x in r)-s['cost'])<1e-6 and s['illegal']==s['ambiguous']==0
    regret=(s['cost']-full['cost'])/max(1,abs(full['cost'])) if s['steps']==full['steps'] else None
    passed=bool(s['complete']==full['complete'] and s['steps']>=full['steps'] and regret is not None and regret<=.005)
    results.append({'method':s['method'],'steps':s['steps'],'complete':s['complete'],'cost_regret':regret,'passed_quality':passed,
       'controller_s':s['controller_s'],'time_reduction_full':1-s['controller_s']/full['controller_s'] if s['steps']==full['steps'] else None,'queries':s['n1_forecasts']})
own=[r for r in results if r['method'].startswith('FREE_GNN')];assert len(own)==2
gate=all(r['passed_quality'] for r in own)
A={'passed_saved_engineering':True,'all_rankers_qualified':gate,'both_new_seeds_qualified':gate,'results':results,
   'warning':'Continuation of15olddevelopment trajectories only if BOTH newseeds meet original.5%quality/noearlierfailure gate. No seed selection,GRID44reuse or criterion adjustment;oldfastmodels remain default.'}
(OUT/'pilot_audit.json').write_text(json.dumps(A,indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('PILOT '+('PASSED:15olddevelopment permitted.' if gate else 'NOT PASSED:preserve5runs;no automatic remaining15;oldfastdefault unchanged.')+'\n',encoding='utf-8')
print(json.dumps(A,indent=2))
