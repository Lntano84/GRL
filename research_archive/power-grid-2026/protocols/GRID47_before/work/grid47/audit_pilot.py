"""Own two-seed quality/positive-speed gate; baseline losses remain in the report."""
import gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid47'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
M=read(OUT/'manifest.json');assert len(M['summaries'])==6
for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
full=next(s for s in M['summaries'] if s['method']=='FULL');results=[]
for s in M['summaries']:
    folder=OUT/'runs'/f"r0__{s['week']}__{s['method']}"
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:r=[json.loads(l) for l in f]
    assert len(r)==s['steps'] and abs(math.fsum(x['raw_cost'] for x in r)-s['cost'])<1e-6
    assert s['illegal']==s['ambiguous']==0
    regret=(s['cost']-full['cost'])/max(1,abs(full['cost'])) if s['complete'] and full['complete'] else None
    sameh=s['steps']==full['steps'];speed=1-s['controller_s']/full['controller_s'] if sameh else None
    passed=bool(s['complete']==full['complete'] and s['steps']>=full['steps'] and regret is not None and regret<=.005)
    results.append({'method':s['method'],'steps':s['steps'],'complete':s['complete'],'cost_regret':regret,
       'passed_quality':passed,'controller_s':s['controller_s'],'time_reduction':speed,'queries':s['n1_forecasts']})
own=[r for r in results if r['method'] in ['LOCAL_GNN0','LOCAL_GNN1']];assert len(own)==2
gate=all(r['passed_quality'] and r['time_reduction'] is not None and r['time_reduction']>0 for r in own)
A={'passed_saved_engineering':True,'own_rankers_qualified':gate,'all_rankers_qualified':gate,
   'field_note':'Compatibility field all_rankers_qualified is the predeclared OWN two-seed continuation gate;not every baseline passing quality.',
   'results':results,'warning':'Single timing development pilot;remaining18 only when own2seeds quality<=.5%andpositive control reduction. All baseline losses retained;no bestseed selection.'}
(OUT/'pilot_audit.json').write_text(json.dumps(A,indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('PILOT '+('PASSED:remaining18permitted.' if gate else 'NOT PASSED:preserve6pilot outputs;no automatic remaining18.')+'\n',encoding='utf-8')
print(json.dumps(A,indent=2))
