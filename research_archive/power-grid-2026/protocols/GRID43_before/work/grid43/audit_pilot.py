"""Bounded engineering/quality gate, not a new method or statistical claim."""
import gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid43'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
M=read(OUT/'manifest.json');D=read(OUT/'design.json');assert len(M['summaries'])==6
for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
full=next(s for s in M['summaries'] if s['method']=='FULL');results=[]
for s in M['summaries']:
    p=OUT/'runs'/f"r0__{s['week']}__{s['method']}"
    with gzip.open(p/'steps.jsonl.gz','rt',encoding='utf-8') as f:r=[json.loads(t) for t in f]
    assert len(r)==s['steps'] and abs(math.fsum(x['raw_cost'] for x in r)-s['cost'])<1e-6
    assert s['illegal']==s['ambiguous']==0 and all(not x['illegal'] and not x['ambiguous'] for x in r)
    cost=(s['cost']-full['cost'])/max(1,abs(full['cost'])) if s['steps']==full['steps'] else None
    passed=bool(s['complete']==full['complete'] and s['steps']>=full['steps'] and cost is not None and cost<=.005)
    results.append({'method':s['method'],'steps':s['steps'],'complete':s['complete'],'cost_regret':cost,'passed':passed,
                    'controller_s':s['controller_s'],'queries':s['n1_forecasts']})
A={'passed_saved_engineering':True,'all_rankers_qualified':all(x['passed'] for x in results),'results':results,
   'warning':'6March pilot trajectories; .5%quality screening,not equivalence; no seed selection or final dates.'}
(OUT/'pilot_audit.json').write_text(json.dumps(A,indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('PILOT QUALITY '+('PASSED:remaining18 permitted.' if A['all_rankers_qualified'] else 'NOT PASSED:retain outputs; no automatic remaining runs.')+'\n',encoding='utf-8')
print(json.dumps(A,indent=2))
