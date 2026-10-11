"""Saved-vector March quality and full fixed-time account; no independent new solves."""
import gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid34'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f]
D=load(OUT/'design.json');S=load(OUT/'manifest.json')['summaries']
pilot=[x for x in S if x['week']==D['weeks'][0]];assert len(pilot)==12
for p,h in load(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
comparisons=[]
for s in pilot:
    folder=OUT/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
    actual=rows(folder/'steps.jsonl.gz');assert len(actual)==s['steps']
    assert abs(math.fsum(r['raw_cost'] for r in actual)-s['cost'])<1e-6
    assert abs(math.fsum(r['decision_s'] for r in actual)-s['controller_s'])<1e-6
    assert sum(r['simulations'] for r in actual)==s['public_forecasts']
    assert s['illegal']==0 and s['ambiguous']==0
    baseline=next(x for x in pilot if x['repeat']==s['repeat'] and x['method']=='FULL')
    regret=(s['cost']-baseline['cost'])/max(1,abs(baseline['cost']))
    qualified=s['complete']==baseline['complete'] and s['steps']==baseline['steps'] and regret<=.005
    calls=[load(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]
    assert len(calls)==s['n1_calls'] and sum(c['public_queries'] for c in calls)==s['n1_forecasts']
    for c in calls:
        assert len({x['pool_id'] for x in c['outcomes']})==c['public_queries']
        valid=[x for x in c['outcomes'] if x['strict_admissible']]
        chosen=max(valid,key=lambda r:(__import__('struct').unpack('f',__import__('struct').pack('f',r['reward']))[0],-r['pool_id']))['pool_id'] if valid else None
        assert chosen==c['chosen_pool_id']
    comparisons.append({'repeat':s['repeat'],'method':s['method'],'controller_s':s['controller_s'],
        'n1_queries':s['n1_forecasts'],'n1_source_s':s['n1_source_s'],'cost':s['cost'],
        'cost_regret_to_full':regret,'steps':s['steps'],'complete':s['complete'],'quality_qualified':qualified,
        'exact_full_trajectory':s['exact_full_trajectory']})
result={'passed_saved_data':True,'all_rankers_qualified':all(c['quality_qualified'] for c in comparisons),
    'rows':comparisons,'warning':'Exposed March pilot2repeats; quality0.5%investment threshold,not equivalence. All controllers share caches. No independent refit/ACsolve.'}
(OUT/'pilot_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('PILOT AUDITED: '+('remainingqualified' if result['all_rankers_qualified'] else 'diagnosefailedpilotbeforeremaining')+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
