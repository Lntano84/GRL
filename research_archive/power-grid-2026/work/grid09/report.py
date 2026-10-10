"""Compare final policies with continued PPO and immutable fixed restoration."""
import json,gzip,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid09';D=json.loads((OUT/'design.json').read_text())
records=[]
for version in ['physical','decision']:
    audit=json.loads((OUT/version/'audit.json').read_text());assert audit['passed']
    records.extend({**r,'version':version} for r in audit['records'] if '/evaluate/' in r['path'].replace('\\','/'))
base=json.loads((ROOT/'outputs/grid08/v0/evaluate/baselines/manifest.json').read_text())
v2=json.loads((ROOT/'outputs/grid08/v2/evaluate/gnn/manifest.json').read_text())
reference={name:{r['scenario']:r for r in base if r['policy']==name} for name in ['NN20','ALWAYS_RESTORE','IMMEDIATE_COST']}
reference['V2_GNN']={r['scenario']:r for r in v2}
reference['PHYSICAL_CONTINUE']={r['scenario']:r for r in records if r['version']=='physical'}
comparisons=[]
def trajectory(item):
    return [(r['action_hash'],r['after_hash']) for r in (json.loads(x) for x in gzip.open(ROOT/item['path']/'steps.jsonl.gz','rt',encoding='utf-8'))]
for version in ['physical','decision']:
    models={r['scenario']:r for r in records if r['version']==version}
    for name,refs in reference.items():
        if version=='physical' and name=='PHYSICAL_CONTINUE':continue
        pairs=[]
        for week in D['evaluation_weeks']:
            a,b=models[week],refs[week]
            pairs.append({'week':week,'model_steps':a['steps'],'reference_steps':b['steps'],
                'model_complete':a['complete'],'reference_complete':b['complete'],
                'gain':(b['raw_cost']-a['raw_cost'])/abs(b['raw_cost']) if a['complete'] and b['complete'] else None,
                'identical_trajectory':trajectory(a)==trajectory(b)})
        values=[r['gain'] for r in pairs if r['gain'] is not None]
        comparisons.append({'version':version,'reference':name,'pairs':pairs,'joint_complete':len(values),
            'mean_gain':float(np.mean(values)) if values else None,'positive':sum(g>1e-9 for g in values),
            'negative':sum(g< -1e-9 for g in values),'identical':sum(r['identical_trajectory'] for r in pairs)})
resources={version:json.loads((OUT/version/'audit.json').read_text())['physical_steps'] for version in ['physical','decision']}
resources['total_physical']=sum(resources.values());resources['bytes']=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
assert resources['total_physical']<=50000 and resources['bytes']<=1073741824
f=json.loads((OUT/'code_freeze.json').read_text())
for p,h in f.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
result={'comparisons':comparisons,'records':records,'resources':resources,'final_test_used':False,
    'limits':['One training seed and four repeatedly used development weeks; no independent confirmation.',
              'Only eligibility-trace timing changes between continuation arms. This is not a full semi-Markov or SMAAC replication.',
              'Costs ranked only jointly completed episodes; native costs are simulated benchmark accounting.',
              'Saved-vector audit is not independent power-flow rerun or retraining.']}
(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines=['# GRID09: decision-linked credit experiment','',
       '| Version | Reference | Jointly complete | Mean cost reduction | Positive / negative | Exact same trajectory |',
       '|---|---|---:|---:|---:|---:|']
for r in comparisons:
    g='NA' if r['mean_gain'] is None else f"{100*r['mean_gain']:.4f}%"
    lines.append(f"| {r['version']} | {r['reference']} | {r['joint_complete']} | {g} | {r['positive']} / {r['negative']} | {r['identical']} |")
lines+=['','Both arms start from the same V2 GNN, reset Adam identically and complete two passes of the same training weeks. Decision-linked GAE retains gamma per physical step and applies lambda only on links to an actual next offered state; physical rollout boundaries and the original critic are retained. No candidate, safety screen, feature, reward or final-test change.','',json.dumps(resources),'']
lines+=['- '+x for x in result['limits']]
(OUT/'GRID09_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'comparisons':comparisons,'resources':resources},indent=2))
