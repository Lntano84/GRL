"""Read training logs only; quantify decision spacing and GAE trace, not policy value."""
import json,gzip,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid08'
D=json.loads((OUT/'v2/design.json').read_text());gamma=D['ppo']['gamma'];lam=D['ppo']['gae_lambda']
rows=[];gaps=[]
for kind in ['gnn','mlp']:
    folder=OUT/'v2/train'/kind
    manifest=json.loads((folder/'manifest.json').read_text())
    for episode in manifest:
        path=ROOT/episode['path'];steps=[]
        for line in gzip.open(path/'steps.jsonl.gz','rt',encoding='utf-8'):
            row=json.loads(line)
            if row['proposal']['offered']:steps.append(row['step'])
        local=np.diff(steps);gaps.extend(local.tolist())
        rows.append({'kind':kind,'episode':path.name,'physical_steps':episode['steps'],
            'offered':len(steps),'offered_fraction':len(steps)/episode['steps'],
            'decision_gap_quantiles':np.quantile(local,[0,.5,.9,1]).tolist() if len(local) else []})
for d in [6,24,96]:
    assert math.isclose((gamma*lam)**d, gamma**d*lam**d,rel_tol=1e-12)
trace=[{'physical_steps':d,'minutes':d*5,
        'current_physical_step_gae_trace':(gamma*lam)**d,
        'illustrative_event_gae_trace_at_cadence6':gamma**d*lam**(d/6)} for d in [6,24,96]]
report={'scope':'Training-only descriptive opportunity counts and algebraic trace weights. Not causal diagnosis, actual gradient influence, counterfactual value, or proof event-time training will help.',
        'gamma':gamma,'lambda':lam,'completed_episode_records':len(rows),'rows':rows,
        'decision_gap_quantiles':np.quantile(gaps,[0,.5,.9,.99,1]).tolist() if gaps else [],
        'trace':trace,'fixed_cadence_illustration':'Event trace uses exactly one decision per six steps, while actual offered intervals can be much longer.',
        'limitation':'A correct value function can transmit long effects through bootstrapping; short GAE trace alone does not prove the policy cannot learn them.'}
(OUT/'credit_diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
