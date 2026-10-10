"""Second arithmetic path from step/event logs; no extra simulation or fitting."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid05';OLD=ROOT/'outputs/grid04'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows=lambda p:[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
design=load(OUT/'design_freeze.json');summary=load(OUT/'GRID05_summary.json')
sources={};counts={};costs={};calibration=[]
for job in design['plan']:
    scene,policy=job['scenario'],job['policy']
    base=OUT/'runs'/f'{scene}__{policy}'
    step=rows(base/'steps.jsonl');events=rows(base/'events.jsonl')
    sources[(scene,policy)]=step
    counts[(scene,policy)]=len(step)
    costs[(scene,policy)]=sum(s['raw_operational_cost'] for s in step)
    assert sum(s['simulate_calls'] for s in step)==sum(e['kind']=='simulate' for e in events)
    fallback=[e for e in events if e['kind']=='fallback']
    short=[e for e in events if e['kind']=='shortlist']
    if policy=='NN20':
        oldrows=rows(OLD/'runs_v3'/f'{scene}__NN20'/'steps.jsonl')
        calibration.append(len(step)==len(oldrows) and all(s['action_hash']==o['action_hash'] and s['observation_hash']==o['observation_hash'] and s['raw_operational_cost']==o['raw_operational_cost'] for s,o in zip(step,oldrows)))
        assert not fallback
    else:
        assert all(not e['forced_test'] for e in fallback)
        assert len(short)==len(fallback)+sum(e['expanded'] for e in fallback)
        calls=sum(e['kind']=='simulate' and e['module']=='topo_12_unsafe' for e in events)
        assert calls==20*len(fallback)+332*sum(e['expanded'] for e in fallback)
    matching=next(s for s in load(OUT/'per_run.json') if s['scenario']==scene and s['policy']==policy)
    assert counts[(scene,policy)]==matching['steps'] and costs[(scene,policy)]==matching['native_raw_cost_sum']

first,april=design['selected_development_weeks'][:2]
old_april=rows(OLD/'runs_v3'/f'{april}__NN352'/'steps.jsonl')
old_cost=sum(s['raw_operational_cost'] for s in old_april)
recovery=(costs[(april,'NN20')]-costs[(april,'FALLBACK90')])/(costs[(april,'NN20')]-old_cost)
jointly_complete=[s for s in design['selected_development_weeks'] if counts[(s,'NN20')]==counts[(s,'FALLBACK90')]==2017]
screen=dict(exact_NN20_calibration=all(calibration),
    no_earlier_end=all(counts[(s,'FALLBACK90')]>=counts[(s,'NN20')] for s in design['selected_development_weeks']),
    january_end_at_least_1719=counts[(first,'FALLBACK90')]>=1719,
    april_joint_completion_and_80pct_gap_recovery=april in jointly_complete and recovery>=.8,
    no_more_than_2pct_cost_regression=all(costs[(s,'FALLBACK90')]<=1.02*costs[(s,'NN20')] for s in jointly_complete))
assert screen==summary['screen']
assert abs(recovery-summary['april_gap_recovery_fraction'])<1e-12
if not screen['exact_NN20_calibration']:verdict='CALIBRATION_FAILED'
elif not screen['no_earlier_end'] or not screen['no_more_than_2pct_cost_regression']:verdict='DO_NOT_PROMOTE_RULE_HERE'
elif all(screen.values()):verdict='SUPPORTED_ON_THIS_DEVELOPMENT_SET_ONLY'
else:verdict='MIXED_OR_INCONCLUSIVE'
assert verdict==summary['verdict']
for p,h in load(OUT/'delivery_manifest.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
review=dict(passed=True,runs=8,physical_steps=sum(counts.values()),
    raw_step_costs_and_event_counts_agree=True,screen=screen,verdict=verdict,
    april_gap_recovery_fraction=recovery,delivery_hashes_verified=True,
    scope='Second arithmetic implementation from the same persisted feedback; not an independent simulator, policy run, neural inference or training.')
(OUT/'GRID05_postrun_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
print(json.dumps(review,indent=2))
