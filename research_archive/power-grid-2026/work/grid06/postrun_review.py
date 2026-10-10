"""Second arithmetic path from raw saved logs; seal only after all writers ended."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid06'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
design=load(OUT/'design_freeze.json');summary=load(OUT/'GRID06_summary.json')
meta={};steps={};cost={};perrun={(r['scenario'],r['policy']):r for r in load(OUT/'per_run.json')}
for job in design['plan']:
    s,p=job['scenario'],job['policy'];folder=OUT/'runs'/f'{s}__{p}'
    rows=lines(folder/'steps.jsonl');events=lines(folder/'events.jsonl')
    meta[(s,p)]=load(folder/'completion.json');steps[(s,p)]=len(rows)
    cost[(s,p)]=sum(r['raw_operational_cost'] for r in rows)
    assert cost[(s,p)]==perrun[(s,p)]['native_raw_cost_sum'] and steps[(s,p)]==perrun[(s,p)]['steps']
    assert sum(r['simulate_calls'] for r in rows)==sum(e['kind']=='simulate' for e in events)
    fb=[e for e in events if e['kind']=='fallback'];short=[e for e in events if e['kind']=='shortlist']
    topo_sims=sum(e['kind']=='simulate' and e['module']=='topo_12_unsafe' for e in events)
    if p=='FALLBACK90':
        assert all(not e['forced_test'] for e in fb)
        assert len(short)==len(fb)+sum(e['expanded'] for e in fb)
        assert topo_sims==20*len(fb)+332*sum(e['expanded'] for e in fb)
    else:
        assert not fb and topo_sims==(20 if p=='NN20' else 352)*len(short)
retreat20=retreat352=False;benefits=0
for s in design['selected_weeks']:
    for p in ['NN20','NN352']:
        joint=meta[(s,p)]['reached_native_horizon'] and meta[(s,'FALLBACK90')]['reached_native_horizon']
        retreat=steps[(s,'FALLBACK90')]<steps[(s,p)] or (joint and cost[(s,'FALLBACK90')]>1.02*cost[(s,p)])
        if p=='NN20':
            retreat20|=retreat
            benefit=(meta[(s,'FALLBACK90')]['reached_native_horizon'] and not meta[(s,p)]['reached_native_horizon']) or (joint and cost[(s,'FALLBACK90')]<=.95*cost[(s,p)])
            benefits+=int(benefit)
        else:retreat352|=retreat
assert (not retreat20)==summary['no_regression_vs_NN20'] and (not retreat352)==summary['full_search_quality_retained_screen']
assert benefits==summary['meaningful_benefit_weeks_vs_NN20']
if retreat20:verdict='DO_NOT_PROMOTE_RULE_AS_GENERAL_IMPROVEMENT'
elif benefits>=2 and not retreat352:verdict='VALIDATION_QUALITY_SCREEN_SUPPORTED'
elif benefits>=2:verdict='BENEFIT_VS_NN20_WITH_FULLSEARCH_QUALITY_TRADEOFF'
elif benefits==0:verdict='NO_PRACTICAL_INCREMENTAL_BENEFIT_CONFIRMED'
else:verdict='LIMITED_SIGNAL_NOT_CONFIRMED'
assert verdict==summary['verdict']
review=dict(passed=True,runs=12,physical_steps=sum(steps.values()),raw_costs_and_events_verified=True,
    verdict=verdict,meaningful_benefits_vs_NN20=benefits,
    scope='Separate arithmetic implementation on the same saved feedback; not a second simulator, policy inference or training run.')
(OUT/'GRID06_postrun_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
# This process stdout is intentionally excluded: it is still being written.
files={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.glob('*')) if p.is_file() and p.suffix!='.log' and p.name!='delivery_manifest.json'}
files.update({str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'work/grid06').glob('*.py'))})
(OUT/'delivery_manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
for p,h in load(OUT/'delivery_manifest.json').items():assert sha(ROOT/p)==h,p
print(json.dumps(review,indent=2))
