"""Independent saved-vector ledger and guard-decision accounting."""
import json,gzip,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid08/guard_qualification'
finish=json.loads((OUT/'finished.json').read_text());results=[];checks=0;maximum=0.;physical=0
freeze=json.loads((OUT/'code_freeze.json').read_text())
for p,h in freeze.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
for item in finish['records']:
    folder=ROOT/item['path'];metadata=json.loads((folder/'metadata.json').read_text())
    rows=[json.loads(x) for x in gzip.open(folder/'steps.jsonl.gz','rt')]
    with np.load(folder/'vectors.npz') as f:arrays={k:f[k] for k in f.files}
    assert len(rows)==item['steps']==len(arrays['action']);physical+=len(rows)
    total=0.;offered=checks_guard=veto=forecast_steps=0
    previous=float(rows[0]['ledger']['previous_curtailed_mw']) if rows[0]['ledger'] else 0.
    price=np.asarray(metadata['gen_cost_per_MW'],dtype=float)
    for i,row in enumerate(rows):
        assert row['step']==i+1
        assert hashlib.sha256(np.ascontiguousarray(arrays['action'][i]).tobytes()).hexdigest()==row['action_hash']
        assert not row['illegal'] and not row['ambiguous']
        total+=row['raw_cost'];offered+=row['proposal']['offered']
        guard=row['proposal'].get('two_step_guard',{});checks_guard+=guard.get('checked',False)
        if guard.get('checked'):
            safe=[]
            for branch in guard['branches']:
                safe.append(len(branch)==2 and all(not x['done'] and not x['illegal'] and not x['ambiguous'] and
                    not x['exceptions'] and np.isfinite(x['rho_max']) and x['rho_max']>0 for x in branch))
            assert safe==guard['safe'];forecast_steps+=sum(map(len,guard['branches']))
            assert guard['forecast_steps']==sum(map(len,guard['branches']))
            assert bool(row['proposal']['offered'])==bool(safe[1])
            veto+=not safe[1]
        if row['ledger']:
            generation=np.asarray(arrays['gen_p'][i],dtype=float);loads=np.asarray(arrays['load_p'][i],dtype=float)
            dispatch=np.asarray(arrays['actual_dispatch'][i],dtype=float);storage=np.asarray(arrays['storage_power'][i],dtype=float)
            curt=float(np.asarray(arrays['curtailment_mw'][i],dtype=float).sum())
            value=float(price[generation>0].max())*metadata['dt_hours']*(generation.sum()-loads.sum()+np.abs(dispatch).sum()+curt-previous+np.abs(storage).sum())
            error=abs(value-row['raw_cost']);maximum=max(maximum,error);checks+=1
            assert error<=row['ledger_tolerance']
        previous=float(np.asarray(arrays['curtailment_mw'][i],dtype=float).sum())
    assert abs(total-item['raw_cost'])<1e-6
    assert offered==item['offered']==item['selected']
    assert checks_guard==item['guard_checks'] and veto==item['vetoes']
    assert forecast_steps==item['explicit_two_step_forecast_steps']
    assert not item['complete'] or item['steps']==metadata['horizon']
    results.append(item)
assert physical==finish['physical_steps'] and len(results)==4
pairs=[]
for week in sorted(set(x['week'] for x in results)):
    a=next(x for x in results if x['week']==week and x['rule']=='ALWAYS_ONE_STEP')
    b=next(x for x in results if x['week']==week and x['rule']=='ALWAYS_TWO_STEP')
    pairs.append({'week':week,'one_step_steps':a['steps'],'two_step_steps':b['steps'],
                  'vetoes':b['vetoes'],'jointly_complete':a['complete'] and b['complete'],
                  'cost_reduction':(a['raw_cost']-b['raw_cost'])/abs(a['raw_cost']) if a['complete'] and b['complete'] else None})
report={'passed':True,'physical_steps':physical,'ledger_checks':checks,'max_abs_residual':maximum,
        'records':results,'pairs':pairs,'scope':'Training-only engineering comparison; independent vector arithmetic, not independently rerun power flow. No final-test/validation results used.'}
(OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['# Two-step guard: training-only qualification','',
       '| Training week | One-step survival | Two-step survival | Vetoes | Cost reduction on jointly complete episodes |',
       '|---|---:|---:|---:|---:|']
for p in pairs:
    reduction='NA' if p['cost_reduction'] is None else f"{p['cost_reduction']*100:.4f}%"
    lines.append(f"| {p['week']} | {p['one_step_steps']} | {p['two_step_steps']} | {p['vetoes']} | {reduction} |")
lines+=['','This is a training-only failure/interface check. A two-step no-op continuation is not a full-week safety guarantee and does not rescue unsafe actions made by the unchanged NN20 base. Costs of incomplete episodes are not ranked. No learned model was changed or fitted.','',
        json.dumps({k:report[k] for k in ['passed','physical_steps','ledger_checks','max_abs_residual']})]
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
