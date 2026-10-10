"""Independent persisted-vector arithmetic and frozen-source audit, no simulator."""
import json,gzip,hashlib,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid10'
ap=argparse.ArgumentParser();ap.add_argument('--version',choices=['raw','normalized'],required=True);args=ap.parse_args()
version=OUT/args.version;records=[];checks=0;maximum=0.;physical=0
freeze=json.loads((OUT/'code_freeze.json').read_text())
for p,h in freeze.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
for phase in ['train','evaluate']:
    finish=json.loads((version/phase/'gnn/finished.json').read_text())
    phase_steps=0
    for item in finish['summaries']:
        folder=ROOT/item['path'];m=json.loads((folder/'metadata.json').read_text())
        rows=[json.loads(x) for x in gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8')]
        with np.load(folder/'vectors.npz') as f:arrays={k:f[k] for k in f.files}
        assert len(rows)==item['steps']==len(arrays['gen_p']);phase_steps+=len(rows)
        cost=0.;choices=simulations=0
        previous=float(rows[0]['ledger']['previous_curtailed_mw']) if rows[0]['ledger'] else 0.
        price=np.asarray(m['gen_cost_per_MW'],dtype=float)
        for i,row in enumerate(rows):
            assert row['step']==i+1 and not row['illegal'] and not row['ambiguous']
            assert hashlib.sha256(np.ascontiguousarray(arrays['action'][i]).tobytes()).hexdigest()==row['action_hash']
            assert not row['choice'] or row['proposal']['offered']
            cost+=row['raw_cost'];choices+=row['choice'];simulations+=row['simulations']
            if row['ledger']:
                generation=np.asarray(arrays['gen_p'][i],dtype=float);loads=np.asarray(arrays['load_p'][i],dtype=float)
                dispatch=np.asarray(arrays['actual_dispatch'][i],dtype=float);storage=np.asarray(arrays['storage_power'][i],dtype=float)
                curt=float(np.asarray(arrays['curtailment_mw'][i],dtype=float).sum())
                value=float(price[generation>0].max())*m['dt_hours']*(generation.sum()-loads.sum()+np.abs(dispatch).sum()+curt-previous+np.abs(storage).sum())
                error=abs(value-row['raw_cost']);maximum=max(maximum,error);checks+=1
                assert error<=row['ledger_tolerance']
            previous=float(np.asarray(arrays['curtailment_mw'][i],dtype=float).sum())
        assert abs(cost-item['raw_cost'])<1e-6 and choices==item['selected'] and simulations==item['simulations']
        assert not item['complete'] or item['steps']==m['horizon']
        records.append(item)
    assert phase_steps==finish['physical_steps'];physical+=phase_steps
assert len(records)==12
updates=[json.loads(x) for x in (version/'train/gnn/updates.jsonl').read_text().splitlines()]
assert [r['update'] for r in updates]==list(range(1,len(updates)+1))
report={'passed':True,'version':args.version,'records':records,'physical_steps':physical,
        'ledger_checks':checks,'max_abs_residual':maximum,'updates':len(updates),
        'optimizer_wall_s':sum(r['wall_s'] for r in updates),
        'scope':'Frozen-source and independent persisted-vector arithmetic, not independent rerun power flow or training.'}
(version/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
