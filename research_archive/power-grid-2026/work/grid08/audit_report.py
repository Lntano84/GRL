"""Independent saved-vector accounting, frozen assets, and transparent paired reporting."""
import argparse,json,gzip,hashlib,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid08'
ap=argparse.ArgumentParser();ap.add_argument('--version',default='v0');args=ap.parse_args()
D=json.loads((OUT/'design.json').read_text());version=OUT/args.version
results=[];checks=0;residual=0.;failures=[];native_steps=0
for file in sorted(version.rglob('summary.json')):
    folder=file.parent;m=json.loads((folder/'metadata.json').read_text());s=json.loads(file.read_text())
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(l) for l in f]
    with np.load(folder/'vectors.npz') as saved:v={k:saved[k] for k in saved.files}
    assert len(rows)==s['steps']==len(v['gen_p'])
    price=np.asarray(m['gen_cost_per_MW'],dtype=float);dt=m['dt_hours']
    previous=float(rows[0]['ledger']['previous_curtailed_mw']) if rows[0]['ledger'] else 0.
    total=0.;chosen=0;sim=0
    for j,row in enumerate(rows):
        assert row['step']==j+1
        assert hashlib.sha256(np.ascontiguousarray(v['action'][j]).tobytes()).hexdigest()==row['action_hash']
        assert not row['choice'] or row['proposal']['offered']
        chosen+=row['choice'];sim+=row['simulations'];total+=row['raw_cost']
        if row['ledger']:
            production=np.asarray(v['gen_p'][j],dtype=float);load=np.asarray(v['load_p'][j],dtype=float)
            dispatch=np.asarray(v['actual_dispatch'][j],dtype=float)
            curt=float(np.asarray(v['curtailment_mw'][j],dtype=float).sum())
            storage=np.asarray(v['storage_power'][j],dtype=float)
            marginal=float(price[production>0].max())
            calculated=marginal*dt*(production.sum()-load.sum()+np.abs(dispatch).sum()+curt-previous+np.abs(storage).sum())
            error=abs(calculated-row['raw_cost']);residual=max(residual,error)
            assert error<=row['ledger_tolerance'],(str(folder),j,error,row['ledger_tolerance'])
            checks+=1
        previous=float(np.asarray(v['curtailment_mw'][j],dtype=float).sum())
    assert abs(total-s['raw_cost'])<1e-6 and chosen==s['selected'] and sim==s['simulations']
    assert not s['complete'] or s['steps']==m['horizon']
    native_steps+=len(rows)
    if s['illegal'] or s['ambiguous']:failures.append(str(folder))
    results.append(s)
original=json.loads((OUT/'original_assets.json').read_text())
for p,h in original.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
audit={'passed':True,'version':args.version,'episodes':len(results),'physical_steps':native_steps,
    'ledgers_checked':checks,'max_abs_ledger_residual':residual,'actual_illegal_or_ambiguous_runs':failures,
    'original_assets_unchanged':True,'scope':'Independent persisted vector accounting; not independent reruns or fits.'}
(version/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
evaluation=[r for r in results if '/evaluate/' in r['path'].replace('\\','/')]
by={(r['scenario'],r['policy']):r for r in evaluation}
comparisons=[]
for method in sorted(set(r['policy'] for r in evaluation)):
    if method=='NN20':continue
    for baseline in D['baselines']:
        if method==baseline:continue
        pairs=[]
        for scenario in D['evaluation_weeks']:
            if (scenario,method) not in by or (scenario,baseline) not in by:continue
            a,b=by[scenario,method],by[scenario,baseline]
            pairs.append({'scenario':scenario,'method_steps':a['steps'],'baseline_steps':b['steps'],
                'jointly_complete':a['complete'] and b['complete'],
                'relative_cost_reduction':(b['raw_cost']-a['raw_cost'])/abs(b['raw_cost']) if a['complete'] and b['complete'] else None})
        usable=[p['relative_cost_reduction'] for p in pairs if p['relative_cost_reduction'] is not None]
        comparisons.append({'method':method,'baseline':baseline,'pairs':pairs,
            'jointly_complete_pairs':len(usable),'mean_relative_cost_reduction':float(np.mean(usable)) if usable else None,
            'positive_cost_pairs':sum(g>1e-9 for g in usable),
            'earlier_ending_pairs':sum(p['method_steps']<p['baseline_steps'] for p in pairs)})
(version/'comparisons.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
with (version/'per_episode.csv').open('w',newline='',encoding='utf-8') as f:
    keys=['scenario','policy','steps','complete','raw_cost','eligible','offered','selected','simulations','wall_s','path']
    writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(results)
lines=[f'# GRID08 {args.version}: residual-control training pilot','',
 'Binary policy chooses whether to execute a separately generated safe-continuous restoration. Original NN20, action library and unsafe controller are preserved. No learned topology ranking has been attempted.',
 '', '| Week | Method | Steps | Completed | Raw cost | Offered / selected |', '|---|---|---:|---|---:|---:|']
for r in evaluation:lines.append(f"| {r['scenario']} | {r['policy']} | {r['steps']} | {r['complete']} | {r['raw_cost']:.3f} | {r['offered']} / {r['selected']} |")
lines.extend(['','Raw costs are benchmark accounting units. Do not compare lower cumulative cost from earlier failures. Only jointly completed pairs are cost-ranked.',
 '', 'One training seed, four synthetic validation weeks, author pretraining identities unknown. These results cannot establish publication novelty, robust safety, or population improvement.',
 '', 'MLP receives node/global arrays and a padded edge table; GNN receives the same information through message passing. Parameter counts differ and are recorded.',
 '', 'First logger-only attempt failed on NumPy float serialization after one physical step per process, before any PPO update; v0_LOGGING_FAILED and the source are preserved. It is not a negative model result.',
 '', 'Pre-outcome design revision replaced unextracted archive names with the original extracted monthly validation subset, and set a terminal penalty to discourage premature termination. Both versions retained.',
 '', 'Audit: '+json.dumps(audit)])
(version/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'audit':audit,'comparisons':comparisons},indent=2))
