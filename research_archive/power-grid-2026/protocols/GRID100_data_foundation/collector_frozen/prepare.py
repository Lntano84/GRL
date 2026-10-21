from pathlib import Path
import json,datetime as dt,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid100_data_foundation';ENV=W/'trainval_env'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):(OUT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    assert not (OUT/'design.json').exists()
    # First and third Monday of each month: no adjacent/overlapping seven-day episodes.
    dates=[]
    for month in range(1,13):
        first=dt.date(2037,month,1);first+=dt.timedelta(days=(-first.weekday())%7);dates += [first,first+dt.timedelta(days=14)]
    rng=np.random.default_rng(20261021);roles={d.isoformat():'train' for d in dates}
    for q in range(4):
        months=rng.permutation(np.arange(1+3*q,4+3*q))
        for role,month in zip(['validation','test'],months[:2]):
            options=[d for d in dates if d.month==month];roles[options[int(rng.integers(2))].isoformat()]=role
    counts={k:sum(v==k for v in roles.values()) for k in ['train','validation','test']};assert counts==dict(train=16,validation=4,test=4)
    assert len(set(dates))==24 and all((b-a).days>=7 for a,b in zip(sorted(dates),sorted(dates)[1:]))
    sources=ROOT/'work/grid03/formal_env_initialized';ENV.mkdir(exist_ok=False);static={}
    for p in sources.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv','.py']:shutil.copy2(p,ENV/p.name);static[p.name]=sha(p)
    (ENV/'chronics').mkdir();bootstrap=sources/'chronics/2035-01-01_8';shutil.copytree(bootstrap,ENV/'chronics'/bootstrap.name)
    params=read(ENV/'scenario_params.json');params['all_dates']=[d.isoformat() for d in dates if roles[d.isoformat()]!='test'];(ENV/'scenario_params.json').write_text(json.dumps(params,indent=2)+'\n',encoding='utf8')
    seed_rngs=[np.random.default_rng(20261021),np.random.default_rng(20261022)];items=[]
    for day in sorted(roles):
        for rep,seed_rng in enumerate(seed_rngs):
            seeds=[int(v) for v in seed_rng.integers(2**32-1,size=3)]
            items.append(dict(date=day,calendar_group=day,partition=roles[day],replicate=rep,scenario_id=100+rep,scenario=f'{day}_{100+rep}',seeds=seeds))
    pilot=next(z for z in items if z['partition']=='train' and z['replicate']==0)
    d=dict(stage='GRID100_LEARNING_DATA_FOUNDATION',utc=dt.datetime.now(dt.timezone.utc).isoformat(),authorization='User starts part1: new data coverage and train/validation/final evaluation organization.',
        environment=ENV.relative_to(ROOT).as_posix(),static_source_sha256=static,calendar_selection='2037 first/third Mondays monthly, 24 disjoint seven-day calendar groups, two independent generator seed streams per date.',
        partition_selection='Within each quarter, seed20261021 permutes months; validation and test take one date each in different months; remaining4 dates train. Date replicates and all decision checkpoints inherit same role.',
        calendar_counts=counts,scenario_counts={k:2*v for k,v in counts.items()},scenario_plan=items,pilot=pilot,
        generation='Same official Chronix2Grid wrapper and existing package/runtime. Master seeds20261021/20261022, per-item frozen triples, scen_id100/101; no auto-ID increment, date replacement or output-based selection.',
        final_test_gate='Eight test scenarios remain planned only: no series generation, teacher outcomes, inputs or labels until a separate model/method freeze. Current generation commands only accept pilot/trainval.',
        collection='Initial local-gain teacher path uses common FULL controller to pay for all candidate feedback once at each physical state. Offline prefixes at128,192,... create pre-query inputs and local completion targets. Multiple prefixes are correlated records, not independent scenarios. FULL trajectories differ from FRAME trajectories: later application requires explicit FRAME distribution/closed-loop validation.',
        checkpoints='Every64 after initial128 before pool exhaustion; retain both original FRAME-triggered and nontriggered checkpoints when admissible incumbent and remaining local candidates exist. Exclusions counted, never silently converted to negatives.',
        target='Primary continuous nonnegative float32 best-admissible reward gain from completing incumbent station. Auxiliary choice change and rho reduction. Known query count is input/cost, not a learned unknown. Teacher query timing is diagnostic, not deploy latency or new speed claim. No conversion of old11 MIXED into good/bad.',
        input_boundary='Only prefix paid feedback, current observation/base vector and public action structures. Per-sample input and target in different directories. Future feedback never copied into learner input; identifiers/date/group used for association and split only.',
        old_data='All existing2035/2036 datasets and56 physical omission labels are development/debug only, not new validation or final test. Author LJN pretraining provenance is not independently verified; new generated date is not a guarantee of unrelated physical source.',
        sample_scope='Same public grid, existing generator/forecast/maintenance configuration. No new grid, production-data claim or extra artificially injected fault model. Two seed repeats are not independent date groups.',
        qualification='Before large collection: pure saved-data label check, no-label suffix perturbation, fixed tie rules and no-incumbent boundaries; one new training scenario generation smoke. Failed scenarios preserved, no replacement. Collection runtime integration is separate from generation and must be qualified before starting teacher replays.',
        generation_caps=dict(per_scenario_seconds=1800,pilot_scenarios=1,trainval_scenarios=40,total_seconds=24*3600),
        scope_now='Build/freeze dataset protocol and generator/label entrypoints, validate historical engineering fixtures and start one train-only generation pilot. No models, no final-test materialization and no undefined mass physical replay.')
    save('design.json',d);save('split_manifest.json',dict(calendar_role=roles,scenarios=items,final_test_materialized=False));save('generation_manifest.json',dict(completed=[],planned_trainval=40,pilot_scenarios=1))
    source_files=list(W.glob('*.py'))+[OUT/'design.json',OUT/'split_manifest.json']
    source_files += [ROOT/p for p in ['work/grid89_review/generate.py','work/grid89_review/adapter.py','work/grid89_review/policies.py','work/grid55/run.py','work/grid55/worker.py','work/grid91_mechanism/policies.py','outputs/grid91_mechanism/design.json','outputs/grid28/offline.json']]
    for p in ENV.iterdir():
        if p.is_file():source_files.append(p)
    packages=ROOT/'work/grid80/generator_packages/chronix2grid';source_files+=list(packages.rglob('*.py'))
    for p in bootstrap.iterdir():
        if p.is_file():source_files.append(p)
    save('freeze.json',dict(sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source_files},before_new_generation=True))
    a=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID100_data_foundation/frozen';a.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+[OUT/'design.json',OUT/'split_manifest.json',OUT/'freeze.json']:shutil.copy2(p,a/p.name)
    print(json.dumps(dict(calendar_groups=24,calendar_counts=counts,scenario_counts=d['scenario_counts'],pilot=pilot),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
