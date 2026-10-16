"""Prospective fresh scenario generation, isolated from controller dependencies."""
from pathlib import Path
import datetime as dt, hashlib, json, shutil
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/grid80'; ENV=ROOT/'work/grid80/fresh_env'
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    assert not (OUT/'design.json').exists()
    OUT.mkdir(exist_ok=True); ENV.mkdir(exist_ok=True)
    qualification=json.loads((ROOT/'outputs/grid66/fresh_scene_qualification.json').read_text(encoding='utf-8'))
    index=json.loads((ROOT/'outputs/grid03/archive_index.json').read_text(encoding='utf-8'))
    # The exact 52-family exposure audit is retained separately; do not relabel old variants as new dates.
    dates=['2036-01-07','2036-03-03','2036-05-05','2036-07-07','2036-09-01','2036-11-03']
    assert all(dt.date.fromisoformat(d).weekday()==0 for d in dates)
    files={}
    source=ROOT/'work/grid03/formal_env_initialized'
    for p in source.iterdir():
        if p.is_file() and p.suffix in ('.json','.csv','.py'):
            shutil.copy2(p,ENV/p.name);files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    params=json.loads((ENV/'scenario_params.json').read_text());params['all_dates']=dates;save(ENV/'scenario_params.json',params)
    # A single existing scenario is only a grid2op construction bootstrap, never an evaluation case.
    (ENV/'chronics').mkdir(exist_ok=True)
    bootstrap=source/'chronics/2035-01-01_8'
    shutil.copytree(bootstrap,ENV/'chronics'/bootstrap.name,copy_function=shutil.copy2)
    design=dict(stage='GRID80_FRESH_GENERATION_QUALIFICATION',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        official_existing_date_families=52,already_exposed_families=52,unexposed_existing_families=0,
        dates=dates,seed=20261016,nb_scenario=1,nb_core=1,with_loss=True,day_lag=0,
        generator='Chronix2Grid 1.2.0.post1; official Grid2Op generate_data API',
        source_static_sha256=files,environment='work/grid80/fresh_env',
        selection='Six first Mondays in alternating months of 2036, frozen before generated series or controller outcomes. No resampling failed cases or outcome filtering.',
        scope='New generated stochastic scenarios on the same public grid. Not another real-world grid, not independently certified operational data.',
        dependency_isolation='generator_packages added only to generation subprocess; scientific controller runtime unchanged',
        failure_rule='Preserve failures. Generation engineering failure blocks fresh confirmation, not gate ablation on already exposed cases. No silently replacing dates or generator settings.',
        model_sha256=hashlib.sha256((ROOT/'outputs/grid75/models/plain.json').read_bytes()).hexdigest())
    save(OUT/'design.json',design)
    (OUT/'RUN_STATE.md').write_text('FROZEN: new scenario dates/seeds; generator dependency qualification in progress.\n',encoding='utf-8')
    print(json.dumps(dict(dates=dates,existing_unexposed=0)))
if __name__=='__main__': main()
