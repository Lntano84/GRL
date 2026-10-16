"""Resume missing frozen dates using the original official wrapper and original seed stream."""
from pathlib import Path
import argparse,datetime as dt,hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid80';REC=ROOT/'outputs/grid81/recovery_01.json'
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'tmp')
sys.path.insert(0,str(ROOT/'work/grid80/generator_packages'))
os.environ['PATH']=str(ROOT/'work/grid80/generator_packages/pulp/solverdir/cbc/win/i64')+os.pathsep+os.environ['PATH']
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def plan():
    import numpy as np,pandas as pd
    D=read(OUT/'design.json');E=ROOT/D['environment']
    for p,h in read(ROOT/'outputs/grid81/code_freeze.json').items():assert sha(ROOT/p)==h,p
    rng=np.random.default_rng(D['seed']);items=[];preserved={}
    expected=['load_p','load_q','prod_p','load_p_forecasted','load_q_forecasted','prod_p_forecasted']
    for day in D['dates']:
        seeds=[int(x) for x in rng.integers(2**32-1,size=3)];folder=E/'chronics'/(day+'_9')
        found=folder.exists()
        if found:
            actual=read(folder/'_seeds_info.json')
            assert actual==dict(zip(['load_seed','renew_seed','gen_p_forecast_seed'],seeds))
            quality=read(folder/'generation_quality.json');assert quality['max']<=.05
            for key in expected:
                x=pd.read_csv(folder/(key+'.csv.bz2'),sep=';');assert len(x)>=2017 and np.isfinite(x.to_numpy()).all(),key
            for key in ['maintenance_meta.json','params_load.json','params_forecasts.json']:assert sha(folder/key)==sha(E/key)
            assert (folder/'start_datetime.info').exists() and (folder/'time_interval.info').exists()
            preserved.update({p.relative_to(ROOT).as_posix():sha(p) for p in folder.iterdir() if p.is_file()})
        items.append(dict(date=day,scenario=day+'_9',seeds=seeds,already_complete=found))
    assert sum(x['already_complete'] for x in items)==1 and items[0]['already_complete']
    return dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),original_generator_process_stopped=True,
        benchmark_runs=0,completed_dates_retained=1,remaining_dates=5,items=items,preserved_files=preserved,
        generator='Same official generate_a_scenario_wrapper and exact args/RNG stream as add_data; fixed scen_id9. No call to add_data (would increment ID and regenerate completed date).',
        interruption='Original process ended without Python exception or final receipt; cause unconfirmed. Pipeline stopped on absent receipt. Completed January files pass seeds/schema/static/finite checks and are retained byte-for-byte.',
        scope='Engineering continuation only; original freeze, dates, master/per-date seeds, generator code, model and all protocol parameters unchanged; interrupted generation time is not a zero-cost completed attempt.')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    if a.preflight:
        assert not REC.exists();r=plan();save(REC,r);print(json.dumps({k:r[k] for k in ['completed_dates_retained','remaining_dates','benchmark_runs']}));return
    r=read(REC);D=read(OUT/'design.json');E=ROOT/D['environment'];stamp=time.perf_counter()
    for p,h in r['preserved_files'].items():assert sha(ROOT/p)==h,p
    import grid2op
    from lightsim2grid import LightSimBackend
    from chronix2grid.grid2op_utils.add_data import generate_a_scenario_wrapper
    env=grid2op.make(str(E),test=True,backend=LightSimBackend())
    try:
        for item in r['items']:
            if item['already_complete']:continue
            assert not (E/'chronics'/item['scenario']).exists(),'Existing partial/new folder requires manual inspection'
            args=(str(E),env.name_gen,env.gen_type,str(E/'chronics'),item['date'],'5','9',*item['seeds'],True,
                ['maintenance_meta.json','params_load.json','params_forecasts.json'],False,0,1e-3,False)
            save(OUT/'resume_progress.json',dict(phase='generating',date=item['date'],completed=[x['scenario'] for x in r['items'] if (E/'chronics'/x['scenario']/'generation_quality.json').exists()],pid=os.getpid()))
            error,*_=generate_a_scenario_wrapper(args);assert error is None,(item['date'],str(error))
            save(OUT/(item['scenario']+'_resume_complete.json'),dict(date=item['date'],seeds=item['seeds'],unchanged_official_wrapper=True))
        for p,h in r['preserved_files'].items():assert sha(ROOT/p)==h,p
        folders=[E/'chronics'/x['scenario'] for x in r['items']]
        save(OUT/'generation.json',dict(passed=True,wall_s=time.perf_counter()-stamp,wall_scope='Resume process only; original interrupted work retained separately and not counted zero.',resumed=True,scenarios=[p.name for p in folders],files={p.relative_to(ROOT).as_posix():sha(p) for folder in folders for p in folder.iterdir() if p.is_file()}))
    finally:env.close()
if __name__=='__main__':
    try:main()
    except BaseException:
        folder=OUT/'engineering_errors';folder.mkdir(exist_ok=True);(folder/(str(time.time_ns())+'_resume.txt')).write_text(traceback.format_exc(),encoding='utf-8');raise
