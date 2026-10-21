"""Frozen train-only pilot or train/validation generation; final test unavailable."""
from pathlib import Path
import argparse,datetime as dt,hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid100_data_foundation'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'tmp/grid100')
sys.path.insert(0,str(ROOT/'work/grid80/generator_packages'));os.environ['PATH']=str(ROOT/'work/grid80/generator_packages/pulp/solverdir/cbc/win/i64')+os.pathsep+os.environ['PATH']
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['pilot','trainval'],default='pilot');a=ap.parse_args();D=read(OUT/'design.json')
    for p,h in read(OUT/'freeze.json')['sha256'].items():assert sha(ROOT/p)==h,p
    assert read(OUT/'preflight.json')['passed'],'Do not generate before saved-data preflight'
    if a.mode=='trainval':assert read(OUT/'generation_pilot_finished.json')['passed'],'Pilot qualification first'
    items=[D['pilot']] if a.mode=='pilot' else [z for z in D['scenario_plan'] if z['partition']!='test']
    import ctypes
    mutex=ctypes.windll.kernel32.CreateMutexW(None,True,'GRL_GRID100_GENERATION');assert ctypes.windll.kernel32.GetLastError()!=183,'Existing generator'
    import grid2op,numpy as np,pandas as pd
    from lightsim2grid import LightSimBackend
    from chronix2grid.grid2op_utils.add_data import generate_a_scenario_wrapper
    E=ROOT/D['environment'];env=grid2op.make(str(E),test=True,backend=LightSimBackend());started=time.perf_counter();completed=[]
    try:
        for item in items:
            assert item['partition']!='test';folder=E/'chronics'/item['scenario'];receipt=OUT/'generation_receipts'/(item['scenario']+'.json')
            if receipt.exists():
                r=read(receipt);assert r['item']==item
                for p,h in r['files'].items():assert sha(ROOT/p)==h,p
                completed.append(r);continue
            assert not folder.exists(),'Partial output preserved; recovery review required, no automatic replacement'
            assert time.perf_counter()-started<D['generation_caps']['total_seconds'];stamp=time.perf_counter()
            save('generation_progress.json',dict(mode=a.mode,pid=os.getpid(),item=item,completed=len(completed),phase='generating',utc=dt.datetime.now(dt.timezone.utc).isoformat()))
            args=(str(E),env.name_gen,env.gen_type,str(E/'chronics'),item['date'],'5',str(item['scenario_id']),*item['seeds'],True,['maintenance_meta.json','params_load.json','params_forecasts.json'],False,0,1e-3,False)
            err,*_=generate_a_scenario_wrapper(args);assert err is None,str(err)
            q=read(folder/'generation_quality.json');assert q['max']<=.05
            for k in ['load_p','load_q','prod_p','load_p_forecasted','load_q_forecasted','prod_p_forecasted']:
                x=pd.read_csv(folder/(k+'.csv.bz2'),sep=';');assert len(x)>=2017 and np.isfinite(x.to_numpy()).all(),k
            assert read(folder/'_seeds_info.json')==dict(zip(['load_seed','renew_seed','gen_p_forecast_seed'],item['seeds']))
            assert time.perf_counter()-stamp<D['generation_caps']['per_scenario_seconds'],'Scenario wall budget exceeded; preserve produced files'
            r=dict(item=item,quality_max=q['max'],wall_s=time.perf_counter()-stamp,files={p.relative_to(ROOT).as_posix():sha(p) for p in folder.iterdir() if p.is_file()})
            save('generation_receipts/'+item['scenario']+'.json',r);completed.append(r);save('generation_manifest.json',dict(mode=a.mode,completed=completed,planned=len(items)));print(item['scenario'],'generated',flush=True)
        save('generation_'+a.mode+'_finished.json',dict(passed=True,mode=a.mode,completed=len(completed),wall_s=time.perf_counter()-started,scenarios=[r['item']['scenario'] for r in completed],final_test_materialized=False,physical_teacher_replays=0,model_fits=0))
        save('generation_progress.json',dict(mode=a.mode,pid=os.getpid(),completed=len(completed),phase='complete',utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    finally:env.close();ctypes.windll.kernel32.ReleaseMutex(mutex);ctypes.windll.kernel32.CloseHandle(mutex)
if __name__=='__main__':
    try:main()
    except BaseException:save('generation_error.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),traceback=traceback.format_exc()));raise
