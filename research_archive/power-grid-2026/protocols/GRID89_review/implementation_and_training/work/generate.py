"""Serial official generation with per-date receipts; no automatic replacement."""
from pathlib import Path
import datetime as dt,hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid89_review'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'tmp')
sys.path.insert(0,str(ROOT/'work/grid80/generator_packages'))
os.environ['PATH']=str(ROOT/'work/grid80/generator_packages/pulp/solverdir/cbc/win/i64')+os.pathsep+os.environ['PATH']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def main():
 D=read(OUT/'design.json');E=ROOT/D['environment'];start=time.perf_counter()
 for p,h in read(OUT/'generation_code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 import numpy as np,pandas as pd,grid2op
 from lightsim2grid import LightSimBackend
 from chronix2grid.grid2op_utils.add_data import generate_a_scenario_wrapper
 rng=np.random.default_rng(D['generation_seed']);env=grid2op.make(str(E),test=True,backend=LightSimBackend());items=[]
 try:
  for day in D['dates']:
   seeds=[int(x) for x in rng.integers(2**32-1,size=3)];name=day+'_'+str(D['scenario_id']);folder=E/'chronics'/name;receipt=OUT/(name+'_generation_receipt.json')
   if receipt.exists():
    r=read(receipt);assert r['seeds']==seeds
    for p,h in r['files'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    items.append(r);continue
   assert not folder.exists(),'Partial generation requires explicit recovery audit; never replace scenario'
   save(OUT/'generation_progress.json',dict(pid=os.getpid(),date=day,completed=len(items),phase='generating',utc=dt.datetime.now(dt.timezone.utc).isoformat()))
   args=(str(E),env.name_gen,env.gen_type,str(E/'chronics'),day,'5',str(D['scenario_id']),*seeds,True,['maintenance_meta.json','params_load.json','params_forecasts.json'],False,0,1e-3,False)
   error,*_=generate_a_scenario_wrapper(args);assert error is None,str(error)
   quality=read(folder/'generation_quality.json');assert quality['max']<=.05
   for k in ['load_p','load_q','prod_p','load_p_forecasted','load_q_forecasted','prod_p_forecasted']:
    x=pd.read_csv(folder/(k+'.csv.bz2'),sep=';');assert len(x)>=2017 and np.isfinite(x.to_numpy()).all(),k
   assert read(folder/'_seeds_info.json')==dict(zip(['load_seed','renew_seed','gen_p_forecast_seed'],seeds))
   r=dict(date=day,scenario=name,seeds=seeds,quality_max=quality['max'],files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()})
   save(receipt,r);items.append(r);print(json.dumps(dict(completed=len(items),date=day)),flush=True)
  save(OUT/'generation.json',dict(passed=True,wall_s=time.perf_counter()-start,scenarios=items,controller_evaluations_during_generation=0))
 finally:env.close()
if __name__=='__main__':
 try:main()
 except BaseException:
  save(OUT/'generation_error.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),traceback=traceback.format_exc()));raise
