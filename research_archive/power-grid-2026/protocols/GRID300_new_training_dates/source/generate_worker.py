"""Original fixed Chronix2Grid wrapper, only after new scope authorization."""
import argparse,os,sys,time
from guard import R,O,read,sha,save,claim,check
def main():
 p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);index=p.parse_args().index;ctx=claim('generation',index);D=read(O/'design.json');item=read(O/'calendar.json')['dates'][index];E=R/D['environment'];folder=E/'chronics'/item['scenario'];assert not folder.exists()
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
 os.environ['TEMP']=os.environ['TMP']=str(R/'tmp/grid100');sys.path.insert(0,str(R/'work/grid80/generator_packages'));os.environ['PATH']=str(R/'work/grid80/generator_packages/pulp/solverdir/cbc/win/i64')+os.pathsep+os.environ['PATH']
 import grid2op,numpy as np,pandas as pd
 from lightsim2grid import LightSimBackend
 from chronix2grid.grid2op_utils.add_data import generate_a_scenario_wrapper
 # Old training-only environment supplies unchanged generator ordering/types.
 # No old/test chronics are copied and no policy/step/simulate is called here.
 env=grid2op.make(str(R/'work/grid100_data_foundation/trainval_env'),test=True,backend=LightSimBackend());names=env.name_gen.copy();types=env.gen_type.copy();schema_forecasts=int(env.nb_highres_called);env.close();check(ctx)
 seeds=item['generator_seeds'];triple=[seeds[k] for k in ['load_seed','renew_seed','gen_p_forecast_seed']];stamp=time.perf_counter()
 args=(str(E),names,types,str(E/'chronics'),item['date'],'5',str(item['scenario_id']),*triple,True,['maintenance_meta.json','params_load.json','params_forecasts.json'],False,0,1e-3,False)
 err,*_=generate_a_scenario_wrapper(args);assert err is None,str(err);check(ctx);q=read(folder/'generation_quality.json');assert q['max']<=.05
 for k in ['load_p','load_q','prod_p','load_p_forecasted','load_q_forecasted','prod_p_forecasted']:
  x=pd.read_csv(folder/(k+'.csv.bz2'),sep=';');assert len(x)>=2017 and np.isfinite(x.to_numpy()).all(),k
 assert read(folder/'_seeds_info.json')==seeds
 save(O/'generation_receipts'/f'{index:03d}.json',{'item':item,'wall_s':time.perf_counter()-stamp,'quality_max':q['max'],'files':{p.relative_to(R).as_posix():sha(p) for p in folder.iterdir() if p.is_file()},'schema_forecast_calls':schema_forecasts,'schema_env_setup_in_total_budget':True,'new_model_predictions':0,'new_policy_steps':0})
if __name__=='__main__':main()
