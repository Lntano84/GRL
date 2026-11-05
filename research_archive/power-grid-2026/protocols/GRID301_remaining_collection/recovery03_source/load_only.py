"""One original FrozenModels load, no assembled worker and no environment."""
import argparse,importlib.util,os,sys,time
from guard import R,W,O,read,save,require_authorization,verify_frozen,check,ancestor
def main():
 require_authorization();p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);index=p.parse_args().index
 assert index in range(12,144,12);ctx=read(O/'load_only_context.json');assert ctx['index']==index and os.environ.get('GRL_GRID300_TOKEN')==ctx['token'] and ancestor(ctx['manager_pid']);check(ctx);verify_frozen()
 with (O/f'load_only_recovery03_{index:03d}.claimed').open('x',encoding='utf8') as f:f.write(str(os.getpid()))
 # Same original common initialization used by the real worker, no make_env call.
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
 sys.path[:0]=[str(R/'work/grid30'),str(R/'work/grid29'),str(R/'work/grid08')]
 import common
 s=importlib.util.spec_from_file_location('g301_original_frozen_models',R/'work/grid291_control_preparation/models.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 item=read(O/'design.json')['baseline_runtime']['execution_order'][index];t=time.perf_counter();payload=m.FrozenModels(item['pool']);elapsed=time.perf_counter()-t
 assert payload.pool==item['pool'] and not payload.net.training and payload.gate.event.n_features_in_==payload.gate.amount.n_features_in_==124
 assert not (O/'runs').exists()
 save(O/f'load_only_recovery03_{index:03d}.json',{'pool':item['pool'],'model_payload_loaded':True,'original_FrozenModels_direct':True,'payload_load_s':elapsed,'new_forward':0,'new_environment':0,'new_physics':0})
 check(ctx)
if __name__=='__main__':main()
