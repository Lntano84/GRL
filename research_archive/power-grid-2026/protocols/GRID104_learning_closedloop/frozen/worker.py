"""Reuse frozen common-controller worker; no fitting or teacher targets enter control."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid104_learning_closedloop'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);args=ap.parse_args()
D=json.loads((OUT/'design.json').read_text(encoding='utf8'));item=D['execution_order'][args.index]
source=ROOT/'work/grid55/run.py';code=source.read_text(encoding='utf8')
assert code.count("OUT=ROOT/'outputs/grid55'")==1
code=code.replace("OUT=ROOT/'outputs/grid55'","OUT=ROOT/'outputs/grid104_learning_closedloop'")
marker='    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose';assert code.count(marker)==1
patch="""    import importlib.util
    adapter_spec=importlib.util.spec_from_file_location('grid104_online_adapter',ROOT/'work/grid104_learning_closedloop/adapter.py')
    adapter=importlib.util.module_from_spec(adapter_spec);adapter_spec.loader.exec_module(adapter);adapter.install(engine.Search,item)
    import common
    def learning_development_env():
        return common.grid2op.make(str(ROOT/'work/grid100_data_foundation/trainval_env'),test=True,backend=common.LightSimBackend(),reward_class=common.MaxRhoReward,other_rewards=common.operational_other_rewards())
    common.make_env=learning_development_env
    # Same previously qualified diagnostic-logger repair, applied equally to all arms.
    optimizer_module=sys.modules[OptimModule.__module__]
    if not hasattr(OptimModule,'logger'):OptimModule.logger=optimizer_module.logger
"""
code=code.replace(marker,patch+marker);sys.argv=[str(source),'--worker',str(args.index)]
exec(compile(code,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
