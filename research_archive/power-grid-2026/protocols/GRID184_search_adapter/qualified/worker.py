"""Guarded original common-controller worker. Physical approval required."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path('D:/Codex/GRL研究');OUT=ROOT/'outputs/grid184_search_adapter'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);args=ap.parse_args()
sys.path.insert(0,str(Path(__file__).parent));import run as registry
registry.verify();authorization=registry.read(OUT/'physical_authorization.json');assert authorization['approved_by_user'] is True and authorization['design_sha256']==registry.sha(OUT/'design.json')
assert registry.read(OUT/'offline_review.json')['passed']
D=registry.read(OUT/'design.json');item=D['execution_order'][args.index]
source=ROOT/'work/grid55/run.py';code=source.read_text(encoding='utf8');assert code.count("OUT=ROOT/'outputs/grid55'")==1
code=code.replace("OUT=ROOT/'outputs/grid55'","OUT=ROOT/'outputs/grid184_search_adapter'")
marker='    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose';assert code.count(marker)==1
patch="""    import importlib.util
    adapter_spec=importlib.util.spec_from_file_location('grid184_live_adapter',ROOT/'work/grid184_search_adapter/adapter.py')
    adapter=importlib.util.module_from_spec(adapter_spec);adapter_spec.loader.exec_module(adapter)
    adapter.install(engine.Search,item)
    import common
    def prepared_development_env():
        return common.grid2op.make(str(ROOT/D['environment']),test=True,backend=common.LightSimBackend(),reward_class=common.MaxRhoReward,other_rewards=common.operational_other_rewards())
    common.make_env=prepared_development_env
    optimizer_module=sys.modules[OptimModule.__module__]
    if not hasattr(OptimModule,'logger'):OptimModule.logger=optimizer_module.logger
"""
code=code.replace(marker,patch+marker);sys.argv=[str(source),'--worker',str(args.index)]
exec(compile(code,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
