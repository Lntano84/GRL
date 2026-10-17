"""Reuse existing controller/timers; each item runs in a fresh serial process."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True);ap.add_argument('--index',type=int,required=True);args=ap.parse_args()
OUT=ROOT/'outputs/grid89_review'/args.phase;D=json.loads((OUT/'design.json').read_text());ITEM=D['execution_order'][args.index]
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
source=ROOT/'work/grid55/run.py';code=source.read_text(encoding='utf-8').replace("OUT=ROOT/'outputs/grid55'",f"OUT=ROOT/'outputs/grid89_review/{args.phase}'")
marker='    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose';assert code.count(marker)==1
patch="    import importlib.util\n    ss=importlib.util.spec_from_file_location('review_adapter',ROOT/'work/grid89_review/adapter.py')\n    aa=importlib.util.module_from_spec(ss);ss.loader.exec_module(aa);aa.install(engine.Search,item)\n    import common\n    def review_env():\n        return common.grid2op.make(str(ROOT/D['environment']),test=True,backend=common.LightSimBackend(),reward_class=common.MaxRhoReward,other_rewards=common.operational_other_rewards())\n    common.make_env=review_env\n"
code=code.replace(marker,patch+marker);sys.argv=[str(source),'--worker',str(args.index)]
exec(compile(code,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
