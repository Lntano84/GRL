"""One predeclared train-only FULL teacher pilot using the existing controller."""
from pathlib import Path
import json,sys,argparse,hashlib
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid100_data_foundation'
ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);a=ap.parse_args()
read=lambda p:json.loads(p.read_text(encoding='utf8'))
for n,h in read(OUT/'collector_code_freeze.json').items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
assert read(OUT/'collector_qualification.json')['passed'] and read(OUT/'generation_pilot_finished.json')['passed']
D=read(OUT/'design.json');runtime=read(OUT/'teacher_runtime/design.json');item=runtime['execution_order'][a.index]
assert a.index==0 and item['partition']=='train' and item['week']==D['pilot']['scenario'],'Only fixed pilot collector enabled now'
source=ROOT/'work/grid55/run.py';code=source.read_text(encoding='utf8').replace("OUT=ROOT/'outputs/grid55'","OUT=ROOT/'outputs/grid100_data_foundation/teacher_runtime'")
marker='    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose';assert code.count(marker)==1
patch="""    import importlib.util
    ss=importlib.util.spec_from_file_location('grid100_runtime_adapter',ROOT/'work/grid89_review/adapter.py')
    aa=importlib.util.module_from_spec(ss);ss.loader.exec_module(aa);aa.install(engine.Search,item)
    import common
    def learning_env():
        return common.grid2op.make(str(ROOT/'work/grid100_data_foundation/trainval_env'),test=True,backend=common.LightSimBackend(),reward_class=common.MaxRhoReward,other_rewards=common.operational_other_rewards())
    common.make_env=learning_env
"""
code=code.replace(marker,patch+marker);sys.argv=[str(source),'--worker',str(a.index)]
exec(compile(code,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
