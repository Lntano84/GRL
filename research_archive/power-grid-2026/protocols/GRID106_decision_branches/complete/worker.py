"""Reuse the original worker and numeric controller; instrument one fixed omission."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid106_decision_branches'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);a=ap.parse_args()
item=json.loads((OUT/'design.json').read_text(encoding='utf8'))['execution_order'][a.index]
sys.path.insert(0,str(W));import hooks106
source=ROOT/'work/grid55/run.py';code=source.read_text(encoding='utf8')
assert code.count("OUT=ROOT/'outputs/grid55'")==1
code=code.replace("OUT=ROOT/'outputs/grid55'","OUT=ROOT/'outputs/grid106_decision_branches'")
old='worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker);worker.OUT=OUT'
new='''worker=importlib.util.module_from_spec(spec)
    worker_source=(ROOT/'work/grid55/worker.py').read_text(encoding='utf8')
    assert worker_source.count('            rows.append(row)')==1
    worker_source=worker_source.replace('            rows.append(row)',"            __import__('hooks106').on_step(row,env,obs,info)\\n            rows.append(row)")
    exec(compile(worker_source,str(ROOT/'work/grid55/worker.py'),'exec'),worker.__dict__);worker.OUT=OUT'''
assert code.count(old)==1;code=code.replace(old,new)
marker='    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose'
patch='''    __import__('hooks106').install(engine.Search,item)
    import common
    def branch_environment():
        return common.grid2op.make(str(ROOT/D['environment']),test=True,backend=common.LightSimBackend(),reward_class=common.MaxRhoReward,other_rewards=common.operational_other_rewards())
    common.make_env=branch_environment
    optimizer_module=sys.modules[OptimModule.__module__]
    if not hasattr(OptimModule,'logger'):OptimModule.logger=optimizer_module.logger
'''
assert code.count(marker)==1;code=code.replace(marker,patch+marker)
needle='        summary=worker.worker(item,D)';assert code.count(needle)==1
code=code.replace(needle,needle+"\n        __import__('hooks106').finished(item)")
sys.argv=[str(source),'--worker',str(a.index)]
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
