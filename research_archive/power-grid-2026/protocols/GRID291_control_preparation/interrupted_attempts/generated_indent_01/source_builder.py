"""Pure source construction, no imports of controller, models or environment."""
from pathlib import Path
R=Path('D:/Codex/GRL研究')
def instrument_worker(text):
 marker='            rows.append(row)'
 assert text.count(marker)==1
 text=text.replace(marker,marker+"\n            with (path/'partial_steps.jsonl').open('a',encoding='utf8') as jf:jf.write(json.dumps(row,allow_nan=False)+chr(10))")
 marker='            assert time.perf_counter()-entered < design["caps"]["episode_s"]'
 assert text.count(marker)==1
 return text.replace(marker,marker+'\n            assert time.perf_counter() < _global_deadline, "original manager window exhausted"')
def build():
 source=R/'work/grid55/run.py';code=source.read_text(encoding='utf8');assert code.count("OUT=ROOT/'outputs/grid55'")==1;code=code.replace("OUT=ROOT/'outputs/grid55'","OUT=ROOT/'outputs/grid291_control_preparation'")
 # Keep a per-step public journal so hard deadline termination retains paid outputs.
 marker='worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker);worker.OUT=OUT';assert code.count(marker)==1
 replacement="""worker=importlib.util.module_from_spec(spec)
     from source_builder import instrument_worker
     worker_code=instrument_worker((ROOT/'work/grid55/worker.py').read_text(encoding='utf8'))
     exec(compile(worker_code,str(ROOT/'work/grid55/worker.py'),'exec'),worker.__dict__);worker.OUT=OUT
     worker._global_deadline=load(OUT/'run_context.json')['deadline_monotonic']"""
 code=code.replace(marker,replacement)
 marker='    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose';assert code.count(marker)==1
 patch="""    import importlib.util,torch
      spec291=importlib.util.spec_from_file_location('g291_live_models',ROOT/'work/grid291_control_preparation/models.py');models291=importlib.util.module_from_spec(spec291);spec291.loader.exec_module(models291)
     t291=time.perf_counter();payload291=models291.FrozenModels(item['pool']);model_load_s291=time.perf_counter()-t291
     spec291=importlib.util.spec_from_file_location('g291_live_adapter',ROOT/'work/grid291_control_preparation/adapter.py');adapter291=importlib.util.module_from_spec(spec291);spec291.loader.exec_module(adapter291);adapter291.install(engine.Search,item,payload291)
     save(OUT/f"worker_setup_{args.worker:02d}.json",dict(pool=item['pool'],model_load_s=model_load_s291,warmup_predictions=0))
     import common
     def prepared_development_env():
         return common.grid2op.make(str(ROOT/D['environment']),test=True,backend=common.LightSimBackend(),reward_class=common.MaxRhoReward,other_rewards=common.operational_other_rewards())
     common.make_env=prepared_development_env
     optimizer_module=sys.modules[OptimModule.__module__]
     if not hasattr(OptimModule,'logger'):OptimModule.logger=optimizer_module.logger
 """
 code=code.replace(marker,patch+marker)
 return code
