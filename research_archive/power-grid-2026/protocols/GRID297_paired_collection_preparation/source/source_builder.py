"""Static assembly only. No live imports at preparation time."""
from pathlib import Path
R=Path('D:/Codex/GRL研究')
def replace_once(text,old,new):
 if text.count(old)!=1:raise ValueError('frozen source marker differs: '+old)
 return text.replace(old,new)
def instrument_worker(text):
 text=replace_once(text,'            rows.append(row)',"            rows.append(row)\n            _intervention.accept(row)\n            with (path/'partial_steps.jsonl').open('a',encoding='utf8') as jf:jf.write(json.dumps(row,allow_nan=False)+chr(10))")
 text=replace_once(text,'    def measured_simulate(observation, *args, **kwargs):','    def measured_simulate(observation, *args, **kwargs):\n        check_runtime_budget()\n        if _intervention.forecasts >= _forecast_remaining: raise RuntimeError("global forecast cap")\n        _intervention.forecasts += 1')
 text=replace_once(text,'            assert time.perf_counter()-entered < design["caps"]["episode_s"]','            assert time.perf_counter()-entered < design["caps"]["episode_s"]\n            check_runtime_budget()\n            if _intervention.steps >= _physical_remaining: raise RuntimeError("global physical step cap")')
 text=replace_once(text,'            t = time.perf_counter(); obs, reward, done, info = env.step(action)','            check_runtime_budget()\n            t = time.perf_counter(); obs, reward, done, info = env.step(action)')
 text=replace_once(text,'        write_json(path / "summary.json", summary)','        write_json(path / "intervention.json", _intervention.record())\n        write_json(path / "summary.json", summary)')
 text=replace_once(text,'    except Exception:\n','    except Exception:\n        write_json(path / "partial_intervention.json", _intervention.record())\n')
 # Install the same counter before controller construction so setup forecasts
 # also consume the global cap. No model, solver or action numeric code changes.
 begin=text.index('    initial = int(env.nb_highres_called)')
 end=text.index('    obs_class.simulate = measured_simulate',begin)+len('    obs_class.simulate = measured_simulate\n')
 block=text[begin:end];text=text[:begin]+text[end:]
 text=replace_once(text,'    ctl = ResidualControl(env, obs)',block+'    ctl = ResidualControl(env, obs)')
 return text
def build():
 text=(R/'work/grid55/run.py').read_text(encoding='utf8');text=replace_once(text,"OUT=ROOT/'outputs/grid55'","OUT=ROOT/'outputs/grid297_paired_collection_preparation'")
 old='worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker);worker.OUT=OUT'
 new="worker=importlib.util.module_from_spec(spec)\n    from source_builder import instrument_worker\n    exec(compile(instrument_worker((ROOT/'work/grid55/worker.py').read_text(encoding='utf8')),str(ROOT/'work/grid55/worker.py'),'exec'),worker.__dict__);worker.OUT=OUT\n    worker.check_runtime_budget=check_runtime_budget\n    worker._forecast_remaining=run_ctx['forecast_remaining']\n    worker._physical_remaining=run_ctx['physical_remaining']"
 text=replace_once(text,old,new)
 marker='    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose'
 lines=["    import importlib.util,torch", "    spec297=importlib.util.spec_from_file_location('g297_models',ROOT/'work/grid291_control_preparation/models.py');models297=importlib.util.module_from_spec(spec297);spec297.loader.exec_module(models297)","    model_started=time.perf_counter();payload297=models297.FrozenModels(item['pool']);model_load_s=time.perf_counter()-model_started", "    import paired_adapter", "    expected=None", "    if item['within_pair']==1:","        peer=D['execution_order'][args.worker-1]", "        peer_path=OUT/'runs'/f\"r{peer['repeat']}__{peer['week']}__{peer['method']}\"", "        expected_record=load(peer_path/'intervention.json')", "        expected=expected_record['event']['common'] if expected_record['event'] else {'status':'NO_ELIGIBLE_EVENT'}", "    worker._intervention=paired_adapter.install(engine.Search,item,payload297,expected)","    save(OUT/f'worker_setup_{args.worker:03d}.json',dict(pool=item['pool'],model_load_s=model_load_s,warmup_predictions=0))", "    import common", "    def fixed_own_training_env():", "        return common.grid2op.make(str(ROOT/D['environment']),test=True,backend=common.LightSimBackend(),reward_class=common.MaxRhoReward,other_rewards=common.operational_other_rewards())", "    common.make_env=fixed_own_training_env", "    optimizer_module=sys.modules[OptimModule.__module__]", "    if not hasattr(OptimModule,'logger'):OptimModule.logger=optimizer_module.logger"]
 return replace_once(text,marker,'\n'.join(lines)+'\n'+marker)
