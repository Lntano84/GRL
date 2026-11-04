"""Static generation only; compile during offline prep, never execute there."""
from pathlib import Path
import importlib.util
R=Path('D:/Codex/GRL研究')
s=importlib.util.spec_from_file_location('g300_original_builder',R/'work/grid297_paired_collection_preparation/source_builder.py');old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
def instrument_worker(text):
 text=old.instrument_worker(text)
 return old.replace_once(text,'            t = time.perf_counter(); obs, reward, done, info = env.step(action)','            _intervention.before(step,before,digest(action_vector(action)))\n            t = time.perf_counter(); obs, reward, done, info = env.step(action)')
def build():
 text=old.build();text=old.replace_once(text,"OUT=ROOT/'outputs/grid297_paired_collection_preparation'","OUT=ROOT/'outputs/grid300_new_training_dates'")
 return old.replace_once(text,"D=load(OUT/'design.json')","D=load(OUT/'runtime_design.json')")
