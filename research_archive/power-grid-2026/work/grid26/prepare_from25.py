"""Preserve pilot instrumentation and remove per-observation bound-method cycles."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/grid26"


def write(name, text):
    path = OUT / name
    assert not path.exists(), path
    path.write_text(text, encoding="utf-8")


def replace_once(text, a, b):
    assert text.count(a) == 1, a
    return text.replace(a, b)


for name in ["engine.py", "prepare.py", "preflight.py", "audit.py", "review.py"]:
    text = (ROOT / "work/grid25" / name).read_text(encoding="utf-8")
    text = text.replace("work/grid25", "work/grid26").replace("outputs/grid25", "outputs/grid26")
    if name == "prepare.py":
        text = replace_once(text, '"stage": "GRID25_CONTROLLED_SERIAL_TIMING"', '"stage": "GRID26_CONTROLLED_SERIAL_TIMING_CLASS_TIMER"')
        text = replace_once(text, '"instrumentation": "Light simulation timers',
            '"instrumentation": "Class-level simulation timer delegates to the original unbound function without setting a bound method on each observation. No per-observation reference cycle. Light simulation timers')
    if name == "review.py": text = text.replace("GRID25", "GRID26")
    write(name, text)

text = (ROOT / "work/grid25/run.py").read_text(encoding="utf-8")
text = text.replace("work/grid25", "work/grid26").replace("outputs/grid25", "outputs/grid26")
a = '''            original_simulate = obs.simulate
            def measured_simulate(*args, **kwargs):
                t0 = time.perf_counter()
                try: return original_simulate(*args, **kwargs)
                finally:
                    simulation["count"] += 1
                    simulation["seconds"] += time.perf_counter()-t0
            obs.simulate = measured_simulate
'''
text = replace_once(text, a, '            active_simulation = simulation\n')
text = replace_once(text, '                obs.simulate = original_simulate', '                active_simulation = None')
text = replace_once(text, '    initial = int(env.nb_highres_called)\n', '''    initial = int(env.nb_highres_called)
    active_simulation = None
    obs_class = type(obs)
    owned_simulate = "simulate" in obs_class.__dict__
    original_simulate = obs_class.simulate
    def measured_simulate(observation, *args, **kwargs):
        if active_simulation is None:
            return original_simulate(observation, *args, **kwargs)
        t0 = time.perf_counter()
        try: return original_simulate(observation, *args, **kwargs)
        finally:
            active_simulation["count"] += 1
            active_simulation["seconds"] += time.perf_counter()-t0
    obs_class.simulate = measured_simulate
''')
text = replace_once(text, '    finally: env.close()', '''    finally:
        if owned_simulate: obs_class.simulate = original_simulate
        else: delattr(obs_class, "simulate")
        env.close()''')
write("run.py", text)
print("GRID26 sources prepared; GRID25 source/results unchanged.")
