"""Zero-step diagnosis of parameter copying; no training, forecasts or env.step."""
from run_probe import *
env = make_env(); env.seed(0); env.set_id(D['training_weeks'][0]); obs = env.reset()
try:
    ctl = ResidualControl(env, obs)
    source = ctl.base.optim; saved = memory(source)
    clone = type(source)(env, env.action_space, config=dict(source.config), verbose=False)
    for name, value in saved.items():
        target = getattr(clone, name)
        if isinstance(target, cp.Parameter): target.value = value.copy()
        else: setattr(clone, name, value.copy())
    copied = memory(clone)
    rows = []
    for name in sorted(saved):
        equal = bool(np.array_equal(saved[name], copied[name], equal_nan=True))
        rows.append({'name': name, 'source_dtype': str(saved[name].dtype),
                     'clone_dtype': str(copied[name].dtype), 'same_numeric_values': equal,
                     'same_bytes': digest(saved[name]) == digest(copied[name]),
                     'shape': list(saved[name].shape)})
    write_json(OUT / 'clone_copy_diagnosis.json', {'physical_steps': 0, 'public_forecasts': 0,
               'rows': rows, 'all_numerically_equal': all(r['same_numeric_values'] for r in rows)})
    print(json.dumps([r for r in rows if not r['same_bytes']], indent=2))
finally:
    env.close()
