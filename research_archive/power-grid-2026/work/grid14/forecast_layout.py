"""One public forecast for serialized forecast schema qualification; no physical step."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import *
OUT = ROOT / 'outputs/grid14'
env = make_env(); env.seed(0); env.set_id('2035-01-29_4'); initial = env.reset()
try:
    future, _, _, _ = initial.simulate(env.action_space(), time_step=1)
    def schema(obs):
        return [{'name': name, 'shape': list(np.asarray(getattr(obs, name)).shape),
                 'size': int(np.asarray(getattr(obs, name)).size)} for name in obs.attr_list_vect]
    result = {'native_size': len(initial.to_vect()), 'forecast_size': len(future.to_vect()),
              'native_schema': schema(initial), 'forecast_schema': schema(future),
              'physical_steps': 0, 'public_forecasts': int(env.nb_highres_called)}
    write_json(OUT / 'forecast_schema.json', result)
    print(json.dumps(result, indent=2))
finally:
    env.close()
