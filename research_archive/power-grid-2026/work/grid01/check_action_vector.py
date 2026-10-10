import json
from pathlib import Path
import numpy as np
import grid2op
from lightsim2grid import LightSimBackend
from trace_agent import action_vector

ROOT=Path(__file__).resolve().parents[2]
env=grid2op.make(str(Path(grid2op.__file__).parent/'data/l2rpn_idf_2023'),test=True,backend=LightSimBackend())
try:
    act=env.action_space()
    native=act.to_vect().copy()
    cache=act._vectorized
    gen=int(np.where(env.gen_redispatchable)[0][0])
    act.redispatch=[(gen,1.)]
    fresh=action_vector(act)
    assert not np.array_equal(fresh,native)
    assert act._vectorized is cache and np.array_equal(cache,native)
    assert np.array_equal(act.to_vect(),native),'Expected historical stale cache is no longer reproducible'
    reconstructed=env.action_space()
    reconstructed.from_vect(fresh)
    assert np.array_equal(reconstructed.redispatch,act.redispatch)
    assert np.array_equal(action_vector(reconstructed),fresh)
    untouched=env.action_space()
    assert untouched._vectorized is None
    assert np.array_equal(action_vector(untouched),native)
    assert untouched._vectorized is None
    result={'status':'PASS','historical_stale_cache_reproduced':True,'fresh_vector_roundtrip':True,
            'live_cache_unchanged':True,'uncached_action_stays_uncached':True,'actual_steps':0}
    (ROOT/'outputs/grid01/action_vector_regression.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
finally:
    env.close()
