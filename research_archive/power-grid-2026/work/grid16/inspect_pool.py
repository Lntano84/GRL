"""Zero-step pretrained policy/library compatibility check, no fits or forecast."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/grid08"))
from common import np, make_env, ResidualControl, action_vector, digest, write_json
env = make_env(); env.seed(0); env.set_id("2035-01-29_4"); obs = env.reset()
try:
    ctl = ResidualControl(env, obs)
    module = ctl.base.topo_12_unsafe
    neural_pool = module.gym_env.action_space.topo_actions_list
    hashes = [digest(action_vector(a)) for a in neural_pool]
    n1_hashes = [digest(action_vector(a)) for a in ctl.base.topo_n1_unsafe.topo_act_list]
    available = set(hashes)
    result = {"neural_pool_size": len(hashes), "n1_pool_size": len(n1_hashes),
              "overlap": len(set(n1_hashes)&available), "n1_covered_by_neural": [i for i, h in enumerate(n1_hashes) if h in available],
              "neural_hashes": hashes, "physical_steps": 0, "forecasts": int(env.nb_highres_called), "model_fits": 0,
              "note": "Coverage only, not effectiveness in outage states. Pretrained NN is an existing learning baseline."}
    write_json(ROOT / "outputs/grid16/pool_preflight.json", result)
    print({k:v for k,v in result.items() if k not in ("neural_hashes", "n1_covered_by_neural")})
finally:
    env.close()
