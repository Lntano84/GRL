"""Same fixed physical model/rewards; only chronics directory changes."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def make_env():
    from common import grid2op,LightSimBackend,MaxRhoReward,operational_other_rewards
    return grid2op.make(str(ROOT/'work/grid03/formal_env_initialized'),test=True,backend=LightSimBackend(),
        reward_class=MaxRhoReward,other_rewards=operational_other_rewards())
