"""Single train-scenario forecast qualifies existing physical interface, outside timing."""
import sys
from registry import ROOT,OUT,save,test_sealed
test_sealed()
sys.path[:0]=[str(ROOT/'work/grid30'),str(ROOT/'work/grid29'),str(ROOT/'work/grid08'),str(ROOT/'work/grid28')]
from common import grid2op,LightSimBackend,MaxRhoReward,operational_other_rewards,ResidualControl,OptimModule,flags
from engine import Search
module=sys.modules[OptimModule.__module__]
if not hasattr(OptimModule,'logger'):OptimModule.logger=module.logger
env=grid2op.make(str(ROOT/'work/grid100_data_foundation/trainval_env'),test=True,backend=LightSimBackend(),reward_class=MaxRhoReward,other_rewards=operational_other_rewards())
try:
    env.seed(0);env.set_id('2037-01-05_100');obs=env.reset();ctl=ResidualControl(env,obs);search=Search(env,ctl,'STATIC_ID',128,None)
    eligible,reps,aliases,actions,changed=search.prepare(obs,env.action_space());assert reps
    n=int(env.nb_highres_called);future,reward,done,info=obs.simulate(actions[sorted(reps)[0]])
    assert int(env.nb_highres_called)-n==1
    # Grid2Op forecast observations carry extra fields; validate using that class.
    restored=future.copy();restored.from_vect(future.to_vect())
    assert float(restored.rho.max())==float(future.rho.max())
    save(OUT/'live_qualification.json',dict(passed=True,scenario='2037-01-05_100',partition='train',actual_forecasts=1,
        observation_size=len(obs.to_vect()),forecast_observation_size=len(future.to_vect()),representatives=len(reps),flags=flags(info),
        scope='Physical interface engineering qualification, not a benchmark result or independent AC re-solve.'))
finally:env.close()
