"""Shared, public-forecast-only guard. Not a learned policy or a safety proof."""
from common import *

def branch_safe(outcomes):
    return len(outcomes)==2 and all(not r['done'] and not r['illegal'] and
        not r['ambiguous'] and not r['exceptions'] and
        np.isfinite(r['rho_max']) and r['rho_max']>0 for r in outcomes)

def forecast_branch(obs,act):
    forecast=obs.get_forecast_env()
    results=[]
    try:
        forecast.reset()
        for action in [act,forecast.action_space()]:
            future,_,done,info=forecast.step(action)
            results.append({'done':bool(done),**flags(info),'rho_max':float(future.rho.max())})
            if done:break
    finally:forecast.close()
    return results

class TwoStepResidualControl(ResidualControl):
    def propose(self,obs,reward=0.,cadence=6):
        base,rest,result=super().propose(obs,reward,cadence)
        result['two_step_guard']={'checked':False,'forecast_steps':0}
        if rest is None:return base,rest,result
        original=digest(obs.to_vect())
        branches=[forecast_branch(obs,a) for a in [base,rest]]
        safe=[branch_safe(x) for x in branches]
        assert digest(obs.to_vect())==original
        assert digest(self.env.get_obs().to_vect())==original
        result['two_step_guard']={'checked':True,'safe':safe,'branches':branches,
                                  'forecast_steps':sum(map(len,branches))}
        # Only veto the offered restoration. The unchanged base agent remains
        # responsible outside this auxiliary action; both-failing is not rescued.
        if not safe[1]:
            result.update(offered=False,reason='two_step_restore_veto');rest=None
        return base,rest,result

if __name__=='__main__':
    report=json.loads((OUT/'ramp_diagnostic.json').read_text())
    assert report['qualified']
    assert not branch_safe(report['branches']['public_forecast_restore'])
    assert branch_safe(report['branches']['public_forecast_hold'])
    good=[{'done':False,'illegal':False,'ambiguous':False,'exceptions':[],'rho_max':.5}]*2
    assert branch_safe(good)
    for key,value in [('done',True),('illegal',True),('ambiguous',True),('exceptions',['failure']),('rho_max',0.)]:
        bad=[dict(x) for x in good];bad[1][key]=value;assert not branch_safe(bad)
    assert not branch_safe(good[:1])
    print('Two-step guard checks passed; no environment transitions in this check')
