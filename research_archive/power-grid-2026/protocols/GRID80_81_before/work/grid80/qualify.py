"""Schema/static qualification for every frozen generated scenario, no controller outcome screening."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid80';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    D=read(OUT/'design.json');R=read(OUT/'generation.json');assert R['passed']
    E=ROOT/D['environment'];static=[]
    for name,h in D['source_static_sha256'].items():
        if name=='scenario_params.json':continue
        assert hashlib.sha256((E/name).read_bytes()).hexdigest()==h,name;static.append(name)
    sys.path[:0]=[str(ROOT/'work/grid81'),str(ROOT/'work/grid08')]
    from environment import make_env
    import numpy as np,pandas as pd
    env=make_env();rows=[]
    try:
        env.seed(0)
        for name in R['scenarios']:
            env.set_id(name);o=env.reset();folder=E/'chronics'/name
            assert np.isfinite(o.to_vect()).all()
            times={}
            for key in ['load_p','load_q','prod_p','load_p_forecasted','load_q_forecasted','prod_p_forecasted']:
                a=pd.read_csv(folder/(key+'.csv.bz2'),sep=';');assert len(a)>=2017 and np.isfinite(a.to_numpy()).all();times[key]=dict(rows=len(a),columns=len(a.columns))
            quality=read(folder/'generation_quality.json')
            assert np.isfinite(quality['max']) and quality['max']<=.05
            rows.append(dict(scenario=name,steps=int(env.max_episode_duration()),initial_rho=float(o.rho.max()),series=times,generator_ac_balance_max_mw=quality['max'],generator_forecast_failures=quality['forecasts_nb_fail'],generator_quality=quality))
        assert len(rows)==6
    finally:env.close()
    receipt=dict(passed=True,static_unchanged=static,scenarios=rows,controller_outcomes_screened=False,selection_unchanged=True,independent_real_grid=False)
    (OUT/'qualification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('COMPLETE: six frozen generated 2036 scenarios, static/schema qualification passed; no controller outcomes screened.\n',encoding='utf-8')
    print(json.dumps(receipt))
if __name__=='__main__':main()
