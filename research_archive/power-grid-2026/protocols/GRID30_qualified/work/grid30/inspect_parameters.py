"""Diagnose replica setter failure on the retained public pre-control state."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(ROOT/'work/grid08'))
from common import np,make_env,ResidualControl,write_json
import cvxpy as cp
env=make_env();env.seed(0);env.set_id('2035-03-05_0');obs=env.reset()
try:
    ctl=ResidualControl(env,obs)
    state=np.load(ROOT/'outputs/grid29c/runs/r0__2035-03-05_0__ORIGINAL/vectors.npz')['observation'][334]
    obs.from_vect(state);assert obs.current_step==335
    ctl.safe.reset(obs);ctl.safe._update_storage_power_obs(obs);ctl.safe.update_parameters(obs,safe=True)
    invalid=[]
    for name,param in vars(ctl.safe).items():
        if isinstance(param,cp.Parameter) and param.value is not None:
            try:param._validate_value(np.array(param.value,copy=True))
            except ValueError as e:
                v=np.asarray(param.value);invalid.append({'name':name,'shape':list(v.shape),
                    'min':float(v.min()),'max':float(v.max()),'negative_count':int((v<0).sum()),'error':str(e)})
    write_json(ROOT/'outputs/grid30/parameter_copy_diagnostic.json',{'state':335,'invalid_setter_values':invalid,
        'scope':'Existing author-populated Parameters inspected on saved public state. No values modified/clipped, no hidden date, no full grid replay.'})
    print(json.dumps(invalid,indent=2))
finally:env.close()
