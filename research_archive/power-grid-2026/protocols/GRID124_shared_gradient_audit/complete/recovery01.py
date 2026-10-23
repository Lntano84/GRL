"""Preserve frozen source; correct only NumPy boolean negation in BCE reference."""
import diagnose as d
import sys
np=d.np

def fixed_metrics(q,y,event,p,arm,prevalence):
    p=p/p.sum();err=q[:,1].astype(float)-y;hub=np.where(abs(err)<1,.5*err**2,abs(err)-.5);go=q[:,1]>0
    out=dict(states=len(y),main_huber=float(p@hub),sign_regret=float(p@(np.maximum(y,0)-go*y)),false_stop_mass=float(p@((y>0)&~go)),unprofitable_continue_mass=float(p@((y<=0)&go)))
    if arm!='NET_ZERO':
        t=(event>0) if arm=='AUX_GAIN' else y>0;z=q[:,0].astype(float);pred=1/(1+np.exp(-z));base=np.clip(prevalence,1e-15,1-1e-15)
        constant_bce=(-np.log(base))*t+(-np.log1p(-base))*(~t)
        out.update(aux_prevalence=float(p@t),aux_auc=d.auc(z,t,p),aux_bce=float(p@(np.logaddexp(0,z)-t*z)),constant_fit_prevalence_bce=float(p@constant_bce),aux_brier=float(p@((pred-t)**2)),constant_fit_prevalence_brier=float(p@((base-t)**2)))
    return out

if __name__=='__main__':
    assert d.sha(d.W/'recovery01.py')==d.read(d.O/'engineering_recovery_01.json')['correction_sha256']
    d.metrics=fixed_metrics
    {'run':d.run,'audit':d.audit}[sys.argv[1]]()
