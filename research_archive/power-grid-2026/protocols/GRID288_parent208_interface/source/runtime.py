"""Original208 saved-public parent interface; does not fit or construct environments."""
import numpy as np
def pack_parent(raw83,raw41,scaler):
 h=np.asarray(raw83);r=np.asarray(raw41)
 if h.ndim!=2 or h.shape[1]!=83 or r.shape!=(len(h),41):raise ValueError("raw public shape")
 if not np.isfinite(h).all() or not np.isfinite(r).all():raise ValueError("raw nonfinite")
 for name,n in [("mean",83),("scale",83),("raw_mean",41),("raw_scale",41)]:
  if scaler[name].shape!=(n,) or not np.isfinite(scaler[name]).all():raise ValueError("scaler")
 if not (scaler["scale"]>0).all() or not (scaler["raw_scale"]>0).all():raise ValueError("scaler zero")
 if not (h[:,82]>0).all():raise ValueError("charge")
 return np.c_[(h-scaler["mean"])/scaler["scale"],(r-scaler["raw_mean"])/scaler["raw_scale"]]
def predict_parent(x124,charge,event,amount):
 if x124.ndim!=2 or x124.shape[1]!=124 or charge.shape!=(len(x124),):raise ValueError("parent shape")
 p=event.predict_proba(x124)[:,1]
 mu=amount.predict(x124)
 if not np.isfinite(p).all() or not np.isfinite(mu).all() or not ((p>=0)&(p<=1)).all() or not (mu>0).all():raise ValueError("native output")
 gain=p*mu;b=(gain-charge)/(gain+charge)
 return dict(event_p=p,mu=mu,charge=charge,b=b,base_CONT=b>0)
