"""Frozen261 pure public input adapter; no labels, forecasts, model fitting or file IO."""
import numpy as np
PUBLIC_KEEP=[j for j in range(41) if j not in [14,18,19]]
def pack_public(raw41,mean41,scale41,b):
 raw=np.asarray(raw41);mean=np.asarray(mean41);scale=np.asarray(scale41);b=np.asarray(b)
 if raw.ndim!=2 or raw.shape[1]!=41 or mean.shape!=(41,) or scale.shape!=(41,) or b.shape!=(len(raw),):raise ValueError("public shape")
 if not all(np.isfinite(a).all() for a in [raw,mean,scale,b]) or not (scale>0).all():raise ValueError("public finite/scale")
 standardized=(raw-mean)/scale
 return np.c_[standardized[:,PUBLIC_KEEP],b,np.ones(len(b))]
def score_public40(X,b,base_CONT,theta,alpha):
 X=np.asarray(X);b=np.asarray(b);base_CONT=np.asarray(base_CONT);theta=np.asarray(theta)
 if X.shape!=(len(b),40) or theta.shape!=(41,) or base_CONT.shape!=b.shape or base_CONT.dtype!=bool:raise ValueError("score shape/types")
 if not all(np.isfinite(a).all() for a in [X,b,theta]) or not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError("score finite/alpha")
 if not np.array_equal(base_CONT,b>0) or not np.array_equal(X[:,38],b) or not (X[:,39]==1).all():raise ValueError("parent/nativeSTOP contract")
 Z=X.copy();Z[:,:38]=np.tanh(Z[:,:38])
 end=b+np.sum(Z*theta[:-1],axis=1)+theta[-1]
 logit=b+alpha*(end-b)
 return logit,base_CONT&(logit>=0)
def parent_margin(event_p,positive_mu,charge):
 p=np.asarray(event_p);m=np.asarray(positive_mu);c=np.asarray(charge)
 if p.shape!=m.shape or p.shape!=c.shape or not all(np.isfinite(a).all() for a in [p,m,c]) or not ((p>=0)&(p<=1)).all() or not (m>0).all() or not (c>0).all():raise ValueError("parent fields")
 return (p*m-c)/(p*m+c)
