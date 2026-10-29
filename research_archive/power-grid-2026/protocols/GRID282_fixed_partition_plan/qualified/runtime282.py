"""Fixed-partition leaf-frequency arithmetic; REAL updates only after GRID282."""
import numpy as np

def route_forest(model,X32):
 assert X32.ndim==2 and X32.shape[1]==39 and X32.dtype==np.float32 and np.isfinite(X32).all()
 assert model.n_features_in_==39 and len(model.estimators_)==64 and np.array_equal(model.classes_,[0,1])
 return np.column_stack([t.apply(X32).astype(np.int32) for t in model.estimators_])

def original_leaf_parameters(model):
 priors=[];masks=[]
 for t in model.estimators_:
  v=t.tree_.value[:,0,:];den=v.sum(axis=1);assert v.shape[1]==2 and (den>0).all() and np.isfinite(v).all() and (v>=0).all()
  p=(v[:,1]/den).copy();mask=(t.tree_.children_left==-1).copy();assert (p>=0).all() and (p<=1).all();priors.append(p);masks.append(mask)
 return priors,masks

def validate_routes(routes,masks):
 assert routes.ndim==2 and routes.shape[1]==len(masks) and np.issubdtype(routes.dtype,np.integer)
 for j,mask in enumerate(masks):
  ids=routes[:,j];assert (ids>=0).all() and (ids<len(mask)).all() and mask[ids].all()

def fit_leaf_tables(routes,y,weight,masks,priors):
 validate_routes(routes,masks);assert len(y)==len(weight)==len(routes) and len(priors)==len(masks)
 assert np.isin(y,[0,1]).all() and np.isfinite(weight).all() and (weight>=0).all()
 tables=[];statistics=[]
 for j,(mask,prior) in enumerate(zip(masks,priors)):
  n=len(prior);assert len(mask)==n and np.isfinite(prior).all() and (prior>=0).all() and (prior<=1).all()
  mass=np.bincount(routes[:,j],weights=weight,minlength=n);positive=np.bincount(routes[:,j],weights=weight*(y>0),minlength=n);count=np.bincount(routes[:,j],minlength=n)
  observed=mask&(mass>0);table=prior.copy();table[observed]=positive[observed]/mass[observed]
  assert np.isfinite(table).all() and (table>=0).all() and (table<=1).all() and np.array_equal(table[mask&~observed],prior[mask&~observed])
  tables.append(table);statistics.append(dict(mass=mass,positive_mass=positive,count=count,empty_mask=mask&~observed))
 return tables,statistics

def score_routes(routes,tables,masks):
 validate_routes(routes,masks);assert len(tables)==len(masks)>0
 p=np.mean(np.column_stack([table[routes[:,j]] for j,table in enumerate(tables)]),axis=1)
 assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
 return p

def as_input(X):
 assert X.ndim==2 and X.shape[1]==39
 out=np.asarray(X,dtype=np.float32);assert np.isfinite(out).all()
 return out

def protect_cutoff(scores,positive,KEEP=False):
 assert np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1))
 if KEEP or not np.any(positive):return 0.0,True
 tau=float(np.min(scores[positive]));assert np.all(scores[positive]>=tau)
 return tau,False

def apply_gate(parent_continue,scores,tau,KEEP=False):
 assert np.isfinite(tau) and 0<=tau<=1
 return parent_continue.copy() if KEEP else parent_continue&(scores>=tau)
