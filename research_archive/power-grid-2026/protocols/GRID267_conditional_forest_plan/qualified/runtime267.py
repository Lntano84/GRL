"""Fixed recipe and numerical rules; REAL fit/predict/cutoff only in later GRID268."""
import numpy as np
PARAMS=dict(n_estimators=64,max_depth=4,min_samples_leaf=10,max_features=1.0,bootstrap=False,n_jobs=1,criterion='gini',class_weight=None,warm_start=False)
def make_classifier(seed):
 from sklearn.ensemble import ExtraTreesClassifier
 return ExtraTreesClassifier(**PARAMS,random_state=int(seed))
def as_input(X):
 assert X.ndim==2 and X.shape[1]==40
 out=np.asarray(X,dtype=np.float32);assert np.isfinite(out).all()
 return out
def positive_score(model,X32):
 assert X32.dtype==np.float32 and np.array_equal(model.classes_,[0,1])
 p=model.predict_proba(X32)[:,1];assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
 return p

def protect_cutoff(scores,positive,KEEP=False):
 assert np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1))
 if KEEP or not np.any(positive):return 0.0,True
 tau=float(np.min(scores[positive]));assert np.all(scores[positive]>=tau)
 return tau,False

def apply_gate(parent_continue,scores,tau,KEEP=False):
 assert np.isfinite(tau) and 0<=tau<=1
 return parent_continue.copy() if KEEP else parent_continue&(scores>=tau)
