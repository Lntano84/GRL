"""Qualified floating operations for the one fixed guarded score-bias hypothesis."""
import numpy as np

def base_score(X,b,theta,alpha):
    assert X.dtype==b.dtype==theta.dtype==np.float64 and X.shape[1]==40 and theta.shape==(41,)
    endpoint=b+np.sum(X*theta[:-1],axis=1)+theta[-1]
    return b+float(alpha)*(endpoint-b)

def guarded_shift(scores,positive,KEEP=False):
    # Call on REAL scores only in the separately published GRID265 fitting stage.
    assert scores.dtype==np.float64 and np.isfinite(scores).all()
    if KEEP or not np.any(positive):return 0.0,True
    assert np.all(scores[positive]>=0)
    delta=float(np.min(scores[positive]))
    assert delta>=0 and np.all((scores-delta)[positive]>=0)
    return delta,False

def apply_shift(scores,base_CONT,delta):
    assert np.isfinite(delta) and delta>=0
    shifted=scores-delta
    return shifted,base_CONT&(shifted>=0)
