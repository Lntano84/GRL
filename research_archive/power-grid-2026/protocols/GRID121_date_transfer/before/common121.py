"""Date-held-out diagnostic conditional on the historically fixed task price."""
from pathlib import Path
import sys,collections
R=Path('D:/Codex/GRL研究')
sys.path.insert(0,str(R/'work/grid120_decision_objective'))
from common120 import *
from common120 import verify as verify120
W=Path(__file__).parent;O=R/'outputs/grid121_date_transfer'

def verify():
    for n in ['code_freeze.json','data_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    verify120();sealed()

def arrays():
    with np.load(O/'train_arrays.npz') as z:return z['raw'],z['target'],z['original_probability']

def fold_data(fold):
    raw,y,p=arrays();meta=read(O/'metadata.json');d=read(O/'design.json');held=set(d['folds'][fold]['held_dates']);hold=np.array([m['date'] in held for m in meta]);train=~hold
    z=np.load(O/f'fold{fold}_normalization.npz');x=np.clip((raw-z['mean'])/z['scale'],-20,20).astype('f')
    ids=np.flatnonzero(train);prob=p[ids]/p[ids].sum();Z=float(prob@abs(y[ids]).astype(float))
    return x,y,ids,np.flatnonzero(hold),prob,Z

def normalize_stats(raw,meta,mask):
    counts=collections.Counter(m['date'] for m,keep in zip(meta,mask) if keep)
    w=np.array([1/counts[m['date']] for m,keep in zip(meta,mask) if keep]);w/=w.mean()
    X=raw[mask].astype(float);mean=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-mean)**2,axis=0,weights=w));scale[scale<1e-8]=1.
    return mean,scale
