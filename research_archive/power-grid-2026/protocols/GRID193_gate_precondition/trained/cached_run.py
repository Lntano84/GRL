"""Cache immutable training matrices; same objective,gradient and fit protocol."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
import numpy as np
cache={}
def cached_objective(z,u,d):
    key=(str(z.zip.filename),d.tobytes())
    if key not in cache:
        w=z['sample_weight'];cache[key]=(z['standard']/d,w,z['y'],w.sum())
    x,w,y,S=cache[key];logit=x@u[:-1]+u[-1];beta=u[:-1]/d;p=np.exp(-np.logaddexp(0.,-logit));e=w*(p-y)/S
    return float(w@(np.logaddexp(0.,logit)-y*logit)/S+beta@beta/(2*S)),np.r_[x.T@e+u[:-1]/(S*d*d),e.sum()]
if __name__=='__main__':
    C.verify();C.save('engineering_io_note.json',dict(original_source_unchanged=True,wrapper_sha256=C.sha(Path(__file__)),change='Cache immutable decompressed standard/weights/labels and Z/d once perpool. No objective/gradient/init/stopping/budget change; prevents repeated NPZ member decompression inside L-BFGS.',before_any_new_training=True))
    C.objective=cached_objective;getattr(C,sys.argv[1])()
