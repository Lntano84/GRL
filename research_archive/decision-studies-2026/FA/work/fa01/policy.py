"""FA01 changes only batch acquisition; FA00 models and eligibility are retained."""
import sys
import time
from pathlib import Path
import numpy as np

FA00=Path(__file__).resolve().parent.parent/"fa00"
sys.path.insert(0,str(FA00))
from core import Selector, keyed_seed

def choose_indices(candidates, size, seed, round_no, mode):
    """Canonical ordering prevents random selection from depending on scores.

    Cost-only ties are randomized with the same acquisition stream. All selected
    sets are shuffled for execution using FA00's seed/round/batch stream.
    """
    assert mode in ("RANDOM-100","CHEAP-100")
    q=min(int(size),len(candidates))
    canonical=np.lexsort((candidates[:,1],candidates[:,0]))
    acquire=np.random.default_rng(keyed_seed(seed,"acquire",round_no))
    order=canonical[acquire.permutation(len(candidates))]
    if mode=="CHEAP-100":
        costs=candidates[order,3]
        assert np.all(np.isfinite(costs)) and np.all(costs>=0)
        order=order[np.argsort(costs,kind="stable")]
    picked=order[:q].copy()
    execute=np.random.default_rng(keyed_seed(seed,"batch",round_no))
    execute.shuffle(picked)
    return picked

class ControlSelector(Selector):
    def __init__(self,*args,mode,**kwargs):
        super().__init__(*args,**kwargs)
        self.mode=mode

    def select_batch(self,candidates,batch_size,round_no):
        started=time.perf_counter()
        indices=choose_indices(candidates,batch_size,self.seed,round_no,self.mode)
        self.selection_s+=time.perf_counter()-started
        return candidates[indices]
