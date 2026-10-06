import time
import numpy as np
import pandas as pd
from common import Selector, keyed_seed

def lexical_indices(candidates,size,seed,round_no):
    assert candidates.shape[1]==5
    canonical=np.lexsort((candidates[:,1],candidates[:,0]))
    rng=np.random.default_rng(keyed_seed(seed,'acquire',round_no))
    order=canonical[rng.permutation(len(candidates))]
    for column,ascending in [(4,False),(2,False),(3,True)]:
        values=candidates[order,column]
        order=order[np.argsort(values if ascending else -values,kind='stable')]
    chosen=order[:min(size,len(order))].copy()
    np.random.default_rng(keyed_seed(seed,'batch',round_no)).shuffle(chosen)
    return chosen

def cheap_indices(candidates,size,seed,round_no):
    canonical=np.lexsort((candidates[:,1],candidates[:,0]))
    rng=np.random.default_rng(keyed_seed(seed,'acquire',round_no))
    order=canonical[rng.permutation(len(candidates))]
    order=order[np.argsort(candidates[order,3],kind='stable')]
    chosen=order[:min(size,len(order))].copy()
    np.random.default_rng(keyed_seed(seed,'batch',round_no)).shuffle(chosen)
    return chosen

def row_batch(cache,train_rows,seed,round_no,size=11,cap=100):
    pool=np.array(sorted(int(i) for i in train_rows if any(
        cache.can_query(int(i),a,cap) for a in range(cache.kind.shape[1]))),dtype=int)
    if not len(pool):return pool,[],[]
    rng=np.random.default_rng(keyed_seed(seed,'row_acquire',round_no))
    selected=pool[rng.permutation(len(pool))[:min(size,len(pool))]].copy()
    np.random.default_rng(keyed_seed(seed,'row_execute',round_no)).shuffle(selected)
    work=[]
    for i in selected:
        algorithms=np.arange(cache.kind.shape[1])
        np.random.default_rng(keyed_seed(seed,'row_alg',round_no,int(i))).shuffle(algorithms)
        work.extend([[int(i),int(a)] for a in algorithms if cache.can_query(int(i),int(a),cap)])
    return pool,selected.tolist(),work

class LexSelector(Selector):
    def candidates(self,cap):
        common=super().candidates(cap)
        if not len(common):return np.empty((0,5))
        start=time.perf_counter()
        raw=np.column_stack([m.predict(pd.DataFrame(self.X[self.train_rows],columns=self.features))
                             for m in self.regressors])
        mapping={int(i):k for k,i in enumerate(self.train_rows)}
        differences=[]
        for i,pi,_,_ in common:
            a,b=self.pairs[int(pi)];k=mapping[int(i)]
            differences.append(abs(float(raw[k,a])-float(raw[k,b])))
        self.selection_s+=time.perf_counter()-start
        return np.column_stack((common,np.array(differences)))

