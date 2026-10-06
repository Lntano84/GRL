"""Published coarse-PC formula under the frozen legal row-query protocol."""
import numpy as np
from common import keyed_seed

def mean_uncertainty(max_probabilities, missing_models):
    m = np.asarray(max_probabilities, dtype=float)
    missing = np.asarray(missing_models, dtype=bool)
    assert m.ndim == 2 and m.shape[1] == len(missing) and len(missing) > 0
    assert np.all(np.isfinite(m)) and np.all(m >= .5-1e-12) and np.all(m <= 1+1e-12)
    assert np.all(m[:,missing] == .5)
    return np.mean(1-m, axis=1)

def select_rows(pool, scores, seed, round_no, size=11):
    assert len(pool) == len(scores)
    order = np.random.default_rng(keyed_seed(seed,'row_acquire',round_no)).permutation(len(pool))
    order = order[np.argsort(-scores[order],kind='stable')]
    selected = pool[order[:min(size,len(pool))]].copy()
    np.random.default_rng(keyed_seed(seed,'row_execute',round_no)).shuffle(selected)
    return selected

def coarse_batch(selector, seed, round_no, size=11, cap=100):
    cache = selector.cache
    pool = np.array(sorted(int(i) for i in selector.train_rows if any(
        cache.can_query(int(i),a,cap) for a in range(cache.kind.shape[1]))),dtype=int)
    missing = np.array([model is None for model in selector.pair_models], dtype=bool)
    max_probs = np.full((len(pool),len(missing)),.5)
    if len(pool):
        for j, model in enumerate(selector.pair_models):
            if model is not None:
                max_probs[:,j] = model.get_learner().predict_proba(selector.X[pool]).max(axis=1)
    scores = mean_uncertainty(max_probs,missing)
    selected = select_rows(pool,scores,seed,round_no,size)
    work = []
    for i in selected:
        algorithms = np.arange(cache.kind.shape[1])
        np.random.default_rng(keyed_seed(seed,'row_alg',round_no,int(i))).shuffle(algorithms)
        work.extend([[int(i),int(a)] for a in algorithms if cache.can_query(int(i),int(a),cap)])
    return pool,scores,max_probs,missing,selected.tolist(),work
