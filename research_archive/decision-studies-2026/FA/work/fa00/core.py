"""Legal-feedback FA00 adaptation. Hidden outcomes live only in ReplayEnvironment.

The upstream ActiveRFModel and PassiveRFRegressor are reused. A small adapter
replaces modAL's only_new=True refit with the identical sklearn fit operation.
"""
from __future__ import annotations
import hashlib
import itertools
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from paretoset import paretoset
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))
from ActiveRFModel import ActiveRFModel
from PassiveRFRegressor import PassiveRFRegressor

T = 600.0
PARAMS = dict(n_estimators=100, max_features="sqrt", max_depth=2**31,
              min_samples_split=2, bootstrap=True, n_jobs=1)

def keyed_seed(seed, *keys):
    return int.from_bytes(hashlib.sha256(repr((seed, keys)).encode()).digest()[:4], "big")

class Exhausted(Exception):
    pass

class WallLimit(Exception):
    pass

class ObservationCache:
    def __init__(self, shape):
        self.kind = np.zeros(shape, dtype=np.int8)  # 0 unknown, 1 lower bound, 2 completion, 3 native timeout
        self.value = np.zeros(shape, dtype=float)

    def snapshot(self):
        return hashlib.sha256(self.kind.tobytes() + self.value.tobytes()).hexdigest()

    def resolved(self, i, a):
        return self.kind[i, a] >= 2

    def can_query(self, i, a, cap):
        return not self.resolved(i, a) and (self.kind[i, a] == 0 or self.value[i, a] < cap - 1e-10)

    def score(self, i, a):
        if self.kind[i, a] == 2:
            return float(self.value[i, a])
        if self.kind[i, a] == 3:
            return 10.0 * T
        raise ValueError("A censored observation is not an exact validation score")

class Ledger:
    def __init__(self, budget, writer=None):
        self.budget = float(budget)
        self.train = self.validation = 0.0
        self.cache_hits = self.executions = self.completions = self.cancellations = 0
        self.writer = writer
        self.event_id = 0
        self.phase = "init"
        self.round = 0

    @property
    def remaining(self):
        return max(0.0, self.budget - self.train - self.validation)

    def event(self, event):
        event.update(event_id=self.event_id, phase=self.phase, round=self.round,
                     train_cost_s=self.train, validation_cost_s=self.validation,
                     remaining_s=self.remaining)
        self.event_id += 1
        if self.writer:
            self.writer(event)

class ReplayEnvironment:
    def __init__(self, runtimes, ok, ledger):
        self._runtimes = np.array(runtimes, dtype=float, copy=True)
        self._ok = np.array(ok, dtype=bool, copy=True)
        self.ledger = ledger

    def observe(self, cache, i, a, cap, role):
        """Restart, charge min(runtime, cap), never peek to decide dispatch."""
        cap = min(T, float(cap))
        before_kind, before_value = int(cache.kind[i, a]), float(cache.value[i, a])
        if not cache.can_query(i, a, cap):
            self.ledger.cache_hits += 1
            self.ledger.event(dict(type="observation", role=role, i=int(i), a=int(a),
                                   planned_cap_s=cap, actual_cap_s=0.0, charged_s=0.0,
                                   before_kind=before_kind, before_value=before_value,
                                   kind=before_kind, value=before_value, cached=True))
            return
        if self.ledger.remaining <= 1e-9:
            raise Exhausted
        actual_cap = min(cap, self.ledger.remaining)
        r = float(self._runtimes[i, a])
        completed = bool(self._ok[i, a] and r <= actual_cap)
        charged = r if completed else actual_cap
        if role == "train":
            self.ledger.train += charged
        else:
            self.ledger.validation += charged
        self.ledger.executions += 1
        if completed:
            cache.kind[i, a] = 2
            cache.value[i, a] = r
            self.ledger.completions += 1
        elif actual_cap >= T - 1e-9:
            cache.kind[i, a] = 3
            cache.value[i, a] = T
            self.ledger.cancellations += 1
        else:
            cache.kind[i, a] = 1
            cache.value[i, a] = max(before_value, actual_cap)
            self.ledger.cancellations += 1
        self.ledger.event(dict(type="observation", role=role, i=int(i), a=int(a),
                               planned_cap_s=cap, actual_cap_s=actual_cap, charged_s=charged,
                               before_kind=before_kind, before_value=before_value,
                               kind=int(cache.kind[i, a]), value=float(cache.value[i, a]), cached=False))
        assert self.ledger.train + self.ledger.validation <= self.ledger.budget + 1e-7

    def free_difference(self, old, new, rows):
        # Explicitly privileged arm only. No path from this method to legal arms.
        selected = np.asarray(rows)[old[np.asarray(rows)] != new[np.asarray(rows)]]
        vals = []
        for i in selected:
            a, b = int(old[i]), int(new[i])
            old_score = self._runtimes[i, a] if self._ok[i, a] else 10*T
            new_score = self._runtimes[i, b] if self._ok[i, b] else 10*T
            vals.append(new_score - old_score)
        return float(sum(vals)), len(selected)

def paid_difference(env, cache, old, new, rows):
    changed = [int(i) for i in rows if old[i] != new[i]]
    delta = 0.0
    for i in changed:
        a, b = int(old[i]), int(new[i])
        env.observe(cache, i, a, T, "validation")
        env.observe(cache, i, b, T, "validation")
        if not cache.resolved(i, a) or not cache.resolved(i, b):
            raise Exhausted
        delta += cache.score(i, b) - cache.score(i, a)
    return float(delta), len(changed)

class RefitLearner:
    def __init__(self, estimator):
        self.estimator = estimator

    def teach(self, X, y, sample_weight=None, only_new=True):
        assert only_new
        self.estimator.fit(np.asarray(X), y, sample_weight=sample_weight)

    def predict(self, X):
        return self.estimator.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(np.asarray(X))

class Selector:
    def __init__(self, X, algorithms, train_rows, cache, seed):
        self.X = X
        self.algorithms = algorithms
        self.train_rows = np.asarray(train_rows)
        self.cache = cache
        self.seed = seed
        self.pairs = list(itertools.combinations(range(len(algorithms)), 2))
        self.features = [f"f{k}" for k in range(X.shape[1])]
        self.label_rows = [dict() for _ in self.pairs]
        self.pair_models = [None for _ in self.pairs]
        self.regressors = [None for _ in algorithms]
        self.fit_s = self.predict_s = self.selection_s = 0.0

    def synchronize_labels(self, label_writer=None):
        """Infer a pair only from two paid observations; retain first acquisition values."""
        for pi, (a, b) in enumerate(self.pairs):
            for i0 in self.train_rows:
                i = int(i0)
                if i in self.label_rows[pi]:
                    continue
                ka, kb = int(self.cache.kind[i, a]), int(self.cache.kind[i, b])
                if not ka or not kb or not (ka == 2 or kb == 2):
                    continue
                va, vb = float(self.cache.value[i, a]), float(self.cache.value[i, b])
                # A lower bound is sufficient only when it resolves the preference.
                if ka != 2 and va < vb - 1e-10:
                    continue
                if kb != 2 and vb < va - 1e-10:
                    continue
                # At an equal bound, the completed algorithm is no worse than
                # the unresolved one; do not label the unresolved one faster.
                if ka != 2 and va == vb:
                    va = float(np.nextafter(va, np.inf))
                if kb != 2 and vb == va:
                    vb = float(np.nextafter(vb, np.inf))
                label = 0 if va <= vb else 1
                # Same weighted pairwise classification as upstream. A censor is
                # penalized at its observed bound, not assigned its hidden runtime.
                pa = va if ka == 2 else 10*va
                pb = vb if kb == 2 else 10*vb
                weight = abs(pa-pb)
                self.label_rows[pi][i] = (va, vb, label, weight)
                if label_writer:
                    label_writer(dict(type="pair_label", pair=pi, i=i, a=a, b=b,
                                      va=va, vb=vb, ka=ka, kb=kb, label=label, weight=weight))

    def fit(self, round_no, deadline=math.inf):
        start = time.perf_counter()
        for pi, (a, b) in enumerate(self.pairs):
            if time.perf_counter() >= deadline:
                raise WallLimit
            labels = self.label_rows[pi]
            rows = [i for i in sorted(labels) if labels[i][3] > 0]
            if not rows:
                self.pair_models[pi] = None
                continue
            data = pd.DataFrame(self.X[rows], columns=self.features)
            names = [self.algorithms[a], self.algorithms[b]]
            data[names[0]] = [labels[i][0] for i in rows]
            data[names[1]] = [labels[i][1] for i in rows]
            encoder = LabelEncoder().fit(names)
            model = ActiveRFModel(RefitLearner(RandomForestClassifier(
                **PARAMS, random_state=keyed_seed(self.seed, "pair", round_no, pi))),
                "Algorithm_predictor", tuple(names), encoder, data,
                pd.DataFrame(), np.array([labels[i][3] for i in rows]))
            model.teach(self.features, "Weighted")
            self.pair_models[pi] = model
        for a, name in enumerate(self.algorithms):
            if time.perf_counter() >= deadline:
                raise WallLimit
            rows = self.train_rows[self.cache.kind[self.train_rows, a] > 0]
            data = pd.DataFrame(self.X[rows], columns=self.features)
            data[name] = self.cache.value[rows, a]
            model = PassiveRFRegressor(RandomForestRegressor(
                **PARAMS, random_state=keyed_seed(self.seed, "reg", round_no, a)),
                "Algorithm_regressor", name, None, data, data[name].to_numpy())
            # Single-algorithm regression has no pairwise sample weight.
            model.fit(self.features, "No_Weight")
            self.regressors[a] = model
        self.fit_s += time.perf_counter()-start

    def predict(self):
        start = time.perf_counter()
        predictions = []
        for pi, (a, b) in enumerate(self.pairs):
            model = self.pair_models[pi]
            if model is None:
                continue
            pred = model.get_learner().predict(self.X).astype(int)
            predictions.append(np.where(pred == 0, a, b))
        if not predictions:
            raise ValueError("No informative pairwise classifier")
        # statistics.mode, as in upstream: ties use first occurring vote.
        out = np.array([Counter(row).most_common(1)[0][0]
                        for row in np.array(predictions).T], dtype=np.int16)
        self.predict_s += time.perf_counter()-start
        return out

    def candidates(self, cap):
        start = time.perf_counter()
        self.synchronize_labels()
        rows, pis, uncertainty, costs = [], [], [], []
        pred_cost = np.column_stack([m.predict(pd.DataFrame(self.X[self.train_rows],
                                                 columns=self.features)) for m in self.regressors])
        # Expected cost follows restart semantics. A censor is a lower bound,
        # never previously paid CPU that may be subtracted from a restarted run.
        pred_cost = np.maximum(pred_cost, self.cache.value[self.train_rows])
        pred_cost = np.minimum(np.maximum(pred_cost, 0), cap)
        for pi, (a, b) in enumerate(self.pairs):
            known = self.label_rows[pi]
            valid = [k for k, i in enumerate(self.train_rows) if int(i) not in known and
                     (self.cache.can_query(int(i), a, cap) or self.cache.can_query(int(i), b, cap))]
            if not valid:
                continue
            indices = self.train_rows[valid]
            model = self.pair_models[pi]
            u = np.ones(len(indices))*0.5 if model is None else 1-model.get_learner().predict_proba(self.X[indices]).max(axis=1)
            for k, i, unc in zip(valid, indices, u):
                cost = sum(pred_cost[k, alg] if self.cache.can_query(int(i), alg, cap) else 0.0
                           for alg in (a, b))
                rows.append(int(i)); pis.append(pi); uncertainty.append(float(unc)); costs.append(float(cost))
        result = np.array([rows, pis, uncertainty, costs], dtype=float).T
        if len(result):
            result = result[np.argsort(-result[:, 2], kind="stable")]
        else:
            result = np.empty((0, 4))
        self.selection_s += time.perf_counter()-start
        return result

    def select_batch(self, candidates, batch_size, round_no):
        start = time.perf_counter()
        remaining = np.arange(len(candidates))
        accepted = []
        while len(accepted) < batch_size and len(remaining):
            front = paretoset(candidates[remaining, 2:4], sense=["max", "min"])
            layer = remaining[front]
            if not accepted:
                # Upstream samples the entire first front when it exceeds q.
                accepted.extend(layer.tolist())
            else:
                accepted.extend(layer[:batch_size-len(accepted)].tolist())
            remaining = remaining[~front]
        accepted = np.array(accepted, dtype=int)
        rng = np.random.default_rng(keyed_seed(self.seed, "batch", round_no))
        rng.shuffle(accepted)
        self.selection_s += time.perf_counter()-start
        return candidates[accepted[:batch_size]]
