import csv
import hashlib
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path
import arff
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / "outputs" / "fa00"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "upstream" / "DATASETS" / "ASP-POTASSCO"

def read(name):
    d = arff.load((DATA / (name + ".arff")).open())
    return [a[0] for a in d["attributes"]], [r for r in d["data"] if r[1] == 1]

def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def main():
    _, runs = read("algorithm_runs")
    statuses = dict(Counter(r[4] for r in runs))
    assert set(statuses) <= {"ok", "timeout"}, statuses
    assert all(r[3] is not None and math.isfinite(r[3]) and r[3] >= 0 for r in runs)
    ids = sorted(set(r[0] for r in runs))
    algos = sorted(set(r[2] for r in runs))
    lookup = {(r[0],r[2]):r for r in runs}
    assert len(lookup) == len(runs) == len(ids)*len(algos)
    times = np.array([[lookup[(i,a)][3] for a in algos] for i in ids], dtype=float)
    ok = np.array([[lookup[(i,a)][4] == "ok" for a in algos] for i in ids])
    assert np.all(times[ok] <= 600) and np.all(times[~ok] == 600)
    names, feats = read("feature_values")
    fd = {r[0]:r[2:] for r in feats}
    assert len(fd) == len(ids) and set(fd) == set(ids)
    Xraw = np.array([[float(v) if v is not None else np.nan for v in fd[i]] for i in ids])
    assert not np.any(np.isinf(Xraw))
    _, cv = read("cv")
    folds = {r[0]:int(r[2]) for r in cv}
    assert len(folds) == len(cv) == len(ids) and set(folds) == set(ids)
    test = np.array([k for k,i in enumerate(ids) if folds[i] == 1])
    pool = np.array([k for k,i in enumerate(ids) if folds[i] != 1])
    shuffled = np.random.default_rng(20261001).permutation(pool)
    n_val = math.ceil(len(pool)*0.1)
    val, train = np.sort(shuffled[:n_val]), np.sort(shuffled[n_val:])
    small32 = np.sort(np.random.default_rng(20261001+1).choice(val, 32, replace=False))
    assert not(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    keep = ~np.all(np.isnan(Xraw[train]), axis=0)
    imputer = SimpleImputer(strategy="median").fit(Xraw[train][:,keep])
    X = imputer.transform(Xraw[:,keep])
    scaler = StandardScaler().fit(X[train])
    X = scaler.transform(X)
    assert np.all(np.isfinite(X))
    cost_names, costs = read("feature_costs")
    cost_dict = {r[0]:r[2:] for r in costs}
    assert set(cost_dict) == set(ids)
    known_cost = sum(v for r in costs for v in r[2:] if v is not None)
    missing_cost = sum(v is None for r in costs for v in r[2:])
    _, frs = read("feature_runstatus")
    feature_status = dict(Counter(v for r in frs for v in r[2:]))
    budget = 0.05*len(train)*len(algos)*600
    pairs = len(algos)*(len(algos)-1)//2
    batch_size = math.ceil((len(train)-20)*pairs*0.01)
    initials = {str(seed):np.random.default_rng(seed).choice(train,20,replace=False).tolist()
                for seed in [7,42,99]}
    versions = {p:importlib.metadata.version(p) for p in
                ["numpy","scipy","scikit-learn","pandas","liac-arff","paretoset"]}
    protocol = dict(name="FA00", source_commit="7a5727651a92fd2fa4960dcbbd7f6dab94130028",
        dataset="ASP-POTASSCO", native_cutoff_s=600, success_boundary="ok and runtime <= dispatched cap",
        timeout_score=6000, execution="restart; successful and terminal native-timeout outcomes cached",
        ids=ids, algorithms=algos, train=train.tolist(), validation=val.tolist(), small32=small32.tolist(),
        test=test.tolist(), initial_rows=initials, seeds=[7,42,99], budget_s=budget,
        initial_cutoff_s=100, batch_size=batch_size, batch_formula="ceil(.01*(N_train-20)*number_of_algorithm_pairs)",
        arms=["FREE-DYN","PAID-DYN","SMALL32-DYN","FIXED-100","FIXED-600","GEOMETRIC"],
        validation_rule="exact new-minus-old PAR10 on disagreements; cache reuse; native 600-second cap",
        model="upstream ActiveRFModel + PassiveRFRegressor; 100-tree RF and hard voting",
        pair_weight="upstream absolute observed penalized runtime difference; zero-weight ties excluded",
        regressor_weight="No_Weight; a single-algorithm regressor has no pairwise cost difference",
        cost_model="primary budget counts solver acquisition CPU-seconds; common feature cost and local strategy wall time separate",
        preprocessing="all 1294 instances retained; train-only median imputation and scaling; no outcome-based filtering",
        tie_break="upstream mode: first encountered vote; pairs and algorithm identifiers sorted",
        rng="counter-based seed/round/model keys, identical definitions across arms",
        remaining_budget="truncate dispatch; keep latest completely fitted selector; incomplete validation gives no cutoff feedback",
        candidate_cache="derive pair preferences only from two paid observations; no redundant same-or-shorter censor queries",
        no_candidates="compare unchanged selector (zero delta), advance permitted cutoff; terminate at cap 600 or fixed-cap exhaustion",
        clipping="only observed runtime targets, never feature columns",
        adaptation_notes=["sklearn 1.3.2 wheel used for Python 3.12 instead of upstream 1.3.1",
            "minimal modAL refit adapter; upstream model wrappers retained",
            "unweighted single-algorithm regression; restart-aware predicted costs",
            "native statuses/cutoff preserved, rather than replacing failed runs by 3600",
            "no instance deletion from missing features; common cost is only a known lower bound where feature costs are missing"],
        actual_wall_limit_s=7200, infrastructure_limit="half day; no automatic model downscaling",
        final_model="last completely fitted selector; no test-based checkpoint selection",
        metrics="mean test PAR10; continuation uses ratio of seed mean PAR10 and paired direction counts",
        continuation="free >=5% better than each fixed arm with >=2/3 paired directions; pooled paid validation fraction >=10%",
        closure="any legal arm's seed mean <=1.02 * free seed mean (including a better legal arm)",
        rule_priority="closure before continuation when both conditions are true; otherwise undetermined",
        test_feedback="test outcomes read only by offline evaluator after each trajectory finishes",
        versions=versions)
    dump("FA00_protocol.json",protocol)
    np.savez_compressed(OUT/"FA00_data.npz", X=X, runtimes=times, ok=ok)
    np.savez_compressed(OUT/"FA00_preprocessing.npz", raw_features=Xraw, keep=keep,
                        medians=imputer.statistics_, mean=scaler.mean_, scale=scaler.scale_)
    dump("FA00_data_audit.json",dict(instances=len(ids),algorithms=len(algos),
        pairs=pairs,rows=len(runs),runstatus=statuses,missing_run_pairs=0,duplicate_run_pairs=0,
        train=len(train),validation=len(val),small32=len(small32),test=len(test),
        features=X.shape[1],missing_feature_cells=int(np.isnan(Xraw).sum()),
        dropped_train_all_missing_columns=int((~keep).sum()),feature_status=feature_status,
        feature_known_cost_lower_bound_s=known_cost, feature_missing_cost_cells=missing_cost,
        common_feature_cost_exact=missing_cost==0, budget_s=budget,batch_size=batch_size,
        source_manifest_sha256=hashlib.sha256((ROOT/"source_manifest.json").read_bytes()).hexdigest(),
        warning="Missing feature costs are not zero. Feature presolving is not exploited as an extra selection policy."))
    with (OUT/"FA00_split_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["row","instance_id","split","small32"])
        for k,i in enumerate(ids):
            w.writerow([k,i,"train" if k in set(train) else "validation" if k in set(val) else "test",int(k in set(small32))])
    print(json.dumps(json.loads((OUT/"FA00_data_audit.json").read_text()),indent=2))

if __name__ == "__main__": main()
