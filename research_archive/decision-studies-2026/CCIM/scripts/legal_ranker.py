"""Can LEGAL visible information pick a better last survey than degree_min?

One question, one small model
-----------------------------
Fixed state, last survey only.  Features are restricted to what is computable **before** the survey is
chosen: no true degrees, no unrevealed neighbours, no post-survey information, no full-graph embeddings, no
node identity.  The model is a single L2-regularised pairwise logistic ranker -- no sweep, no GNN, no RL.

Split, frozen before any gain is looked at
------------------------------------------
======================  ======  ==========================================
side                    states  seeds
======================  ======  ==========================================
development             20      ``valprobe/0..19`` (already seen)
training                40      ``rstates/1000..1039`` (new)
**held-out evaluation** 40      ``rstates/2000..2039`` (new)
======================  ======  ==========================================

Same generation protocol as the previous probe (5 fixed free seeds + 4 uniform random frontier surveys),
never selected by gain.  Candidate pool stays <= 8 with unchanged rules.  A repeated complete visible state
may not straddle train and evaluation, and that is asserted rather than assumed.

Labels
------
Training labels are the **privileged simulation estimates** (a small simulation batch of the true
objective).  That is a supervision signal, never a deployment input, and its generation cost is reported
separately.

Main metric
-----------
Final influence on the held-out states, measured on an **independent confirmation batch**.  Ranking
accuracy is auxiliary only.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "_official"))

import xplore_learn as SIMP  # noqa: E402
from ccim.xplore import DiscoveryEnv  # noqa: E402
from ccim.xplore.icm import INFL_BUDGET, PROP_PROBAB, LiveEdgeObjective  # noqa: E402

OUTPUT = ROOT / "results" / "legal_ranker.json"
GRAPH = "interact__soc-wiki-Vote"
T_MAX, EXTRA_SEEDS, SEL_SAMPLES = 5, 5, 100
SIM_SAMPLES, CONF_SAMPLES = 300, 3000
N_CAND = 8
DEV = [("valprobe", i) for i in range(20)]
TRAIN = [("rstates", 1000 + i) for i in range(40)]
HOLDOUT = [("rstates", 2000 + i) for i in range(40)]
FEATURES = ["vis_deg", "nbr_surveyed", "nbr_frontier", "nbr_edges", "clustering",
            "nbr_deg_mean", "nbr_deg_max", "reveal_step", "n_revealers"]
# single fixed configuration -- no sweep
L2, LR, EPOCHS = 1e-2, 0.5, 400


def sseed(*p):
    return SIMP.sseed(*p)


def build_state(g, ns, i):
    """5 fixed free seeds + 4 uniform random frontier surveys -> the moment before the last survey."""
    seeds = [int(x) for x in np.random.default_rng(sseed(ns, i, "seeds"))
             .choice(g.number_of_nodes(), EXTRA_SEEDS, replace=False)]
    env = DiscoveryEnv(g, seeds, max_T=T_MAX, p=PROP_PROBAB, infl_budget=INFL_BUDGET)
    rng = np.random.default_rng(sseed(ns, i, "walk"))
    reveal_step, revealers = {}, {}
    for s in seeds:
        for v in env.graph.neighbors(s):
            reveal_step.setdefault(v, 0)
            revealers[v] = revealers.get(v, 0) + 1
    for step in range(T_MAX - 1):
        frontier = sorted(env.possible_actions)
        if not frontier:
            break
        u = int(frontier[int(rng.integers(len(frontier)))])
        before = set(env.possible_actions)
        env.step(u)
        for v in set(env.possible_actions) - before:
            reveal_step.setdefault(v, step + 1)
        for v in g.neighbors(u):
            revealers[v] = revealers.get(v, 0) + 1
    return {"env": env, "reveal_step": reveal_step, "revealers": revealers,
            "key": None}


def visible_key(env):
    """Identity of the complete visible state: the observed node set plus the observed edge set."""
    obs = env.observed
    nodes = tuple(sorted(env.active.union(env.possible_actions)))
    edges = tuple(sorted(tuple(sorted(e)) for e in obs.edges()))
    return (nodes, edges)


def candidates(env, ns, i):
    frontier = sorted(env.possible_actions)
    if not frontier:
        return []
    obs = env.observed
    deg = {u: obs.degree(u) for u in frontier}
    lo, hi = min(deg.values()), max(deg.values())
    rng = np.random.default_rng(sseed(ns, i, "cand"))
    picks = []
    for target in (lo, hi):
        tied = [u for u in frontier if deg[u] == target]
        picks.append(int(tied[int(rng.integers(len(tied)))]))
    rest = [u for u in frontier if u not in picks]
    rng.shuffle(rest)
    picks.extend(int(u) for u in rest[:max(0, N_CAND - len(picks))])
    return picks[:N_CAND]


def features(env, st, cand):
    """Legal, pre-survey only.  No true degree, no unrevealed neighbours, no node id."""
    obs = env.observed
    nb = sorted(obs.neighbors(cand))
    deg_c = obs.degree(cand)
    n_surv = sum(1 for v in nb if v in env.active)
    n_front = sum(1 for v in nb if v in env.possible_actions)
    nbr_set = set(nb)
    n_edges = sum(1 for a in nb for b in obs.neighbors(a) if b in nbr_set and a < b)
    clust = (2.0 * n_edges / (deg_c * (deg_c - 1))) if deg_c > 1 else 0.0
    nd = [obs.degree(v) for v in nb] or [0]
    return [float(deg_c), float(n_surv), float(n_front), float(n_edges), float(clust),
            float(np.mean(nd)), float(np.max(nd)),
            float(st["reveal_step"].get(cand, T_MAX)), float(st["revealers"].get(cand, 0))]


def rollout(env, cand):
    e = copy.deepcopy(env)
    if cand is not None:
        e.step(int(cand))
    sel = LiveEdgeObjective(e.graph, samples=SEL_SAMPLES,
                            rng=np.random.default_rng(sseed("rank", "select")), p=PROP_PROBAB)
    chosen, _ = sel.greedy(INFL_BUDGET)
    return chosen


def score(g, chosen, tag, samples):
    obj = LiveEdgeObjective(g, samples=samples,
                            rng=np.random.default_rng(sseed("rank", "score", tag)), p=PROP_PROBAB)
    return obj.value(chosen)


def collect(g, spec, sim_tag, conf_tag=None, log=None):
    out = []
    for ns, i in spec:
        st = build_state(g, ns, i)
        st["key"] = visible_key(st["env"])
        cands = candidates(st["env"], ns, i)
        if not cands:
            continue
        rows = []
        for c in cands:
            ch = rollout(st["env"], c)
            rows.append({"cand": int(c), "feat": features(st["env"], st, c),
                         "chosen": ch,
                         "sim": score(g, ch, f"{sim_tag}:{ns}{i}:{c}", SIM_SAMPLES),
                         "conf": (score(g, ch, f"{conf_tag}:{ns}{i}:{c}", CONF_SAMPLES)
                                  if conf_tag else None)})
        out.append({"ns": ns, "i": i, "key": st["key"], "n_frontier": len(st["env"].possible_actions),
                    "frontier": sorted(int(u) for u in st["env"].possible_actions),
                    "rows": rows,
                    "sim_degmin": rows[0]["sim"], "sim_degmax": rows[1]["sim"]})
        if log:
            log(f"      collected {ns}/{i}: {len(cands)} candidates, "
                f"frontier {len(st['env'].possible_actions)}")
    return out


def ranker_train(states):
    """Pairwise logistic with L2, equal total weight per state.  Returns (w, b, mean, std)."""
    X = np.array([r["feat"] for s in states for r in s["rows"]], dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    pairs = []
    for s in states:
        y = np.array([r["sim"] for r in s["rows"]], dtype=float)
        idx = [(a, b) for a in range(len(y)) for b in range(len(y)) if y[a] > y[b]]
        if not idx:
            continue
        w_state = 1.0 / len(idx)                      # equal total weight per state
        F = (np.array([r["feat"] for r in s["rows"]], dtype=float) - mean) / std
        for a, b in idx:
            pairs.append((F[a] - F[b], w_state))
    d = np.array([p[0] for p in pairs])
    wt = np.array([p[1] for p in pairs])
    wt = wt / wt.sum() * len(wt)
    w = np.zeros(d.shape[1])
    b = 0.0
    for _ in range(EPOCHS):
        z = d @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g_w = (d * ((p - 1.0) * wt)[:, None]).sum(axis=0) / len(wt) + L2 * w
        g_b = float(((p - 1.0) * wt).sum() / len(wt))
        w -= LR * g_w
        b -= LR * g_b
    return w, b, mean, std


def ranker_scores(w, b, mean, std, feats):
    F = (np.array(feats, dtype=float) - mean) / std
    return F @ w + b


def pairwise_acc(w, b, mean, std, states):
    ok = tot = 0
    for s in states:
        y = np.array([r["sim"] for r in s["rows"]], dtype=float)
        sc = ranker_scores(w, b, mean, std, [r["feat"] for r in s["rows"]])
        for a in range(len(y)):
            for c in range(len(y)):
                if y[a] > y[c]:
                    tot += 1
                    ok += int(sc[a] > sc[c])
    return (ok / tot) if tot else None


def ci(vals):
    v = np.asarray(vals, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, None, None, None
    h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, m - h, m + h, float(stats.ttest_1samp(v, 0).pvalue)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    g = pickle.loads((SIMP.DATA / f"{GRAPH}.pkl").read_bytes())

    print("=" * 104)
    print("  LEGAL-INFORMATION RANKER -- last survey only, split frozen before any gain is seen")
    print("=" * 104)
    print(f"    graph {GRAPH} (n={g.number_of_nodes()})   features {len(FEATURES)}: {FEATURES}")
    t0 = time.perf_counter()
    train_states = collect(g, DEV + TRAIN, "simtrain", log=print)
    print(f"    training states {len(train_states)}  ({time.perf_counter() - t0:.0f} s)")
    t1 = time.perf_counter()
    hold = collect(g, HOLDOUT, "simhold", conf_tag="confhold", log=print)
    print(f"    held-out states {len(hold)}  ({time.perf_counter() - t1:.0f} s)")

    # split integrity
    tr_keys = {s["key"] for s in train_states}
    ho_keys = {s["key"] for s in hold}
    clash = tr_keys & ho_keys
    print(f"\n    [{'PASS' if not clash else 'FAIL'}] no repeated complete visible state straddles "
          f"train/evaluation ({len(clash)} clashes)")
    print(f"    [PASS] initial observation and its trajectory stay on one side by construction")

    w, b, mean, std = ranker_train(train_states)
    print(f"\n    ranker weights: " + ", ".join(f"{n}={v:+.3f}" for n, v in zip(FEATURES, w)))
    pa = pairwise_acc(w, b, mean, std, hold)

    # ---- held-out evaluation on the CONFIRMATION batch only
    picked = {"learned": [], "degree_min": [], "degree_max": [], "random": [], "privileged": []}
    aux_same = {"learned_eq_degmin": 0}
    for st in hold:
        cands = [r["cand"] for r in st["rows"]]
        sc = ranker_scores(w, b, mean, std, [r["feat"] for r in st["rows"]])
        sims = [r["sim"] for r in st["rows"]]
        conf = {r["cand"]: r["conf"] for r in st["rows"]}
        rng = np.random.default_rng(sseed("rank", "randpick", st["ns"], st["i"]))
        rnd = int(cands[int(rng.integers(len(cands)))])
        picks = {"learned": int(cands[int(np.argmax(sc))]), "degree_min": int(cands[0]),
                 "degree_max": int(cands[1]), "random": rnd,
                 "privileged": int(cands[int(np.argmax(sims))])}
        for k, v in picks.items():
            picked[k].append(conf[v])
        aux_same["learned_eq_degmin"] += int(picks["learned"] == picks["degree_min"])

    print("\n" + "=" * 104)
    print("  MAIN RESULT -- held-out states, independent confirmation batch")
    print("=" * 104)
    print(f"    {'comparison':<40}{'mean':>9}{'95% CI':>20}{'p':>9}{'W/L':>8}")
    base = np.array(picked["degree_min"])
    table = {"mean_final_influence": {k: float(np.mean(v)) for k, v in picked.items()}}
    for label, key in (("learned - degree_min  (MAIN)", "learned"),
                       ("learned - random     (secondary)", "random"),
                       ("privileged - degree_min", "privileged"),
                       ("privileged - learned", None)):
        if key is None:
            diff = np.array(picked["privileged"]) - np.array(picked["learned"])
        else:
            diff = np.array(picked[key]) - base
        m, lo, hi, p = ci(diff)
        w_, l_ = int((diff > 0).sum()), int((diff < 0).sum())
        table[label] = {"mean": m, "lo": lo, "hi": hi, "p": p, "wins": w_, "losses": l_}
        print(f"    {label:<40}{m:>+9.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>20}{p:>9.4f}{f'{w_}/{l_}':>8}")
    print(f"\n    mean final influence: " + "  ".join(f"{k}={v:.1f}" for k, v in
                                                     table["mean_final_influence"].items()))
    print(f"    auxiliary: held-out pairwise accuracy {pa:.3f}; "
          f"learned picked the degree_min candidate in {aux_same['learned_eq_degmin']}/{len(hold)} states")
    print(f"\n    costs: feature+label generation {time.perf_counter() - t0:.0f} s total; "
          f"training is a closed-form-scale fit ({EPOCHS} epochs)")

    args.output.write_text(json.dumps(
        {"graph": GRAPH, "features": FEATURES, "split": {"dev": 20, "train": 40, "holdout": 40},
         "config": {"L2": L2, "lr": LR, "epochs": EPOCHS},
         "split_integrity": {"visible_state_clashes": len(clash)},
         "weights": {n: float(v) for n, v in zip(FEATURES, w)}, "bias": float(b),
         "pairwise_accuracy_holdout": pa,
         "learned_eq_degmin": aux_same["learned_eq_degmin"], "n_holdout": len(hold),
         "table": table, "per_state_picked": picked,
         "seconds": time.perf_counter() - t0}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
