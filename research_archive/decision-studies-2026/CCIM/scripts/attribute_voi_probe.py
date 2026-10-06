"""Pre-registered last-survey value-of-information probe on the high-school contact graph.

The protocol and stopping rules are in results/attribute_voi_plan.json.  All policy inputs
are derived from the registered member list, class labels, and already queried pairs.
The full graph is used to generate simulated survey outcomes in the evaluation environment
and to score final seed sets; the planner samples its own graphs from its posterior.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccim.xplore.icm import LiveEdgeObjective  # noqa: E402

PLAN = ROOT / "results" / "attribute_voi_plan.json"
OUT = ROOT / "results" / "attribute_voi"
DATA = ROOT / "_data" / "highschool"
DAY = "2013-12-03"
P_IC = 0.1
K = 10
PRIOR = 1.0
N_DEV = 20
N_CONFIRM = 80
N_DRAW = 16
N_SEED_MC = 100
N_PLAN_EVAL_MC = 100
N_CONFIRM_MC = 3000
POOL = 8
CPU_CAP = 21600
DELTA = 3.29
RULES = ("random_global", "observed_degree", "residual_degree",
         "seed_uncertainty", "residual_seed_distance")


def seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_graph():
    meta = {}
    for line in (DATA / "HighSchool2013_metadata.txt").read_text(encoding="utf-8").splitlines():
        cols = line.split()
        assert len(cols) >= 2
        node = int(cols[0])
        assert node not in meta
        meta[node] = cols[1]
    ids = sorted(meta)
    index = {v: j for j, v in enumerate(ids)}
    cls = np.array([meta[v] for v in ids])
    graph = nx.Graph()
    graph.add_nodes_from(range(len(ids)))
    rows, first, last = 0, None, None
    with gzip.open(DATA / "HighSchool2013_proximity_net.csv.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            cols = line.split()
            assert len(cols) == 5
            stamp = int(cols[0])
            if datetime.fromtimestamp(stamp, timezone.utc).date().isoformat() != DAY:
                continue
            u, v = index[int(cols[1])], index[int(cols[2])]
            rows += 1
            first = stamp if first is None else min(first, stamp)
            last = stamp if last is None else max(last, stamp)
            if u != v:
                graph.add_edge(u, v)
    assert len(ids) == 329 and rows == 47338 and graph.number_of_edges() == 2573
    i, j = np.triu_indices(len(ids), 1)
    full = np.zeros(len(i), dtype=bool)
    full_index = np.zeros((len(ids), len(ids)), dtype=np.int32)
    full_index[i, j] = np.arange(len(i))
    full_index[j, i] = np.arange(len(i))
    for u, v in graph.edges():
        full[full_index[u, v]] = True
    same = cls[i] == cls[j]
    info = {"n": len(ids), "m": graph.number_of_edges(), "rows": rows,
            "first_utc": datetime.fromtimestamp(first, timezone.utc).isoformat(),
            "last_utc": datetime.fromtimestamp(last, timezone.utc).isoformat(),
            "classes": {str(c): int(np.sum(cls == c)) for c in sorted(set(cls))}}
    return graph, ids, cls, i, j, same, full, full_index, info


def known_pairs(queried: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return queried[i] | queried[j]


def posterior(known: np.ndarray, edges: np.ndarray, same: np.ndarray) -> tuple[np.ndarray, dict]:
    """Every unordered observed pair contributes once, including confirmed nonedges."""
    q = np.empty(len(known), dtype=float)
    rates = {}
    for label, block in (("same", same), ("different", ~same)):
        seen = known & block
        n_seen, n_edge = int(seen.sum()), int(np.count_nonzero(edges & seen))
        rate = (PRIOR + n_edge) / (2 * PRIOR + n_seen)
        q[block] = rate
        rates[label] = {"seen_pairs": n_seen, "edges": n_edge, "rate": rate}
    q[known] = edges[known].astype(float)
    return q, rates


def sampled_graph(q: np.ndarray, known: np.ndarray, known_edge: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    draw = rng.random(len(q)) < q
    draw[known] = known_edge[known]
    return draw


class WeightedObjective:
    """Live-edge objective with edge-existence uncertainty folded into IC probability."""

    def __init__(self, n: int, i: np.ndarray, j: np.ndarray, existence: np.ndarray,
                 samples: int, rng: np.random.Generator):
        self.n, self.samples = n, samples
        labels = np.empty((samples, n), dtype=np.int32)
        sizes = np.empty((samples, n), dtype=np.int32)
        probs = P_IC * existence
        for s in range(samples):
            keep = rng.random(len(probs)) < probs
            adj = coo_matrix((np.ones(int(keep.sum()), dtype=np.int8), (i[keep], j[keep])), shape=(n, n))
            count, lab = connected_components(adj, directed=False)
            labels[s] = lab
            sizes[s] = np.bincount(lab, minlength=count)[lab]
        self.labels, self.sizes = labels, sizes

    def greedy(self, k: int = K) -> list[int]:
        covered = np.zeros((self.samples, self.n), dtype=bool)
        chosen = []
        taken = np.zeros(self.n, dtype=bool)
        row = np.arange(self.samples)[:, None]
        for _ in range(k):
            gains = np.where(~covered[row, self.labels], self.sizes, 0).mean(axis=0)
            gains[taken] = -1
            u = int(np.argmax(gains))
            chosen.append(u)
            taken[u] = True
            covered[np.arange(self.samples), self.labels[:, u]] = True
        return chosen


def solve(n, i, j, q, stream) -> list[int]:
    return WeightedObjective(n, i, j, q, N_SEED_MC,
                             np.random.default_rng(stream)).greedy()


def build_state(split: str, number: int, n: int, i, j, same, full):
    rng = np.random.default_rng(seed("state", split, number))
    ordered = rng.permutation(n)
    queried = np.zeros(n, dtype=bool)
    queried[ordered[:9]] = True
    known = known_pairs(queried, i, j)
    q, rates = posterior(known, full, same)
    assert int(queried.sum()) == 9
    return {"split": split, "number": number, "query_order": ordered[:9].tolist(),
            "queried": queried, "known": known, "q": q, "rates": rates}


def argmax_legal(values, legal, *stream):
    valid = np.flatnonzero(legal)
    top = values[valid].max()
    tied = valid[np.isclose(values[valid], top, rtol=0, atol=1e-12)]
    rng = np.random.default_rng(seed(*stream))
    return int(tied[int(rng.integers(len(tied)))])


def simple_rules(state, n, i, j, full, full_index, base_seeds):
    known, q, queried = state["known"], state["q"], state["queried"]
    legal = ~queried
    rng = np.random.default_rng(seed("rule", state["split"], state["number"], "random"))
    choices = {"random_global": int(rng.choice(np.flatnonzero(legal)))}
    observed = np.bincount(np.r_[i[known & full], j[known & full]], minlength=n)
    residual = np.bincount(np.r_[i[~known], j[~known]],
                           weights=np.r_[q[~known], q[~known]], minlength=n)
    st = (state["split"], state["number"])
    choices["observed_degree"] = argmax_legal(observed, legal, "rule", *st, "observed")
    choices["residual_degree"] = argmax_legal(residual, legal, "rule", *st, "residual")
    # Direct seed coverage is a legal, low-cost approximation, not the IC objective.
    noncovered = np.ones(n, dtype=float)
    for u in base_seeds:
        pair = full_index[u, np.arange(n)]
        noncovered *= (1.0 - P_IC * q[pair])
        noncovered[u] = 0.0
    weight = q * (1 - q)
    uncertainty = np.bincount(np.r_[i[~known], j[~known]],
                              weights=np.r_[weight[~known] * noncovered[j[~known]],
                                            weight[~known] * noncovered[i[~known]]], minlength=n)
    choices["seed_uncertainty"] = argmax_legal(uncertainty, legal, "rule", *st, "uncertainty")
    observed_graph = nx.Graph()
    observed_graph.add_nodes_from(range(n))
    observed_graph.add_edges_from(zip(i[known & full].tolist(), j[known & full].tolist()))
    dist = nx.multi_source_dijkstra_path_length(observed_graph, base_seeds)
    proximity = np.array([1.0 / (1.0 + dist.get(v, math.inf)) for v in range(n)])
    proxy = residual * (0.5 + 0.5 * proximity)
    choices["residual_seed_distance"] = argmax_legal(proxy, legal, "rule", *st, "distance")
    assert all(legal[v] for v in choices.values())
    return choices


def candidate_pool(choices, legal, split, number):
    pool = list(dict.fromkeys(choices.values()))
    rng = np.random.default_rng(seed("candidate_pool", split, number))
    remain = np.array([v for v in np.flatnonzero(legal) if v not in pool], dtype=int)
    rng.shuffle(remain)
    pool.extend(map(int, remain[:POOL - len(pool)]))
    assert len(pool) <= POOL and set(choices.values()) <= set(pool)
    return pool


def after_query(state, node, observed_edges, i, j, same):
    updated = state["queried"].copy()
    assert not updated[node]
    updated[node] = True
    mask = known_pairs(updated, i, j)
    return posterior(mask, observed_edges, same)[0]


def truth_objective(n, i, j, edges, samples, *stream):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(zip(i[edges].tolist(), j[edges].tolist()))
    return LiveEdgeObjective(graph, samples=samples,
                             rng=np.random.default_rng(seed(*stream)), p=P_IC)


def plan_choice(state, pool, base_seeds, n, i, j, same, full):
    split, number = state["split"], state["number"]
    q, known = state["q"], state["known"]
    gains = np.zeros(len(pool), dtype=float)
    for draw in range(N_DRAW):
        rng = np.random.default_rng(seed("posterior_draw", split, number, draw))
        hypothetical = sampled_graph(q, known, full, rng)
        # Distinct IC stream from the seed solver; shared across all candidates.
        evaluator = truth_objective(n, i, j, hypothetical, N_PLAN_EVAL_MC,
                                    "planning_evaluation", split, number, draw)
        baseline = evaluator.value(base_seeds)
        solver_stream = seed("planning_solver", split, number, draw)
        for a, node in enumerate(pool):
            updated_q = after_query(state, node, hypothetical, i, j, same)
            chosen = solve(n, i, j, updated_q, solver_stream)
            gains[a] += evaluator.value(chosen) - baseline
    gains /= N_DRAW
    # Fixed pool order breaks exact ties; it was determined without target gains.
    return int(pool[int(np.argmax(gains))]), [float(x) for x in gains]


def one_state(split, number, graph, i, j, same, full, full_index, *, preflight=False):
    n = graph.number_of_nodes()
    state = build_state(split, number, n, i, j, same, full)
    t0 = time.process_time()
    base = solve(n, i, j, state["q"], seed("base_solver", split, number))
    choices = simple_rules(state, n, i, j, full, full_index, base)
    pool = candidate_pool(choices, ~state["queried"], split, number)
    plan_node, q_scores = plan_choice(state, pool, base, n, i, j, same, full)
    planning_cpu = time.process_time() - t0
    if preflight:
        # The timing probe writes no decision or outcome; no gain-based adaptation is possible.
        return {"cpu_seconds": time.process_time() - t0, "planning_cpu_seconds": planning_cpu}
    choices["voi"] = plan_node
    t1 = time.process_time()
    chosen_seeds = {}
    for node in sorted(set(choices.values())):
        q_after = after_query(state, node, full, i, j, same)
        chosen_seeds[node] = solve(n, i, j, q_after, seed("final_solver", split, number))
    # One fresh true-graph live-edge batch shared by every method in this state.
    evaluator = truth_objective(n, i, j, full, N_CONFIRM_MC,
                                "true_confirmation", split, number)
    scores = {str(node): float(evaluator.value(seeds)) for node, seeds in chosen_seeds.items()}
    values = {name: scores[str(node)] for name, node in choices.items()}
    confirm_cpu = time.process_time() - t1
    return {"split": split, "state": number, "query_order": state["query_order"],
            "posterior_rates": state["rates"], "pool": pool,
            "choices": choices, "chosen_seed_sets": {name: chosen_seeds[node] for name, node in choices.items()},
            "planner_scores": {str(node): value for node, value in zip(pool, q_scores)},
            "values": values, "cpu_seconds": time.process_time() - t0,
            "planning_cpu_seconds": planning_cpu, "confirmation_cpu_seconds": confirm_cpu}


def paired(values):
    x = np.asarray(values, dtype=float)
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    radius = float(stats.t.ppf(0.975, len(x) - 1) * se)
    p = 1.0 if np.all(x == x[0]) and x[0] == 0 else float(stats.ttest_1samp(x, 0).pvalue)
    return {"mean": m, "ci95": [m-radius, m+radius],
            "p_two_sided": p,
            "wins": int(np.sum(x > 0)), "losses": int(np.sum(x < 0)),
            "ties": int(np.sum(x == 0)), "n": len(x)}


def load_existing(path: Path, plan_hash: str, script_hash: str):
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["plan_sha256"] == plan_hash and row["script_sha256"] == script_hash
    return row["result"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preflight", action="store_true", help="Time one full planning state; do not show gains")
    args = ap.parse_args()
    plan_hash, script_hash = digest(PLAN), digest(Path(__file__))
    graph, ids, cls, i, j, same, full, full_index, info = load_graph()
    print(f"graph {DAY}: n={info['n']} m={info['m']} rows={info['rows']} "
          f"UTC={info['first_utc']}..{info['last_utc']}", flush=True)
    if args.preflight:
        result = one_state("timing", 0, graph, i, j, same, full, full_index, preflight=True)
        atomic_json(OUT / "preflight_timing.json", {"plan_sha256": plan_hash,
                    "script_sha256": script_hash, **result})
        print(f"one-state planning CPU seconds: {result['planning_cpu_seconds']:.2f}", flush=True)
        print(f"100-state extrapolation: {100 * result['cpu_seconds'] / 3600:.2f} CPU hours "
              "(one-state estimate, not measured total)", flush=True)
        return
    start = time.process_time()
    dev = []
    for split, count, target in (("development", N_DEV, dev),):
        for number in range(count):
            path = OUT / split / f"state_{number:03d}.json"
            result = load_existing(path, plan_hash, script_hash)
            if result is None:
                result = one_state(split, number, graph, i, j, same, full, full_index)
                atomic_json(path, {"plan_sha256": plan_hash, "script_sha256": script_hash,
                                   "result": result})
            target.append(result)
            print(f"{split} {number+1}/{count} CPU={result['cpu_seconds']:.1f}s", flush=True)
            if sum(r["cpu_seconds"] for r in dev) > CPU_CAP:
                print("CPU cap reached: incomplete, result uncertain", flush=True)
                return
    means = {rule: float(np.mean([r["values"][rule] for r in dev])) for rule in RULES}
    best = max(RULES, key=lambda rule: means[rule])
    atomic_json(OUT / "development_choice.json", {"plan_sha256": plan_hash,
                "script_sha256": script_hash, "means": means, "baseline": best})
    print(f"frozen development comparator: {best}", flush=True)
    confirm = []
    for number in range(N_CONFIRM):
        path = OUT / "confirmation" / f"state_{number:03d}.json"
        result = load_existing(path, plan_hash, script_hash)
        if result is None:
            result = one_state("confirmation", 1000 + number, graph, i, j, same, full, full_index)
            atomic_json(path, {"plan_sha256": plan_hash, "script_sha256": script_hash,
                               "result": result})
        confirm.append(result)
        print(f"confirmation {number+1}/{N_CONFIRM} CPU={result['cpu_seconds']:.1f}s", flush=True)
        if sum(r["cpu_seconds"] for r in dev + confirm) > CPU_CAP:
            print("CPU cap reached: incomplete, result uncertain", flush=True)
            return
    contrasts = {rule: paired([r["values"]["voi"] - r["values"][rule] for r in confirm])
                 for rule in RULES}
    primary = contrasts[best]
    total_cpu = sum(r["cpu_seconds"] for r in dev + confirm)
    if primary["mean"] >= DELTA and primary["ci95"][0] > 0 and total_cpu <= CPU_CAP:
        decision = "continue_to_cost_and_cross_graph_tests"
    elif primary["ci95"][1] < DELTA:
        decision = "stop_for_practical_gain_in_this_configuration"
    else:
        decision = "uncertain"
    summary = {"plan_sha256": plan_hash, "script_sha256": script_hash,
               "graph": info, "baseline": best, "development_means": means,
               "confirmation_means": {name: float(np.mean([r["values"][name] for r in confirm]))
                                      for name in ("voi",) + RULES},
               "paired_contrasts": contrasts, "primary": primary, "delta": DELTA,
               "decision": decision, "measured_cpu_seconds": total_cpu,
               "mean_planning_cpu_seconds": float(np.mean([r["planning_cpu_seconds"] for r in confirm])),
               "state_count": len(confirm)}
    atomic_json(OUT / "summary.json", summary)
    print(json.dumps({"baseline": best, "primary": primary, "decision": decision,
                      "cpu_seconds": total_cpu}, indent=2), flush=True)


if __name__ == "__main__":
    main()
