"""Small, frozen TAIM strong-baseline probe. See PLAN.md before running."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "06_CCIM" / "_data"
GRAPHS = ("football", "polbooks")
PROBS = (0.1, 0.2)
DEV_N, TEST_N = 4, 8
GREEDY_L, FF_L, SOF_LO, SOF_HI = 32, 32, 8, 64
DEV_EVAL, TEST_EVAL = 128, 512
THETAS = (0.2, 0.4, 0.6, 0.8)
FRONTIER_THRESHOLDS = (0.0, 0.02, 0.05, 0.1, 0.2)


def rng_for(*parts):
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask -= bit


@dataclass(frozen=True)
class State:
    active: int
    frontier: int
    sid: str
    graph: str
    p: float


class IC:
    def __init__(self, graph_name, p):
        source = DATA / f"{graph_name}.gml"
        raw = nx.read_gml(source)
        labels = sorted(raw.nodes, key=str)
        index = {v: i for i, v in enumerate(labels)}
        self.n = len(labels)
        self.graph_name, self.p = graph_name, p
        arcs = sorted((index[u], index[v]) for u, v in raw.edges for u, v in ((u, v), (v, u)))
        self.src = np.array([u for u, _ in arcs], dtype=np.int16)
        self.dst = np.array([v for _, v in arcs], dtype=np.int16)
        self.edge_p = np.full(len(arcs), p, dtype=np.float64)
        self.P = np.zeros((self.n, self.n), dtype=np.float64)
        self.P[self.src, self.dst] = p
        self.degree = np.asarray((self.P > 0).sum(axis=1)).reshape(-1)
        self.allmask = (1 << self.n) - 1
        self.sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

    @classmethod
    def from_arcs(cls, n, arcs):
        """Tiny heterogeneous graph for mechanism regression tests only."""
        obj = cls.__new__(cls)
        obj.n = n
        obj.graph_name, obj.p = "hand_example", -1.0
        obj.src = np.array([u for u, _, _ in arcs], dtype=np.int16)
        obj.dst = np.array([v for _, v, _ in arcs], dtype=np.int16)
        obj.edge_p = np.array([p for _, _, p in arcs], dtype=np.float64)
        obj.P = np.zeros((n, n), dtype=np.float64)
        obj.P[obj.src, obj.dst] = obj.edge_p
        obj.degree = np.asarray((obj.P > 0).sum(axis=1)).reshape(-1)
        obj.allmask = (1 << n) - 1
        obj.sha256 = "hand-example"
        return obj

    def world(self, rng):
        live = rng.random(len(self.src)) < self.edge_p
        rows = [0] * self.n
        for u, v in zip(self.src[live], self.dst[live]):
            rows[int(u)] |= 1 << int(v)
        return rows

    def spread(self, active, frontier, seeds, world, rounds):
        added = 0
        for v in seeds:
            bit = 1 << int(v)
            if not (active & bit):
                added |= bit
        active |= added
        wave = frontier | added
        for _ in range(rounds):
            hits = 0
            for v in bits(wave):
                hits |= world[v]
            wave = hits & ~active
            active |= wave
        return active, wave

    def expected_last(self, active, frontier, seeds):
        source = list(bits(frontier)) + list(seeds)
        active2 = active | sum((1 << v for v in seeds), 0)
        available = np.fromiter((v for v in range(self.n) if not (active2 >> v) & 1), dtype=np.int16)
        if not len(available):
            return float(active2.bit_count())
        if source:
            no = np.prod(1.0 - self.P[np.array(source, dtype=int), :], axis=0)
        else:
            no = np.ones(self.n)
        return float(active2.bit_count() + np.sum(1.0 - no[available]))

    def final_greedy(self, active, frontier, budget):
        chosen = []
        active2 = active
        if frontier:
            no = np.prod(1.0 - self.P[np.fromiter(bits(frontier), dtype=int), :], axis=0)
        else:
            no = np.ones(self.n)
        for _ in range(budget):
            avail = np.fromiter((v for v in range(self.n) if not (active2 >> v) & 1), dtype=int)
            if not len(avail):
                break
            gain = no[avail] + self.P[np.ix_(avail, avail)] @ no[avail]
            v = int(avail[int(np.argmax(gain))])
            chosen.append(v)
            active2 |= 1 << v
            no *= 1.0 - self.P[v, :]
        return chosen, self.expected_last(active, frontier, chosen)

    def greedy_now(self, s, budget, worlds):
        prefix = []
        for _ in range(budget):
            avail = [v for v in range(self.n) if not (s.active >> v) & 1 and v not in prefix]
            scores = []
            for v in avail:
                candidate = prefix + [v]
                scores.append(sum(self.spread(s.active, s.frontier, candidate, w, 2)[0].bit_count() for w in worlds))
            prefix.append(avail[int(np.argmax(scores))])
        return prefix

    def sof(self, s, prefix, budget, worlds):
        values = []
        for i in range(budget + 1):
            total = 0.0
            for w in worlds:
                active, frontier = self.spread(s.active, s.frontier, prefix[:i], w, 1)
                _, value = self.final_greedy(active, frontier, budget - i)
                total += value
            values.append(total / len(worlds))
        return int(np.argmax(values)), values

    def ff_metrics(self, s, prefix, worlds):
        base = sum(self.spread(s.active, s.frontier, [], w, 2)[0].bit_count() for w in worlds)
        single = {}
        metrics = []
        for i, v in enumerate(prefix):
            if v not in single:
                single[v] = sum(self.spread(s.active, s.frontier, [v], w, 2)[0].bit_count() for w in worlds)
            with_v = sum(self.spread(s.active, s.frontier, prefix[: i + 1], w, 2)[0].bit_count() for w in worlds)
            without_v = sum(self.spread(s.active, s.frontier, prefix[:i], w, 2)[0].bit_count() for w in worlds)
            denom = single[v] - base
            ma = (with_v - without_v) / denom if denom > 0 else 0.0
            h1 = h2 = 0.0
            for w in worlds:
                after, _ = self.spread(s.active, s.frontier, prefix[:i], w, 2)
                if (after >> v) & 1:
                    continue
                one, _ = self.spread(after, 0, [v], w, 1)
                two, _ = self.spread(after, 0, [v], w, 2)
                h1 += one.bit_count() - after.bit_count()
                h2 += two.bit_count() - after.bit_count()
            mt = (h2 - h1) / h2 if h2 > 0 else 0.0
            metrics.append(0.5 * max(0.0, min(1.0, ma)) + 0.5 * max(0.0, min(1.0, mt)))
        return metrics

    def confirm(self, s, prefix, budget, i, worlds):
        values = []
        for w in worlds:
            active, frontier = self.spread(s.active, s.frontier, prefix[:i], w, 1)
            last, _ = self.final_greedy(active, frontier, budget - i)
            final, _ = self.spread(active, frontier, last, w, 1)
            values.append(final.bit_count())
        return np.asarray(values, dtype=float)


def ff_choice(metrics, theta):
    for i, score in enumerate(metrics):
        if score < theta:
            return i
    return len(metrics)


def cheap_choice(name, b, state, metrics):
    if name == "all_now":
        return b
    if name == "wait":
        return 0
    if name == "balanced":
        return (b + 1) // 2
    if name.startswith("frontier_"):
        parts = name.split("_")
        theta = float(parts[b])
        return 0 if state.frontier.bit_count() / max(1, state.active.bit_count()) > theta else b
    if name.startswith("ff_"):
        return ff_choice(metrics, float(name.split("_")[1]))
    raise ValueError(name)


def make_states(model, count, split):
    order = np.argsort(-model.degree, kind="stable")[:20]
    states, seen = [], set()
    for j in range(count):
        rng = rng_for("generation", split, model.graph_name, model.p, j)
        v = int(order[int(rng.integers(len(order)))])
        active, frontier = model.spread(0, 0, [v], model.world(rng), 1)
        # Repeated state is retained: no outcome-dependent filtering or retry.
        sid = f"{model.graph_name}_p{model.p}_{split}{j}"
        states.append(State(active, frontier, sid, model.graph_name, model.p))
        seen.add((active, frontier))
    return states, len(seen)


def decisions(model, s, b):
    t0 = time.perf_counter()
    # Reuse the same stream for b=1/2, so the first candidate is truly shared
    # when examining a budget-dependent timing decision on the same state.
    prefix = model.greedy_now(s, b, [model.world(rng_for("greedy", s.sid, j)) for j in range(GREEDY_L)])
    greedy_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    metrics = model.ff_metrics(s, prefix, [model.world(rng_for("ff", s.sid, j)) for j in range(FF_L)])
    ff_sec = time.perf_counter() - t0
    worlds = [model.world(rng_for("sof", s.sid, b, j)) for j in range(SOF_HI)]
    t0 = time.perf_counter()
    low, lowvalues = model.sof(s, prefix, b, worlds[:SOF_LO])
    low_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    high, highvalues = model.sof(s, prefix, b, worlds)
    high_sec = time.perf_counter() - t0
    return dict(prefix=prefix, metrics=metrics, low=low, high=high,
                lowvalues=lowvalues, highvalues=highvalues,
                time=dict(greedy=greedy_sec, ff=ff_sec, low=low_sec, high=high_sec))


def candidates():
    return ("all_now", "wait", "balanced") + tuple(
        f"frontier_{x}_{y}" for x in FRONTIER_THRESHOLDS for y in FRONTIER_THRESHOLDS
    ) + tuple(f"ff_{x}" for x in THETAS)


def stratum_key(s):
    return f"{s.graph}_p{s.p}"


def cluster_interval(rows, field_a, field_b):
    # Each base state contributes the average across b=1,2. Bootstrap within
    # graph/probability strata to avoid treating the two budgets independently.
    grouped = {}
    for row in rows:
        key = (row["stratum"], row["sid"])
        grouped.setdefault(key, []).append(row["values"][field_a] - row["values"][field_b])
    clusters = {}
    for (stratum, _), diffs in grouped.items():
        clusters.setdefault(stratum, []).append(float(np.mean(diffs)))
    point = float(np.mean([v for vals in clusters.values() for v in vals]))
    rng = rng_for("cluster-bootstrap")
    boot = []
    for _ in range(5000):
        boot.append(float(np.mean([x for vals in clusters.values() for x in rng.choice(vals, size=len(vals), replace=True)])))
    return dict(mean=point, ci95=[float(x) for x in np.quantile(boot, [0.025, 0.975])],
                stratum_means={k: float(np.mean(v)) for k, v in clusters.items()},
                clusters=sum(map(len, clusters.values())))


def run(smoke=False):
    t_start = time.perf_counter()
    models = {(g, p): IC(g, p) for g in GRAPHS for p in PROBS}
    dev_states, test_states, uniqueness = [], [], {}
    for model in models.values():
        a, ua = make_states(model, 1 if smoke else DEV_N, "dev")
        z, uz = make_states(model, 1 if smoke else TEST_N, "test")
        overlap = {(s.active, s.frontier) for s in a} & {(s.active, s.frontier) for s in z}
        if overlap:
            raise AssertionError(f"development/test state overlap: {model.graph_name} p={model.p}: {len(overlap)}")
        dev_states += a
        test_states += z
        uniqueness[f"{model.graph_name}_p{model.p}"] = dict(dev_unique=ua, test_unique=uz)
    full_states = dev_states + test_states
    cache = {}
    for j, s in enumerate(full_states):
        model = models[(s.graph, s.p)]
        for b in (1, 2):
            dec = decisions(model, s, b)
            n = 32 if smoke else (DEV_EVAL if s in dev_states else TEST_EVAL)
            worlds = [model.world(rng_for("confirm", s.sid, b, q)) for q in range(n)]
            outcome = {i: model.confirm(s, dec["prefix"], b, i, worlds) for i in range(b + 1)}
            cache[(s.sid, b)] = (dec, outcome)
        print(f"decided {j+1}/{len(full_states)} {s.sid}", flush=True)
    # One globally selected cheap configuration; never select separately by
    # held-out state, graph, or budget.
    scores = {}
    for name in candidates():
        vals = []
        for s in dev_states:
            for b in (1, 2):
                dec, outcome = cache[(s.sid, b)]
                i = cheap_choice(name, b, s, dec["metrics"])
                vals.append(float(outcome[i].mean()))
        scores[name] = float(np.mean(vals))
    selected = max(candidates(), key=lambda name: (scores[name], -candidates().index(name)))
    rows = []
    for s in test_states:
        for b in (1, 2):
            dec, outcome = cache[(s.sid, b)]
            picks = dict(cheap=cheap_choice(selected, b, s, dec["metrics"]), low=dec["low"], high=dec["high"])
            sec = dec["time"]
            rows.append(dict(sid=s.sid, stratum=stratum_key(s), b=b, active=s.active.bit_count(),
                             frontier=s.frontier.bit_count(), prefix=dec["prefix"], picks=picks,
                             values={k: float(outcome[v].mean()) for k, v in picks.items()},
                             decision_seconds=dict(cheap=sec["greedy"] + (sec["ff"] if selected.startswith("ff_") else 0.0),
                                                   low=sec["greedy"] + sec["low"], high=sec["greedy"] + sec["high"]),
                             paired_trial_differences={"high_minus_cheap": (outcome[picks["high"]] - outcome[picks["cheap"]]).tolist(),
                                                       "high_minus_low": (outcome[picks["high"]] - outcome[picks["low"]]).tolist()}))
    contrasts = {"high_minus_cheap": cluster_interval(rows, "high", "cheap"),
                 "low_minus_cheap": cluster_interval(rows, "low", "cheap"),
                 "high_minus_low": cluster_interval(rows, "high", "low")}
    result = dict(protocol="PLAN.md", graph_sha256={g: models[(g, PROBS[0])].sha256 for g in GRAPHS},
                  uniqueness=uniqueness, dev_scores=scores, selected_cheap=selected,
                  contrasts=contrasts, rows=rows, seconds=time.perf_counter()-t_start,
                  smoke=smoke, stream_note="generation, greedy, ff, sof, confirmation are domain-separated SHA256 seeds")
    path = HERE / ("smoke.json" if smoke else "result.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_cheap": selected, "contrasts": contrasts, "seconds": result["seconds"]}, ensure_ascii=False, indent=2))
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.smoke)
