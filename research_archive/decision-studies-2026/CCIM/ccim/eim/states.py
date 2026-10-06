"""One episode: the survey history, every arm's decision, and the confirmatory score.

Protocol (frozen before results)
--------------------------------
1. The organisation starts with a **cold** model: the roster and the attributes, no contacts at all.
2. ``FREE_SURVEYS`` members are surveyed.  They are drawn uniformly from the roster with a dedicated
   random stream, so the starting observation is *not* chosen by any clever rule and not chosen by the
   method being tested.  This represents the contacts an organisation has before it starts paying.
3. ``FIXED_RANDOM_SURVEYS`` further members are surveyed, again uniformly from the roster with the same
   dedicated stream.  Both steps use the **observation stream only**, so the record every method sees
   is identical and is unaffected by which arm is being evaluated.
4. The **one paid survey** is the decision under test: each arm chooses one still-unsurveyed member
   from the full roster, surveys them, and the fixed solver turns the resulting record into ``k`` seeds.
5. A ``no survey`` arm is scored as well: it solves on the record from step 3 and buys nothing.  It is
   the control that says whether the paid survey is worth anything at all.

Stream separation.  Four independent streams per state -- observation, look-ahead, rule sampling, and
confirmatory scoring -- are seeded from the state id.  A method therefore cannot alter the data it is
judged on, and adding or removing candidates cannot move the observations, which is what makes the
"candidate pool" step reproducible instead of self-referential.
"""

from __future__ import annotations

import time

import numpy as np

from . import lookahead, rules
from .gm import HomophilyModel
from .ic import RRSolver, TrueInfluence

FREE_SURVEYS = 10          # surveyed before any method acts (the "already known" contacts)
FIXED_RANDOM_SURVEYS = 4   # protocol surveys, uniformly random, identical for every arm
PRIMARY_K = 10             # seed set size
PRIMARY_P = 0.1            # IC probability, the official exploratory-IM value


def _rng(seed: int, tag: int):
    return np.random.default_rng([int(seed), int(tag)])


def true_neighbours_fn(edges, n):
    nb = [[] for _ in range(n)]
    for a, b in edges:
        nb[int(a)].append(int(b))
        nb[int(b)].append(int(a))
    return lambda u: nb[int(u)]


class State:
    """The frozen record of one state, plus the seed set each arm produced."""

    def __init__(self, sid: int, nodes, edges, attr, cfg: dict):
        self.sid = int(sid)
        self.cfg = dict(cfg)
        self.nodes = list(nodes)
        self.n = len(nodes)
        self.attr = list(attr)
        self.edges = np.asarray(edges, dtype=np.int64)
        self.truth_nb = true_neighbours_fn(self.edges, self.n)
        self.records: dict = {"sid": self.sid, "n": self.n, "m": int(len(self.edges))}

    # ---------------------------------------------------------------- observation phase
    def observe(self) -> None:
        rng = _rng(self.cfg["seed"], 101)
        model = HomophilyModel(self.n, self.attr)
        surveyed: list[int] = []
        seq = list(rng.choice(self.n, size=FREE_SURVEYS + FIXED_RANDOM_SURVEYS, replace=False))
        for u in seq:
            u = int(u)
            model.resolve(u, self.truth_nb(u))
            surveyed.append(u)
        self.model = model
        self.observed_nodes = surveyed
        self.records["observed_nodes"] = surveyed
        self.records["model_after_observation"] = model.summary()

    # ---------------------------------------------------------------- arms
    def _legal(self):
        return rules.legal_nodes(self.model)

    def _scores(self):
        rng = _rng(self.cfg["seed"], 103)
        solver = RRSolver(self.n, PRIMARY_K, self.cfg["rule_solver_samples"], rng)
        out = {
            "B2_pred_residual_degree": rules.predicted_residual_degree(self.model),
            "B1_observed_degree": rules.observed_degree(self.model),
        }
        out["B3_seed_coverage"] = rules.coverage_probability(
            self.model, rng, self.cfg["rule_samples"], PRIMARY_K, solver)
        out["B4_seed_distance"] = rules.seed_distance_weight(
            self.model, rng, PRIMARY_K, self.cfg["rule_samples"], solver)
        return out

    def run_arms(self) -> None:
        legal = self._legal()
        scores = self._scores()
        pool_rng = _rng(self.cfg["seed"], 104)
        pool = rules.candidate_pool(scores, legal, pool_rng, size=self.cfg["pool_size"])
        self.records["candidate_pool"] = pool
        self.records["legal_count"] = int(len(legal))

        # each arm's own solver/sampling stream
        own = {name: RRSolver(self.n, PRIMARY_K, self.cfg["own_solver_samples"],
                              _rng(self.cfg["seed"], 200 + i))
               for i, name in enumerate(["B0_random", "B1_observed_degree", "B2_pred_residual_degree",
                                         "B3_seed_coverage", "B4_seed_distance", "Q_lookahead"])}

        arm_scores = dict(scores)
        arm_scores["B0_random"] = np.zeros(self.n)     # uniform selection handled by `uniform=True`

        def finish(name: str, u: int, t0: float, extra: dict):
            child = self.model.copy()
            child.resolve(u, self.truth_nb(u))
            edges, probs = child.edges_for_solver(own[name].rng)
            idx, value = own[name].solve(edges, probs)
            seeds = [self.nodes[int(i)] for i in idx]
            rec = {"chosen_node": self.nodes[u], "chosen_index": int(u),
                   "seeds": seeds,
                   "survey_yield": int(len(self.truth_nb(u))),
                   "predicted_value": float(value),
                   "seconds": time.perf_counter() - t0}
            rec.update(extra)
            return rec

        # ---- no-survey control
        ctrl = self.model.copy()
        e0, p0 = ctrl.edges_for_solver(own["B1_observed_degree"].rng)
        idx0, val0 = own["B1_observed_degree"].solve(e0, p0)
        self.records["arms"] = {"no_survey": {
            "chosen_node": None, "seeds": [self.nodes[int(i)] for i in idx0],
            "predicted_value": float(val0), "seconds": 0.0}}

        # ---- cheap rules
        for name in ["B0_random", "B1_observed_degree", "B2_pred_residual_degree",
                     "B3_seed_coverage", "B4_seed_distance"]:
            t0 = time.perf_counter()
            if name == "B0_random":
                u = rules.pick(arm_scores[name], legal, _rng(self.cfg["seed"], 150), uniform=True)
            else:
                u = rules.pick(arm_scores[name], legal)
            self.records["arms"][name] = finish(name, int(u), t0, {})

        # ---- legal look-ahead
        timing = {"base_seconds": 0.0, "candidate_seconds": 0.0}
        t0 = time.perf_counter()
        q, diag = lookahead.q_values(self.model, pool, _rng(self.cfg["seed"], 201),
                                     self.cfg["lookahead_samples"], own["Q_lookahead"],
                                     timing=timing)
        u = max(q, key=lambda k: (q[k], -k))
        rec = finish("Q_lookahead", int(u), t0, {
            "lookahead_scores": {self.nodes[k]: v for k, v in sorted(q.items())},
            "expected_survey_yield": {self.nodes[k]: v for k, v in sorted(diag["expected_survey_yield"].items())},
            "pool_in_pool": [self.nodes[k] for k in pool],
            "timer_internal": dict(timing),
        })
        self.records["arms"]["Q_lookahead"] = rec

        # ---- privileged diagnostic: the survey RESULT is assumed known before choosing
        t0 = time.perf_counter()
        pg = lookahead.privileged_gain(self.model.copy(), pool, _rng(self.cfg["seed"], 301),
                                       self.cfg["lookahead_samples"], own["Q_lookahead"],
                                       self.truth_nb)
        up = max(pg, key=lambda k: (pg[k], -k))
        self.records["arms"]["P_privileged"] = finish("Q_lookahead", int(up), t0, {
            "lookahead_scores": {self.nodes[k]: v for k, v in sorted(pg.items())},
            "note": "diagnostic only: uses the real survey result before choosing. Not deployable, "
                    "not an upper bound."})

        # ---- what the legal look-ahead thought versus what actually happened
        realised = {self.nodes[k]: float(
            self.records["arms"]["Q_lookahead"]["lookahead_scores"].get(self.nodes[k], float("nan")))
            for k in pool}
        self.records["lookahead_realised"] = realised

    # ---------------------------------------------------------------- confirmation
    def confirm(self) -> None:
        rng = _rng(self.cfg["seed"], 401)
        scorer = TrueInfluence(self.n, self.edges, PRIMARY_P, self.cfg["confirm_samples"], rng)
        for name, rec in self.records["arms"].items():
            idx = [self.nodes.index(s) for s in rec["seeds"]]
            rec["sigma_true"] = scorer.value(idx)
            rec["sigma_true_normalized"] = rec["sigma_true"] / self.n
        self.scorer = scorer


def run_state(sid: int, nodes, edges, attr, cfg: dict) -> dict:
    st = State(sid, nodes, edges, attr, cfg)
    st.observe()
    st.run_arms()
    st.confirm()
    return st.records
