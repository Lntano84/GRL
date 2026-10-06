"""Paired baseline curves on three development graphs: does survey ORDER matter under equal access?

Scope of this round, fixed before any result exists
---------------------------------------------------
Three **development** graphs, one per family, with the choice and the reasons written to
``results/xplore_dev_plan.json`` by ``--plan-only`` before a single number is produced.  This is a
developmental check on three graphs.  It is not a summary of the data directory, and no other graph is
promoted or demoted on the strength of it.

The main comparison holds **access privilege fixed**
---------------------------------------------------
``random`` / ``degree_max`` / ``degree_min`` all walk the frontier: they may only survey a node they have
already heard of.  Only those three appear in the main table.

CHANGE is reported **separately**, because the published implementation lets it survey uniformly over all
``n`` nodes (``random.sample(range(len(self.graph)), ...)``).  That is a different access privilege, so a
CHANGE-versus-frontier gap mixes privilege with strategy and is attributed to neither here.  Splitting the
two would need a controlled experiment, which this round does not attempt.

Two different privileges, kept apart
------------------------------------
* **global surveying** -- CHANGE only.  May reveal the neighbourhood of a node never heard of.
* **global seeding** -- *every* method.  This follows the official code: the discovered graph keeps all
  ``n`` nodes, and ``greedy(list(range(len(graph))), ...)`` selects over all of them.  So the protocol is
  stated explicitly: **all node identities are known; the relations of unsurveyed nodes are unknown; the
  final seed candidates may cover all nodes.**  To make that visible rather than implicit, every
  checkpoint records how many of the chosen seeds were *not yet discovered* at that moment.

How a checkpoint is read
------------------------
Each policy runs once to ``T = 20`` and the prefixes ``0,1,2,3,5,8,10,15,20`` are read off.  Three separate
random streams are used, so evaluating a checkpoint cannot perturb the survey stream:

* **policy stream** -- the policy's own choices, advanced only by stepping the environment;
* **selection stream** -- the 100 live-edge samples behind the greedy seed choice, seeded per
  ``(graph, group, T)`` so it is identical across the three frontier methods;
* **evaluation stream** -- 1000 live-edge samples on the complete graph, fixed per ``(graph, group)`` and
  shared by *every* method and *every* budget, which is what makes the paired differences tight.

If the frontier empties, the run stops there, the actual number of surveys is recorded, and later
checkpoints carry the frozen final value forward.  The policy never silently gains global access.

What the greedy guarantee does and does not say
-----------------------------------------------
Greedy's ``(1 - 1/e)`` guarantee applies to the coverage objective *defined by those 100 sampled
live-edge graphs*.  It is not a guarantee about the true expected spread, and it is not reported as one.
"""

from __future__ import annotations

import argparse
import hashlib
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

from ccim.xplore import DiscoveryEnv  # noqa: E402
from ccim.xplore.icm import INFL_BUDGET, PROP_PROBAB, LiveEdgeObjective  # noqa: E402

DATA = ROOT / "_data" / "xplore"
PLAN = ROOT / "results" / "xplore_dev_plan.json"
OUTPUT = ROOT / "results" / "xplore_curve.json"

DEV_GRAPHS = ("mammal__bhp", "interact__soc-wiki-Vote", "rt__damascus")
DEV_REASONS = {
    "mammal__bhp": ("mammal family; n=1686, m=4623, mean degree 5.5 -- the densest of the mammal set, "
                    "chosen to cover a moderately dense, non-star-like structure"),
    "interact__soc-wiki-Vote": ("interact family; n=889, m=2914, mean degree 6.6 -- mid-size, densely "
                                "connected, structurally unlike both the mammal and the retweet families"),
    "rt__damascus": ("retweet family; n=3052, m=3869, mean degree 2.5 -- sparse and star-like.  Already "
                     "inspected in a smoke test, so it is a DEVELOPMENT graph and must never be presented "
                     "as an unseen test graph."),
}
GROUPS_PER_GRAPH = 20
BUDGETS = (0, 1, 2, 3, 5, 8, 10, 15, 20)
T_MAX = 20
PRIMARY_T = 5
SELECT_SAMPLES = 100
EVAL_SAMPLES = 1000
EXTRA_SEEDS = 5
MAIN_POLICIES = ("random", "degree_max", "degree_min")
REFERENCE_POLICIES = ("change",)
PLAN_RNG_SEED = 20240119


def sseed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 63)


def make_policy(kind: str, diag: dict):
    """Frontier policies.  Ties break uniformly at random from a fixed rule, never by node id."""
    def fn(env: DiscoveryEnv, rng):
        frontier = sorted(env.possible_actions)
        if not frontier:
            return None
        diag["frontier_sizes"].append(len(frontier))
        obs = env.observed
        if kind == "random":
            return int(frontier[int(rng.integers(len(frontier)))])
        deg = {u: obs.degree(u) for u in frontier}
        target = max(deg.values()) if kind == "degree_max" else min(deg.values())
        tied = [u for u in frontier if deg[u] == target]
        diag["tie_fractions"].append(len(tied) / len(frontier))
        diag["distinct_degrees"].append(len(set(deg.values())))
        return int(tied[int(rng.integers(len(tied)))])
    fn.__name__ = kind
    return fn


def change_survey_plan(full_graph, budget_queries: int, rng):
    """The published CHANGE, kept as a different-privilege reference (global surveys)."""
    if budget_queries <= 0:
        return []
    nodes = list(full_graph.nodes())
    half = max(1, int((2 * budget_queries) / 2))
    v1 = [nodes[i] for i in rng.choice(len(nodes), size=min(half, len(nodes)), replace=False)]
    plan = list(v1)
    for u in v1:
        nb = sorted(set(full_graph.neighbors(u)) - set(v1))
        if nb:
            plan.append(nb[int(rng.integers(len(nb)))])
    seen, out = set(), []
    for u in plan:
        if u not in seen:
            seen.add(u)
            out.append(int(u))
    return out


def evaluate_checkpoint(env, eval_obj, graph, group, t, kind) -> dict:
    """Greedy on the discovered graph (own stream), then the reward on the complete graph (shared stream)."""
    sel = LiveEdgeObjective(env.graph, samples=SELECT_SAMPLES,
                            rng=np.random.default_rng(sseed(graph, group, t, "select")), p=PROP_PROBAB)
    seeds, _ = sel.greedy(INFL_BUDGET)
    infl, se = eval_obj.value_with_error(seeds)
    discovered = env.active | env.possible_actions
    return {"influence": infl, "influence_se": se,
            "undiscovered_seeds": int(sum(1 for s in seeds if s not in discovered)),
            "discovered_nodes": env.discovered_nodes,
            "frontier": len(env.possible_actions),
            "surveys_done": len(env.queried),
            "seeds": [int(s) for s in seeds]}


def run_prefixes(g, kind, seeds, rng_policy, eval_obj, graph, group, diag) -> dict:
    env = DiscoveryEnv(g, seeds, max_T=T_MAX, p=PROP_PROBAB, infl_budget=INFL_BUDGET)
    fn = make_policy(kind, diag) if kind in MAIN_POLICIES else None
    plan_iter = iter(change_survey_plan(g, T_MAX, rng_policy)) if kind == "change" else None

    checkpoints, t = {}, 0
    while t <= T_MAX:
        if t in BUDGETS:
            checkpoints[t] = evaluate_checkpoint(env, eval_obj, graph, group, t, kind)
        if t == T_MAX:
            break
        if not env.done():
            if kind == "change":
                u = next(plan_iter, None)
                if u is not None:
                    env._enlarge(u)
                    env.queried.append(u)
            else:
                u = fn(env, rng_policy)
                if u is not None:
                    env.step(u)
        t += 1
    return {"checkpoints": checkpoints, "actual_surveys": len(env.queried)}


def build_plan() -> dict:
    generated = {}
    for graph in DEV_GRAPHS:
        g = pickle.loads((DATA / f"{graph}.pkl").read_bytes())
        n = g.number_of_nodes()
        entries = []
        for i in range(GROUPS_PER_GRAPH):
            seeds = [int(x) for x in np.random.default_rng(sseed(graph, i, "initial")).choice(
                n, EXTRA_SEEDS, replace=False)]
            entries.append({"group": i, "initial_nodes": sorted(seeds),
                            "policy_seed_offset": i})
        generated[graph] = {"n": n, "m": g.number_of_edges(), "entries": entries}
    return {"plan_rng_seed": PLAN_RNG_SEED, "graphs": list(DEV_GRAPHS), "reasons": DEV_REASONS,
            "groups_per_graph": GROUPS_PER_GRAPH, "budgets": list(BUDGETS), "t_max": T_MAX,
            "primary_checkpoint": PRIMARY_T, "select_samples": SELECT_SAMPLES,
            "eval_samples": EVAL_SAMPLES, "extra_seeds": EXTRA_SEEDS,
            "main_policies": list(MAIN_POLICIES), "reference_policies": list(REFERENCE_POLICIES),
            "generated": generated,
            "note": ("initial survey nodes, policy seeds and the graph choice are fixed here, before any "
                     "result exists; every method shares the same initial observation")}


def cells(records, policy, T, graph=None):
    return [r for r in records if r["policy"] == policy and r["budget"] == T
            and (graph is None or r["graph"] == graph)]


def paired(records, a, b, T, graph=None):
    ca = {(r["graph"], r["group"]): r["influence"] for r in cells(records, a, T, graph)}
    cb = {(r["graph"], r["group"]): r["influence"] for r in cells(records, b, T, graph)}
    keys = sorted(set(ca) & set(cb))
    d = [ca[k] - cb[k] for k in keys]
    if len(d) < 2:
        return {"n": len(d), "mean": (d[0] if d else None), "lo": None, "hi": None,
                "wins": None, "losses": None}
    mean = statistics.mean(d)
    half = stats.t.ppf(0.975, len(d) - 1) * statistics.stdev(d) / len(d) ** 0.5
    return {"n": len(d), "mean": mean, "lo": mean - half, "hi": mean + half,
            "wins": sum(1 for x in d if x > 0), "losses": sum(1 for x in d if x < 0)}


def report(records, artifact):
    print("\n" + "=" * 100)
    print("  (1) SURVEY BUDGET vs INFLUENCE ON THE COMPLETE NETWORK  (mean over 20 paired groups)")
    print("=" * 100)
    for graph in artifact["plan"]["graphs"]:
        n = next(r["n"] for r in records if r["graph"] == graph)
        print(f"\n    {graph}  (n={n})")
        print(f"      {'T':>3}" + "".join(f"{p:>15}" for p in MAIN_POLICIES)
              + f"{'change(global)':>16}{'evalSE':>8}{'discovered r/max/min':>24}")
        for T in BUDGETS:
            row = f"      {T:>3}"
            for p in MAIN_POLICIES:
                c = cells(records, p, T, graph)
                row += f"{statistics.mean(x['influence'] for x in c):>15.2f}" if c else f"{'-':>15}"
            ch = cells(records, "change", T, graph)
            if ch:
                row += f"{statistics.mean(x['influence'] for x in ch):>16.2f}"
            se = statistics.mean(x["influence_se"] for x in cells(records, "random", T, graph))
            row += f"{se:>8.2f}"
            row += "".join(
                f"{statistics.mean(x['discovered_nodes'] for x in cells(records, p, T, graph)):>8.0f}"
                for p in MAIN_POLICIES)
            print(row)

    print("\n" + "=" * 100)
    print("  (2) PAIRED DIFFERENCES WITHIN EQUAL FRONTIER ACCESS  (vs random, 95% t interval)")
    print("=" * 100)
    print(f"    {'graph':<26}{'T':>4}{'comparison':>22}{'mean':>9}{'lo':>8}{'hi':>8}{'W/L':>8}")
    for graph in artifact["plan"]["graphs"]:
        for T in (PRIMARY_T, 20):
            for p in ("degree_max", "degree_min"):
                r = paired(records, p, "random", T, graph)
                wl = f"{r['wins']}/{r['losses']}" if r.get("wins") is not None else "-"
                lo = f"{r['lo']:.2f}" if r["lo"] is not None else "-"
                hi = f"{r['hi']:.2f}" if r["hi"] is not None else "-"
                mean = f"{r['mean']:.2f}" if r["mean"] is not None else "-"
                print(f"    {graph:<26}{T:>4}{p + ' - random':>22}{mean:>9}{lo:>8}{hi:>8}{wl:>8}")

    print("\n" + "=" * 100)
    print("  (3) DID THE STRATEGIES DEGRADE?  frontier size, degree ties, exhaustion, undiscovered seeds")
    print("=" * 100)
    print(f"    {'graph':<26}{'policy':>12}{'frontier@T=5':>13}{'tie frac':>10}"
          f"{'distinct deg':>13}{'exhausted':>11}{'undisc seeds':>14}")
    for graph in artifact["plan"]["graphs"]:
        for p in MAIN_POLICIES:
            c = cells(records, p, PRIMARY_T, graph)
            ex = sum(1 for x in c if x["episode_exhausted_early"])
            tie = [x["mean_tie_fraction"] for x in c if x["mean_tie_fraction"] is not None]
            deg = [x["mean_distinct_frontier_degrees"] for x in c
                   if x["mean_distinct_frontier_degrees"] is not None]
            tie_s = f"{statistics.mean(tie):.3f}" if tie else "n/a"
            deg_s = f"{statistics.mean(deg):.2f}" if deg else "n/a"
            print(f"    {graph:<26}{p:>12}"
                  f"{statistics.mean(x['mean_frontier_size'] for x in c):>13.1f}{tie_s:>10}{deg_s:>13}"
                  f"{ex:>11}{statistics.mean(x['undiscovered_seeds'] for x in c):>14.2f}")
    und = [r["undiscovered_seeds"] for r in records if r["budget"] == PRIMARY_T]
    print(f"\n    undiscovered seeds at T={PRIMARY_T}, all graphs and methods: mean "
          f"{statistics.mean(und):.2f} of 10   max {max(und)}")
    print("    (global seeding is the official protocol: the seed set may name nodes never surveyed)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.plan_only:
        plan = build_plan()
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print("=" * 100)
        print("  DEVELOPMENT PLAN FROZEN -- written before any result exists")
        print("=" * 100)
        for gname in DEV_GRAPHS:
            info = plan["generated"][gname]
            print(f"    {gname:<26} n={info['n']:<5} m={info['m']:<6} groups={GROUPS_PER_GRAPH}")
            print(f"        why: {DEV_REASONS[gname]}")
        print(f"\n    budgets {list(BUDGETS)}, primary checkpoint T={PRIMARY_T}, run to T={T_MAX}")
        print(f"    main (frontier access)   : {list(MAIN_POLICIES)}")
        print(f"    separate (global access) : {list(REFERENCE_POLICIES)}")
        print(f"    selection samples {SELECT_SAMPLES} (own stream) / evaluation samples "
              f"{EVAL_SAMPLES} (shared, paired)")
        print(f"\n    written to {args.plan}")
        return 0

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    print("=" * 100)
    print("  PAIRED BASELINE CURVES -- 3 development graphs, equal frontier access in the main table")
    print("=" * 100)
    print("    protocol: all node identities known; relations of unsurveyed nodes unknown; final seed")
    print("              candidates may cover all nodes (the official code selects over range(n))")
    print(f"    streams : policy | selection {SELECT_SAMPLES} per (graph,group,T) | evaluation "
          f"{EVAL_SAMPLES} per (graph,group) shared by all methods and budgets")

    records, t_start = [], time.perf_counter()
    for graph in plan["graphs"]:
        g = pickle.loads((DATA / f"{graph}.pkl").read_bytes())
        info = plan["generated"][graph]
        for entry in info["entries"]:
            grp = entry["group"]
            seeds = entry["initial_nodes"]
            eval_obj = LiveEdgeObjective(g, samples=EVAL_SAMPLES,
                                         rng=np.random.default_rng(sseed(graph, grp, "eval")),
                                         p=PROP_PROBAB)
            for kind in list(MAIN_POLICIES) + list(REFERENCE_POLICIES):
                diag = {"frontier_sizes": [], "tie_fractions": [], "distinct_degrees": []}
                rng_policy = np.random.default_rng(sseed(graph, grp, kind, "policy"))
                out = run_prefixes(g, kind, seeds, rng_policy, eval_obj, graph, grp, diag)
                for T, cell in out["checkpoints"].items():
                    records.append({
                        "graph": graph, "family": graph.split("__")[0], "group": grp, "budget": T,
                        "policy": kind, "n": info["n"],
                        "access": "global" if kind == "change" else "frontier",
                        "actual_surveys": min(T, out["actual_surveys"]),
                        "episode_exhausted_early": out["actual_surveys"] < T,
                        **{k: v for k, v in cell.items() if k != "seeds"},
                        "seed_set": cell["seeds"],
                        "mean_frontier_size": (statistics.mean(diag["frontier_sizes"])
                                               if diag["frontier_sizes"] else None),
                        "mean_tie_fraction": (statistics.mean(diag["tie_fractions"])
                                              if diag["tie_fractions"] else None),
                        "mean_distinct_frontier_degrees": (statistics.mean(diag["distinct_degrees"])
                                                           if diag["distinct_degrees"] else None),
                    })
        print(f"    {graph:<26} done", flush=True)
    seconds = time.perf_counter() - t_start

    artifact = {"script": Path(__file__).name, "plan": plan,
                "protocol": {
                    "fixed_access": "random / degree_max / degree_min all walk the frontier",
                    "different_privilege_reference": "change surveys uniformly over all n nodes",
                    "seeding": ("all methods select over all n nodes (official behaviour); the count of "
                                "seeds not yet discovered is recorded at every checkpoint"),
                    "select_samples": SELECT_SAMPLES, "eval_samples": EVAL_SAMPLES,
                    "streams": "policy, selection and evaluation are three independent streams",
                    "frontier_exhaustion": ("the run stops, the actual survey count is recorded, later "
                                            "checkpoints carry the final value forward"),
                    "greedy_guarantee": ("applies to the coverage objective of the 100 sampled live-edge "
                                         "graphs, not to the true expected spread"),
                },
                "run_seconds": seconds, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    report(records, artifact)
    print(f"\n    run took {seconds:.1f} s   ({len(records):,} checkpoint records)")
    print(f"    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
