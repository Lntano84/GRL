"""Is there any value to learn in the survey DECISION?  A no-training local-opportunity probe.

The question this answers, and the one it does not
--------------------------------------------------
Everything measured so far compares whole policies.  This fixes the visible state and changes **one**
action -- the last survey -- then asks whether the choice of that one target can move the final influence
materially.  It is a **local, last-step, finite-candidate-pool** opportunity check.  It is **not** a
deployable method and **not** a true optimum: the candidate pool is capped and only the last step moves.

Discipline that makes the number meaningful
-------------------------------------------
* **States are fixed in advance by a rule, never by gain.**  20 states, each built by drawing 5 free seeds
  and applying 4 surveys under fixed RNG streams, so "the state before the last survey" is reproducible and
  cannot be cherry-picked.
* **Candidate pool is fixed and capped at 8.**  It always contains the ``degree_min`` and ``degree_max``
  choices on the visible graph; the rest are filled from a fixed random seed.
* **Simulation and confirmation are separate batches.**  The candidate is *chosen* on a small simulation
  batch and *scored* on an independent confirmation batch.  Taking the maximum of the same noisy batch
  would manufacture an improvement out of noise, so it is never done.
* **Budgets are fixed before running** (``--sim-samples``, ``--conf-samples``, ``--states``, ``--candidates``).
  If the budget is exhausted the answer is "uncertain", not "extend until significant".
* **The complete graph is used only to score this diagnostic.**  Nothing here may become an input to any
  future policy; the visible state is the only thing a policy could ever see.
"""

from __future__ import annotations

import argparse
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
import xplore_official as OFF  # noqa: E402
from ccim.xplore import DiscoveryEnv  # noqa: E402
from ccim.xplore.icm import INFL_BUDGET, PROP_PROBAB, LiveEdgeObjective  # noqa: E402

OUTPUT = ROOT / "results" / "survey_value_probe.json"
GRAPH = "interact__soc-wiki-Vote"
N_STATES = 20
N_CANDIDATES = 8
T_MAX = 5
EXTRA_SEEDS = 5
SELECT_SAMPLES = 100          # official seed-selection setting
SIM_SAMPLES = 300             # budget for CHOOSING a candidate
CONF_SAMPLES = 3000           # independent budget for SCORING it
STATE_RULE = "5 fixed free seeds + 4 uniform-random-frontier surveys under fixed RNG streams"


def sseed(*parts) -> int:
    return SIMP.sseed(*parts)


def build_states(g, n: int) -> list:
    """The 20 states, each the moment before its last survey.  Rule fixed here, not chosen by gain."""
    out = []
    for i in range(n):
        seeds = [int(x) for x in np.random.default_rng(sseed("valprobe", i, "seeds"))
                 .choice(g.number_of_nodes(), EXTRA_SEEDS, replace=False)]
        env = DiscoveryEnv(g, seeds, max_T=T_MAX, p=PROP_PROBAB, infl_budget=INFL_BUDGET)
        rng = np.random.default_rng(sseed("valprobe", i, "walk"))
        for _ in range(T_MAX - 1):                     # 4 surveys -> one left
            frontier = sorted(env.possible_actions)
            if not frontier:
                break
            env.step(int(frontier[int(rng.integers(len(frontier)))]))
        out.append({"i": i, "seeds": seeds, "env": env})
    return out


def candidates_for(env, i: int) -> list:
    """At most 8 legal frontier nodes: the degree_min and degree_max picks, rest from a fixed seed."""
    frontier = sorted(env.possible_actions)
    if not frontier:
        return []
    obs = env.observed
    deg = {u: obs.degree(u) for u in frontier}
    lo, hi = min(deg.values()), max(deg.values())
    pick = []
    rng = np.random.default_rng(sseed("valprobe", i, "cand"))
    for target in (lo, hi):                            # degree_min and degree_max choices
        tied = [u for u in frontier if deg[u] == target]
        pick.append(int(tied[int(rng.integers(len(tied)))]))
    pool = [u for u in frontier if u not in pick]
    rng.shuffle(pool)
    pick.extend(int(u) for u in pool[:max(0, N_CANDIDATES - len(pick))])
    return pick[:N_CANDIDATES]


def rollout_after(env, cand) -> dict:
    """Survey ``cand`` as the last action and take the official greedy seed set on the result."""
    import copy
    e = copy.deepcopy(env)
    if cand is not None:
        e.step(int(cand))
    sel = LiveEdgeObjective(e.graph, samples=SELECT_SAMPLES,
                            rng=np.random.default_rng(sseed("valprobe", "select")), p=PROP_PROBAB)
    chosen, _ = sel.greedy(INFL_BUDGET)
    return {"chosen": chosen, "discovered": len(e.active | e.possible_actions)}


def score(g, chosen, tag, samples) -> float:
    obj = LiveEdgeObjective(g, samples=samples,
                            rng=np.random.default_rng(sseed("valprobe", "score", tag)), p=PROP_PROBAB)
    return obj.value(chosen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", type=int, default=N_STATES)
    ap.add_argument("--candidates", type=int, default=N_CANDIDATES)
    ap.add_argument("--sim-samples", type=int, default=SIM_SAMPLES)
    ap.add_argument("--conf-samples", type=int, default=CONF_SAMPLES)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    g = pickle.loads((SIMP.DATA / f"{GRAPH}.pkl").read_bytes())
    print("=" * 104)
    print("  SURVEY-DECISION VALUE PROBE -- no training, last survey only, fixed states and budgets")
    print("=" * 104)
    print(f"    graph      : {GRAPH} (n={g.number_of_nodes()})")
    print(f"    state rule : {STATE_RULE}")
    print(f"    budgets    : states={args.states} candidates<={args.candidates} "
          f"sim={args.sim_samples} confirmation={args.conf_samples} samples")

    states = build_states(g, args.states)
    t0 = time.perf_counter()
    rows = []
    for st in states:
        env = st["env"]
        cands = candidates_for(env, st["i"])
        if not cands:
            continue
        # ---- per-candidate outcome and seed set (identical work for every candidate)
        outs = {c: rollout_after(env, c) for c in cands}
        base = rollout_after(env, None)               # stop without spending the last survey
        # simulation batch: used ONLY to pick
        sim = {c: score(g, outs[c]["chosen"], f"{st['i']}:sim:{c}", args.sim_samples) for c in cands}
        sim_base = score(g, base["chosen"], f"{st['i']}:sim:base", args.sim_samples)
        chosen_c = max(cands, key=lambda c: sim[c] - sim_base)
        # independent confirmation batch: used ONLY to score
        conf = {c: score(g, outs[c]["chosen"], f"{st['i']}:conf:{c}", args.conf_samples) for c in cands}
        conf_base = score(g, base["chosen"], f"{st['i']}:conf:base", args.conf_samples)
        rng = np.random.default_rng(sseed("valprobe", st["i"], "randomsurvey"))
        rand_c = int(cands[int(rng.integers(len(cands)))])
        rows.append({
            "state": st["i"], "n_candidates": len(cands), "frontier": len(env.possible_actions),
            "selected": int(chosen_c),
            "conf_selected": conf[chosen_c], "conf_degree_min": conf[cands[0]],
            "conf_degree_max": conf[cands[1]], "conf_random": conf[rand_c],
            "conf_no_survey": conf_base,
            "gain_selected_vs_no": conf[chosen_c] - conf_base,
            "gain_selected_vs_degree_min": conf[chosen_c] - conf[cands[0]],
            "gain_selected_vs_random": conf[chosen_c] - conf[rand_c],
            "sim_spread": max(sim.values()) - min(sim.values()),
            "conf_spread": max(conf.values()) - min(conf.values()),
            "selected_is_degree_min": bool(chosen_c == cands[0]),
            "selected_is_degree_max": bool(chosen_c == cands[1]),
        })
    seconds = time.perf_counter() - t0

    def ci(vals):
        v = np.asarray(vals, dtype=float)
        m = float(v.mean())
        if len(v) < 2:
            return m, None, None, None
        h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
        return m, m - h, m + h, float(stats.ttest_1samp(v, 0).pvalue)

    print("\n" + "=" * 104)
    print("  CORE TABLE -- independent confirmation batch only (selection used a separate sim batch)")
    print("=" * 104)
    print(f"    {'comparison':<42}{'mean':>9}{'95% CI':>20}{'p':>9}{'W/L':>9}")
    table = {}
    for label, key in (("selected - no last survey", "gain_selected_vs_no"),
                       ("selected - degree_min", "gain_selected_vs_degree_min"),
                       ("selected - random survey", "gain_selected_vs_random")):
        vals = [r[key] for r in rows]
        m, lo, hi, p = ci(vals)
        w = sum(1 for v in vals if v > 0)
        l = sum(1 for v in vals if v < 0)
        table[label] = {"mean": m, "lo": lo, "hi": hi, "p": p, "wins": w, "losses": l,
                        "per_state": vals}
        cis = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "n/a"
        print(f"    {label:<42}{m:>+9.2f}{cis:>20}{p:>9.4f}{f'{w}/{l}':>9}" if p is not None else "")
    print(f"\n    selection quality: sim-batch spread within a state "
          f"{statistics.mean(r['sim_spread'] for r in rows):.2f}; "
          f"confirmation spread {statistics.mean(r['conf_spread'] for r in rows):.2f}")
    print(f"    the selected candidate was the degree_min pick in "
          f"{sum(r['selected_is_degree_min'] for r in rows)}/{len(rows)} states, "
          f"degree_max in {sum(r['selected_is_degree_max'] for r in rows)}/{len(rows)}")
    print(f"    cost: {len(rows)} states x <= {args.candidates} candidates, "
          f"{len(rows) * (args.candidates * 2 + 4)} evaluations, {seconds:.1f} s")

    args.output.write_text(json.dumps(
        {"graph": GRAPH, "state_rule": STATE_RULE, "budgets": vars(args) | {"output": str(args.output)},
         "core_table": table, "rows": rows, "seconds": seconds},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
