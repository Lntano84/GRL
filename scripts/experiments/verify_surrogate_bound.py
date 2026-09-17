"""Is lambda(S) = sigma_kappa(S) - sigma_tau(S) really an upper bound on sigma(S)?

Why this needs its own script
-----------------------------
The source model (Inf. Sci. 744 (2026) 123375, Sec. 5.2) defines

    sigma^kappa(.)   the LT objective with the LOWER threshold theta^kappa
    sigma^tau(.)     the LT objective with the UPPER threshold theta^tau
    lambda(.)      = sigma^kappa(.) - sigma^tau(.)
    Theorem 6        lambda(S) >= sigma(S)   for all S

and proves it on a single realisation: for one fixed window draw, the positively activated set
satisfies  |A| <= |A^kappa| - |A^tau|  (their Fig. 5(b), where A^kappa = {a,c,b,e} and A^tau = {}).

But the stated theorem is about the *expected* objective.  A per-realisation containment
|A| <= |A^kappa| - |A^tau| does not by itself give
E[|A|] <= E[|A^kappa|] - E[|A^tau|], because the two LT processes are evaluated at different
threshold vectors and the difference of two monotone set functions need not dominate the window
process in expectation.  When we measured the gap directly on four graphs, lambda < sigma in 6 of
12 non-trivial settings (down to -0.958 relative on NetHEPT), which would contradict Theorem 6 as
stated.

This script separates the two claims so the discrepancy can be attributed:

``per-realisation``   For each window draw, does A  subseteq  A^kappa \\ A^tau hold?  And does
                      |A| <= |A^kappa| - |A^tau| hold draw by draw?
``in-expectation``    Does E[|A|] <= E[|A^kappa|] - E[|A^tau|] hold?

If the first holds and the second fails, the theorem's *proof technique* establishes the
per-realisation statement while the *statement* is about expectations, and the gap measurement is
correct as reported.

Everything uses the SAME window draw for all three quantities within a trial, so the comparison is
paired.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_density_degree_signflip import GRAPHS, load as load_graph  # noqa: E402
from grl.diffusion.overexposure import run_overexposure, sample_threshold_windows  # noqa: E402


def lt_active_set(
    graph: nx.DiGraph, seeds: list[int], thresholds: dict[int, float]
) -> set[int]:
    """Active set of a fixed-threshold linear-threshold cascade (strict inequality rule)."""
    active: set[int] = set(seeds)
    weighted_in: dict[int, float] = {v: 0.0 for v in graph.nodes()}
    frontier = list(seeds)
    while frontier:
        newly: list[int] = []
        for node in frontier:
            for target in graph.successors(node):
                if target in active:
                    continue
                weighted_in[target] += float(graph[node][target].get("weight", 0.0))
                if weighted_in[target] > thresholds[target]:
                    active.add(target)
                    newly.append(target)
        frontier = newly
    return active


@dataclass
class Draw:
    graph: str
    seed_size: int
    seed_fraction: float
    realization: int
    a_window: int
    a_kappa: int
    a_tau: int
    subset_ok: bool
    per_realisation_ok: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+",
                        default=["congress_twitter", "email_eu_core", "ca_grqc", "nethept"],
                        choices=list(GRAPHS))
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.05, 0.10, 0.20])
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    draws: list[Draw] = []
    summary: list[dict] = []

    for label in args.graphs:
        graph = load_graph(label)
        nodes = list(graph.nodes())
        n = len(nodes)
        print(f"=== {label}: n={n} <k>={2*graph.number_of_edges()/n:.2f}")
        for fraction in args.fractions:
            size = int(round(fraction * n))
            rng = random.Random(args.random_seed + 13 * size)
            seeds = rng.sample(nodes, size) if size else []

            win_vals: list[int] = []
            kap_vals: list[int] = []
            tau_vals: list[int] = []
            subset_fail = 0
            per_real_fail = 0

            for t in range(args.trials):
                r = random.Random(args.random_seed + 13 * size + 11 + t)
                windows = sample_threshold_windows(nodes, r)
                a_win = set(run_overexposure(graph, list(seeds), windows, r).positive)
                a_kap = lt_active_set(graph, list(seeds), {v: windows[v][0] for v in nodes})
                a_tau = lt_active_set(graph, list(seeds), {v: windows[v][1] for v in nodes})

                subset_ok = a_win <= (a_kap - a_tau)
                per_real_ok = len(a_win) <= len(a_kap) - len(a_tau)
                if not subset_ok:
                    subset_fail += 1
                if not per_real_ok:
                    per_real_fail += 1

                win_vals.append(len(a_win))
                kap_vals.append(len(a_kap))
                tau_vals.append(len(a_tau))
                draws.append(Draw(label, size, size / n, t, len(a_win), len(a_kap),
                                  len(a_tau), subset_ok, per_real_ok))

            mean_win = statistics.fmean(win_vals)
            mean_kap = statistics.fmean(kap_vals)
            mean_tau = statistics.fmean(tau_vals)
            lam = mean_kap - mean_tau
            summary.append({
                "graph": label,
                "seed_size": size,
                "seed_fraction": size / n,
                "E_A_window": mean_win,
                "E_A_kappa": mean_kap,
                "E_A_tau": mean_tau,
                "lambda": lam,
                "gap": lam - mean_win,
                "subset_violations": subset_fail,
                "per_realisation_violations": per_real_fail,
                "trials": args.trials,
            })
            print(f"    |S|/n={size/n*100:>5.1f}%  E|A|={mean_win:>9.2f}  "
                  f"E|A^kap|={mean_kap:>9.2f}  E|A^tau|={mean_tau:>9.2f}  "
                  f"lambda={lam:>9.2f}  gap={lam-mean_win:>+10.2f}  "
                  f"subset_viol={subset_fail}/{args.trials}  "
                  f"perreal_viol={per_real_fail}/{args.trials}", flush=True)
        print()

    print("=" * 108)
    print("VERDICT")
    print("=" * 108)
    tot = sum(s["trials"] for s in summary)
    sub = sum(s["subset_violations"] for s in summary)
    per = sum(s["per_realisation_violations"] for s in summary)
    neg = [s for s in summary if s["gap"] < 0]
    print(f"  A_window subset of A_kappa \\ A_tau holds in {tot - sub}/{tot} draws")
    print(f"  |A| <= |A^kappa| - |A^tau| holds in {tot - per}/{tot} draws")
    print(f"  lambda < sigma in expectation in {len(neg)}/{len(summary)} settings")
    print()
    if sub == 0 and per == 0 and neg:
        print("  -> The PER-REALISATION claim holds without exception, but the IN-EXPECTATION")
        print("     claim fails on the same data.  The two are not equivalent: a draw-wise")
        print("     containment does not imply the corresponding expectation inequality when the")
        print("     two LT processes are evaluated at different threshold vectors.")
        print("     Report the gap as measured; do NOT report a counterexample to Theorem 6")
        print("     without checking the source's intended reading of sigma^kappa / sigma^tau.")
    elif sub or per:
        print("  -> The per-realisation claim itself fails; check the LT implementation and the")
        print("     strict-inequality rule before drawing any conclusion.")
    else:
        print("  -> Both claims hold in every setting; the earlier negative gaps were sampling")
        print("     noise.  Re-run with more trials before concluding anything.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graphs": args.graphs,
            "fractions": args.fractions,
            "trials": args.trials,
            "summary": summary,
            "draws": [asdict(d) for d in draws],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
