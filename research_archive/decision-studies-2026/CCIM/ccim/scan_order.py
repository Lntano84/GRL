"""Three scan orders for the same local search: does cheap structure improve the real trajectory?

The comparison is deliberately narrow
-------------------------------------
Everything is frozen except **the order in which candidate swaps are examined**: same initialisation,
same restart rule, same acceptance rule (the true ``sigma``), same caching, same tie-break policy, and
the same budget.  The proxy score never decides whether a swap is accepted -- only who gets asked
first.

============================  ==================================================================
``random``                    the original search: a uniformly random scan order
``degree``                    ``degree(v) - degree(u)`` descending, for the swap ``u -> v``
``threshold``                 the one-round threshold proxy below, descending
============================  ==================================================================

The threshold proxy
-------------------
With ``S' = S \\ {u} u {v}``:

    F1(S)     = |S| + |{x not in S : |N(x) cap S| >= K}|
    q(u,v|S)  = F1(S') - F1(S)

``N`` is the same simple undirected neighbourhood the verified diffuser uses.  ``F1`` counts only the
activations that the seeds trigger **in one round**; the full cascade is never run for scoring.  This
proxy is used because it states the CCIM threshold condition explicitly, rather than importing a
degree-discount formula designed for a different diffusion model.

Two things that are NOT done here
---------------------------------
* No candidate is evaluated just to fill in top-k labels.  The scan stops at the first improvement, so
  the candidates behind it were never asked about and the log must not pretend otherwise.
* Old logs are not re-sorted and called a new result.  Accepting a different swap changes the state,
  the restarts and everything after it, so the three arms are run from scratch.

Randomness is split into independent streams (restarts vs scan order) so that spending randomness on
ordering cannot silently change the restart sequence.
"""

from __future__ import annotations

import hashlib
import random
import time

import networkx as nx

from .baselines import Ledger, Result
from .lazy_candidates import candidate_count, decode, make_scanner
from .model import DEFAULT_K, DEFAULT_T

METHODS = ("random", "degree", "threshold")


# ------------------------------------------------------------------------------------ the proxy
def f1_value(graph: nx.Graph, seeds, K: int) -> int:
    """``F1(S) = |S| + |{x not in S : |N(x) cap S| >= K}|`` -- one round, no cascade."""
    S = set(int(s) for s in seeds)
    newly = 0
    for x in graph.nodes():
        if x in S:
            continue
        active = 0
        for nb in graph.neighbors(x):
            if nb in S:
                active += 1
        if active >= K:
            newly += 1
    return len(S) + newly


def swap_scores(graph: nx.Graph, current: list[int], K: int) -> dict:
    """``q(u,v|S)`` for every ``u`` in ``S`` and ``v`` outside it, computed incrementally.

    Only nodes adjacent to ``u`` or ``v`` can change their one-round status, and ``u`` and ``v``
    themselves change membership, so the delta is exact when summed over ``N(u) | N(v) | {u, v}``.
    """
    S = set(current)
    count = {x: 0 for x in graph.nodes()}
    for s in current:
        for x in graph.neighbors(s):
            count[x] += 1

    def counted(x: int, c: dict, members: set) -> bool:
        return (x not in members) and c[x] >= K

    base = sum(1 for x in graph.nodes() if counted(x, count, S))
    neighbour_sets = {v: set(graph.neighbors(v)) for v in graph.nodes()}

    scores: dict[tuple[int, int], int] = {}
    for u in current:
        Nu = neighbour_sets[u]
        for v in graph.nodes():
            if v in S:
                continue
            Nv = neighbour_sets[v]
            affected = Nu | Nv | {u, v}
            Sp = (S - {u}) | {v}
            delta = 0
            for x in affected:
                before = counted(x, count, S)
                new_count = count[x] + (1 if x in Nv else 0) - (1 if x in Nu else 0)
                after = (x not in Sp) and new_count >= K
                delta += (1 if after else 0) - (1 if before else 0)
            scores[(u, v)] = delta
    return scores


# ------------------------------------------------------------------------------------ the search
def ordered_search(graph: nx.Graph, method: str = "random", T: int = DEFAULT_T,
                   K: int = DEFAULT_K, budget: int = 5_000, seed: int = 0, restarts: int = 6,
                   two_swap_passes: int = 200, checkpoints: tuple = (100, 300, 1_000, 3_000, 5_000),
                   collect_logs: bool = True, time_limit: float | None = None,
                   timer=None, scanner: str = "legacy",
                   trace_hash: bool = False) -> Result:
    """Local search whose only manipulated variable is the candidate scan order.

    ``time_limit`` stops the run once that many seconds have elapsed, **keeping the queries already
    spent**; the caller must not top the count up to the query budget.  ``timer`` (``time.perf_counter``
    in the probe) switches on the wall-clock breakdown; passing it changes no result.

    ``scanner``
        ``"legacy"`` builds and shuffles the whole ``T * (n - T)`` pair list up front (the original
        behaviour, kept so earlier artifacts stay reproducible).  ``"lazy"`` and ``"explicit"`` draw
        candidates on demand with the same Fisher-Yates rule -- see :mod:`ccim.lazy_candidates`.  The
        two new modes use **isolated random streams** (scan / 2-swap / restart) so that consuming more
        or fewer scan draws cannot perturb the restart or 2-swap sequences.

    ``trace_hash``
        maintains a running hash of the sequence of evaluated seed sets, plus the accepted-swap
        sequence, so two implementations can be shown to have followed the *same* trajectory rather
        than merely landing on the same value.
    """
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; expected one of {METHODS}")
    if scanner not in ("legacy", "lazy", "explicit"):
        raise KeyError(f"unknown scanner {scanner!r}")

    isolated = scanner in ("lazy", "explicit")
    rng_restart = random.Random(seed * 1_000_003 + 11)   # restarts: independent stream
    rng_order = random.Random(seed * 1_000_003 + 23)     # legacy: shared ordering stream
    rng_scan = random.Random(seed * 1_000_003 + 37) if isolated else rng_order
    rng_twoswap = random.Random(seed * 1_000_003 + 53) if isolated else rng_order
    ledger = Ledger(timer=timer)
    nodes = list(graph.nodes())
    n = graph.number_of_nodes()
    degree = {v: graph.degree(v) for v in nodes}
    now = timer if timer is not None else time.perf_counter

    # ---- timing buckets; all disjoint, checked against the total at the end ------------------
    clock = {"candidate_seconds": 0.0, "control_seconds": 0.0,
             "sparse_entries_max": 0, "drawn_max": 0}
    accepts = {"total": 0}

    def past_limit() -> bool:
        return time_limit is not None and (now() - t_start) >= time_limit

    def over_budget() -> bool:
        return ledger.decision_evaluations >= budget or past_limit()

    t_start = now()
    best_value = -1
    best_seeds: list[int] = []
    best_restart_value = -1
    history: list[tuple] = []      # (queries, best, distinct, seconds, accepts, restarts)
    scan_log: list[dict] = []
    state_counter = 0
    restarts_used = 0
    digest = hashlib.sha256()
    accept_sequence: list[list[int]] = []

    def value(seeds) -> int:
        nonlocal best_value
        if trace_hash:
            digest.update(repr(sorted(int(s) for s in seeds)).encode())
        v = ledger.sigma(graph, seeds, K)
        if v > best_value:
            best_value = v
        history.append((ledger.decision_evaluations, best_value,
                        ledger.as_dict()["distinct_queries"],
                        now() - t_start, accepts["total"], restarts_used))
        return v

    def legacy_pairs(current: list[int]) -> list[tuple[int, int]]:
        S = set(current)
        pairs = [(u, v) for u in current for v in nodes if v not in S]
        if method == "random":
            ordered = pairs[:]
            rng_scan.shuffle(ordered)
            return ordered
        if method == "degree":
            shuffled = pairs[:]
            rng_scan.shuffle(shuffled)
            return sorted(shuffled, key=lambda p: -(degree[p[1]] - degree[p[0]]))
        scores = swap_scores(graph, current, K)
        shuffled = pairs[:]
        rng_scan.shuffle(shuffled)
        return sorted(shuffled, key=lambda p: -scores[p])

    by_degree = sorted(nodes, key=lambda v: (-degree[v], v))
    starts = [tuple(by_degree[:T])]
    for _ in range(restarts):
        starts.append(tuple(sorted(rng_restart.sample(nodes, T))))

    for restart_index, start in enumerate(starts):
        if over_budget():
            break
        restarts_used += 1
        t_ctrl = now()
        current = sorted(start)
        current_value = value(current)
        clock["control_seconds"] += now() - t_ctrl
        improved = True
        while improved and not over_budget():
            improved = False
            state_counter += 1
            t_cand = now()
            if scanner == "legacy":
                ordered = legacy_pairs(current)
                n_candidates = len(ordered)
                iterator = ((p[0], p[1]) for p in ordered)
                clock["drawn_max"] = max(clock["drawn_max"], n_candidates)
            else:
                S = set(current)
                selected = sorted(current)
                outside = [v for v in nodes if v not in S]
                n_candidates = candidate_count(selected, outside)
                sc = make_scanner(scanner, n_candidates, rng_scan)
                iterator = (decode(sc.next_id(), selected, outside)
                            for _ in range(n_candidates))
            score_seconds = now() - t_cand
            clock["candidate_seconds"] += score_seconds
            scanned = 0
            accepted: tuple[int, int] | None = None
            for (u, v) in iterator:
                if over_budget():
                    break
                t_ctrl = now()
                trial = sorted([x for x in current if x != u] + [v])
                clock["control_seconds"] += now() - t_ctrl
                scanned += 1
                val = value(trial)
                if val > current_value:
                    t_ctrl = now()
                    accepted = (u, v)
                    current, current_value, improved = trial, val, True
                    accepts["total"] += 1
                    if trace_hash:
                        accept_sequence.append([int(u), int(v)])
                    clock["control_seconds"] += now() - t_ctrl
                    break
            t_ctrl = now()
            # Guard: the lazy scanner must NOT have materialised the range.  ``moved`` grows only with
            # the positions actually touched, so it stays proportional to what was scanned.
            clock["drawn_max"] = max(clock["drawn_max"], scanned)
            if scanner != "legacy":
                clock["sparse_entries_max"] = max(clock["sparse_entries_max"],
                                                   len(getattr(sc, "moved", {})))
            if collect_logs:
                scan_log.append({
                    "state_id": state_counter, "method": method, "restart": restart_index,
                    "candidates_available": n_candidates, "candidates_scanned": scanned,
                    "first_improvement_position": scanned if accepted else None,
                    "exhausted": accepted is None,
                    "accepted": None if accepted is None else [accepted[0], accepted[1]],
                    "scoring_seconds": score_seconds,
                    "cumulative_queries": ledger.decision_evaluations,
                })
            clock["control_seconds"] += now() - t_ctrl
            if improved:
                continue
            for _ in range(two_swap_passes):
                if over_budget():
                    break
                i, j = rng_twoswap.sample(range(T), 2)
                a, b = rng_twoswap.sample(nodes, 2)
                if a in current or b in current:
                    continue
                trial = sorted([x for x in current if x not in (current[i], current[j])] + [a, b])
                val = value(trial)
                if val > current_value:
                    t_ctrl = now()
                    current, current_value, improved = trial, val, True
                    accepts["total"] += 1
                    if trace_hash:
                        accept_sequence.append([int(a), int(b)])
                    clock["control_seconds"] += now() - t_ctrl
                    break
        t_ctrl = now()
        if current_value > best_restart_value:
            best_restart_value = current_value
            best_seeds = list(current)
        clock["control_seconds"] += now() - t_ctrl
    final = ledger.sigma(graph, best_seeds, K, decision=False)
    total_seconds = now() - t_start

    curve = []
    for c in checkpoints:
        eligible = [h for h in history if h[0] <= c]
        if eligible:
            q, best, distinct, seconds, acc, rst = eligible[-1]
            curve.append({"checkpoint": c, "queries": q, "best_sigma": best,
                          "best_normalized": best / n, "distinct_queries": distinct,
                          "cumulative_seconds": seconds, "accepts": acc, "restarts": rst,
                          "reached": True})
        else:
            curve.append({"checkpoint": c, "queries": 0, "best_sigma": None,
                          "best_normalized": None, "distinct_queries": 0,
                          "cumulative_seconds": None, "accepts": None, "restarts": None,
                          "reached": False})

    scan_summary = {
        "scans": len(scan_log),
        "scans_exhausted": sum(1 for s in scan_log if s["exhausted"]),
        "scans_improved": sum(1 for s in scan_log if not s["exhausted"]),
        "candidates_scanned_total": sum(s["candidates_scanned"] for s in scan_log),
        "first_improvement_positions": [s["first_improvement_position"]
                                        for s in scan_log if s["first_improvement_position"]],
        "scoring_seconds_total": sum(s["scoring_seconds"] for s in scan_log),
    }

    # ---- five disjoint buckets, reconciled against the total ---------------------------------
    led = ledger.timing()
    buckets = {
        "1_diffusion_seconds": led["diffusion_seconds"],
        "2_cache_and_key_seconds": led["cache_and_key_seconds"],
        "3_candidate_generation_seconds": clock["candidate_seconds"],
        "4_search_control_seconds": clock["control_seconds"],
    }
    buckets["5_logging_and_unaccounted_seconds"] = max(
        0.0, total_seconds - sum(buckets.values()))
    buckets["sum_of_buckets"] = sum(v for k, v in buckets.items()
                                    if k.startswith(("1_", "2_", "3_", "4_", "5_")))
    buckets["total_seconds"] = total_seconds
    buckets["reconciliation_error_seconds"] = total_seconds - buckets["sum_of_buckets"]
    buckets["diffusion_share_of_total"] = (led["diffusion_seconds"] / total_seconds
                                           if total_seconds > 0 else None)
    buckets["instrumented"] = led["instrumented"]
    buckets["budget_exhausted"] = ledger.decision_evaluations >= budget
    buckets["time_limit_seconds"] = time_limit
    buckets["stopped_by_time_limit"] = bool(time_limit is not None and past_limit()
                                            and ledger.decision_evaluations < budget)
    buckets["sparse_entries_max"] = clock["sparse_entries_max"]
    buckets["candidates_drawn_max_in_one_scan"] = clock["drawn_max"]

    return Result("ordered_search", sorted(best_seeds), final, n, ledger.as_dict(),
                  {"method": method, "budget": budget, "seed": seed,
                   "budget_used": ledger.decision_evaluations,
                   "checkpoint_curve": curve, "scan_summary": scan_summary,
                   "scan_log": scan_log if collect_logs else [],
                   "timing_buckets": buckets, "accepts": accepts["total"],
                   "restarts_used": restarts_used,
                   "scanner": scanner,
                   "query_sequence_sha256": digest.hexdigest() if trace_hash else None,
                   "accept_sequence": accept_sequence if trace_hash else None})
