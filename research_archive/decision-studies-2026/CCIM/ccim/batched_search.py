"""Batch-of-32 search: rank a batch of candidate swaps, verify them in order, accept the first gain.

Why this shape
--------------
The previous probe compared *whole scan orders* (random / degree / threshold).  That changes the search
globally and makes it hard to say what the ranking contributed.  Here the unit is one batch:

1. draw the next ``batch_size`` candidate swaps from the **same lazy without-replacement scanner** the
   accepted engineering round already uses;
2. rank only those ``batch_size`` swaps;
3. query the *true* ``sigma`` in ranked order, accepting the first **strict** improvement;
4. on acceptance the state changed, so the state context and the scanner are rebuilt and **the rest of
   the batch is discarded**;
5. if the batch contains no improvement, draw the next batch -- never declare a local optimum.

Three rankers, and only the ranking differs
-------------------------------------------
====================================  ==========================================================
``random``                             keep the draw order (the baseline)
``degree``                             sort by ``degree(v) - degree(u)`` descending
``learned``                            sort by the frozen model's predicted swap gain
====================================  ==========================================================

The ranking never decides acceptance.  It only decides who is asked first, and every accepted swap is
still a strict improvement under the verified diffuser.

Paths that do not occur at this budget
--------------------------------------
If a whole state's candidate pool is drawn without any improvement there is no "next batch" left; the
run records ``pool_exhausted_count`` and moves to the next restart.  At ``M = T(n-T) = 269,828``
candidates and a 5,000-query budget this is unreachable.  The counter is reported so the claim can be
checked rather than assumed.

Timing is split into six disjoint buckets.  Bucket 3 of the earlier rounds is split in two, because the
question here is exactly whether feature construction plus inference plus sorting is paid back:
``3a`` candidate id drawing and decoding, ``3b`` state context + features + inference + sort.  Bucket 4
and bucket 5 keep the same meaning as in the accepted engineering round -- ``value()`` itself is *not*
wrapped in the control timer, because the ledger already times the diffusion and the cache inside it;
wrapping both would double count and drive the reconciliation error negative.
"""

from __future__ import annotations

import hashlib
import random
import time

import networkx as nx

from .baselines import Ledger, Result
from .lazy_candidates import candidate_count, decode, make_scanner
from .model import DEFAULT_K, DEFAULT_T
from .swap_features import StateContext, build_context

RANKERS = ("random", "degree", "learned")
BATCH_SIZE = 32


def _order(mode: str, ctx: StateContext | None, pairs, model, device, now) -> tuple[list, float]:
    """Return ``(ordered_pairs, seconds_spent_building_features_and_scoring)``."""
    if mode == "random":
        return list(pairs), 0.0
    t = now()
    if mode == "degree":
        ordered = sorted(pairs, key=lambda p: (-(ctx.degree[p[1]] - ctx.degree[p[0]]), p[0], p[1]))
        return ordered, now() - t
    if mode == "learned":
        import torch
        with torch.no_grad():
            x = torch.tensor([ctx.features(u, v) for (u, v) in pairs],
                             dtype=torch.float32, device=device)
            x = (x - model.mean) / model.std
            pred = model.net(x).squeeze(-1)
        ordered = [pairs[i] for i in torch.argsort(pred, descending=True).tolist()]
        return ordered, now() - t
    raise KeyError(f"unknown ranker {mode!r}; expected one of {RANKERS}")


def batched_search(graph: nx.Graph, ranker="random", T: int = DEFAULT_T, K: int = DEFAULT_K,
                   budget: int = 5_000, seed: int = 0, batch_size: int = BATCH_SIZE,
                   restarts: int = 6, checkpoints: tuple = (100, 300, 1_000, 3_000, 5_000),
                   time_limit: float | None = None, timer=None, collect_logs: bool = True,
                   record_rows: bool = False, trace_hash: bool = False, return_history: bool = False,
                   model=None, device: str = "cpu") -> Result:
    """Run the batched local search.  See the module docstring for the contract.

    ``ranker`` is one of :data:`RANKERS` or a callable ``(ctx, pairs) -> ordered pairs``.
    ``record_rows`` stores ``(state_id, u, v, before, after)`` for every query issued plus each state's
    ``S`` once.  That is the only way the training rows are produced, and it is a logging change --
    never an extra query, never a reordering.
    ``return_history`` attaches the per-query ``(queries, best, distinct, seconds, accepts, restarts)``
    trace, which is what a quality-versus-time curve is read from.
    """
    if callable(ranker):
        mode, custom = "custom", ranker
    elif ranker in RANKERS:
        mode, custom = ranker, None
    else:
        raise KeyError(f"unknown ranker {ranker!r}; expected one of {RANKERS}")

    rng_restart = random.Random(seed * 1_000_003 + 11)
    rng_scan = random.Random(seed * 1_000_003 + 37)
    ledger = Ledger(timer=timer)
    nodes = list(graph.nodes())
    n = graph.number_of_nodes()
    degree = {v: graph.degree(v) for v in nodes}
    now = timer if timer is not None else time.perf_counter

    clock = {"draw_seconds": 0.0, "rank_seconds": 0.0, "control_seconds": 0.0,
             "drawn_max": 0, "pool_exhausted": 0, "candidates_drawn": 0,
             "queries_in_batches": 0}

    t_start = now()

    def past_limit() -> bool:
        return time_limit is not None and (now() - t_start) >= time_limit

    def over_budget() -> bool:
        return ledger.decision_evaluations >= budget or past_limit()

    best_value = -1
    best_seeds: list[int] = []
    best_restart_value = -1
    history: list[tuple] = []
    scan_log: list[dict] = []
    rows: list[dict] = []
    state_sets: dict[int, list[int]] = {}
    state_counter = 0
    restarts_used = 0
    accepts_total = 0
    current_value = 0.0
    digest = hashlib.sha256()
    accept_sequence: list[list[int]] = []

    def value(seeds, u=None, v=None, state_id=None):
        """One true diffusion query.  Not wrapped in the control timer -- see the module docstring."""
        nonlocal best_value
        if trace_hash:
            digest.update(repr(sorted(int(s) for s in seeds)).encode())
        val = ledger.sigma(graph, seeds, K)
        if val > best_value:
            best_value = val
        history.append((ledger.decision_evaluations, best_value,
                        ledger.as_dict()["distinct_queries"],
                        now() - t_start, accepts_total, restarts_used))
        if record_rows and u is not None:
            rows.append({"state_id": state_id, "u": int(u), "v": int(v),
                         "before": float(current_value), "after": float(val)})
        return val

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
        clock["control_seconds"] += now() - t_ctrl
        current_value = float(value(current))
        improved = True
        while improved and not over_budget():
            improved = False
            state_counter += 1
            t_ctx = now()
            S = set(current)
            selected = sorted(current)
            outside = [v for v in nodes if v not in S]
            n_candidates = candidate_count(selected, outside)
            sc = make_scanner("lazy", n_candidates, rng_scan)
            ctx = (build_context(graph, selected, K)
                   if mode in ("degree", "learned", "custom") else None)
            clock["rank_seconds"] += now() - t_ctx
            if record_rows:
                state_sets[state_counter] = list(selected)

            drawn = 0
            ran_out = False
            scanned = 0
            accepted = None
            batch_index = 0
            while not over_budget():
                if drawn >= n_candidates:
                    ran_out = True
                    break
                t_draw = now()
                want = min(batch_size, n_candidates - drawn)
                ids = [sc.next_id() for _ in range(want)]
                pairs = [decode(i, selected, outside) for i in ids]
                drawn += want
                clock["draw_seconds"] += now() - t_draw
                clock["drawn_max"] = max(clock["drawn_max"], want)

                t_rank = now()
                if mode == "custom":
                    pairs = list(custom(ctx, pairs))
                    clock["rank_seconds"] += now() - t_rank
                else:
                    pairs, spent = _order(mode, ctx, pairs, model, device, now)
                    clock["rank_seconds"] += spent

                batch_index += 1
                for (u, v) in pairs:
                    if over_budget():
                        break
                    t_ctrl = now()
                    trial = sorted([x for x in current if x != u] + [v])
                    clock["control_seconds"] += now() - t_ctrl
                    scanned += 1
                    clock["queries_in_batches"] += 1
                    val = value(trial, u, v, state_counter)
                    if val > current_value:
                        t_ctrl = now()
                        accepted = (u, v)
                        current, current_value, improved = trial, val, True
                        accepts_total += 1
                        if trace_hash:
                            accept_sequence.append([int(u), int(v)])
                        clock["control_seconds"] += now() - t_ctrl
                        break
                if accepted is not None:
                    break

            clock["candidates_drawn"] += drawn
            if ran_out:
                clock["pool_exhausted"] += 1
            if collect_logs:
                scan_log.append({
                    "state_id": state_counter, "ranker": mode, "restart": restart_index,
                    "candidates_available": n_candidates, "candidates_drawn": drawn,
                    "batches": batch_index, "candidates_scanned": scanned,
                    "first_improvement_position": scanned if accepted else None,
                    "pool_exhausted_without_improvement": ran_out,
                    "ended_by_budget": bool(accepted is None and not ran_out and over_budget()),
                    "accepted": None if accepted is None else [accepted[0], accepted[1]],
                    "cumulative_queries": ledger.decision_evaluations,
                })
            if improved:
                continue
            if ran_out:
                continue      # no next batch exists; next restart. Counted in pool_exhausted_count.
            if over_budget():
                continue      # the budget, not the search, ended this scan.
            raise RuntimeError("batch loop ended without improvement and without exhausting the pool")

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
            curve.append({"checkpoint": c, "queries": 0, "best_sigma": None, "best_normalized": None,
                          "distinct_queries": 0, "cumulative_seconds": None, "accepts": None,
                          "restarts": None, "reached": False})

    positions = [s["first_improvement_position"] for s in scan_log if s["first_improvement_position"]]
    scan_summary = {
        "scans": len(scan_log),
        "scans_with_improvement": sum(1 for s in scan_log if s["accepted"]),
        "scans_without_improvement": sum(1 for s in scan_log if not s["accepted"]),
        "candidates_scanned_total": sum(s["candidates_scanned"] for s in scan_log),
        "candidates_drawn_total": sum(s["candidates_drawn"] for s in scan_log),
        "first_improvement_positions": positions,
        "mean_first_improvement_position": (sum(positions) / len(positions)) if positions else None,
        "pool_exhausted_count": clock["pool_exhausted"],
        "ended_by_budget_count": sum(1 for s in scan_log if s["ended_by_budget"]),
    }

    led = ledger.timing()
    named = {
        "1_diffusion_seconds": led["diffusion_seconds"],
        "2_cache_and_key_seconds": led["cache_and_key_seconds"],
        "3a_candidate_draw_seconds": clock["draw_seconds"],
        "3b_features_inference_sort_seconds": clock["rank_seconds"],
        "4_search_control_seconds": clock["control_seconds"],
    }
    buckets = dict(named)
    buckets["5_logging_and_unaccounted_seconds"] = max(0.0, total_seconds - sum(named.values()))
    buckets["sum_of_buckets"] = sum(buckets.values())
    buckets["total_seconds"] = total_seconds
    buckets["reconciliation_error_seconds"] = total_seconds - buckets["sum_of_buckets"]
    buckets["diffusion_share_of_total"] = (led["diffusion_seconds"] / total_seconds
                                           if total_seconds > 0 else None)
    buckets["instrumented"] = led["instrumented"]
    buckets["budget_exhausted"] = ledger.decision_evaluations >= budget
    buckets["time_limit_seconds"] = time_limit
    buckets["stopped_by_time_limit"] = bool(time_limit is not None and past_limit()
                                            and ledger.decision_evaluations < budget)
    buckets["queries_issued_in_batches"] = clock["queries_in_batches"]
    buckets["batch_size"] = batch_size
    buckets["candidates_drawn_max_in_one_batch"] = clock["drawn_max"]

    return Result("batched_search", sorted(best_seeds), final, n, ledger.as_dict(),
                  {"ranker": mode, "budget": budget, "seed": seed, "batch_size": batch_size,
                   "budget_used": ledger.decision_evaluations, "checkpoint_curve": curve,
                   "scan_summary": scan_summary, "scan_log": scan_log if collect_logs else [],
                   "timing_buckets": buckets, "accepts": accepts_total,
                   "restarts_used": restarts_used, "scanner": "lazy",
                   "query_sequence_sha256": digest.hexdigest() if trace_hash else None,
                   "accept_sequence": accept_sequence if trace_hash else None,
                   "rows": rows, "state_sets": state_sets,
                   "history": ([list(h) for h in history] if return_history else None)})
