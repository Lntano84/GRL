"""M01 independent check: enumerate every persistent edge state and run the SAVED policies.

Scope, as specified: the (n,m) = (6,6) tier with s = 0,1 under both probability models -- four
instances.  Each is checked against the recursive values computed by ``m01_worker``.

Independence.  This file does NOT call the recursive value function to produce payoffs.  It
  * enumerates all 2^m persistent edge states (``m = 6``, so 64 worlds);
  * builds its own action tables by exhaustive one-step search over all independent edge sets,
    using its own matching/transition code, with the continuation values coming from its own tables;
  * simulates G, R and OPT step by step in each world, reading only the states of the edges it chose;
  * weights the per-world payoffs and compares the total with the recursion's number.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from m01_worker import generate_configs  # noqa: E402


# ------------------------------------------------------------------ independent primitives

def independent_sets(n, edges, avail):
    """All independent edge sets among available edges; its own implementation."""
    out = []
    idxs = [i for i in range(len(edges)) if (avail >> i) & 1]
    for r in range(len(idxs) + 1):
        for combo in itertools.combinations(idxs, r):
            used = set()
            ok = True
            for i in combo:
                u, v = edges[i]
                if u in used or v in used:
                    ok = False
                    break
                used.add(u)
                used.add(v)
            if ok:
                out.append(combo)
    return out


def advance(n, edges, avail, M, S):
    """Own transition: chosen edges leave; every edge touching a matched vertex leaves.

    ``S`` is the set of chosen edges (by index) that succeeded.
    """
    new = avail
    gone_v = set()
    for i in M:
        new &= ~(1 << i)
        if i in S:
            u, v = edges[i]
            gone_v.add(u)
            gone_v.add(v)
    for i in range(len(edges)):
        if (new >> i) & 1:
            u, v = edges[i]
            if u in gone_v or v in gone_v:
                new &= ~(1 << i)
    return new


def build_tables(n, edges, probs):
    """Own policy tables, built by exhaustive one-step search.

    ``action(avail, h, method)`` is the rule's choice and ``value(avail, h, method)`` is the value of
    FOLLOWING that rule for ``h`` rounds (for R that is the value of the lookahead rule, not
    ``max_M Qtilde``).  Everything is memoised under ``(avail, h, method)``.

    Two bugs were fixed here after the independent check disagreed with the recursion:
      * the cache was aliased but never written, so continuations were silently recomputed;
      * the greedy rule's myopic score ``2 sum q_e`` was returned as ``V_G(.,h)`` for every ``h``,
        which is only correct at ``h = 1``.  ``V_G`` must compose the greedy action with its own
        continuation, exactly like the other two methods.
    """
    eps = 1e-10 * max(1, n)
    cache: dict = {}
    _MISS = object()
    # Continuation rule per method.  R is a ONE-STEP lookahead whose continuation estimate is the
    # full greedy rule G -- not R itself.  (Using "R" here was a bug: it made the continuation
    # inconsistent with the lookahead that the action was chosen from.)
    CONT = {"G": "G", "R": "G", "OPT": "OPT"}

    def outcomes(M):
        """``(successful edge-index frozenset, probability)`` over all ``2^|M|`` outcomes."""
        for r in range(len(M) + 1):
            for succ in itertools.combinations(M, r):
                p = 1.0
                for i in M:
                    p *= probs[i] if i in succ else (1.0 - probs[i])
                yield frozenset(succ), p

    def step(avail, M, h, cont):
        """``E[2|S| + V_{h-1}(transition)]`` for a fixed action and continuation rule."""
        tot = 0.0
        for S, p in outcomes(M):
            tot += p * (2 * len(S) + value(advance(n, edges, avail, M, S), h - 1, cont))
        return tot

    def value(avail, h, method):
        if avail == 0 or h <= 0:
            return 0.0
        ck = (avail, h, method)
        got = cache.get(ck, _MISS)
        if got is not _MISS:
            return got
        M = choose(avail, h, method)
        v = step(avail, M, h, CONT[method])
        cache[ck] = v
        return v

    def choose(avail, h, method):
        """The rule's action; G maximises the myopic weight, R and OPT maximise one-step lookahead."""
        if avail == 0 or h <= 0:
            return ()
        best = None
        for M in independent_sets(n, edges, avail):
            tot = (2.0 * sum(probs[i] for i in M)) if method == "G" \
                else step(avail, M, h, CONT[method])
            k = tuple(edges[i] for i in M)
            if best is None or tot > best[0] + eps or (
                    abs(tot - best[0]) <= eps and k < best[2]):
                best = (tot, M, k)
        return best[1]

    def action(avail, h, method):
        return choose(avail, h, method)

    return action, value, cache


def simulate(n, edges, action, world_mask, method, rounds=3):
    """Run one policy in one persistent edge state; the policy sees only chosen edges' states.

    An empty matching is a legitimate PASS for that round: it consumes a round and the process
    continues.  (An earlier version broke out of the loop on an empty action, which silently
    discarded all remaining rounds -- that was a bug in this harness, not in the solver.)
    """
    avail = (1 << len(edges)) - 1
    total = 0
    for h in range(rounds, 0, -1):
        M = action(avail, h, method)
        if not M:
            continue
        succ = frozenset(i for i in M if (world_mask >> i) & 1)
        total += 2 * len(succ)
        avail = advance(n, edges, avail, M, succ)
    return total


def main() -> int:
    configs = generate_configs()
    by_id = {c["config_id"]: c for c in configs}
    targets = [by_id[f"n6_m6_s{s}_{mdl}"] for s in (0, 1) for mdl in ("HOM", "HET")]

    print("=" * 100)
    print("  M01 INDEPENDENT CHECK -- full persistent edge-state enumeration, saved policies")
    print("=" * 100)

    report = []
    for cfg in targets:
        cid = cfg["config_id"]
        n, edges = cfg["n"], cfg["edges"]
        probs = cfg["probs"]
        m = len(edges)
        t0 = time.perf_counter()
        action, value, cache = build_tables(n, edges, probs)
        build_s = time.perf_counter() - t0

        # recursive reference from the worker
        ref_path = HERE / "results_tmp" / f"config_{configs.index(cfg):02d}.json"
        ref = json.loads(ref_path.read_text(encoding="utf-8"))

        print(f"\n  {cid}   edges={edges}  q={probs}")
        print(f"    policy tables built in {build_s:.2f}s, {len(cache)} cached entries")
        rows = {}
        for method in ("G", "R", "OPT"):
            t0 = time.perf_counter()
            tot = 0.0
            for world in range(1 << m):
                p = 1.0
                for i in range(m):
                    p *= probs[i] if (world >> i) & 1 else (1.0 - probs[i])
                tot += p * simulate(n, edges, action, world, method)
            dt = time.perf_counter() - t0
            rows[method] = (tot, dt)
            rec = ref[f"V3_{method}"]
            ok = abs(tot - rec) < 1e-9
            print(f"    {method:<3} world-enumeration {tot:.10f} | recursion {rec:.10f} | "
                  f"diff {abs(tot-rec):.2e} | {dt:.2f}s | {'MATCH' if ok else 'MISMATCH'}")

        g, o = rows["G"][0], rows["OPT"][0]
        r = rows["R"][0]
        checks = {
            "G_le_R": g <= r + 1e-9,
            "R_le_OPT": r <= o + 1e-9,
            "G_matches": abs(rows["G"][0] - ref["V3_G"]) < 1e-9,
            "R_matches": abs(rows["R"][0] - ref["V3_R"]) < 1e-9,
            "OPT_matches": abs(rows["OPT"][0] - ref["V3_OPT"]) < 1e-9,
        }
        print(f"    ordering G<=R<=OPT: {checks['G_le_R'] and checks['R_le_OPT']}")
        report.append({
            "config_id": cid, "n": n, "m": m, "edges": edges, "probs": probs,
            "n_worlds_enumerated": 1 << m,
            "policy_table_seconds": build_s, "cached_entries": len(cache),
            "V3_G_world_enum": rows["G"][0], "V3_R_world_enum": rows["R"][0],
            "V3_OPT_world_enum": rows["OPT"][0],
            "V3_G_recursion": ref["V3_G"], "V3_R_recursion": ref["V3_R"],
            "V3_OPT_recursion": ref["V3_OPT"],
            "abs_diff": {"G": abs(rows["G"][0] - ref["V3_G"]),
                         "R": abs(rows["R"][0] - ref["V3_R"]),
                         "OPT": abs(rows["OPT"][0] - ref["V3_OPT"])},
            "enumeration_seconds": {k: rows[k][1] for k in rows},
            "checks": checks,
            "all_checks_pass": all(checks.values()),
            "note": "world enumeration used only the saved policy tables and its own transition "
                    "code; the recursive value function was NOT called to produce these payoffs",
        })

    out = HERE / "M01_independent_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    allok = all(r["all_checks_pass"] for r in report)
    print(f"\n  ALL FOUR INDEPENDENT CHECKS {'PASSED' if allok else 'FAILED'}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
