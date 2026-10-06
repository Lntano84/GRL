"""
Stage 06 neighbourhood construction: the eight frozen configurations.

Every configuration changes ONLY the short-term Y release set. The far-horizon Y and every Z
stay free in all eight, exactly as in stage 04b/05.

    FULL            release every legal short-term Y
    EMPTY           release none
    SHORTAGE-24     the historical shortage ranking, kept as a control
    WINDOW-24       recovery-window priority, then shortage score, then index
    DEPENDENCY-24   shortage seeds expanded along capacity competition and adjacent periods
    MATCHED-RANDOM  three sets stratified to match DEPENDENCY-24's
                    (machine, period, repair Y_r in {0,1}) profile

The two new rules are frozen here and are NOT tuned. Neither is claimed to be a fully tuned
dynamic-window algorithm or a complete dependency closure.

Eligibility rule (unchanged from stage 03/04/04b/05): a candidate (i, j, t) is legal when
w_i,j != 0 and s_i <= c'_jt. Candidates ruled out by compatibility or capacity are never
selected by any rule.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Set, Tuple

from lsp_model import Instance, Solution

YIndex = Tuple[int, int, int]

CONFIGS = ("FULL", "EMPTY", "SHORTAGE-24", "WINDOW-24", "DEPENDENCY-24",
           "MATCHED-RANDOM-1", "MATCHED-RANDOM-2", "MATCHED-RANDOM-3")
RANDOM_SEEDS = {"MATCHED-RANDOM-1": 6101, "MATCHED-RANDOM-2": 6102,
                "MATCHED-RANDOM-3": 6103}


def legal_candidates(inst: Instance, tau: int) -> List[YIndex]:
    out = []
    for i in range(inst.N):
        for j in range(inst.M):
            if inst.w[i][j] == 0:
                continue
            for t in range(1, tau + 1):
                if inst.s[i] <= inst.c[j][t - 1] + 1e-9:
                    out.append((i, j, t))
    return out


def short_term_slots(inst: Instance, tau: int) -> List[YIndex]:
    return [(i, j, t) for i in range(inst.N) for j in range(inst.M)
            for t in range(1, tau + 1)]


def shortage_scores(inst: Instance, repair: Solution, tau: int) -> Dict[YIndex, float]:
    """a(i,j,t) = l_i * sum_{r>=t} L^r_ir / (s_i + b_i m_i), from the repaired plan only."""
    scores: Dict[YIndex, float] = {}
    for (i, j, t) in short_term_slots(inst, tau):
        shortage = sum(float(repair.L[i, r - 1]) for r in range(t, inst.T + 1))
        scores[(i, j, t)] = (inst.l[i] * shortage) / (inst.s[i] + inst.b[i] * inst.m[i])
    return scores


def recovery_period(dis, j: int, T: int) -> int:
    """delta + 1 when machine j is disrupted and delta < T; otherwise T + 1 (never)."""
    delta = dis.delta_for(j)
    if delta == 0 or delta >= T:
        return T + 1
    return delta + 1


def rank_shortage(inst, repair, tau, legal) -> List[YIndex]:
    sc = shortage_scores(inst, repair, tau)
    return sorted(legal, key=lambda u: (-sc[u], u))


def rank_window(inst, repair, tau, legal, dis) -> List[YIndex]:
    """WINDOW-24: distance to the machine's recovery period first, then shortage, then index."""
    sc = shortage_scores(inst, repair, tau)
    rec = {j: recovery_period(dis, j, inst.T) for j in range(inst.M)}

    def key(u: YIndex):
        i, j, t = u
        dist = abs(t - rec[j])
        return (dist, -sc[u], u)

    return sorted(legal, key=key)


def build_dependency_24(inst, repair, tau, legal, dis) -> Tuple[List[YIndex], Dict[str, Any]]:
    """DEPENDENCY-24, exactly as frozen:

    1. take the top 8 by shortage score as seeds;
    2. expand round-robin, each round adding at most ONE new candidate per seed;
    3. expansion priority:
         a. same machine, same period, other products that already occupy a setup on that
            machine in the repaired plan;
         b. same product, same machine, adjacent periods;
    4. stop at 24; if expansion cannot continue, fill by shortage rank.
    """
    legal_set = set(legal)
    order = rank_shortage(inst, repair, tau, legal)
    seeds = order[:8]
    chosen: List[YIndex] = list(seeds)
    chosen_set: Set[YIndex] = set(seeds)
    log: Dict[str, Any] = {"seeds": [list(s) for s in seeds], "added": [],
                           "filled_by_shortage": []}

    def occupied_setup(i: int, j: int, t: int) -> bool:
        return float(repair.Y[i, j, t - 1]) > 0.5

    def neighbours_a(seed: YIndex) -> List[YIndex]:
        i, j, t = seed
        cand = [u for u in legal
                if u[1] == j and u[2] == t and u[0] != i and occupied_setup(u[0], j, t)]
        return sorted(cand, key=lambda u: (-shortage_scores(inst, repair, tau)[u], u))

    def neighbours_b(seed: YIndex) -> List[YIndex]:
        i, j, t = seed
        cand = [u for u in legal
                if u[0] == i and u[1] == j and u[2] != t and u not in chosen_set]
        return sorted(cand, key=lambda u: (abs(u[2] - t), u))

    target = 24
    while len(chosen) < target:
        progressed = False
        for seed in list(seeds):
            if len(chosen) >= target:
                break
            for kind, nf in (("a", neighbours_a), ("b", neighbours_b)):
                added = None
                for u in nf(seed):
                    if u not in chosen_set and u in legal_set:
                        added = u
                        break
                if added is not None:
                    chosen.append(added)
                    chosen_set.add(added)
                    log["added"].append({"seed": list(seed), "kind": kind,
                                         "added": list(added)})
                    progressed = True
                    break
        if not progressed:
            break

    if len(chosen) < target:
        for u in order:
            if len(chosen) >= target:
                break
            if u not in chosen_set:
                chosen.append(u)
                chosen_set.add(u)
                log["filled_by_shortage"].append(list(u))
    log["n_dependency_added"] = len(log["added"])
    log["n_shortage_filled"] = len(log["filled_by_shortage"])
    return chosen[:target], log


def match_random(inst, repair, tau, legal, reference: Sequence[YIndex],
                 seed: int) -> Tuple[List[YIndex], Dict[str, Any]]:
    """MATCHED-RANDOM: keep the (machine, period, repair-value) profile of `reference`
    stratum by stratum and resample the PRODUCT inside each stratum from the legal pool."""
    rng = random.Random(seed)
    ref_set = list(reference)
    strata: Dict[Tuple[int, int, float], List[YIndex]] = {}
    for (i, j, t) in ref_set:
        key = (j, t, float(repair.Y[i, j, t - 1]))
        strata.setdefault(key, []).append((i, j, t))

    out: List[YIndex] = []
    taken: Set[YIndex] = set()
    swaps = 0
    for key, members in strata.items():
        j, t, yval = key
        pool = [u for u in legal
                if u[1] == j and u[2] == t
                and abs(float(repair.Y[u[0], u[1], u[2] - 1]) - yval) < 0.5
                and u not in taken]
        if len(pool) < len(members):
            # stratum too small to resample: keep whatever is available
            pool_names = {u for u in pool}
            for u in members:
                if u in pool_names:
                    out.append(u)
                    taken.add(u)
            continue
        picks = rng.sample(pool, len(members))
        for p in picks:
            out.append(p)
            taken.add(p)
        if set(picks) != set(members):
            swaps += 1

    # top up if strata could not cover the target
    if len(out) < len(ref_set):
        sc = shortage_scores(inst, repair, tau)
        for u in sorted(legal, key=lambda u: (-sc[u], u)):
            if len(out) >= len(ref_set):
                break
            if u not in taken:
                out.append(u)
                taken.add(u)

    overlap = len(set(out) & set(ref_set))
    info = {
        "n_strata": len(strata),
        "n_strata_resampled": swaps,
        "overlap_with_dependency": overlap,
        "overlap_fraction": (overlap / len(ref_set)) if ref_set else None,
        "profile_preserved": _profile(out, repair) == _profile(ref_set, repair),
    }
    return out[:len(ref_set)], info


def _profile(s: Sequence[YIndex], repair: Solution) -> Dict[Any, int]:
    from collections import Counter
    return dict(Counter((j, t, float(repair.Y[i, j, t - 1])) for (i, j, t) in s))


def build_release_sets(inst: Instance, repair: Solution, tau: int, dis,
                       size: int = 24) -> Dict[str, Any]:
    """Return every configuration's release set plus diagnostics."""
    legal = legal_candidates(inst, tau)
    slots = short_term_slots(inst, tau)
    out: Dict[str, Any] = {"legal": legal, "slots": slots,
                           "n_legal": len(legal), "n_slots": len(slots)}

    out["FULL"] = {"release": list(legal), "note": "all legal short-term Y"}
    out["EMPTY"] = {"release": [], "note": "none"}

    s_order = rank_shortage(inst, repair, tau, legal)
    out["SHORTAGE-24"] = {"release": s_order[:size],
                          "note": "top %d by shortage score" % size}

    w_order = rank_window(inst, repair, tau, legal, dis)
    out["WINDOW-24"] = {"release": w_order[:size],
                        "note": "recovery-window distance, then shortage, then index"}

    dep, dep_log = build_dependency_24(inst, repair, tau, legal, dis)
    out["DEPENDENCY-24"] = {"release": dep, "note": "shortage seeds + dependency expansion",
                            "log": dep_log}

    for name, seed in RANDOM_SEEDS.items():
        rs, info = match_random(inst, repair, tau, legal, dep, seed)
        out[name] = {"release": rs, "note": "matched random, seed %d" % seed, "match": info}
    return out
