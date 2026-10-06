"""Supplementary probe: ranking reversal when the two sides of the comparison are ALLOWED to couple.

Why this probe exists
---------------------
The frozen N01 condition requires ``(R2(a) u R2(b)) n (R2(c) u R2(d)) = empty``.  That condition makes
the reversal test vacuous: it forces ``R(a) n R(d) = empty`` and ``R(b) n R(c) = empty``, and therefore

    X_c = |R(a) u R(c)| - |R(b) u R(c)| = |R(a)| - |R(b)| = |R(a) u R(d)| - |R(b) u R(d)| = X_d

in EVERY live-edge world, for real diffusion as well as for the proxy.  So the frozen condition cannot
produce a reversal, and the main run measures a null by construction.

This probe keeps exactly the two constraints that make the comparison meaningful --
``R2(a) n R2(b) = empty`` and ``R2(c) n R2(d) = empty``, i.e. the two nodes being compared never
influence each other's region -- and DROPS the cross-side exclusion, so that ``a`` may couple to ``d``
and ``b`` to ``c``.  Then

    X_c - X_d = (|R(a) n R(d)| - |R(a) n R(c)|) - (|R(b) n R(d)| - |R(b) n R(c)|)

can be non-zero, and the proxy -- which must output ``delta_hat_c == delta_hat_d`` -- may genuinely
fail to express the true ordering.

This is a POST-HOC supplementary diagnostic, generated after the frozen run was specified.  It is
reported separately from the frozen results and never replaces them.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from n01_common import K_SEEDS, Graph, parse_ca_grqc, within_hops  # noqa: E402

SAMPLING_SEED = 20261013          # distinct from the frozen 20261012
N_PER_GROUP = 20
MAX_ATTEMPTS = 200_000


def main() -> int:
    arcs, undirected, node_ids, manifest = parse_ca_grqc()
    graph = Graph(arcs, undirected, node_ids)
    lcc = graph.components()[0]
    print(f"  n={graph.n} m={graph.m} lcc={len(lcc)}")

    r2 = {v: within_hops(graph, v, 2) for v in range(graph.n)}
    rng = random.Random(SAMPLING_SEED)
    degree_ranked = graph.degree_ranking()

    seen: set = set()
    groups = {"uniform_U": [], "degree_U": []}
    attempts = 0
    rej_same_side = 0
    rej_cross = 0

    while attempts < MAX_ATTEMPTS and any(len(groups[g]) < N_PER_GROUP for g in groups):
        attempts += 1
        a, b, c, d = rng.sample(lcc, 4)
        quad = tuple(sorted((a, b, c, d)))
        if quad in seen:
            continue
        # RELAXED condition: same-side 2-hop regions disjoint, cross-side coupling allowed.
        if (r2[a] & r2[b]) or (r2[c] & r2[d]):
            rej_same_side += 1
            continue
        if not ((r2[a] & r2[d]) or (r2[b] & r2[c])):
            rej_cross += 1          # we WANT cross coupling, so this is a rejection here
            continue
        seen.add(quad)
        for group in ("uniform_U", "degree_U"):
            if len(groups[group]) >= N_PER_GROUP:
                continue
            if group == "uniform_U":
                pool = [v for v in lcc if v not in quad]
                u = sorted(rng.sample(pool, K_SEEDS - 2))
            else:
                u = sorted([v for v in degree_ranked if v not in quad][:K_SEEDS - 2])
            groups[group].append({
                "quadruple": [a, b, c, d], "U": u,
                "same_side_disjoint": {"r2_a_and_r2_b": len(r2[a] & r2[b]),
                                       "r2_c_and_r2_d": len(r2[c] & r2[d])},
                "cross_coupling_deterministic": {
                    "r2_a_and_r2_d": len(r2[a] & r2[d]),
                    "r2_b_and_r2_c": len(r2[b] & r2[c])},
            })
            break

    cases = []
    for group, rows in groups.items():
        for row in rows:
            a, b, c, d = row["quadruple"]
            u = row["U"]
            cases.append({
                "group": group, "a": a, "b": b, "c": c, "d": d, "U": u,
                "S_ac": sorted([*u, a, c]), "S_bc": sorted([*u, b, c]),
                "S_ad": sorted([*u, a, d]), "S_bd": sorted([*u, b, d]),
                "degrees": {"a": graph.degree[a], "b": graph.degree[b],
                            "c": graph.degree[c], "d": graph.degree[d],
                            "U": [graph.degree[v] for v in u]},
                "struct": {
                    "same_side_disjoint": True,
                    "r2_a_and_r2_b": row["same_side_disjoint"]["r2_a_and_r2_b"],
                    "r2_c_and_r2_d": row["same_side_disjoint"]["r2_c_and_r2_d"],
                    "cross_r2_a_and_r2_d": row["cross_coupling_deterministic"]["r2_a_and_r2_d"],
                    "cross_r2_b_and_r2_c": row["cross_coupling_deterministic"]["r2_b_and_r2_c"],
                    "frozen_condition_holds": False,
                },
            })

    # sanity: exact seed-set sizes
    problems = []
    for i, cs in enumerate(cases):
        cs["index"] = i
        for key in ("S_ac", "S_bc", "S_ad", "S_bd"):
            if len(cs[key]) != K_SEEDS or len(set(cs[key])) != K_SEEDS:
                problems.append(f"case {i}: {key} bad size/uniqueness")
        if set(cs["U"]) & {cs["a"], cs["b"], cs["c"], cs["d"]}:
            problems.append(f"case {i}: candidate inside U")

    print(f"  attempts {attempts}, accepted {len(seen)}, rejected same-side {rej_same_side}, "
          f"rejected no-cross-coupling {rej_cross}")
    print(f"  produced { {g: len(v) for g, v in groups.items()} }, problems {len(problems)}")
    cad = [cs["degrees"][x] for cs in cases for x in "abcd"]
    print(f"  candidate degrees: min {min(cad)}, mean {sum(cad)/len(cad):.2f}, max {max(cad)}")

    art = {
        "task": "N01-S probe -- reversal under a relaxed condition that PERMITS cross-side coupling",
        "post_hoc": True,
        "supersedes_frozen_run": False,
        "graph": {"sha256": manifest["sha256"], "n": graph.n, "m": graph.m,
                  "node_mapping": "ascending original id -> contiguous index"},
        "relaxed_condition": {
            "kept": ["R2(a) n R2(b) = empty", "R2(c) n R2(d) = empty"],
            "dropped": ["(R2(a) u R2(b)) n (R2(c) u R2(d)) = empty"],
            "rationale": "the dropped clause forces X_c == X_d identically and makes the test vacuous",
        },
        "protocol": {
            "propagation_model": "independent cascade (live-edge view)",
            "arc_probability": "p_{u,v} = 1/deg(v)",
            "rounds_cap": 100, "seed_budget_k": K_SEEDS,
            "activation_count_includes_seeds": True,
            "common_random_numbers": "one live/dead state per directed arc per world, shared by all "
                                     "four sets",
        },
        "generation": {"sampling_seed": SAMPLING_SEED, "attempts": attempts,
                       "accepted": len(seen), "max_attempts": MAX_ATTEMPTS,
                       "rejected_same_side": rej_same_side,
                       "rejected_missing_cross_coupling": rej_cross},
        "verification": {"problems": problems},
        "cases": cases,
    }
    out = HERE / "N01S_cases.json"
    out.write_text(json.dumps(art, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
